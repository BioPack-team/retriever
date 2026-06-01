import traceback
import uuid
from typing import Any, Literal, NamedTuple, overload

import ormsgpack
import sentry_sdk
import zstandard
from fastapi import Request
from loguru import logger
from opentelemetry import context, propagate, trace

from retriever.config.general import CONFIG
from retriever.lookup.lookup import async_lookup, lookup
from retriever.lookup.utils import get_submitter
from retriever.metadata.metadata import get_metadata
from retriever.metadata.trapi_metakg import trapi_metakg
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import APIInfo, ErrorDetail, QueryInfo
from retriever.types.trapi import (
    AsyncQueryDict,
    AsyncQueryResponseDict,
    MetaKnowledgeGraphDict,
    QueryDict,
    ResponseDict,
)
from retriever.types.trapi_pydantic import AsyncQuery as TRAPIAsyncQuery
from retriever.types.trapi_pydantic import Query as TRAPIQuery
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import telemetry
from retriever.utils.logs import TRAPILogger, structured_log_to_trapi
from retriever.utils.mongo import JobDocs, MongoClient, MongoQueue, QueryState

tracer = trace.get_tracer("lookup.execution.tracer")
MONGO_QUEUE = MongoQueue()
ZSTD_COMPRESSOR = zstandard.ZstdCompressor()
ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()


class QueryMetadata(NamedTuple):
    """Metadata about a query."""

    job_id: str
    job_timeout: float
    data_tier: int | None
    query_type: str
    submitter: str
    qnodes: int
    qedges: int
    qpaths: int


@overload
async def make_query(
    func: Literal["lookup"],
    ctx: APIInfo,
    *,
    body: TRAPIQuery,
) -> tuple[int, ResponseDict]: ...


@overload
async def make_query(
    func: Literal["lookup"],
    ctx: APIInfo,
    *,
    body: TRAPIAsyncQuery,
) -> tuple[int, AsyncQueryResponseDict]: ...


@overload
async def make_query(
    func: Literal["metakg"],
    ctx: APIInfo,
    *,
    tier: TierNumber | None,
) -> tuple[int, MetaKnowledgeGraphDict]: ...


@overload
async def make_query(
    func: Literal["metadata"],
    ctx: APIInfo,
    *,
    tier: TierNumber,
) -> tuple[int, DINGOMetadata]: ...


async def make_query(
    func: Literal["lookup", "metakg", "metadata"],
    ctx: APIInfo,
    *,
    body: TRAPIQuery | TRAPIAsyncQuery | None = None,
    tier: TierNumber | None = None,  # Guaranteed to be 0 <= x <= 2
) -> tuple[
    int,
    ResponseDict
    | AsyncQueryResponseDict
    | MetaKnowledgeGraphDict
    | DINGOMetadata
    | ErrorDetail,
]:
    """Process a request and await its response before returning.

    Unhandled errors are handled by middleware.
    """
    job_id = uuid.uuid4().hex

    if tier is None and func != "metakg":
        tier = 0
    if deprecated_tiers := body and body.parameters and body.parameters.tiers:
        tier = deprecated_tiers[0]
    if custom_tier := body and body.parameters and body.parameters.tier:
        tier = custom_tier

    custom_timeout = (
        body.parameters.timeout
        if body is not None and body.parameters is not None
        else None
    )
    timeout = custom_timeout
    if timeout is None and func in ("metakg", "metadata"):
        timeout = CONFIG.job.metakg.timeout
    elif timeout is None:
        timeout = {
            0: CONFIG.job.lookup.tier0_timeout,
            1: CONFIG.job.lookup.tier1_timeout,
            2: CONFIG.job.lookup.tier2_timeout,
        }[tier or 0]

    body_transformed: QueryDict | AsyncQueryDict | None = None
    if body is not None:
        if isinstance(body, TRAPIQuery):
            body_transformed = QueryDict(**body.model_dump(mode="json"))
        else:
            body_transformed = AsyncQueryDict(**body.model_dump(mode="json"))

    query = QueryInfo(
        endpoint=ctx.request.url.path,
        method=ctx.request.method,
        body=body_transformed,
        job_id=job_id,
        tier=tier,
        timeout=timeout,
    )

    query_metadata = get_query_metadata(query, func, ctx.background_tasks is not None)
    with logger.catch(
        Exception,
        level="ERROR",
        message="Error while attempting to contextualize telemetry to query.",
    ):
        contextualize_query_telemetry(
            {
                **query_metadata._asdict(),
                "job_timeout": timeout,
                "data_tier": query_metadata,
            }
        )

    query_function = {
        "lookup": lookup,
        "metakg": trapi_metakg,
        "metadata": get_metadata,
    }[func]
    if func == "lookup":
        MONGO_QUEUE.put(
            "job_state",
            QueryState(
                query=ZSTD_COMPRESSOR.compress(ormsgpack.packb(query.body)),
                status="Running",
                is_async=ctx.background_tasks is not None,
                **{
                    k: v
                    for k, v in query_metadata._asdict().items()
                    if k != "query_type"
                },
            ),
        )

    # TRAPI Async vs Sync query (client wants callback vs. will wait)
    if ctx.background_tasks is not None:  # TRAPI Asyncquery lookup
        carrier = dict[str, str]()
        propagate.inject(carrier, context.get_current())
        ctx.background_tasks.add_task(async_lookup, query=query, ctx=carrier)
        return 200, AsyncQueryResponseDict(
            status="Accepted",
            description="Query has been queued for processing.",
            job_id=job_id,
        )
    else:  # Sync query
        status_code, response_body = await query_function(query)
        return status_code, response_body


def get_query_metadata(query: QueryInfo, func: str, is_async: bool) -> QueryMetadata:
    """Obtain useful metrics about the query."""
    qnodes, qedges, qpaths = 0, 0, 0
    body = query.body
    if (
        body is not None
        and "query_graph" in body["message"]
        and body["message"]["query_graph"]
    ):
        qnodes = len(body["message"]["query_graph"]["nodes"])
        if "edges" in body["message"]["query_graph"]:
            qedges = len(body["message"]["query_graph"]["edges"])
        else:
            qpaths = len(body["message"]["query_graph"]["paths"])

    return QueryMetadata(
        job_id=query.job_id,
        job_timeout=query.timeout,
        data_tier=query.tier,
        query_type=func if not is_async else f"{func}-async",
        submitter=get_submitter(query),
        qnodes=qnodes,
        qedges=qedges,
        qpaths=qpaths,
    )


def contextualize_query_telemetry(info: dict[str, Any]) -> None:
    """Provide some advanced information about the query for Telemetry."""
    current_span = trace.get_current_span()
    current_span.set_attributes(info)
    # Have to set separately in Sentry to make searchable tags
    sentry_sdk.set_tags(info)


async def get_job_state(job_id: str, request: Request) -> tuple[int, dict[str, Any]]:
    """Retrieve job information from MongoDB.

    Returns whole job response so it can be used by either `/asyncquery_status` or `/response`.
    """
    job: JobDocs | None = None
    error: Exception | None = None
    ctx_log = TRAPILogger(job_id)

    try:
        job = await MongoClient().get_job_doc(job_id)
        if job is not None:
            ctx_log.debug(f"Got job {job_id} response from MongoDB.", no_mongo_log=True)
    except Exception as e:
        ctx_log.exception(
            f"Encountered exception retrieving job {job_id} from MongoDB.",
            no_mongo_log=True,
        )
        telemetry.capture_exception(e)
        error = e
    if job is None or job.get("completed") is None:
        job_logs = [
            log
            async for log in structured_log_to_trapi(
                MongoClient().get_logs(job_id=job_id)
            )
        ]
        if job is not None or len(job_logs) > 0:
            return 200, {
                "status": "Running",
                "logs": job_logs,
                "description": "Job is running.",
            }

        if error is not None:
            return 500, {
                "status": "Error",
                "description": "An error occurred while attempting to retrieve job status",
                "logs": list(ctx_log.get_logs()),
                "error": str(error),
                "trace": traceback.format_exc(),
            }

        # Otherwise we have no evidence job exists
        return 404, {
            "status": "Not Found",
            "description": f"The job ID you provided ({job_id}) was not found. It may have expired.",
            "logs": list(ctx_log.get_logs()),
        }

    # Job is terminal
    response_bytes = job.get("doc")
    response: dict[str, Any] = (
        ormsgpack.unpackb(ZSTD_DECOMPRESSOR.decompress(response_bytes))
        if response_bytes is not None
        else {}
    )
    return 200, {
        **response,
        "response_url": f"{request.base_url}response/{job_id}",
        "logs": response.get("logs", []),
    }
