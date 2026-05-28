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
    LogEntryDict,
    MetaKnowledgeGraphDict,
    QueryDict,
    ResponseDict,
)
from retriever.types.trapi_pydantic import AsyncQuery as TRAPIAsyncQuery
from retriever.types.trapi_pydantic import Query as TRAPIQuery
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import telemetry, worker
from retriever.utils.job_status import NON_TERMINAL, TERMINAL_SUCCESS
from retriever.utils.logs import TRAPILogger, structured_log_to_trapi
from retriever.utils.mongo import (
    JobDoc,
    JobStatus,
    MongoClient,
    MongoQueue,
    QueryState,
)

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
                worker_pid=worker.get_pid(),
                worker_started_at=worker.get_started_at(),
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


_TERMINAL_STATUS_DESCRIPTIONS: dict[str, str] = {
    "Complete": "Job is complete.",
    "Failed": "Job failed.",
    "QueryNotTraversable": "Query graph was not traversable.",
    "UnsupportedConstraint": "Query graph contained an unsupported constraint.",
}


async def job_logs(job_id: str) -> list[LogEntryDict]:
    """Return all stored logs for a job, formatted as TRAPI LogEntries."""
    return [
        log
        async for log in structured_log_to_trapi(MongoClient().get_logs(job_id=job_id))
    ]


async def fetch_job_status(
    job_id: str, ctx_log: TRAPILogger
) -> tuple[JobStatus | None, Exception | None]:
    """Read the JobStatus + bump TTL."""
    try:
        status_doc = await MongoClient().get_job_status(job_id, touch=True)
        if status_doc is not None:
            ctx_log.debug(f"Got job {job_id} status from MongoDB.", no_mongo_log=True)
    except Exception as e:
        ctx_log.exception(
            f"Encountered exception retrieving job {job_id} status from MongoDB.",
            no_mongo_log=True,
        )
        telemetry.capture_exception(e)
        return None, e
    return status_doc, None


async def in_progress_payload(
    job_id: str,
    ctx_log: TRAPILogger,
    exists: bool,
    error: Exception | None,
) -> tuple[int, dict[str, Any]]:
    """Return an AsyncQueryStatusResponse for in-progress or missing job.

    "Running" if the row exists or there's log evidence, "Error" if the
    fetch raised, otherwise "Not Found".
    """
    logs = await job_logs(job_id)
    if exists or len(logs) > 0:
        return 200, {
            "status": "Running",
            "logs": logs,
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
    return 404, {
        "status": "Not Found",
        "description": f"The provided job ID ({job_id}) was not found. It may have expired.",
        "logs": list(ctx_log.get_logs()),
    }


async def terminal_status_payload(
    job_id: str, request: Request, job_status: str
) -> dict[str, Any]:
    """Build an AsyncQueryStatusResponse a terminal job."""
    return {
        "status": job_status,
        "description": _TERMINAL_STATUS_DESCRIPTIONS.get(
            job_status,
            "Job is complete."
            if job_status in TERMINAL_SUCCESS
            else f"Job ended with status: {job_status}.",
        ),
        "logs": await job_logs(job_id),
        "response_url": f"{request.base_url}response/{job_id}",
    }


def unpack_doc(job: JobDoc) -> dict[str, Any]:
    """Decompress + unpack the stored doc bytes, return {} if absent."""
    doc_bytes = job.get("doc")
    if doc_bytes is None:
        return {}
    return ormsgpack.unpackb(ZSTD_DECOMPRESSOR.decompress(doc_bytes))


async def get_job_status(job_id: str, request: Request) -> tuple[int, dict[str, Any]]:
    """Get the current job status and return an AsyncQueryStatusResponse."""
    ctx_log = TRAPILogger(job_id)
    status_doc, error = await fetch_job_status(job_id, ctx_log)

    job_status = (status_doc or {}).get("status") or "Running"
    if status_doc is None or job_status in NON_TERMINAL:
        return await in_progress_payload(
            job_id, ctx_log, exists=status_doc is not None, error=error
        )

    return 200, await terminal_status_payload(job_id, request, job_status)


async def get_job_response(job_id: str, request: Request) -> tuple[int, dict[str, Any]]:
    """Return a Response if completed, or AsyncQueryStatusResponse otherwise."""
    ctx_log = TRAPILogger(job_id)
    status_doc, error = await fetch_job_status(job_id, ctx_log)

    job_status = (status_doc or {}).get("status") or "Running"
    if status_doc is None or job_status in NON_TERMINAL:
        return await in_progress_payload(
            job_id, ctx_log, exists=status_doc is not None, error=error
        )

    # Terminal status: get the response body, otherwise build a status response.
    job: JobDoc | None = None
    try:
        job = await MongoClient().get_job_doc(job_id)
    except Exception as e:
        ctx_log.exception(
            f"Encountered exception retrieving job {job_id} from MongoDB.",
            no_mongo_log=True,
        )
        telemetry.capture_exception(e)

    # If job is abandoned, respond with the query
    if status_doc.get("abandoned"):
        payload = await terminal_status_payload(job_id, request, job_status)
        if job is not None and job.get("doc") is not None:
            query = unpack_doc(job)
            payload = {
                **{k: v for k, v in query.items() if v is not None},
                **payload,
            }
        return 200, payload

    if job is None or job.get("doc") is None:
        return 200, await terminal_status_payload(job_id, request, job_status)

    response = unpack_doc(job)
    return 200, {
        **response,
        "status": job_status,
        "response_url": f"{request.base_url}response/{job_id}",
        "logs": response.get("logs", []),
    }
