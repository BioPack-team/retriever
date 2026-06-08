from __future__ import annotations

import traceback
import uuid
from http import HTTPStatus
from typing import Any, NamedTuple, cast

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
from retriever.utils import service_health, telemetry, worker
from retriever.utils.job_status import (
    NON_TERMINAL,
    TERMINAL_SUCCESS,
    to_async_lifecycle,
)
from retriever.utils.logs import TRAPILogger, structured_log_to_trapi
from retriever.utils.mongo import (
    JobDoc,
    JobStatus,
    MongoClient,
    MongoOutage,
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


def _resolve_lookup_timeout(
    body: TRAPIQuery | TRAPIAsyncQuery, tier: TierNumber
) -> float:
    """Lookup timeout: explicit `parameters.timeout` else the per-tier default."""
    custom = body.parameters.timeout if body.parameters is not None else None
    if custom is not None:
        return custom
    return {
        0: CONFIG.job.lookup.tier0_timeout,
        1: CONFIG.job.lookup.tier1_timeout,
        2: CONFIG.job.lookup.tier2_timeout,
    }[tier]


def _contextualize(query_metadata: QueryMetadata, timeout: float) -> None:
    """Tag telemetry with the resolved query metadata; failures are logged, not raised."""
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


def _record_initial_state(
    query: QueryInfo,
    ctx: APIInfo,
    query_metadata: QueryMetadata,
) -> None:
    """Persist the initial Running state; drop quietly to the server log on outage.

    Server-side log only, no TRAPI/job-log entry: the final-state write at
    lookup completion is the user-visible signal - it attaches the mongo-outage
    warning to the response. An initial-state drop just means /asyncquery_status
    can't observe the job; logging at the server gives operators the signal
    without polluting the response.
    """
    try:
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
    except MongoOutage:
        logger.warning(
            f"Initial job state for {query.job_id} dropped - MongoDB unavailable.",
            no_mongo_log=True,
        )


async def make_lookup_query(
    ctx: APIInfo,
    body: TRAPIQuery | TRAPIAsyncQuery,
) -> tuple[HTTPStatus, ResponseDict | AsyncQueryResponseDict | ErrorDetail]:
    """Process a TRAPI /query or /asyncquery request.

    Unhandled errors are handled by middleware.
    """
    job_id = uuid.uuid4().hex

    tier: TierNumber = 0
    if deprecated_tiers := body.parameters and body.parameters.tiers:
        tier = deprecated_tiers[0]
    if custom_tier := body.parameters and body.parameters.tier:
        tier = custom_tier

    timeout = _resolve_lookup_timeout(body, tier)

    body_transformed: QueryDict | AsyncQueryDict
    if isinstance(body, TRAPIQuery):
        body_transformed = QueryDict(**body.model_dump(mode="json"))
    else:
        body_transformed = AsyncQueryDict(**body.model_dump(mode="json"))

    resolved = service_health.Snapshot().select_tier(body, tier)
    if isinstance(resolved[0], HTTPStatus):
        return cast("tuple[HTTPStatus, ErrorDetail]", resolved)
    tier, extra_warnings = cast("tuple[TierNumber, list[LogEntryDict]]", resolved)

    query = QueryInfo(
        endpoint=ctx.request.url.path,
        method=ctx.request.method,
        body=body_transformed,
        job_id=job_id,
        tier=tier,
        timeout=timeout,
    )

    query_type = "lookup-async" if ctx.background_tasks is not None else "lookup"
    query_metadata = get_query_metadata(query, query_type)
    _contextualize(query_metadata, timeout)
    _record_initial_state(query, ctx, query_metadata)

    if ctx.background_tasks is not None:
        carrier = dict[str, str]()
        propagate.inject(carrier, context.get_current())
        ctx.background_tasks.add_task(
            async_lookup, query=query, ctx=carrier, extra_warnings=extra_warnings
        )
        return HTTPStatus.OK, AsyncQueryResponseDict(
            status="Accepted",
            description="Query has been queued for processing.",
            job_id=job_id,
        )

    status_code, response = await lookup(query)
    if extra_warnings:
        response["logs"] = [*(response.get("logs") or []), *extra_warnings]
    return status_code, response


async def make_metakg_query(
    ctx: APIInfo,
    tier: TierNumber | None,
) -> tuple[HTTPStatus, MetaKnowledgeGraphDict | ErrorDetail]:
    """Process a /meta_knowledge_graph request.

    Unhandled errors are handled by middleware.
    """
    job_id = uuid.uuid4().hex
    timeout = CONFIG.job.metakg.timeout

    query = QueryInfo(
        endpoint=ctx.request.url.path,
        method=ctx.request.method,
        body=None,
        job_id=job_id,
        tier=tier,
        timeout=timeout,
    )
    query_metadata = get_query_metadata(query, "metakg")
    _contextualize(query_metadata, timeout)

    return await trapi_metakg(query)


async def make_metadata_query(
    ctx: APIInfo,
    tier: TierNumber,
) -> tuple[HTTPStatus, DINGOMetadata | ErrorDetail]:
    """Process a /metadata/tier_N request.

    Unhandled errors are handled by middleware.
    """
    job_id = uuid.uuid4().hex
    timeout = CONFIG.job.metakg.timeout

    query = QueryInfo(
        endpoint=ctx.request.url.path,
        method=ctx.request.method,
        body=None,
        job_id=job_id,
        tier=tier,
        timeout=timeout,
    )
    query_metadata = get_query_metadata(query, "metadata")
    _contextualize(query_metadata, timeout)

    return await get_metadata(query)


def get_query_metadata(query: QueryInfo, query_type: str) -> QueryMetadata:
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
        query_type=query_type,
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
    "Success": "Job is complete.",
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
) -> tuple[HTTPStatus, dict[str, Any]]:
    """Return an AsyncQueryStatusResponse for in-progress or missing job.

    "Running" if the row exists or there's log evidence, "Error" if the
    fetch raised, otherwise "Not Found".
    """
    logs = await job_logs(job_id)
    if exists or len(logs) > 0:
        return HTTPStatus.OK, {
            "status": "Running",
            "logs": logs,
            "description": "Job is running.",
        }
    if error is not None:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "status": "Error",
            "description": "An error occurred while attempting to retrieve job status",
            "logs": list(ctx_log.get_logs()),
            "error": str(error),
            "trace": traceback.format_exc(),
        }
    return HTTPStatus.NOT_FOUND, {
        "status": "Not Found",
        "description": f"The provided job ID ({job_id}) was not found. It may have expired.",
        "logs": list(ctx_log.get_logs()),
    }


async def terminal_status_payload(
    job_id: str, request: Request, status_doc: JobStatus
) -> dict[str, Any]:
    """Build an AsyncQueryStatusResponse for a terminal job.

    Stored status is an outcome shortcode (Success, QueryNotTraversable, ...);
    we emit its TRAPI 1.6 lifecycle equivalent (Completed / Failed) on the
    response. Description prefers the per-job string written by the lookup
    engine (which can include constraint names, edge ids, etc.) and falls
    back to a status-keyed default for legacy docs that pre-date the field.
    """
    job_status = status_doc["status"]
    stored_description = status_doc.get("description")
    description = stored_description or _TERMINAL_STATUS_DESCRIPTIONS.get(
        job_status,
        "Job is complete."
        if job_status in TERMINAL_SUCCESS
        else f"Job ended with status: {job_status}.",
    )
    return {
        "status": to_async_lifecycle(job_status),
        "description": description,
        "logs": await job_logs(job_id),
        "response_url": f"{request.base_url}response/{job_id}",
    }


def unpack_doc(job: JobDoc) -> dict[str, Any]:
    """Decompress + unpack the stored doc bytes, return {} if absent."""
    doc_bytes = job.get("doc")
    if doc_bytes is None:
        return {}
    return ormsgpack.unpackb(ZSTD_DECOMPRESSOR.decompress(doc_bytes))


async def get_job_status(
    job_id: str, request: Request
) -> tuple[HTTPStatus, dict[str, Any]]:
    """Get the current job status and return an AsyncQueryStatusResponse."""
    ctx_log = TRAPILogger(job_id)
    status_doc, error = await fetch_job_status(job_id, ctx_log)

    job_status = (status_doc or {}).get("status") or "Running"
    if status_doc is None or job_status in NON_TERMINAL:
        return await in_progress_payload(
            job_id, ctx_log, exists=status_doc is not None, error=error
        )

    return HTTPStatus.OK, await terminal_status_payload(job_id, request, status_doc)


async def get_job_response(
    job_id: str, request: Request
) -> tuple[HTTPStatus, dict[str, Any]]:
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
        payload = await terminal_status_payload(job_id, request, status_doc)
        if job is not None and job.get("doc") is not None:
            query = unpack_doc(job)
            payload = {
                **{k: v for k, v in query.items() if v is not None},
                **payload,
            }
        return HTTPStatus.OK, payload

    if job is None or job.get("doc") is None:
        return HTTPStatus.OK, await terminal_status_payload(job_id, request, status_doc)

    response = unpack_doc(job)
    return HTTPStatus.OK, {
        **response,
        "status": job_status,
        "response_url": f"{request.base_url}response/{job_id}",
        "logs": response.get("logs", []),
    }
