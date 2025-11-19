import contextlib
import traceback
import uuid
from typing import Any, Literal, overload

import sentry_sdk
from fastapi import Request, Response
from opentelemetry import trace
from reasoner_pydantic import QueryGraph

from retriever.config.general import CONFIG
from retriever.lookup.lookup import async_lookup, lookup
from retriever.metakg.trapi_metakg import trapi_metakg
from retriever.types.general import APIInfo, QueryInfo
from retriever.types.trapi import (
    AsyncQueryResponseDict,
    MetaKnowledgeGraphDict,
    ResponseDict,
)
from retriever.types.trapi_pydantic import AsyncQuery as TRAPIAsyncQuery
from retriever.types.trapi_pydantic import Query as TRAPIQuery
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import telemetry
from retriever.utils.logs import TRAPILogger, structured_log_to_trapi
from retriever.utils.mongo import MONGO_CLIENT, MONGO_QUEUE

tracer = trace.get_tracer("lookup.execution.tracer")


@overload
async def make_query(
    func: Literal["lookup"],
    ctx: APIInfo,
    *,
    body: TRAPIQuery,
) -> ResponseDict: ...


@overload
async def make_query(
    func: Literal["lookup"],
    ctx: APIInfo,
    *,
    body: TRAPIAsyncQuery,
) -> AsyncQueryResponseDict: ...


@overload
async def make_query(
    func: Literal["metakg"],
    ctx: APIInfo,
    *,
    tiers: list[TierNumber],
) -> MetaKnowledgeGraphDict: ...


async def make_query(
    func: Literal["lookup", "metakg"],
    ctx: APIInfo,
    *,
    body: TRAPIQuery | TRAPIAsyncQuery | None = None,
    tiers: list[TierNumber] | None = None,  # Guaranteed to be 0 <= x <= 2
) -> ResponseDict | AsyncQueryResponseDict | MetaKnowledgeGraphDict:
    """Process a request and await its response before returning.

    Unhandled errors are handled by middleware.
    """
    job_id = uuid.uuid4().hex

    timeout: float = {
        "lookup": CONFIG.job.lookup.timeout,
        "metakg": CONFIG.job.metakg.timeout,
    }[func]
    if tiers is None:
        tiers = [0]
    if body is not None and body.parameters is not None:
        if body.parameters.timeout is not None:
            timeout = body.parameters.timeout
        if body.parameters.tiers is not None:
            tiers = body.parameters.tiers

    query = QueryInfo(
        endpoint=ctx.request.url.path,
        method=ctx.request.method,
        body=body,
        job_id=job_id,
        tiers=set(tiers),
        timeout=timeout,
    )

    with contextlib.suppress(Exception):
        contextualize_query_telemetry(query, func, ctx.background_tasks is not None)

    query_function = {"lookup": lookup, "metakg": trapi_metakg}[func]
    if func == "lookup":
        MONGO_QUEUE.put("job_state", {"job_id": job_id, "status": "Running"})

    # TRAPI Async vs Sync query (client wants callback vs. will wait)
    if ctx.background_tasks is not None:  # TRAPI Asyncquery lookup
        ctx.background_tasks.add_task(async_lookup, query=query)
        return AsyncQueryResponseDict(
            status="Accepted",
            description="Query has been queued for processing.",
            job_id=job_id,
        )
    else:  # Sync query
        status_code, response_body = await query_function(query)
        ctx.response.status_code = status_code
        return response_body


def contextualize_query_telemetry(query: QueryInfo, func: str, is_async: bool) -> None:
    """Provide some advanced information about the query for Telemetry."""
    span_tags = {
        "job_id": query.job_id,
        "job_timeout": query.timeout,
        "data_tier": str(sorted(query.tiers)),
        "query_type": func if not is_async else f"{func}-async",
    }

    body = query.body

    if (
        submitter := body is not None
        and body.model_extra is not None
        and body.model_extra.get("submitter")
    ):
        span_tags["submitter"] = str(submitter)
    else:
        span_tags["submitter"] = "not_provided"

    if body is not None and body.message.query_graph is not None:
        span_tags["qnodes"] = len(body.message.query_graph.nodes)
        if isinstance(body.message.query_graph, QueryGraph):
            span_tags["qedges"] = len(body.message.query_graph.edges)
        else:
            span_tags["qpaths"] = len(body.message.query_graph.paths)

    current_span = trace.get_current_span()
    current_span.set_attributes(span_tags)
    # Have to set separately in Sentry to make searchable tags
    sentry_sdk.set_tags(span_tags)


async def get_job_state(
    job_id: str, request: Request, response: Response
) -> dict[str, Any]:
    """Retrieve job information from MongoDB.

    Returns whole job response so it can be used by either `/asyncquery_status` or `/response`.
    """
    job: dict[str, Any] | None = None
    error: Exception | None = None
    ctx_log = TRAPILogger(job_id)

    try:
        job_dict = await MONGO_CLIENT.get_job_doc(job_id)
        if job_dict:
            ctx_log.debug(f"Got job {job_id} result from MongoDB.")
        else:
            ctx_log.debug(f"Job {job_id} not in MongoDB.")
        job = job_dict
    except Exception as e:
        ctx_log.exception(
            f"Encountered exception retrieving job {job_id} from MongoDB."
        )
        telemetry.capture_exception(e)
        error = e
    if job is None:
        job_logs = [
            log
            async for log in structured_log_to_trapi(
                MONGO_CLIENT.get_logs(job_id=job_id)
            )
        ]
        if len(job_logs) > 0:
            job = {
                "status": "Running",
                "logs": job_logs,
                "description": "Job is running.",
            }

    if job is None and error is not None:
        response.status_code = 500
        return {
            "status": "Error",
            "description": "An error occurred while attempting to retrieve job status",
            "logs": list(ctx_log.get_logs()),
            "error": str(error),
            "trace": traceback.format_exc(),
        }
    elif job is None:
        response.status_code = 404
        return {
            "status": "Not Found",
            "description": f"The job ID you provided ({job_id}) was not found. It may have expired.",
            "logs": list(ctx_log.get_logs()),
        }

    # Clean up internal-use values
    for key in ("touched", "completed"):
        if key in job:
            del job[key]

    return {
        **job,
        "response_url": f"{request.base_url}response/{job_id}",
        "logs": job.get("logs", list(ctx_log.get_logs())),
    }
