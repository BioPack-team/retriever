import traceback
import uuid
from typing import Any, Literal, overload

from fastapi import Request, Response
from opentelemetry import trace
from reasoner_pydantic import AsyncQuery as TRAPIAsyncQuery
from reasoner_pydantic import AsyncQueryResponse as TRAPIAsyncQueryResponse
from reasoner_pydantic import MetaKnowledgeGraph as TRAPIMetaKnowledgeGraph
from reasoner_pydantic import Query as TRAPIQuery
from reasoner_pydantic import Response as TRAPIResponse

from retriever.tasks.lookup.lookup import async_lookup, lookup
from retriever.tasks.metakg import metakg
from retriever.type_defs import APIInfo, QueryInfo, TierNumber
from retriever.utils import telemetry
from retriever.utils.logs import TRAPILogger, structured_log_to_trapi
from retriever.utils.mongo import MONGO_CLIENT, MONGO_QUEUE

tracer = trace.get_tracer("lookup.execution.tracer")


@overload
async def make_query(
    func: Literal["lookup"],
    ctx: APIInfo,
    tier: list[int],
    body: TRAPIQuery,
) -> TRAPIResponse: ...


@overload
async def make_query(
    func: Literal["lookup"],
    ctx: APIInfo,
    tier: list[int],
    body: TRAPIAsyncQuery,
) -> TRAPIAsyncQueryResponse: ...


@overload
async def make_query(
    func: Literal["metakg"],
    ctx: APIInfo,
    tier: list[int],
) -> TRAPIMetaKnowledgeGraph: ...


async def make_query(
    func: Literal["lookup", "metakg"],
    ctx: APIInfo,
    tier: list[TierNumber],  # Guaranteed to be 0 <= x <= 2
    body: TRAPIQuery | TRAPIAsyncQuery | None = None,
) -> TRAPIResponse | TRAPIAsyncQueryResponse | TRAPIMetaKnowledgeGraph:
    """Process a request and await its response before returning.

    Unhandled errors are handled by middleware.
    """
    # TODO: Data tier selection
    # For lookup, this should control which tiers to use:
    #   0 is fired off independently, while 1/2 are just used in metakg checks in QGX
    # For metakg, this should control reporting metakg relative to tiers
    job_id = uuid.uuid4().hex
    query = QueryInfo(
        endpoint=ctx.request.url.path,
        method=ctx.request.method,
        body=body,
        job_id=job_id,
        tier=set(tier),
    )
    query_function = {"lookup": lookup, "metakg": metakg}[func]
    if func == "lookup":
        MONGO_QUEUE.put("job_state", {"job_id": job_id, "status": "Running"})
    if ctx.background_tasks is not None:  # Asyncquery lookup
        ctx.background_tasks.add_task(async_lookup, query=query)
        return TRAPIAsyncQueryResponse(
            status="Accepted",
            description="Query has been queued for processing.",
            job_id=job_id,
        )
    else:  # Sync query
        status_code, response_body = await query_function(query)
        ctx.response.status_code = status_code
        return response_body


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
