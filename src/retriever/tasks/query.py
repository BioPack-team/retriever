import traceback
import uuid
from typing import Any, Literal

from fastapi import BackgroundTasks, Request, Response
from opentelemetry import trace

from retriever.tasks.lookup import async_lookup, lookup
from retriever.tasks.metakg import metakg
from retriever.type_defs import Query
from retriever.utils import telemetry
from retriever.utils.logs import TRAPILogger, structured_log_to_trapi
from retriever.utils.mongo import MONGO_CLIENT, MONGO_QUEUE

tracer = trace.get_tracer("lookup.execution.tracer")


# @tracer.start_as_current_span("await_worker_execution")
async def make_query(
    func: Literal["lookup", "metakg"],
    request: Request,
    response: Response,  # Possibly used in the future
    body: dict[str, Any] | None = None,
    background_tasks: BackgroundTasks | None = None,
) -> dict[str, Any]:
    """Process a request and await its response before returning.

    Unhandled errors are handled by middleware.
    """
    if body is None:
        body = {}
    job_id = uuid.uuid4().hex
    query = Query(
        endpoint=request.url.path, method=request.method, body=body, job_id=job_id
    )
    query_function = {"lookup": lookup, "metakg": metakg}[func]
    if func == "lookup":
        MONGO_QUEUE.put("job_state", {"job_id": job_id, "status": "Running"})
    if background_tasks is not None:  # Asyncquery lookup
        background_tasks.add_task(async_lookup, query=query)
        return dict(
            status="Accepted",
            description="Query has been queued for processing.",
            job_id=job_id,
        )
    else:  # Sync query
        status_code, response_body = await query_function(query)
        response.status_code = status_code
        return response_body


async def get_job_state(
    job_id: str, _request: Request, response: Response
) -> dict[str, Any]:
    """Retrieve job information from MongoDB.

    Returns whole job response so it can be used by either asyncquery_status or asyncquery_response.
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
        "logs": job.get("logs", list(ctx_log.get_logs())),
    }
