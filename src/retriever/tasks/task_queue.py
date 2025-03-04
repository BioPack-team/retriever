import multiprocessing
import os
import queue
import traceback
from datetime import datetime
from typing import Any, Literal, override

import redis.asyncio as redis
from fastapi import Request, Response
from loguru import logger as log
from opentelemetry import trace
from saq import Job
from saq.queue.redis import RedisQueue
from saq.types import Context, SettingsDict

from retriever.config.general import CONFIG
from retriever.tasks.lookup import lookup
from retriever.type_defs import Query
from retriever.utils import telemetry
from retriever.utils.logs import add_context_trapi_sink
from retriever.utils.mongo import MongoClient

tracer = trace.get_tracer("lookup.execution.tracer")


class RetrieverQueue(RedisQueue):
    """A custom queue with some ammenities for Retriever behavior."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize parent class and some extra functionality."""
        super().__init__(*args, **kwargs)
        self.mongo_queue: multiprocessing.Queue[tuple[str, dict[str, Any]]] | None = (
            None
        )

    @override
    @classmethod
    def from_url(
        cls: type["RetrieverQueue"], url: str, **kwargs: Any
    ) -> "RetrieverQueue":
        """Create a queue with a redis url a name.

        Overridden from original to prevent type problems.
        """
        return cls(redis.from_url(url), **kwargs)

    @override
    async def enqueue(self, job_or_func: str | Job, **kwargs: Any) -> Job | None:
        """Ensure desired defaults for all enqueued jobs."""
        # TODO: set defaults (timeout, etc.)
        return await super().enqueue(job_or_func, **kwargs)

    @override
    def serialize(self, job: Job) -> bytes | str:
        """Serialize job to an external db for persistence."""
        # try:
        #     if self.mongo_queue is not None:
        #         self.mongo_queue.put_nowait(job.to_dict())
        # except asyncio.QueueFull:
        #     log.error(f"Failed to serialize job {job.key}. Serialization queue full.")
        if self.mongo_queue is not None:
            try:
                self.mongo_queue.put_nowait(("update_job_doc", job.to_dict()))
            except (queue.Empty, ValueError):
                log.error(
                    f"Failed to serialize job {job.key}. MongoDB queue full/closed."
                )
        return super().serialize(job)


async def before_enqueue(job: Job) -> None:
    """Additional hooks for mutating job prior to enqueue."""
    telemetry.inject_context(job)


def before_process(ctx: Context) -> None:
    """Hook to run some setup before a job is started."""
    job = ctx.get("job")
    if not job:
        log.warning("Job process started without job in context.")
        return

    # Ensure the job is being executed with proper context in Otel
    telemetry.align_context(job)


retriever_queue = RetrieverQueue.from_url(f"redis://{CONFIG.redis.host}")
retriever_queue.register_before_enqueue(before_enqueue)


async def worker_startup(_ctx: Context) -> None:
    """Initial setup when worker starts."""
    pass


async def worker_shutdown(_ctx: Context) -> None:
    """Tasks to run when worker stops."""
    log.info(f"Worker process {os.getpid()} stopped.")
    await log.complete()


SETTINGS = SettingsDict(
    queue=retriever_queue,
    concurrency=10,  # number of jobs worker may process concurrently
    startup=worker_startup,
    shutdown=worker_shutdown,
    before_process=before_process,
    functions=[lookup],
)


@tracer.start_as_current_span("await_worker_execution")
async def make_query(
    func: Literal["lookup", "metakg"],
    request: Request,
    response: Response,
    body: dict[str, Any] | None,
    mode: Literal["sync", "async"] = "sync",
) -> dict[str, Any]:
    """Process a request and await its response before returning.

    Unhandled errors are propagated by SAQ and handled by middleware.
    """
    query = Query(endpoint=request.url.path, method=request.method, body=body)
    if mode == "async":
        job = await retriever_queue.enqueue(func, query=query)
        if not job:
            response.status_code = 500
            return dict(
                status="Error",
                description="Query could not be added to queue at this time.",
            )
        return dict(
            status="Accepted",
            description="Query has been queued for processing.",
            job_id=job.key,
        )
    elif mode == "sync":
        return await retriever_queue.apply(func, query=query)


async def get_job_state(
    job_id: str, request: Request, response: Response, mongo_client: MongoClient
) -> dict[str, Any]:
    """Retrieve job information either from SAQ, or from MongoDB if it's been swept from SAQ.

    Returns whole job response so it can be used by either asyncquery_status or asyncquery_response.
    """
    # TODO more strict output type once using reasoner-pydantic/other

    job: dict[str, Any] | None = None
    error: Exception | None = None
    ctx_log, logs, remove_sink = add_context_trapi_sink(
        f"get_job_state:{datetime.now()}:{job_id}"
    )
    try:
        ctx_log.debug(f"Checking SAQ for job {job_id}...")
        saq_job = await retriever_queue.job(job_id)
        if saq_job:
            ctx_log.debug(f"Got job {job_id} state from SAQ.")
            job = saq_job.to_dict()
    except Exception as e:
        ctx_log.exception(
            f"Encountered exception retrieving job {job_id} from SAQ. Trying MongoDB..."
        )
        telemetry.capture_exception(e)
        error = e
    if job is None:
        try:
            ctx_log.debug(f"Job {job_id} not in SAQ. Checking MongoDB...")
            job_dict = await mongo_client.get_job_doc(job_id)
            if job_dict:
                ctx_log.debug(f"Got job {job_id} state from MongoDB.")
            else:
                ctx_log.debug(f"Job {job_id} not in MongoDB.")
            job = job_dict
        except Exception as e:
            ctx_log.exception(
                f"Encountered exception retrieving job {job_id} from MongoDB."
            )
            telemetry.capture_exception(e)
            error = e

    remove_sink()

    if job is None and error is not None:
        response.status_code = 500
        return {
            "status": "Error",
            "description": "An error occurred while attempting to retrieve job status",
            "logs": list(logs),
            "error": str(error),
            "trace": traceback.format_exc(),
        }
    elif job is None:
        response.status_code = 404
        return {
            "status": "Not Found",
            "description": f"The job ID you provided ({job_id}) was not found. It may have expired.",
            "logs": list(logs),
        }
    status = job.get("status", "failed")
    if status in ("failed", "aborting", "aborted"):
        message = {
            "status": "Failed",
            "descripton": "The job failed due to an error, see attached info and try again.",
            "logs": job.get("meta", {}).get("logs", list(logs)),
        }
        if job.get("error", None) is not None:
            message["trace"] = job["error"]
        return message
    elif status == "complete":
        # Ensure dev-friendly item order :)
        message = {
            "status": "Completed",
            "response_url": f"{request.base_url}asyncquery_response/{job_id}",
            **{
                key: value
                for key, value in job["result"].items()
                if key not in ("status", "response_url")
            },
        }
        return message

    return {
        "status": {
            "new": "Queued",
            "queued": "Queued",
            "active": "Running",
        }[status],
        "description": {
            "new": "Job is newly created and entering the queue.",
            "queued": "Job has been queued and is awaiting an open worker.",
            "active": "Job is running.",
        }[status],
        "logs": job.get("meta", {}).get("logs", list(logs)),
    }
