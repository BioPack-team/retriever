import asyncio
import math
import time
from random import random
from typing import Any

import httpx
from loguru import logger as log
from opentelemetry import trace

from retriever.type_defs import Query
from retriever.utils.calls import BASIC_CLIENT
from retriever.utils.logs import TRAPILogger
from retriever.utils.mongo import MONGO_QUEUE

tracer = trace.get_tracer("lookup.execution.tracer")


async def lookup(query: Query) -> tuple[int, dict[str, Any]]:
    """Execute a lookup query.

    Does job state updating regardless of asyncquery for easier debugging.

    Returns:
        A tuple of HTTP status code, response body.
    """
    job_id = query["job_id"]
    job_log = TRAPILogger(job_id)
    try:
        start_time = time.time()
        # TODO implement actual query handling

        job_log.info(f"got request with arg {job_id}...")
        with tracer.start_as_current_span("intermediate_step"):
            a = 0
            await asyncio.sleep(0.1)
            for i in range(10_000):
                a += i
        job_log.info("finished working.")

        chance_of_error = 0.01
        if random() < chance_of_error:
            raise ValueError("Simulated unhandled error.")

        results: list[Any] = []  # placeholder for response

        end_time = time.time()
        response = {
            "job_id": job_id,
            "status": "Complete",
            "description": f"Execution completed, obtained {len(results)} results in {math.ceil((end_time - start_time) * 1000):}ms.",
            "logs": job_log.get_logs(),
            "message": f"This'll be a TRAPI message soon! BTW the number was {a}",
        }
        MONGO_QUEUE.put("job_state", response)
        return 200, response
    except Exception:
        job_log.error(
            f"Uncaught exception during handling of job {job_id}. Job failed."
        )
        job_log.exception("Cause of error:")
        response = {
            "job_id": job_id,
            "status": "Failed",
            "description": "Execution failed due to an unhandled error. See the logs for more details.",
            "logs": job_log.get_logs(),
        }
        MONGO_QUEUE.put("job_state", response)
        return 500, response


async def async_lookup(query: Query) -> None:
    """Handle running lookup as an async query where the client receives a callback."""
    job_id = query["job_id"]
    job_log = log.bind(job_id=job_id)
    _, response = await lookup(query)
    if callback_url := query["body"].get("callback"):
        job_log.debug(f"Sending callback to `{callback_url}`...")
        try:
            callback_response = await BASIC_CLIENT.put(url=callback_url, json=response)
            callback_response.raise_for_status()
            job_log.debug("Request sent successfully.")
        except httpx.HTTPError:
            job_log.exception("Failed to make callback for async query.")
