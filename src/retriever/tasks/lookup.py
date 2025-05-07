import asyncio
import math
import time
from random import random
from typing import Any

import httpx
from loguru import logger as log
from opentelemetry import trace
from reasoner_pydantic import (
    AsyncQuery,
    KnowledgeGraph,
    Message,
    Response,
    Results,
)

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.type_defs import QueryInfo
from retriever.utils.calls import BASIC_CLIENT
from retriever.utils.logs import TRAPILogger
from retriever.utils.mongo import MONGO_QUEUE

tracer = trace.get_tracer("lookup.execution.tracer")


async def lookup(query: QueryInfo) -> tuple[int, Response]:
    """Execute a lookup query.

    Does job state updating regardless of asyncquery for easier debugging.

    Returns:
        A tuple of HTTP status code, response body.
    """
    job_id = query.job_id
    job_log = TRAPILogger(job_id)

    if query.body is None:
        raise TypeError(
            "`lookup()` received body of type None, should have received Query or AsyncQuery."
        )

    # NOTE: Overrides other responses, if you need some base info
    # on sub-levels, you'll want to rewrite how this is used
    base_response: dict[str, Any] = dict(
        biolink_version=OPENAPI_CONFIG.x_translator.biolink_version,
        schema_version=OPENAPI_CONFIG.x_trapi.version,
        workflow=query.body.workflow,
    )

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
        # ^ Above is placeholder

        end_time = time.time()
        response = {
            "job_id": job_id,
            "status": "Complete",
            "description": f"Execution completed, obtained {len(results)} results in {math.ceil((end_time - start_time) * 1000):}ms.",
            "logs": job_log.get_logs(),
            "message": Message(
                query_graph=query.body.message.query_graph,
                knowledge_graph=KnowledgeGraph(),
                results=Results(),
            ),
            **base_response,
        }
        MONGO_QUEUE.put("job_state", response)
        return 200, Response.model_validate(response)

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
            **base_response,
        }
        MONGO_QUEUE.put("job_state", response)
        return 500, Response.model_validate(response)


async def async_lookup(query: QueryInfo) -> None:
    """Handle running lookup as an async query where the client receives a callback."""
    job_id = query.job_id
    job_log = log.bind(job_id=job_id)
    _, response = await lookup(query)
    if isinstance(query.body, AsyncQuery):
        job_log.debug(f"Sending callback to `{query.body.callback}`...")
        try:
            callback_response = await BASIC_CLIENT.put(
                url=str(query.body.callback), json=response
            )
            callback_response.raise_for_status()
            job_log.debug("Request sent successfully.")
        except httpx.HTTPError:
            job_log.exception("Failed to make callback for async query.")
