import asyncio
import math
import time

import httpx
from loguru import logger as log
from opentelemetry import trace
from reasoner_pydantic import (
    AsyncQuery,
    HashableSequence,
    KnowledgeGraph,
    LogEntry,
    Message,
    Response,
    Results,
)

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.tasks.lookup.qgx import execute_query_graph
from retriever.tasks.lookup.validate import validate
from retriever.type_defs import QueryInfo
from retriever.utils.calls import BASIC_CLIENT
from retriever.utils.logs import TRAPILogger
from retriever.utils.mongo import MONGO_QUEUE

tracer = trace.get_tracer("lookup.execution.tracer")


def tracked_response(status: int, body: Response) -> tuple[int, Response]:
    """Utility function for response handling."""
    MONGO_QUEUE.put("job_state", body.model_dump())
    return status, body


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

    # Set up a basic response to avoid some repetition
    response = Response(
        message=Message(
            query_graph=query.body.message.query_graph,
            knowledge_graph=KnowledgeGraph(),
            results=Results(),
        ),
        biolink_version=OPENAPI_CONFIG.x_translator.biolink_version,
        schema_version=OPENAPI_CONFIG.x_trapi.version,
        workflow=query.body.workflow,
        job_id=job_id,  # pyright:ignore[reportCallIssue] Extra is allowed
    )

    try:
        start_time = time.time()
        job_log.info(f"Begin processing job {job_id}.")

        # Query graph validation that isn't handled by reasoner_pydantic
        validation_problems = validate(query.body.message.query_graph)
        if validation_problems:
            job_log.error(
                f"Query validation encountered {len(validation_problems)} errors. Error logs to follow:"
            )
            for problem in validation_problems:
                job_log.error(f"Validation Error: {problem}")
            job_log.error("Due to the above errors, your query terminates.")

            response.status = "QueryNotTraversable"
            response.description = (
                "Query terminated due to validation errors. See logs for more details."
            )
            response.logs = HashableSequence(job_log.get_logs())
            return tracked_response(422, response)

        # TODO: make this `await graph_supported(query.body.meesage.query_graph, job_id)`
        # Do a linear iteration of all edges and check that metaEdges exist for each
        graph_supported, logs = True, list[LogEntry]()
        if not graph_supported:
            job_log.log_deque.extend(logs)
            job_log.error(
                "Query cannot be traversed due to missing metaEdges. See above logs for details."
            )
            response.status = "QueryNotTraversable"
            response.description = "Query cannot be traversed due to missing metaEdges. See logs for details."
            return tracked_response(424, response)

        # Execute query graph
        results, kgraph, logs = await execute_query_graph(query.body, job_id)
        job_log.log_deque.extend(logs)

        # TODO: cleanup (subclass, is_set, kg prune)

        # v Placeholder functionality
        with tracer.start_as_current_span("intermediate_step"):
            a = 0
            await asyncio.sleep(0.1)
            for i in range(10_000):
                a += i
        job_log.info("finished working.")
        # ^ Above is placeholder

        end_time = time.time()

        response.status = "Complete"
        response.description = f"Execution completed, obtained {len(results)} results in {math.ceil((end_time - start_time) * 1000):}ms."
        response.logs = HashableSequence(job_log.get_logs())
        response.message.results = results
        response.message.knowledge_graph = kgraph
        return tracked_response(200, response)

    except Exception:
        job_log.error(
            f"Uncaught exception during handling of job {job_id}. Job failed."
        )
        job_log.exception("Cause of error:")
        response.status = "Failed"
        response.description = (
            "Execution failed due to an unhandled error. See the logs for more details."
        )
        response.logs = HashableSequence(job_log.get_logs())
        return tracked_response(500, response)


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
