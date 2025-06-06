import math
import time

import bmt
import httpx
from loguru import logger as log
from opentelemetry import trace
from reasoner_pydantic import (
    AsyncQuery,
    AuxiliaryGraphs,
    HashableSequence,
    KnowledgeGraph,
    LogEntry,
    LogLevel,
    Message,
    QueryGraph,
    Response,
    Results,
)

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.data_tiers.tier_0 import Tier0Query
from retriever.tasks.lookup.qgx import QueryGraphExecutor
from retriever.tasks.lookup.utils import expand_qnode_categories
from retriever.tasks.lookup.validate import validate
from retriever.type_defs import LookupArtifacts, QueryInfo
from retriever.utils.calls import BASIC_CLIENT
from retriever.utils.logs import TRAPILogger, trapi_level_to_int
from retriever.utils.mongo import MONGO_QUEUE
from retriever.utils.trapi import prune_kg

tracer = trace.get_tracer("lookup.execution.tracer")
biolink = bmt.Toolkit()


async def async_lookup(query: QueryInfo) -> None:
    """Handle running lookup as an async query where the client receives a callback."""
    job_id = query.job_id
    job_log = log.bind(job_id=job_id)
    _, response = await lookup(query)
    if isinstance(query.body, AsyncQuery):
        job_log.debug(f"Sending callback to `{query.body.callback}`...")
        try:
            callback_response = await BASIC_CLIENT.put(
                url=str(query.body.callback), json=response.model_dump()
            )
            callback_response.raise_for_status()
            job_log.debug("Request sent successfully.")
        except httpx.HTTPError:
            job_log.exception("Failed to make callback for async query.")


async def lookup(query: QueryInfo) -> tuple[int, Response]:
    """Execute a lookup query.

    Does job state updating regardless of asyncquery for easier debugging.

    Returns:
        A tuple of HTTP status code, response body.
    """
    if query.body is None:
        raise TypeError(
            "Received body of type None, should have received Query or AsyncQuery."
        )
    job_id, job_log, response = initialize_lookup(query)

    try:
        start_time = time.time()
        job_log.info(f"Begin processing job {job_id}.")

        qgraph = query.body.message.query_graph
        if qgraph is None:
            raise ValueError("Query Graph is None.")

        # Query graph validation that isn't handled by reasoner_pydantic
        if not passes_validation(qgraph, response, job_log):
            return tracked_response(422, response)

        expanded_qgraph = expand_qnode_categories(qgraph.model_copy(), job_log)

        if not await qgraph_supported(expanded_qgraph, response, job_log):
            return tracked_response(424, response)

        results, kgraph, aux_graphs, logs = await run_tiered_lookups(
            query, expanded_qgraph
        )
        job_log.log_deque.extend(logs)

        # TODO: cleanup (subclass, is_set)
        job_log.info(f"Collected {len(results)} results from query tasks.")
        prune_kg(results, kgraph, aux_graphs, job_log)

        # v Placeholder functionality
        # with tracer.start_as_current_span("intermediate_step"):
        #     a = 0
        #     await asyncio.sleep(0.1)
        #     for i in range(10_000):
        #         a += i
        # job_log.info("finished working.")
        # ^ Above is placeholder

        end_time = time.time()
        finish_msg = f"Execution completed, obtained {len(results)} results in {math.ceil((end_time - start_time) * 1000):}ms."
        job_log.info(finish_msg)

        desired_log_level = trapi_level_to_int(query.body.log_level or LogLevel.DEBUG)
        response.status = "Complete"
        response.description = finish_msg
        response.logs = HashableSequence(  # Filter for desired log_level
            [
                log
                for log in job_log.get_logs()
                if trapi_level_to_int(log.level or LogLevel.DEBUG) >= desired_log_level
            ]
        )
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


def initialize_lookup(query: QueryInfo) -> tuple[str, TRAPILogger, Response]:
    """Set up a basic response to avoid some repetition."""
    job_id = query.job_id
    job_log = TRAPILogger(job_id)

    if query.body is None:
        raise TypeError(
            "Received body of type None, should have received Query or AsyncQuery."
        )

    return (
        job_id,
        job_log,
        Response(
            message=Message(
                query_graph=query.body.message.query_graph,
                knowledge_graph=KnowledgeGraph(),
                results=Results(),
            ),
            biolink_version=OPENAPI_CONFIG.x_translator.biolink_version,
            schema_version=OPENAPI_CONFIG.x_trapi.version,
            workflow=query.body.workflow,
            job_id=job_id,  # pyright:ignore[reportCallIssue] Extra is allowed
        ),
    )


def passes_validation(
    qgraph: QueryGraph, response: Response, job_log: TRAPILogger
) -> bool:
    """Ensure a given query graph passes validation.

    Prepares response with appropriate messages if not.
    """
    validation_problems = validate(qgraph)
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
        return False
    return True


async def qgraph_supported(
    qgraph: QueryGraph, response: Response, job_log: TRAPILogger
) -> bool:
    """Check that the given query graph has metakg support for all edges.

    Prepares response with appropriate messages if not.
    """
    # TODO: Do a linear iteration of all edges and check that metaEdges exist for each
    graph_supported, logs = bool(qgraph), list[LogEntry]()
    if not graph_supported:
        job_log.log_deque.extend(logs)
        job_log.error(
            "Query cannot be traversed due to missing metaEdges. See above logs for details."
        )
        response.status = "QueryNotTraversable"
        response.description = (
            "Query cannot be traversed due to missing metaEdges. See logs for details."
        )
        return False
    return True


@tracer.start_as_current_span("execute_query_graph")
async def run_tiered_lookups(
    query: QueryInfo, expanded_qgraph: QueryGraph
) -> LookupArtifacts:
    """Run lookups against requested tier(s) and combine results."""
    results = Results()
    kgraph = KnowledgeGraph()
    aux_graphs = AuxiliaryGraphs()
    logs = list[LogEntry]()

    query_tasks = list[asyncio.Task[LookupArtifacts]]()

    handlers = (Tier0Query, QueryGraphExecutor)
    # FIX: not happening in parallel
    for i, called_for in enumerate(
        (0 in query.tier, not set(query.tier).isdisjoint({1, 2}))
    ):
        if not called_for:
            continue
        query_handler = handlers[i](expanded_qgraph, query.job_id, query.tier)
        query_tasks.append(asyncio.create_task(query_handler.execute()))

    while query_tasks:
        done, pending = await asyncio.wait(
            query_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            findings = await task
            print(len(findings.results))
            results.update(findings.results)
            kgraph.update(findings.kgraph)
            aux_graphs.update(findings.aux_graphs)
            logs.extend(findings.logs)

        query_tasks = pending

    return LookupArtifacts(results, kgraph, aux_graphs, logs)


def tracked_response(status: int, body: Response) -> tuple[int, Response]:
    """Utility function for response handling."""
    MONGO_QUEUE.put("job_state", body.model_dump())
    return status, body
