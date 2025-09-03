import asyncio
import math
import time
from typing import Any, Literal, cast, overload

import bmt
import httpx
from loguru import logger as log
from opentelemetry import trace
from reasoner_pydantic import (
    LogLevel,
    QueryGraph,
)
from reasoner_pydantic.qgraph import PathfinderQueryGraph

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.data_tiers.tier_0.neo4j.query import Neo4jQuery
from retriever.lookup.qgx import QueryGraphExecutor
from retriever.lookup.utils import expand_qgraph
from retriever.lookup.validate import validate
from retriever.types.general import LookupArtifacts, QueryInfo
from retriever.types.trapi import (
    AuxGraphDict,
    KnowledgeGraphDict,
    LogEntryDict,
    MessageDict,
    ParametersDict,
    QueryGraphDict,
    ResponseDict,
    ResultDict,
)
from retriever.types.trapi_pydantic import AsyncQuery
from retriever.utils.calls import CALLBACK_CLIENT
from retriever.utils.logs import TRAPILogger, trapi_level_to_int
from retriever.utils.mongo import MONGO_QUEUE
from retriever.utils.trapi import merge_results, prune_kg, update_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")
biolink = bmt.Toolkit()


async def async_lookup(query: QueryInfo) -> None:
    """Handle running lookup as an async query where the client receives a callback."""
    if not isinstance(query.body, AsyncQuery):
        raise TypeError(f"Expected AsyncQuery, received {type(query.body)}.")

    job_id = query.job_id
    job_log = log.bind(job_id=job_id)
    status, response = await lookup(query)

    job_log.debug(f"Sending callback to `{query.body.callback}`...")
    try:
        callback_response = await CALLBACK_CLIENT.post(
            url=str(query.body.callback), json=response
        )
        callback_response.raise_for_status()
        job_log.debug("Callback sent successfully.")
    except httpx.HTTPStatusError as error:
        job_log.exception(
            f"Callback returned erroneous status {error.response.status_code}"
        )
        job_log.error(f"Response body was {error.response.text}")
    except httpx.HTTPError:
        job_log.exception(
            "An unhandled exception occured while making response callback."
        )

    # update so callback logs are kept with response
    tracked_response(status, response)


async def lookup(query: QueryInfo) -> tuple[int, ResponseDict]:
    """Execute a lookup query.

    Does job state updating regardless of asyncquery for easier debugging.

    Returns:
        A tuple of HTTP status code, response body.
    """
    if query.body is None:
        raise TypeError("Query body is None, should have received Query or AsyncQuery.")
    job_id, job_log, response = initialize_lookup(query)

    try:
        start_time = time.time()
        job_log.info(f"Begin processing job {job_id}.")
        job_log.debug(
            f"Job timeout is {str(query.timeout) + 's' if query.timeout >= 0 else 'disabled'}."
        )

        qgraph = query.body.message.query_graph
        if qgraph is None:
            raise ValueError("Query Graph is None.")

        # Query graph validation that isn't handled by reasoner_pydantic
        if not passes_validation(qgraph, response, job_log) or isinstance(
            qgraph, PathfinderQueryGraph
        ):
            return tracked_response(422, response)

        expanded_qgraph = expand_qgraph(qgraph.model_copy(), job_log)

        if not await qgraph_supported(expanded_qgraph, response, job_log):
            return tracked_response(424, response)

        results, kgraph, aux_graphs, logs, _ = await run_tiered_lookups(
            query, expanded_qgraph
        )
        job_log.log_deque.extend(logs)

        job_log.info(f"Collected {len(results)} results from query tasks.")
        prune_kg(results, kgraph, aux_graphs, job_log)

        end_time = time.time()
        finish_msg = f"Execution completed, obtained {len(results)} results in {math.ceil((end_time - start_time) * 1000):}ms."
        job_log.info(finish_msg)

        response["status"] = "Complete"
        response["description"] = finish_msg
        # Filter for desired log_level
        desired_log_level = trapi_level_to_int(query.body.log_level or LogLevel.DEBUG)
        response["logs"] = [
            log
            for log in job_log.get_logs()
            if trapi_level_to_int(log.get("level", LogLevel.DEBUG)) >= desired_log_level
        ]
        response["message"]["results"] = results
        response["message"]["knowledge_graph"] = kgraph
        response["message"]["auxiliary_graphs"] = aux_graphs
        return tracked_response(200, response)

    except Exception:
        job_log.error(
            f"Uncaught exception during handling of job {job_id}. Job failed."
        )
        job_log.exception("Cause of error:")
        response["status"] = "Failed"
        response["description"] = (
            "Execution failed due to an unhandled error. See the logs for more details."
        )
        response["logs"] = job_log.get_logs()
        return tracked_response(500, response)


def initialize_lookup(query: QueryInfo) -> tuple[str, TRAPILogger, ResponseDict]:
    """Set up a basic response to avoid some repetition."""
    job_id = query.job_id
    job_log = TRAPILogger(job_id)

    if query.body is None:
        raise TypeError(
            "Received body of type None, should have received Query or AsyncQuery."
        )
    if query.body.message.query_graph is None:
        raise TypeError(
            "Received QueryGraph of type None, query graph should be present."
        )

    return (
        job_id,
        job_log,
        ResponseDict(
            message=MessageDict(
                query_graph=QueryGraphDict(
                    **query.body.message.query_graph.model_dump()
                ),
                knowledge_graph=KnowledgeGraphDict(nodes={}, edges={}),
                results=list[ResultDict](),
            ),
            biolink_version=OPENAPI_CONFIG.x_translator.biolink_version,
            schema_version=OPENAPI_CONFIG.x_trapi.version,
            workflow=query.body.workflow.model_dump() if query.body.workflow else None,
            parameters=ParametersDict(tiers=list(query.tiers), timeout=query.timeout),
            job_id=job_id,  # pyright:ignore[reportCallIssue] Extra is allowed
        ),
    )


@overload
def passes_validation(
    qgraph: PathfinderQueryGraph, response: ResponseDict, job_log: TRAPILogger
) -> Literal[False]: ...


@overload
def passes_validation(
    qgraph: QueryGraph,
    response: ResponseDict,
    job_log: TRAPILogger,
) -> bool: ...


def passes_validation(
    qgraph: QueryGraph | PathfinderQueryGraph,
    response: ResponseDict,
    job_log: TRAPILogger,
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

        response["status"] = "QueryNotTraversable"
        response["description"] = (
            "Query terminated due to validation errors. See logs for more details."
        )
        response["logs"] = job_log.get_logs()
        return False
    return True


async def qgraph_supported(
    qgraph: QueryGraph, response: ResponseDict, job_log: TRAPILogger
) -> bool:
    """Check that the given query graph has metakg support for all edges.

    Prepares response with appropriate messages if not.
    """
    # TODO: Do a linear iteration of all edges and check that metaEdges exist for each
    graph_supported, logs = bool(qgraph), list[LogEntryDict]()
    if not graph_supported:
        job_log.log_deque.extend(logs)
        job_log.error(
            "Query cannot be traversed due to missing metaEdges. See above logs for details."
        )
        response["status"] = "QueryNotTraversable"
        response["description"] = (
            "Query cannot be traversed due to missing metaEdges. See logs for details."
        )
        return False
    return True


@tracer.start_as_current_span("execute_lookup")
async def run_tiered_lookups(
    query: QueryInfo, expanded_qgraph: QueryGraph
) -> LookupArtifacts:
    """Run lookups against requested tier(s) and combine results."""
    results = dict[int, ResultDict]()
    kgraph = KnowledgeGraphDict(nodes={}, edges={})
    aux_graphs = dict[str, AuxGraphDict]()
    logs = list[LogEntryDict]()

    query_tasks = list[asyncio.Task[LookupArtifacts]]()
    job_log = TRAPILogger(query.job_id)

    handlers = (Neo4jQuery, QueryGraphExecutor)
    for i, called_for in enumerate(
        (0 in query.tiers, not set(query.tiers).isdisjoint({1, 2}))
    ):
        if not called_for:
            continue
        query_handler = handlers[i](expanded_qgraph, query)
        query_tasks.append(asyncio.create_task(query_handler.execute()))

    async for task in asyncio.as_completed(query_tasks):
        try:
            findings = await task
            logs.extend(findings.logs)
            if not findings.error:
                merge_results(results, findings.results)
                # Small optimization: Iterate the smaller kgraph
                kgraph_size = len(kgraph["nodes"]) + len(kgraph["edges"])
                new_kgraph_size = len(findings.kgraph["nodes"]) + len(
                    findings.kgraph["edges"]
                )
                if kgraph_size > new_kgraph_size:
                    update_kgraph(kgraph, findings.kgraph)
                else:
                    update_kgraph(findings.kgraph, kgraph)
                    kgraph = findings.kgraph

                aux_graphs.update(findings.aux_graphs)
        except Exception:
            # This is a bad exception to get because we lose detail about which tier.
            # Idealy the Tier handler will catch almost any unhandled exception and
            # report it with more detail.
            job_log.exception(
                "Unhandled exception while running Tier lookup. See other logs for details."
            )
            logs.append(job_log.log_deque.popleft())

    return LookupArtifacts(list(results.values()), kgraph, aux_graphs, logs)


def tracked_response(status: int, body: ResponseDict) -> tuple[int, ResponseDict]:
    """Utility function for response handling."""
    # Cast because TypedDict has some *really annoying* interactions with more general dicts
    MONGO_QUEUE.put("job_state", cast(dict[str, Any], cast(object, body)))
    return status, body
