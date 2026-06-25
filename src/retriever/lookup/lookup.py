import math
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from http import HTTPStatus

import bmt
import httpx
import orjson
import ormsgpack
import zstandard
from opentelemetry import context, propagate, trace

from retriever.config.general import CONFIG
from retriever.config.openapi import OPENAPI_CONFIG
from retriever.data_tiers import tier_manager
from retriever.lookup.qgx import QueryGraphExecutor
from retriever.lookup.utils import expand_qgraph, get_submitter
from retriever.lookup.validate import validate
from retriever.metadata.optable import (
    OpTableManager,
    QueryNotTraversable,
    UnsupportedConstraint,
)
from retriever.types.general import QueryInfo
from retriever.types.trapi import (
    KnowledgeGraphDict,
    LogEntryDict,
    LogLevel,
    MessageDict,
    ParametersDict,
    QueryDict,
    QueryGraphDict,
    ResponseDict,
    ResultDict,
)
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import service_health
from retriever.utils.calls import get_callback_client
from retriever.utils.compression import accepts_zstd
from retriever.utils.logs import TRAPILogger, trapi_level_to_int
from retriever.utils.mongo import MongoOutage, MongoQueue, ResponseState

tracer = trace.get_tracer("lookup.execution.tracer")
biolink = bmt.Toolkit()
MONGO_QUEUE = MongoQueue()
OP_TABLE_MANAGER = OpTableManager()

ZSTD_COMPRESSOR = zstandard.ZstdCompressor(level=CONFIG.mongo.compression_level)

CALLBACK_BODY_LOG_LIMIT = 500
"""Truncate logged callback response bodies to this many chars."""


async def async_lookup(
    query: QueryInfo,
    ctx: dict[str, str],
    *,
    extra_warnings: list[LogEntryDict] | None = None,
) -> None:
    """Handle running lookup as an async query where the client receives a callback."""
    token = context.attach(propagate.extract(ctx))  # Ensure context propagates

    try:
        if query.body is None or "callback" not in query.body:
            raise TypeError(f"Expected AsyncQuery, received {type(query.body)}.")

        job_id = query.job_id
        job_log = TRAPILogger(job_id)
        status, response = await lookup(query)
        if extra_warnings:
            response["logs"] = [*(response.get("logs") or []), *extra_warnings]

        job_log.debug(f"Sending callback to `{query.body['callback']}`...")
        try:
            async with get_callback_client() as client:
                callback_json = orjson.dumps(response)
                headers = {"Content-Type": "application/json"}
                if accepts_zstd(query.headers.get("Callback-Accept-Encoding", "")):
                    callback_json = ZSTD_COMPRESSOR.compress(callback_json)
                    headers["Content-Encoding"] = "zstd"
                callback_response = await client.post(
                    url=str(query.body["callback"]),
                    headers=headers,
                    content=callback_json,
                )
                callback_response.raise_for_status()
                job_log.debug("Callback sent successfully.")
        except httpx.HTTPStatusError as error:
            job_log.exception(
                f"Callback returned erroneous status {error.response.status_code}"
            )
            job_log.error(
                f"Response body was {error.response.text[:CALLBACK_BODY_LOG_LIMIT]}"
            )
        except httpx.HTTPError:
            job_log.exception(
                "An unhandled httpx error occurred while making response callback."
            )
        except Exception:
            # Anything not modeled by httpx (e.g. serialization issues)
            # gets logged so the background task doesn't crash unnoticed.
            job_log.exception("Unexpected error during callback delivery.")

        # Effectively tack the callback logs onto the end of the response
        response_logs = response.get("logs", []) or []
        job_log.log_deque = deque(response_logs) + job_log.log_deque

        # Update the stored state with new logs
        tracked_response(status, query, response, job_log)
    finally:
        context.detach(token)


def _summarize_execution(
    ctx: QueryInfo,
    status: str,
    results_count: int,
    duration_ms: int,
    job_log: TRAPILogger,
) -> str:
    """Build the post-execution description and emit a status-appropriate log line.

    `TimedOut` is an internal-only status the query handlers can emit; the
    caller is expected to collapse it to `Failed` before storing / responding.
    """
    match status:
        case "Success":
            finish_msg = (
                f"Execution completed, obtained {results_count} results "
                f"in {duration_ms}ms."
            )
            job_log.success(finish_msg)
        case "QueryNotTraversable":
            finish_msg = (
                "Execution terminated: query graph was not traversable "
                f"with the current MetaKG (took {duration_ms}ms)."
            )
            job_log.warning(finish_msg)
        case "TimedOut":
            finish_msg = f"Execution exceeded timeout ({duration_ms}ms > {ctx.timeout * 1000}ms)."
            job_log.error(finish_msg)
        case "Failed":
            finish_msg = (
                "Execution failed in query handling; see logs "
                f"for details (took {duration_ms}ms)."
            )
            job_log.error(finish_msg)
        case _:
            finish_msg = (
                f"Execution ended with status {status!r} (took {duration_ms}ms)."
            )
            job_log.warning(finish_msg)
    return finish_msg


async def lookup(query: QueryInfo) -> tuple[HTTPStatus, ResponseDict]:
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
        job_log.info(
            f"Begin processing job {job_id} for client {get_submitter(query)}."
        )
        if (query.body.get("parameters") or {}).get("dehydrated"):
            job_log.info(
                "Query running in dehydrated mode. Response will not be stored."
            )

        qgraph = query.body["message"].get("query_graph")
        if qgraph is None:
            raise ValueError("Query Graph is None.")

        # Query graph validation that isn't handled by reasoner_pydantic
        if not passes_validation(query.body, response, job_log) or "paths" in qgraph:
            return tracked_response(
                HTTPStatus.UNPROCESSABLE_ENTITY, query, response, job_log
            )

        expanded_qgraph = expand_qgraph(deepcopy(qgraph), job_log)

        if not await qgraph_supported(
            expanded_qgraph, response, job_log, query.tier or 0
        ):
            return tracked_response(HTTPStatus.OK, query, response, job_log)

        if not tier_manager.get_driver(query.tier or 0).up:
            msg = f"Tier {query.tier or 0} backend connection failed, query terminates."
            job_log.error(msg)
            response["status"] = "Failed"
            response["description"] = msg
            return tracked_response(
                HTTPStatus.FAILED_DEPENDENCY, query, response, job_log
            )

        handlers = (
            tier_manager.QUERY_HANDLERS[CONFIG.tier0.backend],
            QueryGraphExecutor,
        )
        query_handler = handlers[query.tier or 0](expanded_qgraph, query)
        results, kgraph, aux_graphs, logs, status = await query_handler.execute()

        job_log.log_deque.extend(logs)

        job_log.info(f"Collected {len(results)} results from query task.")

        duration_ms = math.ceil((time.time() - start_time) * 1000)
        finish_msg = _summarize_execution(
            query, status, len(results), duration_ms, job_log
        )

        # `TimedOut` is an internal signal driving the description above;
        # the wire/storage status collapses to `Failed` per spec.
        if status == "TimedOut":
            status = "Failed"

        response["status"] = status
        response["description"] = finish_msg
        response["message"]["results"] = results
        response["message"]["knowledge_graph"] = kgraph
        response["message"]["auxiliary_graphs"] = aux_graphs
        return tracked_response(HTTPStatus.OK, query, response, job_log)

    except Exception:
        job_log.error(
            f"Uncaught exception during handling of job {job_id}. Job failed."
        )
        job_log.exception("Cause of error:")
        response["status"] = "Failed"
        response["description"] = (
            "Execution failed due to an unhandled error. See the logs for more details."
        )
        return tracked_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, query, response, job_log
        )


def initialize_lookup(query: QueryInfo) -> tuple[str, TRAPILogger, ResponseDict]:
    """Set up a basic response to avoid some repetition."""
    job_id = query.job_id
    job_log = TRAPILogger(job_id)

    if query.body is None:
        raise TypeError(
            "Received body of type None, should have received Query or AsyncQuery."
        )
    if (
        "query_graph" not in query.body["message"]
        or query.body["message"]["query_graph"] is None
    ):
        raise TypeError(
            "Received QueryGraph of type None, query graph should be present."
        )

    parameters = ParametersDict(tier=query.tier or 0)
    if (
        timeout := "parameters" in query.body
        and query.body["parameters"] is not None
        and query.body["parameters"].get("timeout")
    ):
        parameters["timeout"] = timeout
    return (
        job_id,
        job_log,
        ResponseDict(  # pyright:ignore[reportCallIssue] Extra is allowed
            message=MessageDict(
                query_graph=query.body["message"]["query_graph"],
                knowledge_graph=KnowledgeGraphDict(nodes={}, edges={}),
                results=list[ResultDict](),
            ),
            biolink_version=OPENAPI_CONFIG.x_translator.biolink_version,
            schema_version=OPENAPI_CONFIG.x_trapi.version,
            workflow=query.body.get("workflow"),
            parameters=parameters,
            submitter=get_submitter(query),
            job_id=job_id,
        ),
    )


def passes_validation(
    query: QueryDict,
    response: ResponseDict,
    job_log: TRAPILogger,
) -> bool:
    """Ensure a given query graph passes validation.

    Prepares response with appropriate messages if not.
    """
    warnings, errors = validate(query)
    for warning in warnings:
        job_log.warning(warning)
    if len(errors) > 0:
        for problem in errors:
            job_log.error(f"Validation Error: {problem}")
        job_log.error(
            f"Due to the {len(errors)} above validation error(s), your query terminates."
        )

        response["status"] = "QueryNotTraversable"
        response["description"] = (
            "Query terminated due to validation errors. See logs for more details."
        )
        response["logs"] = job_log.get_logs()
        return False
    return True


async def qgraph_supported(
    qgraph: QueryGraphDict,
    response: ResponseDict,
    job_log: TRAPILogger,
    tiers: TierNumber,
) -> bool:
    """Check that the given query graph has metakg support for all edges.

    Prepares response with appropriate messages if not.
    """
    supported, plan_or_report = await OP_TABLE_MANAGER.create_operation_plan(
        qgraph, tiers
    )
    if not supported:
        missing_edges = [
            qedge_id
            for qedge_id, reason in plan_or_report.items()
            if isinstance(reason, QueryNotTraversable)
        ]
        unsupported_constraints = {
            qedge_id: reason
            for qedge_id, reason in plan_or_report.items()
            if isinstance(reason, UnsupportedConstraint)
        }
        if len(missing_edges) > 0:
            job_log.warning(
                f"MetaEdges could not be found for the following QEdge(s): {missing_edges}"
            )
        if len(unsupported_constraints) > 0:
            for qedge_id, reason in unsupported_constraints.items():
                job_log.warning(
                    f"Unsupported constraint(s) present in QEdge `{qedge_id}` (if multiple, try a smaller combination): {reason.unmet}"
                )
        status = (
            "QueryNotTraversable"
            if "QueryNotTraversable" in plan_or_report.values()
            else "UnsupportedConstraint"
        )
        response["status"] = status
        reason_desc = (
            "missing MetaEdges"
            if status == "QueryNotTraversable"
            else "unsupported constraints"
        )
        response["description"] = (
            f"Query cannot be traversed due to {reason_desc}. See logs for details."
        )

    unsupported_nodes = await OP_TABLE_MANAGER.qnodes_supported(qgraph, tiers)
    if unsupported_nodes is not None:
        for qnode_id, reason in unsupported_nodes.items():
            job_log.warning(
                f"Unsupported constraint(s) present in QNode `{qnode_id}` (if multiple, try a smaller combination): {reason.unmet}"
            )
        if response.get("status") != "QueryNotTraversable":
            response["status"] = "UnsupportedConstraint"
            response["description"] = (
                "Query cannot be traversed due to unsupported constraints. See logs for details."
            )

    supported = supported and unsupported_nodes is None
    return supported


def tracked_response(
    status: HTTPStatus,
    query: QueryInfo,
    response: ResponseDict,
    job_log: TRAPILogger,
) -> tuple[HTTPStatus, ResponseDict]:
    """Utility function for response handling."""
    # Filter for desired log_level
    desired_log_level = trapi_level_to_int(
        LogLevel(
            ((query.body) or {}).get("log_level", LogLevel.DEBUG) or LogLevel.DEBUG
        )
    )
    response["logs"] = [
        log
        for log in job_log.get_logs()
        if trapi_level_to_int(log.get("level", LogLevel.DEBUG) or LogLevel.DEBUG)
        >= desired_log_level
    ]

    # Cast because TypedDict has some *really annoying* interactions with more general dicts
    kgraph = response["message"].get("knowledge_graph") or {}
    # Don't store dehydrated job responses, they're usually huge
    dehydrated = bool(((query.body or {}).get("parameters") or {}).get("dehydrated"))
    state = ResponseState(
        job_id=query.job_id,
        event_time=datetime.now().astimezone(),
        knodes=len(kgraph.get("nodes") or {}),
        kedges=len(kgraph.get("edges") or {}),
        aux_graphs=len(response["message"].get("auxiliary_graphs") or {}),
        results=len(response["message"].get("results") or []),
        status=(response.get("status") or "Running"),
        description=response.get("description"),
    )
    if not dehydrated:
        state["response"] = ZSTD_COMPRESSOR.compress(ormsgpack.packb(response))
    try:
        MONGO_QUEUE.put("job_state", state)
    except MongoOutage:
        response["logs"] = [
            *(response.get("logs") or []),
            service_health.mongo_outage_warning(),
        ]
    return status, response
