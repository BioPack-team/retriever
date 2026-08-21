import os
from typing import NamedTuple

import aiofiles
from loguru import logger
from translator_tom.v1_6 import (
    Biolink,
    QEdge,
    QEdgeID,
    QNodeID,
    QueryGraph,
)

from retriever.types.general import (
    AdjacencyGraph,
    QEdgeIDMap,
    QueryInfo,
)
from retriever.utils.general import BatchedAction
from retriever.utils.logs import TRAPILogger
from retriever.utils.telemetry import contextualize_query_telemetry


def ensure_minimal_types(qg: QueryGraph, job_log: TRAPILogger) -> QueryGraph:
    """Ensure nodes without categories have NamedThing and edges without predicates have related_to."""
    # biolink functions are already LRU cached :)
    for qnode_id, qnode in qg.nodes.items():
        if len(qnode.categories_list) == 0:
            qnode.categories = [Biolink("NamedThing")]
            job_log.info(
                f"QNode {qnode_id}: Inferred NamedThing from empty category list."
            )

    for qedge_id, qedge in qg.edges.items():
        if len(qedge.predicates_list) == 0:
            qedge.predicates = [Biolink("related_to")]
            job_log.info(
                f"QEdge {qedge_id}: Inferred related_to from empty predicate list."
            )

    return qg


def make_mappings(qg: QueryGraph) -> tuple[AdjacencyGraph, QEdgeIDMap]:
    """Make an undirected QGraph representation in which edges are presented by their nodes."""
    agraph: AdjacencyGraph = {}
    edge_id_map: QEdgeIDMap = {}
    for edge_id, edge in qg.edges.items():
        edge_id_map[id(edge)] = QEdgeID(edge_id)
        subject_node = QNodeID(edge.subject)
        object_node = QNodeID(edge.object)
        if subject_node not in agraph:
            agraph[subject_node] = dict[QNodeID, list[QEdge]]()
        if object_node not in agraph:
            agraph[object_node] = dict[QNodeID, list[QEdge]]()
        if object_node not in agraph[subject_node]:
            agraph[subject_node][object_node] = list[QEdge]()
        if subject_node not in agraph[object_node]:
            agraph[object_node][subject_node] = list[QEdge]()
        agraph[subject_node][object_node].append(edge)
        agraph[object_node][subject_node].append(edge)

    return agraph, edge_id_map


def get_submitter(query: QueryInfo) -> str:
    """Extract the submitter from a query, if it's provided."""
    body = query.body

    submitter = body is not None and body.submitter

    if submitter:
        return submitter
    else:
        return "not_provided"


class QueryMetadata(NamedTuple):
    """Metadata about a query."""

    job_id: str
    job_timeout: float
    data_tier: int | None
    query_type: str
    submitter: str
    qnodes: int
    qedges: int
    qpaths: int


def get_query_metadata(query: QueryInfo, query_type: str) -> QueryMetadata:
    """Obtain useful metrics about the query."""
    qnodes, qedges, qpaths = 0, 0, 0
    body = query.body
    if body is not None and body.message.query_graph is not None:
        qnodes = len(body.message.query_graph.nodes)
        if isinstance(body.message.query_graph, QueryGraph):
            qedges = len(body.message.query_graph.edges)
        else:
            qpaths = len(body.message.query_graph.paths)

    return QueryMetadata(
        job_id=query.job_id,
        job_timeout=query.timeout,
        data_tier=query.tier,
        query_type=query_type,
        submitter=get_submitter(query),
        qnodes=qnodes,
        qedges=qedges,
        qpaths=qpaths,
    )


def contextualize_query(query: QueryInfo, query_type: str) -> None:
    """Tag telemetry (Sentry tags + current OTel span) with the query's metadata.

    Safe to call from the async background-task context to re-establish the tags on
    that task's separate Sentry transaction; failures are logged, not raised.
    """
    with logger.catch(
        Exception,
        level="ERROR",
        message="Error while attempting to contextualize telemetry to query.",
    ):
        contextualize_query_telemetry(get_query_metadata(query, query_type)._asdict())


class QueryDumper(BatchedAction):
    """A class for quickly queueing queries to dump to a file."""

    flush_time: float = 60

    async def write_tier0(self, payload: list[bytes]) -> None:
        """Alias for tier 0 specifically."""
        await self.write(0, payload)

    async def write_tier1(self, payload: list[bytes]) -> None:
        """Alias for tier 1 specifically."""
        await self.write(1, payload)

    async def write_tier2(self, payload: list[bytes]) -> None:
        """Alias for tier 2 specifically."""
        await self.write(2, payload)

    async def write(self, tier: int, payload: list[bytes]) -> None:
        """Write a batch of query payloads to the dump.

        Assumes the lines have already been dumped by orjson with a terminating newline.
        """
        async with aiofiles.open(
            f"{os.getpid()}_tier{tier}_dump.jsonl", mode="ab"
        ) as file:
            for line in payload:
                await file.write(line)
        logger.trace(f"Wrote {len(payload)} tier-{tier} queries.")
