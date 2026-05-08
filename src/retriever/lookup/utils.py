import os

import aiofiles
from loguru import logger
from translator_tom import (
    Biolink,
    QEdge,
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


def expand_qgraph(qg: QueryGraph, job_log: TRAPILogger) -> QueryGraph:
    """Ensure all nodes in qgraph have all descendant categories.

    See https://biolink.github.io/biolink-model/categories.html
    """
    # biolink functions are already LRU cached :)
    for qnode_id, qnode in qg.nodes.items():
        categories = set(qnode.categories_list)
        new_categories = set[Biolink.Entity]()
        if len(categories) == 0:
            categories.add("biolink:NamedThing")
            job_log.info(
                f"QNode {qnode_id}: Inferred NamedThing from empty category list."
            )
        # Not necessary because backends check against descendants
        # new_categories = biolink.expand(categories) - categories

        if "biolink:NamedThing" in categories:
            job_log.info(
                f"QNode {qnode_id}: Expanded to all categories (original had NamedThing)."
            )
        elif len(new_categories):
            job_log.info(
                f"QNode {qnode_id}: Added descendant categories {new_categories}."
            )

        qnode.categories = [*categories, *new_categories]

    for qedge_id, qedge in qg.edges.items():
        predicates = set(qedge.predicates_list)
        new_predicates = set[Biolink.Predicate]()
        if len(predicates) == 0:
            predicates.add("biolink:related_to")
            job_log.info(
                f"QEdge {qedge_id}: Inferred related_to from empty predicate list."
            )
        # Not necessary because backends check against descendants
        # new_predicates = biolink.expand(predicates) - predicates

        if "biolink:related_to" in predicates:
            job_log.info(
                f"QEdge {qedge_id}: Expanded to all predicates (original had related_to)."
            )
        elif len(new_predicates):
            job_log.info(
                f"QEdge {qedge_id}: Added descendant predicates {new_predicates}."
            )

        qedge.predicates = [*predicates, *new_predicates]

    return qg


def make_mappings(qg: QueryGraph) -> tuple[AdjacencyGraph, QEdgeIDMap]:
    """Make an undirected QGraph representation in which edges are presented by their nodes."""
    agraph: AdjacencyGraph = {}
    edge_id_map: QEdgeIDMap = {}
    for edge_id, edge in qg.edges.items():
        edge_id_map[id(edge)] = edge_id
        subject_node = edge.subject
        object_node = edge.object
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

    if submitter := body is not None and body.submitter:
        return submitter
    else:
        return "not_provided"


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
