from reasoner_pydantic import (
    BiolinkEntity,
    BiolinkPredicate,
    HashableSequence,
    QEdge,
    QueryGraph,
)

from retriever.lookup.branch import Branch, SuperpositionHop
from retriever.types.general import (
    AdjacencyGraph,
    QEdgeIDMap,
)
from retriever.types.trapi import (
    CURIE,
    EdgeDict,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeID,
    QNodeID,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import hash_edge, hash_hex


def expand_qgraph(qg: QueryGraph, job_log: TRAPILogger) -> QueryGraph:
    """Ensure all nodes in qgraph have all descendant categories.

    See https://biolink.github.io/biolink-model/categories.html
    """
    # biolink functions are already LRU cached :)
    for qnode_id, qnode in qg.nodes.items():
        categories = set(qnode.categories or {})
        new_categories = set[BiolinkEntity]()
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

        qnode.categories = HashableSequence[BiolinkEntity](
            [*categories, *new_categories]
        )

    for qedge_id, qedge in qg.edges.items():
        predicates = set(qedge.predicates or {})
        new_predicates = set[BiolinkPredicate]()
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

        qedge.predicates = HashableSequence[BiolinkPredicate](
            [*predicates, *new_predicates]
        )

    return qg


def make_mappings(qg: QueryGraph) -> tuple[AdjacencyGraph, QEdgeIDMap]:
    """Make an undirected QGraph representation in which edges are presented by their nodes."""
    agraph: AdjacencyGraph = {}
    edge_id_map: QEdgeIDMap = {}
    for edge_id, edge in qg.edges.items():
        edge_id_map[edge] = QEdgeID(edge_id)
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


async def get_subgraph(
    branch: Branch,
    key: SuperpositionHop,
    kedges: dict[SuperpositionHop, list[EdgeDict]],
    kgraph: KnowledgeGraphDict,
) -> tuple[KnowledgeGraphDict, list[LogEntryDict]]:
    """Get a subgraph from a given set of kedges.

    Used to replace subquerying when a given hop has already been completed.
    """
    edges = kedges[key]
    curies = list[str]()
    for edge in edges:
        if not branch.reversed:
            curies.append(edge["object"])
        else:
            curies.append(edge["subject"])

    kg = KnowledgeGraphDict(
        edges={hash_hex(hash_edge(edge)): edge for edge in edges},
        nodes={CURIE(curie): kgraph["nodes"][CURIE(curie)] for curie in curies},
    )

    return kg, list[LogEntryDict]()
