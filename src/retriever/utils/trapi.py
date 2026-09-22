from typing import cast

from translator_tom.v2_0 import CURIE, QueryGraph
from translator_tom.v2_0.model_dicts import (
    KnowledgeGraphDict,
    MessageDict,
    MessageDictUtil,
    NodeDict,
    QueryGraphDict,
    ResultDict,
)

from retriever.utils.logs import TRAPILogger


def initialize_kgraph(qgraph: QueryGraphDict | QueryGraph) -> KnowledgeGraphDict:
    """Initialize a knowledge graph, using nodes from the query graph."""
    kgraph = KnowledgeGraphDict(nodes={}, edges={})
    if isinstance(qgraph, QueryGraph):
        for qnode in qgraph.nodes.values():
            if qnode.ids is None:
                continue
            for curie in qnode.ids:
                kgraph["nodes"][CURIE(curie)] = NodeDict(categories=[])
    else:
        for qnode in qgraph["nodes"].values():
            if "ids" not in qnode or not qnode["ids"]:
                continue
            for curie in qnode["ids"]:
                kgraph["nodes"][CURIE(curie)] = NodeDict(categories=[])
    return kgraph


def _has_set_interpretation(qgraph: QueryGraph) -> bool:
    """True if any QNode declares a non-BATCH set_interpretation."""
    return any(
        (qnode.set_interpretation or "BATCH") != "BATCH"
        for qnode in qgraph.nodes.values()
    )


def solve_set_interpretation(
    qgraph: QueryGraph,
    results: list[ResultDict],
    kgraph: KnowledgeGraphDict,
    job_log: TRAPILogger,
) -> list[ResultDict]:
    """Collapse results per each QNode's set_interpretation via TOM's dict post-solver.

    Wraps the working dicts in a placeholder MessageDict; shortcuts all-BATCH graphs and
    treats MANY as BATCH (TOM does not collate MANY).
    """
    if not results or not _has_set_interpretation(qgraph):
        return results

    message = MessageDict(
        query_graph=cast("QueryGraphDict", cast("object", qgraph.to_dict())),
        knowledge_graph=kgraph,
        results=results,
    )

    try:
        MessageDictUtil.solve_set_interpretation(message, skip_many=True)
    except ValueError as error:
        job_log.error(
            f"Set interpretation failed; returning uncollapsed results: {error}"
        )
        return results

    return MessageDictUtil.results_list(message)
