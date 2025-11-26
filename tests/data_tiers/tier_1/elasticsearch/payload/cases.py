from typing import Any, cast

from retriever.types.trapi import QueryGraphDict
from tests.data_tiers.tier_1.elasticsearch.payload.trapi_qgraphs import SIMPLE_QGRAPH_0, SIMPLE_QGRAPH_1, \
    SIMPLE_QGRAPH_MULTIPLE_IDS


def qg(d: dict[str, Any]) -> QueryGraphDict:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(QueryGraphDict, cast(object, d))


Q_GRAPH_CASES = (
    "q_graph",
    [SIMPLE_QGRAPH_0, SIMPLE_QGRAPH_1, SIMPLE_QGRAPH_MULTIPLE_IDS],
)
Q_GRAPH_CASES_IDS = ["single id 0","single id 1", "multiple ids"]
