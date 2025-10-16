from typing import cast, Any

import pytest

from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.types.trapi import QueryGraphDict


def qg(d: dict[str, Any]) -> QueryGraphDict:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(QueryGraphDict, cast(object, d))

SIMPLE_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:4514"], "constraints": []},
        "n1": {"ids": ["UMLS:C1564592"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["subclass_of"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

SIMPLE_QGRAPH_MULTIPLE_IDS: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125", "CHEBI:53448"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090", "CHEBI:10119"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})


@pytest.fixture
def es_transpiler() -> ElasticsearchTranspiler:
    return ElasticsearchTranspiler()


@pytest.mark.parametrize(
    "q_graph",
    [
        SIMPLE_QGRAPH,
        SIMPLE_QGRAPH_MULTIPLE_IDS,
    ],
    ids = ["single id", "multiple ids"],
)
def test_convert_triple(q_graph: QueryGraphDict, es_transpiler: ElasticsearchTranspiler) -> None:
    generated_payload = es_transpiler.convert_triple(q_graph)
    assert generated_payload is not None

    filter_content = generated_payload["query"]["bool"]["filter"]
    assert filter_content is not None
    assert isinstance(filter_content, list)

    q_edge = next(iter(q_graph["edges"].values()), None)
    in_node = q_graph["nodes"][q_edge["subject"]]
    out_node = q_graph["nodes"][q_edge["object"]]

    for single_filter in filter_content:
        terms = single_filter["terms"]
        if "subject.id" in terms:
            assert in_node["ids"] == terms["subject.id"]
        if "object.id" in terms:
            assert out_node["ids"] == terms["object.id"]
        if "all_predicates" in terms:
            assert q_edge["predicates"] == terms["all_predicates"]

def test_convert_batch_triple() -> None:
    pass