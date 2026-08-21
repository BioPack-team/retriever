from translator_tom.v1_6 import QueryGraph

from retriever.utils.trapi import initialize_kgraph


def test_initialize_kgraph() -> None:
    qgraph_dict = {
        "nodes": {
            "n0": {"categories": ["biolink:Gene"], "ids": ["NCBIGene:3778"]},
            "n1": {"categories": ["biolink:Disease"]},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:causes"],
            }
        },
    }
    qgraph = QueryGraph.from_dict(qgraph_dict)
    kgraph = initialize_kgraph(qgraph)

    assert len(kgraph["nodes"]) == 1
    assert "NCBIGene:3778" in kgraph["nodes"]
    assert len(kgraph["edges"]) == 0
