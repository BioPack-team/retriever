from reasoner_pydantic import Edge, QueryGraph

from retriever.types.trapi import EdgeDict
from retriever.utils.trapi import hash_edge, hash_hex, initialize_kgraph


def test_initialize_kgraph() -> None:
    """Ensure kgraph is initialized properly."""
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
    qgraph = QueryGraph.model_validate(qgraph_dict)
    kgraph = initialize_kgraph(qgraph)

    assert len(kgraph["nodes"]) == 1
    assert "NCBIGene:3778" in kgraph["nodes"]
    assert "biolink:Gene" in kgraph["nodes"]["NCBIGene:3778"]["categories"]
    assert len(kgraph["edges"]) == 0
