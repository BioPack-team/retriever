import pytest
from typing import Any

from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


def test_parse_single_success_case():
    """Test parsing of a well-formed, multi-hop Dgraph response."""
    # The raw response must match the actual Dgraph output format:
    # 1. The root key includes the query index prefix (e.g., "q0_").
    # 2. The value is a LIST of node objects.
    raw_response = {
        "q0_node_n0": [
            {
                "id": "CHEBI:3125",
                "name": "Bisacodyl",
                "in_edges_e0": [
                    {
                        "predicate": "interacts_with",
                        "node_n1": {
                            "id": "UMLS:C0282090",
                            "name": "Laxatives",
                            "out_edges_e2": [
                                {
                                    "predicate": "is_a",
                                    "node_n3": {"id": "UMLS:C12345"},
                                }
                            ],
                        },
                    }
                ],
                "out_edges_e1": [
                    {
                        "predicate": "causes",
                        "node_n2": {
                            "id": "UMLS:C0012345",
                            "name": "SideEffect",
                        },
                    }
                ],
            }
        ]
    }

    # 1. Parse the response
    parsed = dg_models.DgraphResponse.parse(raw_response)
    assert "q0" in parsed.data
    print(f"Parsed data keys: {parsed.data}")
    assert len(parsed.data["q0"]) == 1

    # 2. Assertions for the root node
    root_node = parsed.data["q0"][0]
    assert root_node.binding == "n0"
    assert root_node.id == "CHEBI:3125"
    assert len(root_node.edges) == 2

    # 3. Find and assert the 'in' edge
    in_edge = next((e for e in root_node.edges if e.direction == "in"), None)
    assert in_edge is not None
    assert in_edge.binding == "e0"
    assert in_edge.edge_id == "e0"
    assert in_edge.predicate == "interacts_with"

    # 4. Find and assert the 'out' edge
    out_edge = next((e for e in root_node.edges if e.direction == "out"), None)
    assert out_edge is not None
    assert out_edge.binding == "e1"
    assert out_edge.edge_id == "e1"
    assert out_edge.predicate == "causes"

    # 5. Assertions for the first level of nested nodes
    nested_node_in = in_edge.node
    assert nested_node_in.binding == "n1"
    assert nested_node_in.id == "UMLS:C0282090"
    assert len(nested_node_in.edges) == 1

    # 6. Assertions for the second level of nesting
    deep_edge = nested_node_in.edges[0]
    assert deep_edge.binding == "e2"
    assert deep_edge.direction == "out"
    assert deep_edge.node.binding == "n3"
    assert deep_edge.node.id == "UMLS:C12345"


def test_parse_batch_success_case():
    """Test parsing of a well-formed batch query response."""
    # A batch response with two queries, q0 and q1.
    raw_response = {
        "q0_node_n0": [
            {
                "id": "CHEBI:3125",
                "name": "Bisacodyl",
                "in_edges_e0": [
                    {
                        "predicate": "interacts_with",
                        "node_n1": {"id": "UMLS:C0282090"},
                    }
                ],
            }
        ],
        "q1_node_n0": [
            {
                "id": "GENE:1234",
                "name": "BRCA1",
                "out_edges_e0": [
                    {
                        "predicate": "related_to",
                        "node_n1": {"id": "DISEASE:5678"},
                    }
                ],
            }
        ],
    }

    # 1. Parse the response
    parsed = dg_models.DgraphResponse.parse(raw_response)

    # 2. Assert both query keys are present
    assert "q0" in parsed.data
    assert "q1" in parsed.data
    assert len(parsed.data) == 2

    # 3. Assertions for the first query (q0)
    assert len(parsed.data["q0"]) == 1
    q0_node = parsed.data["q0"][0]
    assert q0_node.binding == "n0"
    assert q0_node.id == "CHEBI:3125"
    assert len(q0_node.edges) == 1
    assert q0_node.edges[0].binding == "e0"
    assert q0_node.edges[0].node.binding == "n1"

    # 4. Assertions for the second query (q1)
    assert len(parsed.data["q1"]) == 1
    q1_node = parsed.data["q1"][0]
    assert q1_node.binding == "n0"
    assert q1_node.id == "GENE:1234"
    assert len(q1_node.edges) == 1
    assert q1_node.edges[0].binding == "e0"
    assert q1_node.edges[0].node.binding == "n1"


def test_parse_with_junk_data():
    """Test that the parser gracefully ignores malformed data."""
    # This raw response simulates a batch query with valid keys,
    # keys that don't match the pattern, and junk data inside valid structures.
    raw_response = {
        "q1_node_n0": [{"id": "CHEBI:111"}],  # Valid node
        "no_node_key": [{"id": "CHEBI:222"}],  # Invalid key, should be ignored
        "q1_node_n1": [
            {
                "id": "CHEBI:333",
                "in_edges_e0": [
                    None,  # Should be ignored
                    {"predicate": "pred1", "node_n2": {"id": "CHEBI:444"}},
                    "a string is not a dict",  # Should be ignored
                ],
            }
        ],
        "q1_node_n_bad": "not a list",  # Invalid value, should be ignored
    }

    parsed = dg_models.DgraphResponse.parse(raw_response)

    # The parser should find one valid query key: "q1"
    assert "q1" in parsed.data
    assert len(parsed.data) == 1

    # It should parse two valid nodes for the "q1" query
    assert len(parsed.data["q1"]) == 2

    # Check the node that should have one valid edge parsed from the junk
    node_with_edge = next((n for n in parsed.data["q1"] if n.binding == "n1"), None)
    assert node_with_edge is not None
    assert len(node_with_edge.edges) == 1
    assert node_with_edge.edges[0].binding == "e0"
    assert node_with_edge.edges[0].node.binding == "n2"


@pytest.mark.parametrize(
    "_case_name, raw_response, expected_key",
    [
        # A query that returns no results should have an empty list.
        ("Empty List", {"q0_node_n0": []}, "q0"),
        # A key with a null value should be gracefully handled.
        ("Null Value", {"q0_node_n0": None}, "q0"),
        # An empty response from Dgraph.
        ("Empty Response", {}, None),
    ],
)
def test_parse_empty_and_null_cases(
    _case_name: str, raw_response: dict[str, Any], expected_key: str | None
):
    """Test that the parser handles empty and null inputs correctly."""
    parsed = dg_models.DgraphResponse.parse(raw_response)

    if expected_key:
        # For cases that should produce a query key,
        # assert the key exists and its value is an empty list.
        assert expected_key in parsed.data
        assert parsed.data[expected_key] == []
    else:
        # For an empty response, the data should be an empty dict.
        assert parsed.data == {}
