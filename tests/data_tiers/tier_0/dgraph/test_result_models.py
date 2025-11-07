import pytest
from typing import Any

from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


def test_parse_single_success_case_versioned():
    """Test parsing of a well-formed, multi-hop Dgraph response."""
    # The raw response must match the actual Dgraph output format.
    # A well-formed response with data for all node and edge properties.
    raw_response = {
        "q0_node_n0": [
            {
                "v8_id": "CHEBI:3125",
                "v8_name": "Bisacodyl",
                "v8_category": "biolink:SmallMolecule",
                "v8_all_names": ["Biscodyl", "Bisacodyl"],
                "v8_all_categories": ["SmallMolecule", "Drug"],
                "v8_iri": "http://purl.obolibrary.org/obo/CHEBI_3125",
                "v8_equivalent_curies": ["PUBCHEM.COMPOUND:2391"],
                "v8_description": "A stimulant laxative.",
                "v8_publications": ["PMID:12345"],
                "in_edges_e0": [
                    {
                        "v8_predicate": "interacts_with",
                        "v8_primary_knowledge_source": "infores:test-ks",
                        "v8_knowledge_level": "knowledge-level-val",
                        "v8_agent_type": "agent-type-val",
                        "v8_kg2_ids": ["kg2:abc"],
                        "v8_domain_range_exclusion": True,
                        "v8_qualified_object_aspect": "aspect-val",
                        "v8_qualified_object_direction": "direction-val",
                        "v8_qualified_predicate": "qualified-pred-val",
                        "v8_publications_info": "pub-info-val",
                        "node_n1": {
                            "v8_id": "UMLS:C0282090",
                            "v8_name": "Laxatives",
                            "v8_category": "biolink:Drug",
                            "v8_all_names": ["Laxative"],
                            "v8_all_categories": ["Drug"],
                            "v8_iri": "http://purl.obolibrary.org/obo/UMLS_C0282090",
                            "v8_equivalent_curies": [],
                            "v8_description": "A substance that promotes defecation.",
                            "v8_publications": [],
                        },
                    }
                ],
            }
        ]
    }

    # 1. Parse the response
    parsed = dg_models.DgraphResponse.parse(raw_response, prefix="v8_")
    assert "q0" in parsed.data
    assert len(parsed.data["q0"]) == 1

    # 2. Assertions for the root node (n0)
    root_node = parsed.data["q0"][0]
    assert root_node.binding == "n0"
    assert root_node.id == "CHEBI:3125"
    assert root_node.name == "Bisacodyl"
    assert root_node.category == "biolink:SmallMolecule"
    assert root_node.all_names == ["Biscodyl", "Bisacodyl"]
    assert root_node.all_categories == ["SmallMolecule", "Drug"]
    assert root_node.iri == "http://purl.obolibrary.org/obo/CHEBI_3125"
    assert root_node.equivalent_curies == ["PUBCHEM.COMPOUND:2391"]
    assert root_node.description == "A stimulant laxative."
    assert root_node.publications == ["PMID:12345"]
    assert len(root_node.edges) == 1

    # 3. Assertions for the incoming edge (e0)
    in_edge = root_node.edges[0]
    assert in_edge.binding == "e0"
    assert in_edge.direction == "in"
    assert in_edge.predicate == "interacts_with"
    assert in_edge.primary_knowledge_source == "infores:test-ks"
    assert in_edge.knowledge_level == "knowledge-level-val"
    assert in_edge.agent_type == "agent-type-val"
    assert in_edge.kg2_ids == ["kg2:abc"]
    assert in_edge.domain_range_exclusion is True
    assert in_edge.edge_id == "e0"  # Derived from the key
    assert in_edge.qualified_object_aspect == "aspect-val"
    assert in_edge.qualified_object_direction == "direction-val"
    assert in_edge.qualified_predicate == "qualified-pred-val"
    assert in_edge.publications_info == "pub-info-val"

    # 4. Assertions for the connected node (n1)
    connected_node = in_edge.node
    assert connected_node.binding == "n1"
    assert connected_node.id == "UMLS:C0282090"
    assert connected_node.name == "Laxatives"
    assert connected_node.category == "biolink:Drug"
    assert connected_node.all_names == ["Laxative"]
    assert connected_node.all_categories == ["Drug"]
    assert connected_node.iri == "http://purl.obolibrary.org/obo/UMLS_C0282090"
    assert connected_node.equivalent_curies == []
    assert connected_node.description == "A substance that promotes defecation."
    assert connected_node.publications == []


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
