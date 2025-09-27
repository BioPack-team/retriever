import json

import pytest

from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


@pytest.mark.parametrize("shape", ["grpc_bytes", "http_wrapped"])
def test_parse_response_full_fields_both_shapes(shape: str) -> None:
    # Build a fully-populated nested node used by an in-edge
    target_node = {
        "id": "UMLS:C0282090",
        "name": "Laxatives",
        "category": "biolink:Drug",
        "all_names": ["Laxatives", "Laxative agents"],
        "all_categories": ["biolink:Drug", "biolink:NamedThing"],
        "iri": "http://example.org/UMLS:C0282090",
        "equivalent_curies": ["RXCUI:12345"],
        "description": "Target description",
        "publications": ["PMID:222", "PMID:333"],
    }

    # Root node (NodeResult) with all fields plus in_edges that include one valid and one junk entry
    root_node_q1 = {
        "id": "CHEBI:3125",
        "name": "Bisacodyl",
        "category": "biolink:SmallMolecule",
        "all_names": ["Bisacodyl", "C18H13Cl2NO5"],
        "all_categories": ["biolink:SmallMolecule", "biolink:NamedThing"],
        "iri": "http://example.org/CHEBI:3125",
        "equivalent_curies": ["CHEMBL:CHEBI_3125"],
        "description": "A stimulant laxative",
        "publications": ["PMID:111", "PMID:112"],
        "in_edges": [
            {
                "predicate": "interacts_with",
                "primary_knowledge_source": "infores:kp",
                "knowledge_level": "prediction",
                "agent_type": "automated_agent",
                "kg2_ids": ["KG2:123", "KG2:456"],
                "domain_range_exclusion": False,
                "edge_id": "edge-1",
                "node": target_node,
            },
            "junk_edge_should_be_ignored",
        ],
    }

    root_node_q2 = {
        "id": "CHEBI:4514",
        "name": "Acetylsalicylic acid",
        "category": "biolink:SmallMolecule",
        "all_names": ["Aspirin"],
        "all_categories": ["biolink:SmallMolecule"],
        "iri": "http://example.org/CHEBI:4514",
        "equivalent_curies": ["CHEMBL:CHEBI_4514"],
        "description": "Pain reliever",
        "publications": ["PMID:444"],
        "in_edges": [
            {
                "predicate": "related_to",
                "primary_knowledge_source": "infores:kg",
                "knowledge_level": "assertion",
                "agent_type": "manual_agent",
                "kg2_ids": ["KG2:777"],
                "domain_range_exclusion": True,
                "edge_id": "edge-2",
                "node": target_node,
            }
        ],
    }

    # Include a non-mapping item in q1 list to exercise ignore logic
    data_map = {
        "q1": [root_node_q1, 123],
        "q2": [root_node_q2],
    }

    if shape == "grpc_bytes":
        raw = json.dumps(data_map).encode("utf-8")
    elif shape == "http_wrapped":
        raw = {"data": data_map, "extensions": {"server_latency": {"total_ns": 1}}}
    else:
        raise AssertionError("Unknown shape")

    result = dg_models.parse_response(raw)

    # We should have both queries present
    assert set(result.data.keys()) == {"q1", "q2"}

    # q1
    q1_nodes = result.data["q1"]
    assert len(q1_nodes) == 1  # non-mapping list item ignored
    n1 = q1_nodes[0]
    assert n1.id == "CHEBI:3125"
    assert n1.name == "Bisacodyl"
    assert n1.category == "biolink:SmallMolecule"
    assert n1.all_names == ["Bisacodyl", "C18H13Cl2NO5"]
    assert n1.all_categories == ["biolink:SmallMolecule", "biolink:NamedThing"]
    assert n1.iri == "http://example.org/CHEBI:3125"
    assert n1.equivalent_curies == ["CHEMBL:CHEBI_3125"]
    assert n1.description == "A stimulant laxative"
    assert n1.publications == ["PMID:111", "PMID:112"]

    # q1 in-edges
    assert len(n1.in_edges) == 1  # junk in_edge ignored
    e1 = n1.in_edges[0]
    assert e1.predicate == "interacts_with"
    assert e1.primary_knowledge_source == "infores:kp"
    assert e1.knowledge_level == "prediction"
    assert e1.agent_type == "automated_agent"
    assert e1.kg2_ids == ["KG2:123", "KG2:456"]
    assert e1.domain_range_exclusion is False
    assert e1.edge_id == "edge-1"

    # nested target node on e1
    t1 = e1.node
    assert t1.id == "UMLS:C0282090"
    assert t1.name == "Laxatives"
    assert t1.category == "biolink:Drug"
    assert t1.all_names == ["Laxatives", "Laxative agents"]
    assert t1.all_categories == ["biolink:Drug", "biolink:NamedThing"]
    assert t1.iri == "http://example.org/UMLS:C0282090"
    assert t1.equivalent_curies == ["RXCUI:12345"]
    assert t1.description == "Target description"
    assert t1.publications == ["PMID:222", "PMID:333"]

    # q2
    q2_nodes = result.data["q2"]
    assert len(q2_nodes) == 1
    n2 = q2_nodes[0]
    assert n2.id == "CHEBI:4514"
    assert n2.name == "Acetylsalicylic acid"
    assert n2.category == "biolink:SmallMolecule"
    assert n2.all_names == ["Aspirin"]
    assert n2.all_categories == ["biolink:SmallMolecule"]
    assert n2.iri == "http://example.org/CHEBI:4514"
    assert n2.equivalent_curies == ["CHEMBL:CHEBI_4514"]
    assert n2.description == "Pain reliever"
    assert n2.publications == ["PMID:444"]

    e2 = n2.in_edges[0]
    assert e2.predicate == "related_to"
    assert e2.primary_knowledge_source == "infores:kg"
    assert e2.knowledge_level == "assertion"
    assert e2.agent_type == "manual_agent"
    assert e2.kg2_ids == ["KG2:777"]
    assert e2.domain_range_exclusion is True
    assert e2.edge_id == "edge-2"

    t2 = e2.node
    assert t2.id == "UMLS:C0282090"
    assert t2.name == "Laxatives"
    assert t2.category == "biolink:Drug"
    assert t2.all_names == ["Laxatives", "Laxative agents"]
    assert t2.all_categories == ["biolink:Drug", "biolink:NamedThing"]
    assert t2.iri == "http://example.org/UMLS:C0282090"
    assert t2.equivalent_curies == ["RXCUI:12345"]
    assert t2.description == "Target description"
    assert t2.publications == ["PMID:222", "PMID:333"]


def test_parse_response_missing_data_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        dg_models.parse_response({"meta": {"ok": True}})
