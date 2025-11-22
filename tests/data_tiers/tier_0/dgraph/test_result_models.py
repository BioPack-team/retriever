import pytest
from typing import Any

from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


def test_parse_single_success_case_versioned():
    """Test parsing of a well-formed, multi-hop Dgraph response."""
    # The raw response must match the actual Dgraph output format.
    # A well-formed response with data for all node and edge properties.
    raw_response = {
        "q1_node_n0": [
            {
                "vA_name": "cytoplasmic vesicle",
                "vA_information_content": 56.8,
                "vA_equivalent_identifiers": [
                    "GO:0031410"
                ],
                "vA_id": "GO:0031410",
                "vA_category": [
                    "NamedThing",
                    "OrganismalEntity",
                    "PhysicalEssence",
                    "PhysicalEssenceOrOccurrent",
                    "CellularComponent",
                    "ThingWithTaxon",
                    "SubjectOfInvestigation",
                    "AnatomicalEntity",
                    "BiologicalEntity"
                ],
                "vA_description": "A vesicle found in the cytoplasm of a cell.",
                "in_edges_e0": [
                    {
                        "vA_knowledge_level": "prediction",
                        "vA_has_evidence": [
                            "ECO:IEA"
                        ],
                        "vA_original_subject": "UniProtKB:Q9UMZ2",
                        "vA_sources": [
                            {
                                "vA_resource_id": "infores:biolink",
                                "vA_resource_role": "aggregator_knowledge_source",
                                "vA_upstream_resource_ids": ["infores:goa"],
                                "vA_source_record_urls": ["https://example.com/record/123"]
                            },
                            {
                                "vA_resource_id": "infores:goa",
                                "vA_resource_role": "primary_knowledge_source"
                            }
                        ],
                        "vA_ecategory": [
                            "Association"
                        ],
                        "vA_predicate": "located_in",
                        "vA_source_inforeses": [
                            "infores:biolink",
                            "infores:goa"
                        ],
                        "vA_predicate_ancestors": [
                            "related_to_at_instance_level",
                            "located_in",
                            "related_to"
                        ],
                        "vA_agent_type": "automated_agent",
                        "vA_original_object": "GO:0031410",
                        "vA_eid": "urn:uuid:0763a393-7cc8-4d80-8720-0efcc0f9245f",
                        "node_n1": {
                            "vA_information_content": 83.6,
                            "vA_category": [
                                "MacromolecularMachineMixin",
                                "NamedThing",
                                "Gene",
                                "ChemicalEntityOrProteinOrPolypeptide",
                                "PhysicalEssence",
                                "PhysicalEssenceOrOccurrent",
                                "OntologyClass",
                                "ChemicalEntityOrGeneOrGeneProduct",
                                "GeneOrGeneProduct",
                                "Polypeptide",
                                "ThingWithTaxon",
                                "GenomicEntity",
                                "GeneProductMixin",
                                "Protein",
                                "BiologicalEntity"
                            ],
                            "vA_equivalent_identifiers": [
                                "PR:Q9UMZ2",
                                "OMIM:607291",
                                "UniProtKB:Q9UMZ2",
                                "ENSEMBL:ENSG00000275066",
                                "UMLS:C1412437",
                                "UMLS:C0893518",
                                "MESH:C121510",
                                "HGNC:557",
                                "NCBIGene:11276"
                            ],
                            "vA_id": "NCBIGene:11276",
                            "vA_name": "SYNRG",
                            "vA_description": "synergin gamma",
                            "vA_in_taxon": [
                                "NCBITaxon:9606"
                            ]
                        }
                    }
                ]
            }
        ]
    }

    # 1. Parse the response
    parsed = dg_models.DgraphResponse.parse(raw_response, prefix="vA_")
    assert "q1" in parsed.data
    assert len(parsed.data["q1"]) == 1

    # 2. Assertions for the root node (n0)
    root_node = parsed.data["q1"][0]
    assert root_node.binding == "n0"
    assert root_node.id == "GO:0031410"
    assert root_node.name == "cytoplasmic vesicle"
    assert root_node.category == [
        "NamedThing",
        "OrganismalEntity",
        "PhysicalEssence",
        "PhysicalEssenceOrOccurrent",
        "CellularComponent",
        "ThingWithTaxon",
        "SubjectOfInvestigation",
        "AnatomicalEntity",
        "BiologicalEntity",
    ]
    assert root_node.information_content == 56.8
    assert root_node.equivalent_identifiers == ["GO:0031410"]
    assert root_node.description == "A vesicle found in the cytoplasm of a cell."
    assert len(root_node.edges) == 1

    # 3. Assertions for the incoming edge (e0)
    in_edge = root_node.edges[0]
    assert in_edge.binding == "e0"
    assert in_edge.direction == "in"
    assert in_edge.predicate == "located_in"
    assert in_edge.knowledge_level == "prediction"
    assert in_edge.agent_type == "automated_agent"
    assert in_edge.has_evidence == ["ECO:IEA"]
    assert in_edge.original_subject == "UniProtKB:Q9UMZ2"
    assert in_edge.original_object == "GO:0031410"
    assert in_edge.source_inforeses == ["infores:biolink", "infores:goa"]
    assert in_edge.predicate_ancestors == [
        "related_to_at_instance_level",
        "located_in",
        "related_to",
    ]
    assert in_edge.id == "urn:uuid:0763a393-7cc8-4d80-8720-0efcc0f9245f"
    assert in_edge.category == ["Association"]
    assert in_edge.sources == [
        dg_models.Source(resource_id="infores:biolink", resource_role="aggregator_knowledge_source", upstream_resource_ids=["infores:goa"], source_record_urls=["https://example.com/record/123"]),
        dg_models.Source(resource_id="infores:goa", resource_role="primary_knowledge_source", upstream_resource_ids=[], source_record_urls=[]),
    ]

    # 4. Assertions for the connected node (n1)
    connected_node = in_edge.node
    assert connected_node.binding == "n1"
    assert connected_node.id == "NCBIGene:11276"
    assert connected_node.name == "SYNRG"
    assert connected_node.description == "synergin gamma"
    assert connected_node.information_content == 83.6
    assert connected_node.in_taxon == ["NCBITaxon:9606"]
    assert connected_node.category == [
        "MacromolecularMachineMixin",
        "NamedThing",
        "Gene",
        "ChemicalEntityOrProteinOrPolypeptide",
        "PhysicalEssence",
        "PhysicalEssenceOrOccurrent",
        "OntologyClass",
        "ChemicalEntityOrGeneOrGeneProduct",
        "GeneOrGeneProduct",
        "Polypeptide",
        "ThingWithTaxon",
        "GenomicEntity",
        "GeneProductMixin",
        "Protein",
        "BiologicalEntity",
    ]
    assert connected_node.equivalent_identifiers == [
        "PR:Q9UMZ2",
        "OMIM:607291",
        "UniProtKB:Q9UMZ2",
        "ENSEMBL:ENSG00000275066",
        "UMLS:C1412437",
        "UMLS:C0893518",
        "MESH:C121510",
        "HGNC:557",
        "NCBIGene:11276",
    ]


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


def test_parse_symmetric_predicate_success_case():
    raw_response = {
        "q0_node_n1": [{
            "vC_id": "NCBIGene:3778",
            "vC_name": "KCNMA1",
            "out_edges_e0": [
                {
                    "vC_eid": "urn:uuid:3281005f-e94e-4e76-8779-01a6e812d397",
                    "vC_predicate": "has_phenotype",
                    "vC_predicate_ancestors": [
                        "related_to_at_instance_level",
                        "related_to",
                        "has_phenotype"
                    ],
                    "node_n0": {
                        "vC_name": "Hypertelorism",
                        "vC_id": "HP:0000316"
                    }
                }
            ],
            "in_edges-symmetric_e0": [
                {
                    "vC_eid": "urn:uuid:f0e89fda-4d67-45a3-a4dc-a6a2aac2e873",
                    "vC_predicate": "has_part",
                    "vC_predicate_ancestors": [
                        "has_part",
                        "related_to_at_instance_level",
                        "overlaps",
                        "related_to"
                    ],
                    "node_n0": {
                        "vC_id": "GO:0042391",
                        "vC_name": "regulation of membrane potential",
                    }
                }
            ],
            "in_edges_e0": [
                {
                    "vC_eid": "urn:uuid:f0e89fda-4d67-45a3-a4dc-a6a2aac2e873",
                    "vC_predicate": "has_part",
                    "vC_predicate_ancestors": [
                        "has_part",
                        "related_to_at_instance_level",
                        "overlaps",
                        "related_to"
                    ],
                    "node_n0": {
                        "vC_id": "GO:0034702",
                        "vC_name": "monoatomic ion channel complex",
                    }
                }
            ]            
        }]
    }

    # 1. Parse the response
    parsed = dg_models.DgraphResponse.parse(raw_response, prefix="vC_")
    assert "q0" in parsed.data
    assert len(parsed.data["q0"]) == 1

    # 2. Assertions for the root node (n1)
    root_node = parsed.data["q0"][0]
    assert root_node.binding == "n1"
    assert root_node.id == "NCBIGene:3778"
    assert root_node.name == "KCNMA1"

    # 3. Assert symmetric predicates: both in_edges_e0 and in_edges_e0_reverse 
    #    are merged under binding "e0" (due to split("_", 3))
    # Total: 1 out_edge + 2 in_edges (one from in_edges_e0, one from in_edges_e0_reverse)
    assert len(root_node.edges) == 3, "Should have 3 edges total: 1 out + 2 in (merged)"

    # All edges with binding "e0"
    e0_edges = [e for e in root_node.edges if e.binding == "e0"]
    assert len(e0_edges) == 3, "All edges should have binding 'e0'"

    # 4. Separate by direction and predicate
    out_edges = [e for e in e0_edges if e.direction == "out"]
    in_edges = [e for e in e0_edges if e.direction == "in"]

    assert len(out_edges) == 1, "Should have 1 outgoing edge"
    assert len(in_edges) == 2, "Should have 2 incoming edges (merged from in_edges_e0 and in_edges_e0_reverse)"

    # 5. Verify the outgoing edge (has_phenotype)
    out_edge = out_edges[0]
    assert out_edge.predicate == "has_phenotype"
    assert "related_to" in out_edge.predicate_ancestors
    assert out_edge.node.binding == "n0"
    assert out_edge.node.id == "HP:0000316"
    assert out_edge.node.name == "Hypertelorism"

    # 6. Verify the incoming edges (both has_part - but pointing to different nodes)
    has_part_edges = [e for e in in_edges if e.predicate == "has_part"]
    assert len(has_part_edges) == 2, "Should have 2 has_part edges"

    # Separate the two incoming edges by node ID
    edge_to_membrane = next((e for e in has_part_edges if e.node.id == "GO:0042391"), None)
    edge_to_channel = next((e for e in has_part_edges if e.node.id == "GO:0034702"), None)

    assert edge_to_membrane is not None, "Should have edge to regulation of membrane potential"
    assert edge_to_channel is not None, "Should have edge to monoatomic ion channel complex"

    # Verify edge to GO:0042391 (from in_edges_e0_reverse)
    assert edge_to_membrane.predicate == "has_part"
    assert "related_to" in edge_to_membrane.predicate_ancestors
    assert edge_to_membrane.node.binding == "n0"
    assert edge_to_membrane.node.name == "regulation of membrane potential"

    # Verify edge to GO:0034702 (from in_edges_e0)
    assert edge_to_channel.predicate == "has_part"
    assert "related_to" in edge_to_channel.predicate_ancestors
    assert edge_to_channel.node.binding == "n0"
    assert edge_to_channel.node.name == "monoatomic ion channel complex"

    # 7. Verify all edges connect to nodes with binding "n0"
    assert all(e.node.binding == "n0" for e in root_node.edges)
