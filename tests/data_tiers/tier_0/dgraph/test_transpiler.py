import pytest
import re

from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler

# Example minimal TRAPI query graph
SIMPLE_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["Q1"]},
        "n1": {"ids": ["Q2"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "interacts_with"}
    }
}

# Example minimal TRAPI query graph with multiple ids per node
SIMPLE_QGRAPH_MULTIPLE_IDS = {
    "nodes": {
        "n0": {"ids": ["Q0", "Q1"]},
        "n1": {"ids": ["Q2", "Q3"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "interacts_with"}
    }
}

# 2-hop query
TWO_HOP_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"}
    }
}

# 3-hop query
THREE_HOP_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
        "n3": {"ids": ["CL:0000185"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "develops_from"}
    }
}

# 4-hop query
FOUR_HOP_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
        "n3": {"ids": ["CL:0000185"]},
        "n4": {"ids": ["CL:0000075"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "develops_from"},
        "e3": {"object": "n3", "subject": "n4", "predicate": "develops_from"}
    }
}

# 5-hop query
FIVE_HOP_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
        "n3": {"ids": ["CL:0000185"]},
        "n4": {"ids": ["CL:0000075"]},
        "n5": {"ids": ["UMLS:C1257909"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "develops_from"},
        "e3": {"object": "n3", "subject": "n4", "predicate": "develops_from"},
        "e4": {"object": "n4", "subject": "n5", "predicate": "associated_with"}
    }
}

# 5-hop query with multiple ids per node
FIVE_HOP_QGRAPH_MULTIPLE_IDS = {
    "nodes": {
        "n0": {"ids": ["Q0, Q1"]},
        "n1": {"ids": ["Q2, Q3"]},
        "n2": {"ids": ["Q4, Q5"]},
        "n3": {"ids": ["Q6, Q7"]},
        "n4": {"ids": ["Q8, Q9"]},
        "n5": {"ids": ["Q10, Q11"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "P0"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "P1"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "P2"},
        "e3": {"object": "n3", "subject": "n4", "predicate": "P3"},
        "e4": {"object": "n4", "subject": "n5", "predicate": "P4"}
    }
}


def test_convert_multihop_simple_query():
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(SIMPLE_QGRAPH)

    assert isinstance(query, str)
    # assert '@cascade' in query
    # assert 'node(func: eq(id, "Q1")) @cascade' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"interacts_with"))' in query
    # assert 'node: target @filter(eq(id, "Q2"))' in query
    # assert "id" in query
    # assert "name" in query
    # assert "category" in query
    # assert "predicate" in query

    # Check full query structure
    expected_structure = """
    {
        node(func: eq(id, "Q1")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "interacts_with")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, "Q2")) {
                    id name category all_names all_categories iri equivalent_curies description publications
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_multihop_multiple_ids_query():
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(SIMPLE_QGRAPH_MULTIPLE_IDS)

    assert isinstance(query, str)
    # assert '@cascade' in query
    # assert 'node(func: eq(id, ["Q0", "Q1"])) @cascade' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"interacts_with"))' in query
    # assert 'node: target @filter(eq(id, ["Q2", "Q3"]))' in query
    # assert "id" in query
    # assert "name" in query
    # assert "category" in query
    # assert "predicate" in query

    # Check full query structure - updated with reversed node direction
    expected_structure = """
    {
        node(func: eq(id, ["Q0", "Q1"])) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "interacts_with")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, ["Q2", "Q3"])) {
                    id name category all_names all_categories iri equivalent_curies description publications
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_2hop_query():
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(TWO_HOP_QGRAPH)

    assert isinstance(query, str)
    # assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"has_part"))' in query
    # assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    # assert 'node: target @filter(eq(id, "CL:1000445"))' in query

    # Check the full query structure for a multi-hop traversal - updated direction
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "has_part")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id name category all_names all_categories iri equivalent_curies description publications
                        in_edges: ~source @filter(eq(predicate, "has_part")) {
                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                        node: target @filter(eq(id, "CL:1000445")) {
                            id name category all_names all_categories iri equivalent_curies description publications
                        }
                    }
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_3hop_query():
    """Test the transpiler with a 3-hop query."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(THREE_HOP_QGRAPH)

    assert isinstance(query, str)
    # assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"has_part"))' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"develops_from"))' in query
    # assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    # assert 'node: target @filter(eq(id, "CL:1000445"))' in query
    # assert 'node: target @filter(eq(id, "CL:0000185"))' in query

    # Check the full query structure for a three-hop traversal
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "has_part")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id name category all_names all_categories iri equivalent_curies description publications
                    in_edges: ~source @filter(eq(predicate, "has_part")) {
                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                        node: target @filter(eq(id, "CL:1000445")) {
                            id name category all_names all_categories iri equivalent_curies description publications
                            in_edges: ~source @filter(eq(predicate, "develops_from")) {
                                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                node: target @filter(eq(id, "CL:0000185")) {
                                    id name category all_names all_categories iri equivalent_curies description publications
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_4hop_query():
    """Test the transpiler with a 4-hop query."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(FOUR_HOP_QGRAPH)

    assert isinstance(query, str)
    # assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    # assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    # assert 'node: target @filter(eq(id, "CL:1000445"))' in query
    # assert 'node: target @filter(eq(id, "CL:0000185"))' in query
    # assert 'node: target @filter(eq(id, "CL:0000075"))' in query

    # Check the full query structure for a four-hop traversal
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "has_part")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id name category all_names all_categories iri equivalent_curies description publications
                    in_edges: ~source @filter(eq(predicate, "has_part")) {
                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                        node: target @filter(eq(id, "CL:1000445")) {
                            id name category all_names all_categories iri equivalent_curies description publications
                            in_edges: ~source @filter(eq(predicate, "develops_from")) {
                                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                node: target @filter(eq(id, "CL:0000185")) {
                                    id name category all_names all_categories iri equivalent_curies description publications
                                    in_edges: ~source @filter(eq(predicate, "develops_from")) {
                                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                        node: target @filter(eq(id, "CL:0000075")) {
                                            id name category all_names all_categories iri equivalent_curies description publications
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_5hop_query():
    """Test the transpiler with a 5-hop query."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(FIVE_HOP_QGRAPH)

    assert isinstance(query, str)
    # # Updated: Start with the last node
    # assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    # # Updated: Check for first node as a target
    # assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    # assert 'node: target @filter(eq(id, "CL:1000445"))' in query
    # assert 'node: target @filter(eq(id, "CL:0000185"))' in query
    # assert 'node: target @filter(eq(id, "CL:0000075"))' in query
    # assert 'node: target @filter(eq(id, "UMLS:C1257909"))' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"associated_with"))' in query

    # Check the full query structure for a five-hop traversal
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "has_part")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id name category all_names all_categories iri equivalent_curies description publications
                    in_edges: ~source @filter(eq(predicate, "has_part")) {
                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                        node: target @filter(eq(id, "CL:1000445")) {
                            id name category all_names all_categories iri equivalent_curies description publications
                            in_edges: ~source @filter(eq(predicate, "develops_from")) {
                                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                node: target @filter(eq(id, "CL:0000185")) {
                                    id name category all_names all_categories iri equivalent_curies description publications
                                    in_edges: ~source @filter(eq(predicate, "develops_from")) {
                                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                        node: target @filter(eq(id, "CL:0000075")) {
                                            id name category all_names all_categories iri equivalent_curies description publications
                                            in_edges: ~source @filter(eq(predicate, "associated_with")) {
                                                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                                node: target @filter(eq(id, "UMLS:C1257909")) {
                                                    id name category all_names all_categories iri equivalent_curies description publications
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_5hop_with_multiple_ids_query():
    """Test the transpiler with a 5-hop multiple IDs query."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(FIVE_HOP_QGRAPH_MULTIPLE_IDS)

    assert isinstance(query, str)
    # assert 'node(func: eq(id, ["Q0", "Q1"])) @cascade' in query
    # assert 'node: target @filter(eq(id, ["Q2", "Q3"]))' in query
    # assert 'node: target @filter(eq(id, ["Q4", "Q5"]))' in query
    # assert 'node: target @filter(eq(id, ["Q6", "Q7"]))' in query
    # assert 'node: target @filter(eq(id, ["Q8", "Q9"]))' in query
    # assert 'node: target @filter(eq(id, ["Q10", "Q11"]))' in query
    # assert 'in_edges: ~source @filter(eq(predicate,"P0"))' in query

    # Check the full query structure for a five-hop traversal with multiple IDs
    expected_structure = """
    {
        node(func: eq(id, ["Q0", "Q1"])) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "P0")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, ["Q2", "Q3"])) {
                    id name category all_names all_categories iri equivalent_curies description publications
                    in_edges: ~source @filter(eq(predicate, "P1")) {
                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                        node: target @filter(eq(id, ["Q4", "Q5"])) {
                            id name category all_names all_categories iri equivalent_curies description publications
                            in_edges: ~source @filter(eq(predicate, "P2")) {
                                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                node: target @filter(eq(id, ["Q6", "Q7"])) {
                                    id name category all_names all_categories iri equivalent_curies description publications
                                    in_edges: ~source @filter(eq(predicate, "P3")) {
                                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                        node: target @filter(eq(id, ["Q8", "Q9"])) {
                                            id name category all_names all_categories iri equivalent_curies description publications
                                            in_edges: ~source @filter(eq(predicate, "P4")) {
                                                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                                                node: target @filter(eq(id, ["Q10", "Q11"])) {
                                                    id name category all_names all_categories iri equivalent_curies description publications
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_batch_multihop_query():
    """Test the batch multihop conversion function with a special batch container."""
    transpiler = DgraphTranspiler()

    # Create a batch container with multiple query graphs
    batch_container = {
        "query_graphs": [
            {
                "nodes": {
                    "n0": {"ids": ["Q0"]},
                    "n1": {"ids": ["Q1"]}
                },
                "edges": {
                    "e0": {"object": "n0", "subject": "n1", "predicate": "P0"}
                }
            },
            {
                "nodes": {
                    "n0": {"ids": ["Q0", "Q1"]},
                    "n1": {"ids": ["Q2", "Q3"]}
                },
                "edges": {
                    "e0": {"object": "n0", "subject": "n1", "predicate": "P0"}
                }
            }
        ]
    }

    query = transpiler._convert_batch_multihop(batch_container)

    assert isinstance(query, str)
    # # Check that both queries are in the result
    # assert 'node0(func: eq(id, "Q0")) @cascade' in query
    # assert 'node1(func: eq(id, ["Q0", "Q1"])) @cascade' in query
    # assert '@filter(eq(predicate,"P0"))' in query
    # assert 'node: target @filter(eq(id, "Q1"))' in query
    # assert 'node: target @filter(eq(id, ["Q2", "Q3"]))' in query

    # Check the full query structure
    expected_structure = """
    {
        node0(func: eq(id, "Q0")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "P0")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, "Q1")) {
                    id name category all_names all_categories iri equivalent_curies description publications
                }
            }
        }

        node1(func: eq(id, ["Q0", "Q1"])) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "P0")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(id, ["Q2", "Q3"])) {
                    id name category all_names all_categories iri equivalent_curies description publications
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust
    query_normalized = re.sub(r'\s+', ' ', query.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_structure.strip())

    assert query_normalized == expected_normalized


def test_convert_results_wraps_dict():
    transpiler = DgraphTranspiler()
    results = {"data": {"some": "value"}}
    backend_results = transpiler.convert_results(SIMPLE_QGRAPH, results)

    assert isinstance(backend_results, dict)
    assert "results" in backend_results
    assert backend_results["results"] == results
