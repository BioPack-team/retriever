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


def test_convert_multihop_query():
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(SIMPLE_QGRAPH)

    assert isinstance(query, str)
    assert '@cascade' in query
    assert 'node(func: eq(id, "Q1")) @cascade' in query
    assert 'in_edges: ~source @filter(eq(predicate,"interacts_with"))' in query
    assert 'node: target @filter(eq(id, "Q2"))' in query
    assert "id" in query
    assert "name" in query
    assert "category" in query
    assert "predicate" in query

    # Check full query structure - updated with reversed node direction
    expected_structure = """
    {
        node(func: eq(id, "Q1")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"interacts_with")) {
            predicate
            primary_knowledge_source
            node: target @filter(eq(id, "Q2")) {
                id
                name
                category
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
    # Updated: Starting with the last node in the chain
    assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    assert 'in_edges: ~source @filter(eq(predicate,"has_part"))' in query
    # Updated node order
    assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    assert 'node: target @filter(eq(id, "CL:1000445"))' in query

    # Check the full query structure for a multi-hop traversal - updated direction
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"has_part")) {
            predicate
            primary_knowledge_source
            node: target @filter(eq(id, "UBERON:0001608")) {
                id
                name
                category
                in_edges: ~source @filter(eq(predicate,"has_part")) {
                predicate
                primary_knowledge_source
                node: target @filter(eq(id, "CL:1000445")) {
                    id
                    name
                    category
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
    # Updated: Starting with the last node in the chain
    assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    assert 'in_edges: ~source @filter(eq(predicate,"has_part"))' in query
    assert 'in_edges: ~source @filter(eq(predicate,"develops_from"))' in query
    assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    assert 'node: target @filter(eq(id, "CL:1000445"))' in query
    assert 'node: target @filter(eq(id, "CL:0000185"))' in query

    # Check the full query structure for a three-hop traversal - updated direction
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"has_part")) {
                predicate
                primary_knowledge_source
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id
                    name
                    category
                    in_edges: ~source @filter(eq(predicate,"has_part")) {
                        predicate
                        primary_knowledge_source
                        node: target @filter(eq(id, "CL:1000445")) {
                            id
                            name
                            category
                            in_edges: ~source @filter(eq(predicate,"develops_from")) {
                                predicate
                                primary_knowledge_source
                                node: target @filter(eq(id, "CL:0000185")) {
                                    id
                                    name
                                    category
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
    # Updated: Start with the last node
    assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    # Updated: Check for the first node as a target
    assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    assert 'node: target @filter(eq(id, "CL:1000445"))' in query
    assert 'node: target @filter(eq(id, "CL:0000185"))' in query
    assert 'node: target @filter(eq(id, "CL:0000075"))' in query

    # Check the full query structure for a four-hop traversal - updated direction
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"has_part")) {
                predicate
                primary_knowledge_source
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id
                    name
                    category
                    in_edges: ~source @filter(eq(predicate,"has_part")) {
                        predicate
                        primary_knowledge_source
                        node: target @filter(eq(id, "CL:1000445")) {
                            id
                            name
                            category
                            in_edges: ~source @filter(eq(predicate,"develops_from")) {
                                predicate
                                primary_knowledge_source
                                node: target @filter(eq(id, "CL:0000185")) {
                                    id
                                    name
                                    category
                                    in_edges: ~source @filter(eq(predicate,"develops_from")) {
                                        predicate
                                        primary_knowledge_source
                                        node: target @filter(eq(id, "CL:0000075")) {
                                            id
                                            name
                                            category
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
    # Updated: Start with the last node
    assert 'node(func: eq(id, "UBERON:0001769")) @cascade' in query
    # Updated: Check for first node as a target
    assert 'node: target @filter(eq(id, "UBERON:0001608"))' in query
    assert 'node: target @filter(eq(id, "CL:1000445"))' in query
    assert 'node: target @filter(eq(id, "CL:0000185"))' in query
    assert 'node: target @filter(eq(id, "CL:0000075"))' in query
    assert 'node: target @filter(eq(id, "UMLS:C1257909"))' in query
    assert 'in_edges: ~source @filter(eq(predicate,"associated_with"))' in query

    # Check the full query structure for a five-hop traversal - updated direction
    expected_structure = """
    {
        node(func: eq(id, "UBERON:0001769")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"has_part")) {
                predicate
                primary_knowledge_source
                node: target @filter(eq(id, "UBERON:0001608")) {
                    id
                    name
                    category
                    in_edges: ~source @filter(eq(predicate,"has_part")) {
                        predicate
                        primary_knowledge_source
                        node: target @filter(eq(id, "CL:1000445")) {
                            id
                            name
                            category
                            in_edges: ~source @filter(eq(predicate,"develops_from")) {
                                predicate
                                primary_knowledge_source
                                node: target @filter(eq(id, "CL:0000185")) {
                                    id
                                    name
                                    category
                                    in_edges: ~source @filter(eq(predicate,"develops_from")) {
                                        predicate
                                        primary_knowledge_source
                                        node: target @filter(eq(id, "CL:0000075")) {
                                            id
                                            name
                                            category
                                            in_edges: ~source @filter(eq(predicate,"associated_with")) {
                                                predicate
                                                primary_knowledge_source
                                                node: target @filter(eq(id, "UMLS:C1257909")) {
                                                    id
                                                    name
                                                    category
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
    transpiler = DgraphTranspiler()
    # Add multiple IDs to test batch
    batch_qgraph = {
        "nodes": {
            "n0": {"ids": ["Q1", "Q2"]},
            "n1": {"ids": ["Q3"]}
        },
        "edges": {
            "e0": {"object": "n0", "subject": "n1", "predicate": "interacts_with"}
        }
    }

    query = transpiler._convert_batch_multihop(batch_qgraph)

    assert isinstance(query, str)
    assert "node0(func: eq(id, \"Q1\"))" in query
    assert '@cascade' in query
    assert 'in_edges: ~source @filter(eq(predicate,"interacts_with"))' in query
    assert "node1(func: eq(id, \"Q2\"))" in query
    assert 'node: target @filter(eq(id, "Q3"))' in query
    assert "id" in query
    assert "name" in query
    assert "category" in query
    assert "predicate" in query

    # Check the full batch query structure - updated direction
    expected_structure = """
    {
        node0(func: eq(id, "Q1")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"interacts_with")) {
                predicate
                primary_knowledge_source
                node: target @filter(eq(id, "Q3")) {
                    id
                    name
                    category
                }
            }
        }

        node1(func: eq(id, "Q2")) @cascade {
            id
            name
            category
            in_edges: ~source @filter(eq(predicate,"interacts_with")) {
                predicate
                primary_knowledge_source
                node: target @filter(eq(id, "Q3")) {
                    id
                    name
                    category
                }
            }
        }
    }
    """.strip()

    # Normalize spaces to make comparison more robust - account for different line breaks
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
