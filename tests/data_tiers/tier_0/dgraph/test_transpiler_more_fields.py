import re
import pytest

from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.types.trapi import QueryGraphDict

# Example TRAPI query with category filter
CATEGORY_FILTER_QGRAPH = {
    "nodes": {
        "n0": {"categories": ["biolink:Gene"]},
        "n1": {"categories": ["biolink:Disease"]}
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "biolink:gene_associated_with_condition"}
    }
}

# Update MULTIPLE_FILTERS_QGRAPH
MULTIPLE_FILTERS_QGRAPH = {
    "nodes": {
        "n0": {
            "ids": ["MONDO:0005148"],
            "categories": ["biolink:Disease"],
            "constraints": [
                {"id": "name", "name": "name", "operator": "matches", "value": "diabetes"}
            ]
        },
        "n1": {
            "categories": ["biolink:Gene"],
            "constraints": [
                {"id": "description", "name": "description", "operator": "matches", "value": "insulin"}
            ]
        }
    },
    "edges": {
        "e0": {
            "object": "n0", 
            "subject": "n1", 
            "predicates": ["biolink:gene_associated_with_condition", "biolink:contributes_to"],
            "attribute_constraints": [
                {"id": "knowledge_level", "name": "knowledge_level", "operator": "==", "value": "primary"}
            ]
        }
    }
}

# Update NEGATED_CONSTRAINT_QGRAPH
NEGATED_CONSTRAINT_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["DOID:14330"]},
        "n1": {
            "categories": ["biolink:Protein"],
            "constraints": [
                {"id": "name", "name": "name", "operator": "matches", "value": "kinase", "not": True}
            ]
        }
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1"}
    }
}

# Update PUBLICATION_FILTER_QGRAPH
PUBLICATION_FILTER_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["DOID:14330"]},
        "n1": {
            "constraints": [
                {"id": "publications", "name": "publications", "operator": "in", "value": "PMID:12345678"}
            ]
        }
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1"}
    }
}

# Update NUMERIC_FILTER_QGRAPH
NUMERIC_FILTER_QGRAPH = {
    "nodes": {
        "n0": {"ids": ["DOID:14330"]},
        "n1": {"categories": ["biolink:Gene"]}
    },
    "edges": {
        "e0": {
            "object": "n0", 
            "subject": "n1",
            "attribute_constraints": [
                {"id": "edge_id", "name": "edge_id", "operator": ">", "value": "100"}
            ]
        }
    }
}


def test_category_filter():
    """Test filtering by category."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(CATEGORY_FILTER_QGRAPH)

    assert isinstance(query, str)
    
    # Updated expected structure
    expected_structure = """
    {
        node(func: eq(category, "biolink:Gene")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "biolink:gene_associated_with_condition")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(category, "biolink:Disease")) {
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


def test_multiple_filters():
    """Test multiple filters on nodes and edges."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(MULTIPLE_FILTERS_QGRAPH)
    
    assert isinstance(query, str)
    
    # Updated expected structure
    expected_structure = """
    {
        node(func: eq(id, "MONDO:0005148")) @filter(eq(category, "biolink:Disease") AND anyoftext(name, "diabetes")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, ["biolink:gene_associated_with_condition", "biolink:contributes_to"]) AND eq(knowledge_level, "primary")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(category, "biolink:Gene") AND anyoftext(description, "insulin")) {
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


def test_negated_constraint():
    """Test negated constraints."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(NEGATED_CONSTRAINT_QGRAPH)
    
    assert isinstance(query, str)
    
    expected_structure = """
    {
        node(func: eq(id, "DOID:14330")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(category, "biolink:Protein") AND NOT(anyoftext(name, "kinase"))) {
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


def test_publication_filter():
    """Test filtering by publication."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(PUBLICATION_FILTER_QGRAPH)
    
    assert isinstance(query, str)
    
    expected_structure = """
    {
        node(func: eq(id, "DOID:14330")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(publications, "PMID:12345678")) {
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


def test_numeric_filter():
    """Test numeric comparison filters."""
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(NUMERIC_FILTER_QGRAPH)
    
    assert isinstance(query, str)
    
    expected_structure = """
    {
        node(func: eq(id, "DOID:14330")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(gt(edge_id, "100")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(category, "biolink:Gene")) {
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


def test_complex_query():
    """Test a complex query with multiple filter types."""
    # Define a complex query with various filter types
    complex_query = {
        "nodes": {
            "n0": {
                "ids": ["MONDO:0005148"],
                "categories": ["biolink:Disease"],
                "constraints": [
                    {"id": "name", "name": "name", "operator": "matches", "value": "diabetes"},
                    {"id": "description", "name": "description", "operator": "matches", "value": "pancreas"}
                ]
            },
            "n1": {
                "categories": ["biolink:Gene", "biolink:Protein"],
                "constraints": [
                    {"id": "equivalent_curies", "name": "equivalent_curies", "operator": "in", "value": ["HGNC:1234", "HGNC:5678"]}
                ]
            },
            "n2": {
                "constraints": [
                    {"id": "publications", "name": "publications", "operator": "in", "value": "PMID:12345678"},
                    {"id": "name", "name": "name", "operator": "==", "value": "insulin", "not": True}
                ]
            }
        },
        "edges": {
            "e0": {
                "object": "n0", 
                "subject": "n1",
                "predicates": ["biolink:gene_associated_with_condition"]
            },
            "e1": {
                "object": "n1", 
                "subject": "n2",
                "attribute_constraints": [
                    {"id": "knowledge_level", "name": "knowledge_level", "operator": "==", "value": "primary"},
                    {"id": "edge_id", "name": "edge_id", "operator": "<", "value": "1000"}
                ]
            }
        }
    }
    
    transpiler = DgraphTranspiler()
    query = transpiler._convert_multihop(complex_query)
    
    assert isinstance(query, str)
    
    expected_structure = """
    {
        node(func: eq(id, "MONDO:0005148")) @filter(eq(category, "biolink:Disease") AND anyoftext(name, "diabetes") AND anyoftext(description, "pancreas")) @cascade {
            id name category all_names all_categories iri equivalent_curies description publications
            in_edges: ~source @filter(eq(predicate, "biolink:gene_associated_with_condition")) {
                predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                node: target @filter(eq(category, ["biolink:Gene", "biolink:Protein"]) AND eq(equivalent_curies, ["HGNC:1234", "HGNC:5678"])) {
                    id name category all_names all_categories iri equivalent_curies description publications
                    in_edges: ~source @filter(eq(knowledge_level, "primary") AND lt(edge_id, "1000")) {
                        predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
                        node: target @filter(eq(publications, "PMID:12345678") AND NOT(eq(name, "insulin"))) {
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
