import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

import pytest

from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.types.trapi import QueryGraphDict


# -----------------------
# Helpers
# -----------------------

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def assert_query_equals(actual: str, expected: str) -> None:
    assert normalize(actual) == normalize(expected)


@dataclass(frozen=True)
class QueryCase:
    """Pair of input TRAPI qgraph and expected Dgraph query."""
    name: str
    qgraph: QueryGraphDict
    expected: str


@dataclass(frozen=True)
class BatchCase:
    """Pair of input TRAPI container or single qgraph for batch API and expected Dgraph query."""
    name: str
    qgraph: dict[str, Any]  # container with "query_graphs" or single qgraph accepted by batch API
    expected: str


@dataclass(frozen=True)
class ResultsCase:
    """Pair for convert_results: input qgraph and raw backend results."""
    name: str
    qgraph: QueryGraphDict
    raw_results: dict[str, Any]
    expected_wrapped: dict[str, Any]


# -----------------------
# Query graph inputs
# -----------------------

SIMPLE_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["Q1"]}, "n1": {"ids": ["Q2"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicate": "interacts_with"}},
}

SIMPLE_QGRAPH_MULTIPLE_IDS: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["Q0", "Q1"]}, "n1": {"ids": ["Q2", "Q3"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicate": "interacts_with"}},
}

TWO_HOP_QGRAPH: QueryGraphDict = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
    },
}

THREE_HOP_QGRAPH: QueryGraphDict = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
        "n3": {"ids": ["CL:0000185"]},
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "develops_from"},
    },
}

FOUR_HOP_QGRAPH: QueryGraphDict = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
        "n3": {"ids": ["CL:0000185"]},
        "n4": {"ids": ["CL:0000075"]},
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "develops_from"},
        "e3": {"object": "n3", "subject": "n4", "predicate": "develops_from"},
    },
}

FIVE_HOP_QGRAPH: QueryGraphDict = {
    "nodes": {
        "n0": {"ids": ["UBERON:0001769"]},
        "n1": {"ids": ["UBERON:0001608"]},
        "n2": {"ids": ["CL:1000445"]},
        "n3": {"ids": ["CL:0000185"]},
        "n4": {"ids": ["CL:0000075"]},
        "n5": {"ids": ["UMLS:C1257909"]},
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "has_part"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "has_part"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "develops_from"},
        "e3": {"object": "n3", "subject": "n4", "predicate": "develops_from"},
        "e4": {"object": "n4", "subject": "n5", "predicate": "associated_with"},
    },
}

FIVE_HOP_QGRAPH_MULTIPLE_IDS: QueryGraphDict = {
    "nodes": {
        "n0": {"ids": ["Q0", "Q1"]},
        "n1": {"ids": ["Q2", "Q3"]},
        "n2": {"ids": ["Q4", "Q5"]},
        "n3": {"ids": ["Q6", "Q7"]},
        "n4": {"ids": ["Q8", "Q9"]},
        "n5": {"ids": ["Q10", "Q11"]},
    },
    "edges": {
        "e0": {"object": "n0", "subject": "n1", "predicate": "P0"},
        "e1": {"object": "n1", "subject": "n2", "predicate": "P1"},
        "e2": {"object": "n2", "subject": "n3", "predicate": "P2"},
        "e3": {"object": "n3", "subject": "n4", "predicate": "P3"},
        "e4": {"object": "n4", "subject": "n5", "predicate": "P4"},
    },
}

CATEGORY_FILTER_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"categories": ["biolink:Gene"]}, "n1": {"categories": ["biolink:Disease"]}},
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicate": "biolink:gene_associated_with_condition",
        },
    },
}

MULTIPLE_FILTERS_QGRAPH: QueryGraphDict = {
    "nodes": {
        "n0": {
            "ids": ["MONDO:0005148"],
            "categories": ["biolink:Disease"],
            "constraints": [{"id": "name", "name": "name", "operator": "matches", "value": "diabetes"}],
        },
        "n1": {
            "categories": ["biolink:Gene"],
            "constraints": [{"id": "description", "name": "description", "operator": "matches", "value": "insulin"}],
        },
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": [
                "biolink:gene_associated_with_condition",
                "biolink:contributes_to",
            ],
            "attribute_constraints": [{"id": "knowledge_level", "name": "knowledge_level", "operator": "==", "value": "primary"}],
        }
    },
}

NEGATED_CONSTRAINT_QGRAPH: QueryGraphDict = {
    "nodes": {
        "n0": {"ids": ["DOID:14330"]},
        "n1": {
            "categories": ["biolink:Protein"],
            "constraints": [{"id": "name", "name": "name", "operator": "matches", "value": "kinase", "not": True}],
        },
    },
    "edges": {"e0": {"object": "n0", "subject": "n1"}},
}

PUBLICATION_FILTER_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["DOID:14330"]}, "n1": {"constraints": [{"id": "publications", "name": "publications", "operator": "in", "value": "PMID:12345678"}]}},
    "edges": {"e0": {"object": "n0", "subject": "n1"}},
}

NUMERIC_FILTER_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["DOID:14330"]}, "n1": {"categories": ["biolink:Gene"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "attribute_constraints": [{"id": "edge_id", "name": "edge_id", "operator": ">", "value": "100"}]}},
}

# Special single-graph inputs to hit specific branches
SINGLE_STRING_WITH_COMMAS_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["Q0, Q1"]}, "n1": {"ids": ["Q2"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicate": "P"}},
}

PREDICATES_SINGLE_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["A"]}, "n1": {"ids": ["B"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicates": ["Ponly"]}},
}

ATTRIBUTES_ONLY_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["A"]}, "n1": {"ids": ["B"]}},
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "attribute_constraints": [{"id": "knowledge_level", "name": "knowledge_level", "operator": "==", "value": "primary"}],
        }
    },
}

START_OBJECT_WITH_IDS_QGRAPH: QueryGraphDict = {
    "nodes": {"n0": {"ids": ["X"]}, "n1": {"ids": ["Y"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicate": "rel"}},
}

# Batch inputs
BATCH_CONTAINER_QGRAPH: dict[str, list[QueryGraphDict]] = {
    "query_graphs": [
        SIMPLE_QGRAPH,
        SIMPLE_QGRAPH_MULTIPLE_IDS,
    ]
}

BATCH_MULTI_IDS_SINGLE_GRAPH: dict[str, Any] = {
    "nodes": {"n0": {"ids": ["A", "B"]}, "n1": {"ids": ["C"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicate": "P"}},
}

BATCH_NO_IDS_SINGLE_GRAPH: dict[str, Any] = {
    "nodes": {"n0": {"categories": ["biolink:Gene"]}, "n1": {"ids": ["D"]}},
    "edges": {"e0": {"object": "n0", "subject": "n1", "predicate": "R"}},
}


# -----------------------
# Expected queries
# -----------------------

EXP_SIMPLE = dedent("""
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
""").strip()

EXP_SIMPLE_MULTIPLE_IDS = dedent("""
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
""").strip()

EXP_TWO_HOP = dedent("""
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
""").strip()

EXP_THREE_HOP = dedent("""
{
    node(func: eq(id, "UBERON:0001769")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "has_part")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "UBERON:0001608")) {
                id name category all_names all_categories iri equivalent_curies description publications
                in_edges: ~source @filter(eq(predicate, "has_part")) {
                    predicate primary_knowledge_source knowledge level agent_type kg2_ids domain_range_exclusion edge_id
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
""").strip().replace("knowledge level", "knowledge_level")

EXP_FOUR_HOP = dedent("""
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
""").strip()

EXP_FIVE_HOP = dedent("""
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
""").strip()

EXP_FIVE_HOP_MULTIPLE_IDS = dedent("""
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
""").strip()

EXP_CATEGORY_FILTER = dedent("""
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
""").strip()

EXP_MULTIPLE_FILTERS = dedent("""
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
""").strip()

EXP_NEGATED_CONSTRAINT = dedent("""
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
""").strip()

EXP_PUBLICATION_FILTER = dedent("""
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
""").strip()

EXP_NUMERIC_FILTER = dedent("""
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
""").strip()

EXP_SINGLE_STRING_WITH_COMMAS = dedent("""
{
    node(func: eq(id, ["Q0", "Q1"])) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "P")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "Q2")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()

EXP_PREDICATES_SINGLE = dedent("""
{
    node(func: eq(id, "A")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "Ponly")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "B")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()

EXP_ATTRIBUTES_ONLY = dedent("""
{
    node(func: eq(id, "A")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(knowledge_level, "primary")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "B")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()

EXP_START_OBJECT_WITH_IDS = dedent("""
{
    node(func: eq(id, "X")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "rel")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "Y")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()

EXP_BATCH_CONTAINER = dedent("""
{
    node0(func: eq(id, "Q1")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "interacts_with")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "Q2")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }

    node1(func: eq(id, ["Q0", "Q1"])) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "interacts_with")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, ["Q2", "Q3"])) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()

EXP_BATCH_MULTI_IDS_SINGLE = dedent("""
{
    node0(func: eq(id, "A")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "P")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "C")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
    node1(func: eq(id, "B")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "P")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "C")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()

EXP_BATCH_NO_IDS_SINGLE = dedent("""
{
    node(func: eq(category, "biolink:Gene")) @cascade {
        id name category all_names all_categories iri equivalent_curies description publications
        in_edges: ~source @filter(eq(predicate, "R")) {
            predicate primary_knowledge_source knowledge_level agent_type kg2_ids domain_range_exclusion edge_id
            node: target @filter(eq(id, "D")) {
                id name category all_names all_categories iri equivalent_curies description publications
            }
        }
    }
}
""").strip()


# -----------------------
# Case pairs
# -----------------------

CASES: list[QueryCase] = [
    QueryCase("simple", SIMPLE_QGRAPH, EXP_SIMPLE),
    QueryCase("simple-multiple-ids", SIMPLE_QGRAPH_MULTIPLE_IDS, EXP_SIMPLE_MULTIPLE_IDS),
    QueryCase("two-hop", TWO_HOP_QGRAPH, EXP_TWO_HOP),
    QueryCase("three-hop", THREE_HOP_QGRAPH, EXP_THREE_HOP),
    QueryCase("four-hop", FOUR_HOP_QGRAPH, EXP_FOUR_HOP),
    QueryCase("five-hop", FIVE_HOP_QGRAPH, EXP_FIVE_HOP),
    QueryCase("five-hop-multiple-ids", FIVE_HOP_QGRAPH_MULTIPLE_IDS, EXP_FIVE_HOP_MULTIPLE_IDS),
    QueryCase("category-filter", CATEGORY_FILTER_QGRAPH, EXP_CATEGORY_FILTER),
    QueryCase("multiple-filters", MULTIPLE_FILTERS_QGRAPH, EXP_MULTIPLE_FILTERS),
    QueryCase("negated-constraint", NEGATED_CONSTRAINT_QGRAPH, EXP_NEGATED_CONSTRAINT),
    QueryCase("publication-filter", PUBLICATION_FILTER_QGRAPH, EXP_PUBLICATION_FILTER),
    QueryCase("numeric-filter", NUMERIC_FILTER_QGRAPH, EXP_NUMERIC_FILTER),
    QueryCase("id-list-from-single-string-with-commas", SINGLE_STRING_WITH_COMMAS_QGRAPH, EXP_SINGLE_STRING_WITH_COMMAS),
    QueryCase("predicates-single", PREDICATES_SINGLE_QGRAPH, EXP_PREDICATES_SINGLE),
    QueryCase("edge-attributes-only", ATTRIBUTES_ONLY_QGRAPH, EXP_ATTRIBUTES_ONLY),
    QueryCase("start-object-with-ids", START_OBJECT_WITH_IDS_QGRAPH, EXP_START_OBJECT_WITH_IDS),
]

BATCH_CASES: list[BatchCase] = [
    BatchCase("batch-container", {"query_graphs": BATCH_CONTAINER_QGRAPH["query_graphs"]}, EXP_BATCH_CONTAINER),
    BatchCase("batch-multi-ids-single-graph", BATCH_MULTI_IDS_SINGLE_GRAPH, EXP_BATCH_MULTI_IDS_SINGLE),
    BatchCase("batch-no-ids-single-graph", BATCH_NO_IDS_SINGLE_GRAPH, EXP_BATCH_NO_IDS_SINGLE),
]

RESULTS_CASES: list[ResultsCase] = [
    ResultsCase(
        "wraps-results",
        SIMPLE_QGRAPH,
        raw_results={"data": {"some": "value"}},
        expected_wrapped={
            "auxiliary_graphs": {},
            "knowledge_graph": {"edges": {}, "nodes": {}, },
            "results": {"data": {"some": "value"}}
        },
    ),
]


# -----------------------
# Tests
# -----------------------

@pytest.fixture
def transpiler() -> DgraphTranspiler:
    return DgraphTranspiler()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_convert_multihop_pairs(transpiler: DgraphTranspiler, case: QueryCase) -> None:
    actual = transpiler._convert_multihop(case.qgraph)  # type: ignore[reportPrivateUsage]
    assert_query_equals(actual, case.expected)


@pytest.mark.parametrize("case", BATCH_CASES, ids=[c.name for c in BATCH_CASES])
def test_convert_batch_multihop_pairs(transpiler: DgraphTranspiler, case: BatchCase) -> None:
    actual = transpiler._convert_batch_multihop(case.qgraph)  # type: ignore[reportPrivateUsage]
    assert_query_equals(actual, case.expected)


@pytest.mark.parametrize("case", RESULTS_CASES, ids=[c.name for c in RESULTS_CASES])
def test_convert_results_pairs(case: ResultsCase) -> None:
    tr = DgraphTranspiler()
    wrapped = tr.convert_results(case.qgraph, case.raw_results)
    assert wrapped == case.expected_wrapped
