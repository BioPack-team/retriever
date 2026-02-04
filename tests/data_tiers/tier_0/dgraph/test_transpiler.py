import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import pytest

from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.types.trapi import QEdgeDict, QNodeDict, QueryGraphDict

# -----------------------
# Helpers
# -----------------------

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
    qgraphs: list[QueryGraphDict]
    expected: str


@dataclass(frozen=True)
class ResultsCase:
    """Pair for convert_results: input qgraph and raw backend results."""
    name: str
    qgraph: QueryGraphDict
    raw_results: dict[str, Any]
    expected_wrapped: dict[str, Any]


class _TestDgraphTranspiler(DgraphTranspiler):
    """Expose protected methods for testing without modifying production code."""
    def convert_multihop_public(self, qgraph: QueryGraphDict) -> str:
        return self.convert_multihop(qgraph)

    def convert_batch_multihop_public(self, qgraphs: list[QueryGraphDict]) -> str:
        return self.convert_batch_multihop(qgraphs)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def assert_query_equals(actual: str, expected: str) -> None:
    assert normalize(actual) == normalize(expected)


def qg(d: dict[str, Any]) -> QueryGraphDict:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(QueryGraphDict, cast(object, d))


def qn(d: dict[str, Any]) -> QNodeDict:
    """Cast a raw node dict into a QNodeDict for type-checking in tests."""
    return cast(QNodeDict, cast(object, d))


def qe(d: dict[str, Any]) -> QEdgeDict:
    """Cast a raw edge dict into a QEdgeDict for type-checking in tests."""
    return cast(QEdgeDict, cast(object, d))


# -----------------------
# Query graph inputs
# -----------------------

SIMPLE_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:4514"], "constraints": []},
        "n1": {"ids": ["UMLS:C1564592"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["subclass_of"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

SIMPLE_QGRAPH_GT_NOT: QueryGraphDict = qg({
    "nodes": {
        "n0": {
            "ids": ["MONDO:0030010", "MONDO:0011766", "MONDO:0009890"],
        },
        "n1": {},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n1",
            "predicates": ["biolink:has_phenotype"],
            "attribute_constraints": [
                {
                    "id": "biolink:has_total",
                    "operator": ">",
                    "value": 2,
                },
                {
                    "id": "biolink:has_total",
                    "operator": ">",
                    "value": 4,
                    "not": True,
                },
            ],
        }
    },
})

SIMPLE_QGRAPH_MULTIPLE_IDS: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125", "CHEBI:53448"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090", "CHEBI:10119"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

SIMPLE_REVERSE_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
        "n1": {"ids": ["NCBIGene:3778"], "constraints": []}
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["biolink:has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        }
    }
})

TWO_HOP_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090"], "constraints": []},
        "n2": {"ids": ["UMLS:C0496995"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

THREE_HOP_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090"], "constraints": []},
        "n2": {"ids": ["UMLS:C0496995"], "constraints": []},
        "n3": {"ids": ["UMLS:C0149720"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

FOUR_HOP_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090"], "constraints": []},
        "n2": {"ids": ["UMLS:C0496995"], "constraints": []},
        "n3": {"ids": ["UMLS:C0149720"], "constraints": []},
        "n4": {"ids": ["UMLS:C0496994"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e3": {
            "object": "n3",
            "subject": "n4",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

FIVE_HOP_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090"], "constraints": []},
        "n2": {"ids": ["UMLS:C0496995"], "constraints": []},
        "n3": {"ids": ["UMLS:C0149720"], "constraints": []},
        "n4": {"ids": ["UMLS:C0496994"], "constraints": []},
        "n5": {"ids": ["UMLS:C2879715"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e3": {
            "object": "n3",
            "subject": "n4",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e4": {
            "object": "n4",
            "subject": "n5",
            "predicates": ["has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

FIVE_HOP_QGRAPH_MULTIPLE_IDS: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["Q0", "Q1"], "constraints": []},
        "n1": {"ids": ["Q2", "Q3"], "constraints": []},
        "n2": {"ids": ["Q4", "Q5"], "constraints": []},
        "n3": {"ids": ["Q6", "Q7"], "constraints": []},
        "n4": {"ids": ["Q8", "Q9"], "constraints": []},
        "n5": {"ids": ["Q10", "Q11"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["P0"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["P1"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["P2"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e3": {
            "object": "n3",
            "subject": "n4",
            "predicates": ["P3"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e4": {
            "object": "n4",
            "subject": "n5",
            "predicates": ["P4"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

CATEGORY_FILTER_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"categories": ["biolink:Gene"], "constraints": []},
        "n1": {"categories": ["biolink:Disease"], "constraints": []}},
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["biolink:gene_associated_with_condition"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

MULTIPLE_FILTERS_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {
            "ids": ["CHEBI:3125"],
            "categories": ["biolink:SmallMolecule"],
            "constraints": [
                {"id": "description", "name": "description", "operator": "matches", "value": "/.*diphenylmethane.*/i", "not": False},
            ],
        },
        "n1": {
            "categories": ["biolink:Drug"],
            "constraints": [
                {"id": "description", "name": "description", "operator": "matches", "value": "/.*laxative.*/i", "not": False},
            ],
        },
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": [
                "has_phenotype",
                "contributes_to",
            ],
            "attribute_constraints": [
                {
                    "id": "knowledge_level",
                    "name": "knowledge_level",
                    "operator": "==",
                    "value": "prediction",
                    "not": False,
                }
            ],
            "qualifier_constraints": [],
        }
    },
})

NEGATED_CONSTRAINT_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125"], "constraints": []},
        "n1": {
            "categories": ["biolink:Drug"],
            "constraints": [
                {
                    "id": "description",
                    "name": "description",
                    "operator": "matches",
                    "value": "/.*diphenylmethane.*/i",
                    "not": True,
                }
            ],
        },
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": [],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        }
    },
})

PUBLICATION_FILTER_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["DOID:14330"], "constraints": []},
        "n1": {
            "constraints": [
                {
                    "id": "publications",
                    "name": "publications",
                    "operator": "in",
                    "value": "PMID:12345678",
                    "not": False,
                }
            ]
        },
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": [],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        }
    },
})

NUMERIC_FILTER_QGRAPH: QueryGraphDict = qg({
    "nodes": {"n0": {"ids": ["DOID:14330"], "constraints": []}, "n1": {"categories": ["biolink:Gene"], "constraints": []}},
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": [],
            "attribute_constraints": [
                {"id": "edge_id", "name": "edge_id", "operator": ">", "value": "100", "not": False}
            ],
            "qualifier_constraints": [],
        }
    },
})

SINGLE_STRING_WITH_COMMAS_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["Q0, Q1"], "constraints": []},
        "n1": {"ids": ["Q2"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["P"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        }
    },
})

PREDICATES_SINGLE_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["A"], "constraints": []},
        "n1": {"ids": ["B"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["Ponly"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        }
    },
})

ATTRIBUTES_ONLY_QGRAPH: QueryGraphDict = qg({
    "nodes": {"n0": {"ids": ["A"], "constraints": []}, "n1": {"ids": ["B"], "constraints": []}},
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": [],
            "attribute_constraints": [
                {"id": "knowledge_level", "name": "knowledge_level", "operator": "==", "value": "primary", "not": False}
            ],
            "qualifier_constraints": [],
        }
    },
})

START_OBJECT_WITH_IDS_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["X"], "constraints": []},
        "n1": {"ids": ["Y"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["rel"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        }
    },
})

# Batch inputs
BATCH_QGRAPHS: list[QueryGraphDict] = [
    SIMPLE_QGRAPH,
    SIMPLE_QGRAPH_MULTIPLE_IDS,
]

BATCH_QGRAPHS_MULTI_HOP: list[QueryGraphDict] = [
    SIMPLE_QGRAPH,
    SIMPLE_QGRAPH_MULTIPLE_IDS,
    TWO_HOP_QGRAPH,
    FIVE_HOP_QGRAPH_MULTIPLE_IDS,
]

BATCH_MULTI_IDS_SINGLE_GRAPH: list[QueryGraphDict] = [
    qg({
        "nodes": {
            "n0": {"ids": ["A", "B"], "constraints": []},
            "n1": {"ids": ["C"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["P"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    })
]

BATCH_NO_IDS_SINGLE_GRAPH: list[QueryGraphDict] = [
    qg({
        "nodes": {
            "n0": {"categories": ["biolink:Gene"], "constraints": []},
            "n1": {"ids": ["D"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["R"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    })
]

TRAPI_FLOATING_OBJECT_QUERY: QueryGraphDict = qg({
    "nodes": {
        "n0": {
            "categories": ["biolink:Gene"],
            "ids": ["NCBIGene:3778"]
        },
        "n1": {
            "categories": ["biolink:Disease"]
        }
    },
    "edges": {
        "e01": {
            "subject": "n0",
            "object": "n1",
            "predicates": ["biolink:causes"]
        }
    }
})

TRAPI_FLOATING_OBJECT_QUERY_TWO_CATEGORIES: QueryGraphDict = qg({
    "nodes": {
        "n0": {
            "categories": ["biolink:Gene", "biolink:Protein"],
            "ids": ["NCBIGene:3778"]
        },
        "n1": {
            "categories": ["biolink:Disease"]
        }
    },
    "edges": {
        "e01": {
            "subject": "n0",
            "object": "n1",
            "predicates": ["biolink:causes"]
        }
    }
})

QUALIFIER_SET_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["MONDO:0030010", "MONDO:0011766", "MONDO:0009890"], "constraints": []},
        "n1": {"constraints": []},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n1",
            "predicates": ["biolink:has_phenotype"],
            "attribute_constraints": [],
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                        {
                            "qualifier_type_id": "biolink:frequency_qualifier",
                            "qualifier_value": "HP:0040280",
                        },
                        {
                            "qualifier_type_id": "biolink:onset_qualifier",
                            "qualifier_value": "ANY_VALUE",
                        },
                        {
                            "qualifier_type_id": "biolink:sex_qualifier",
                            "qualifier_value": "ANY_OTHER_VALUE",
                        },
                    ]
                }
            ],
        }
    },
})

QUALIFIER_SETS_AND_QGRAPH: QueryGraphDict = qg({
    "nodes": {"n0": {"ids": ["X"], "constraints": []}, "n1": {"constraints": []}},
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n1",
            "predicates": ["R"],
            "attribute_constraints": [],
            "qualifier_constraints": [
                {"qualifier_set": [
                    {"qualifier_type_id": "biolink:frequency_qualifier", "qualifier_value": "F1"},
                    {"qualifier_type_id": "biolink:onset_qualifier", "qualifier_value": "O1"},
                ]},
                {"qualifier_set": [
                    {"qualifier_type_id": "biolink:sex_qualifier", "qualifier_value": "S1"},
                ]},
            ],
        }
    },
})


# -----------------------
# Expected queries
# -----------------------

EXP_SIMPLE = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:4514")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C1564592")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_SIMPLE_GT_NOT = dedent("""
{
  q0_node_n0(func: eq(id, ["MONDO:0030010", "MONDO:0011766", "MONDO:0009890"])) @cascade(id, out_edges_e0) {
    expand(Node)
    out_edges_e0: ~subject
      @filter(eq(predicate_ancestors, "has_phenotype") AND
        gt(has_total, "2") AND
        NOT(gt(has_total, "4"))) @cascade(predicate, object) {
      expand(Edge) { sources expand(Source) }
      node_n1: object @cascade(id) {
        expand(Node)
      }
    }
  }
}
""").strip()

EXP_SIMPLE_REVERSE = dedent("""
{
    q0_node_n1(func: eq(id, "NCBIGene:3778")) @cascade(id, out_edges_e0) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(category, "NamedThing")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_SIMPLE_WITH_VERSION = dedent("""
{
    q0_node_n0(func: eq(v1_id, "CHEBI:4514")) @cascade(v1_id, in_edges_e0) {
        expand(v1_Node)
        in_edges_e0: ~v1_object @filter(eq(v1_predicate_ancestors, "subclass_of")) @cascade(v1_predicate, v1_subject) {
            expand(v1_Edge) { v1_sources expand(v1_Source) }
            node_n1: v1_subject @filter(eq(v1_id, "UMLS:C1564592")) @cascade(v1_id) {
                expand(v1_Node)
            }
        }
    }
}
"""
).strip()

EXP_SIMPLE_MULTIPLE_IDS = dedent("""
{
    q0_node_n0(func: eq(id, ["CHEBI:3125", "CHEBI:53448"])) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, ["UMLS:C0282090", "CHEBI:10119"])) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_TWO_HOP = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C0282090")) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id) {
                        expand(Node)
                    }
                }
            }
        }
    }
}
""").strip()

EXP_TWO_HOP_WITH_VERSION = dedent("""
{
    q0_node_n0(func: eq(v1_id, "CHEBI:3125")) @cascade(v1_id, in_edges_e0) {
        expand(v1_Node)
        in_edges_e0: ~v1_object @filter(eq(v1_predicate_ancestors, "has_phenotype")) @cascade(v1_predicate, v1_subject) {
            expand(v1_Edge) { v1_sources expand(v1_Source) }
            node_n1: v1_subject @filter(eq(v1_id, "UMLS:C0282090")) @cascade(v1_id, in_edges_e1) {
                expand(v1_Node)
                in_edges_e1: ~v1_object @filter(eq(v1_predicate_ancestors, "has_phenotype")) @cascade(v1_predicate, v1_subject) {
                    expand(v1_Edge) { v1_sources expand(v1_Source) }
                    node_n2: v1_subject @filter(eq(v1_id, "UMLS:C0496995")) @cascade(v1_id) {
                        expand(v1_Node)
                    }
                }
            }
        }
    }
}
""").strip()

EXP_THREE_HOP = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C0282090")) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id, in_edges_e2) {
                        expand(Node)
                        in_edges_e2: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                            expand(Edge) { sources expand(Source) }
                            node_n3: subject @filter(eq(id, "UMLS:C0149720")) @cascade(id) {
                                expand(Node)
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
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C0282090")) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id, in_edges_e2) {
                        expand(Node)
                        in_edges_e2: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                            expand(Edge) { sources expand(Source) }
                            node_n3: subject @filter(eq(id, "UMLS:C0149720")) @cascade(id, in_edges_e3) {
                                expand(Node)
                                in_edges_e3: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                                    expand(Edge) { sources expand(Source) }
                                    node_n4: subject @filter(eq(id, "UMLS:C0496994")) @cascade(id) {
                                        expand(Node)
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
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C0282090")) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id, in_edges_e2) {
                        expand(Node)
                        in_edges_e2: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                            expand(Edge) { sources expand(Source) }
                            node_n3: subject @filter(eq(id, "UMLS:C0149720")) @cascade(id, in_edges_e3) {
                                expand(Node)
                                in_edges_e3: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                                    expand(Edge) { sources expand(Source) }
                                    node_n4: subject @filter(eq(id, "UMLS:C0496994")) @cascade(id, in_edges_e4) {
                                        expand(Node)
                                        in_edges_e4: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                                            expand(Edge) { sources expand(Source) }
                                            node_n5: subject @filter(eq(id, "UMLS:C2879715")) @cascade(id) {
                                                expand(Node)
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
    q0_node_n0(func: eq(id, ["Q0", "Q1"])) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "P0")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, ["Q2", "Q3"])) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "P1")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, ["Q4", "Q5"])) @cascade(id, in_edges_e2) {
                        expand(Node)
                        in_edges_e2: ~object @filter(eq(predicate_ancestors, "P2")) @cascade(predicate, subject) {
                            expand(Edge) { sources expand(Source) }
                            node_n3: subject @filter(eq(id, ["Q6", "Q7"])) @cascade(id, in_edges_e3) {
                                expand(Node)
                                in_edges_e3: ~object @filter(eq(predicate_ancestors, "P3")) @cascade(predicate, subject) {
                                    expand(Edge) { sources expand(Source) }
                                    node_n4: subject @filter(eq(id, ["Q8", "Q9"])) @cascade(id, in_edges_e4) {
                                        expand(Node)
                                        in_edges_e4: ~object @filter(eq(predicate_ancestors, "P4")) @cascade(predicate, subject) {
                                            expand(Edge) { sources expand(Source) }
                                            node_n5: subject @filter(eq(id, ["Q10", "Q11"])) @cascade(id) {
                                                expand(Node)
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
    q0_node_n0(func: eq(category, "Gene")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "gene_associated_with_condition")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(category, "Disease")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_MULTIPLE_FILTERS = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, ["has_phenotype", "contributes_to"]) AND eq(knowledge_level, "prediction")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(category, "Drug") AND regexp(description, /.*laxative.*/i)) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_NEGATED_CONSTRAINT = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(category, "Drug") AND NOT(regexp(description, /.*diphenylmethane.*/i))) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_PUBLICATION_FILTER = dedent("""
{
    q0_node_n0(func: eq(id, "DOID:14330")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(publications, "PMID:12345678")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_NUMERIC_FILTER = dedent("""
{
    q0_node_n0(func: eq(id, "DOID:14330")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(gt(edge_id, "100")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(category, "Gene")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_SINGLE_STRING_WITH_COMMAS = dedent("""
{
    q0_node_n0(func: eq(id, ["Q0", "Q1"])) @cascade(id, in_edges_e0) {
        expand(Node)

        in_edges_e0: ~object @filter(eq(predicate_ancestors, "P")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "Q2")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_PREDICATES_SINGLE = dedent("""
{
    q0_node_n0(func: eq(id, "A")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "Ponly")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "B")) @cascade(id) {
                expand(Node)
            }
        }
    }
}

""").strip()

EXP_ATTRIBUTES_ONLY = dedent("""
{
    q0_node_n0(func: eq(id, "A")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(knowledge_level, "primary")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "B")) @cascade(id) {
                expand(Node)
            }
        }
    }
}

""").strip()

EXP_START_OBJECT_WITH_IDS = dedent("""
{
    q0_node_n0(func: eq(id, "X")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "rel")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "Y")) @cascade(id) {
                expand(Node)
            }
        }
    }
}

""").strip()

EXP_BATCH_QGRAPHS = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:4514")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C1564592")) @cascade(id) {
                expand(Node)
            }
        }
    }
    q1_node_n0(func: eq(id, ["CHEBI:3125", "CHEBI:53448"])) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, ["UMLS:C0282090", "CHEBI:10119"])) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_BATCH_QGRAPHS_MULTI_HOP = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:4514")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C1564592")) @cascade(id) {
                expand(Node)
            }
        }
    }
    q1_node_n0(func: eq(id, ["CHEBI:3125", "CHEBI:53448"])) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, ["UMLS:C0282090", "CHEBI:10119"])) @cascade(id) {
                expand(Node)
            }
        }
    }
    q2_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, "UMLS:C0282090")) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id) {
                        expand(Node)
                    }
                }
            }
        }
    }
    q3_node_n0(func: eq(id, ["Q0", "Q1"])) @cascade(id, in_edges_e0) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, "P0")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n1: subject @filter(eq(id, ["Q2", "Q3"])) @cascade(id, in_edges_e1) {
                expand(Node)
                in_edges_e1: ~object @filter(eq(predicate_ancestors, "P1")) @cascade(predicate, subject) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: subject @filter(eq(id, ["Q4", "Q5"])) @cascade(id, in_edges_e2) {
                        expand(Node)
                        in_edges_e2: ~object @filter(eq(predicate_ancestors, "P2")) @cascade(predicate, subject) {
                            expand(Edge) { sources expand(Source) }
                            node_n3: subject @filter(eq(id, ["Q6", "Q7"])) @cascade(id, in_edges_e3) {
                                expand(Node)
                                in_edges_e3: ~object @filter(eq(predicate_ancestors, "P3")) @cascade(predicate, subject) {
                                    expand(Edge) { sources expand(Source) }
                                    node_n4: subject @filter(eq(id, ["Q8", "Q9"])) @cascade(id, in_edges_e4) {
                                        expand(Node)
                                        in_edges_e4: ~object @filter(eq(predicate_ancestors, "P4")) @cascade(predicate, subject) {
                                            expand(Edge) { sources expand(Source) }
                                            node_n5: subject @filter(eq(id, ["Q10", "Q11"])) @cascade(id) {
                                                expand(Node)
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

EXP_BATCH_MULTI_IDS_SINGLE = dedent("""
{
    q0_node_n1(func: eq(id, "C")) @cascade(id, out_edges_e0) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "P")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, ["A", "B"])) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_BATCH_NO_IDS_SINGLE = dedent("""
{
    q0_node_n1(func: eq(id, "D")) @cascade(id, out_edges_e0) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "R")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(category, "Gene")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

DGRAPH_FLOATING_OBJECT_QUERY = dedent("""
{
    q0_node_n0(func: eq(id, "NCBIGene:3778")) @cascade(id, out_edges_e0) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "causes")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n1: object @filter(eq(category, "Disease")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

DGRAPH_FLOATING_OBJECT_QUERY_WITH_VERSION = dedent("""
{
    q0_node_n0(func: eq(v1_id, "NCBIGene:3778")) @cascade(v1_id, out_edges_e0) {
        expand(v1_Node)
        out_edges_e0: ~v1_subject @filter(eq(v1_predicate_ancestors, "causes")) @cascade(v1_predicate, v1_object) {
            expand(v1_Edge) { v1_sources expand(v1_Source) }
            node_n1: v1_object @filter(eq(v1_category, "Disease")) @cascade(v1_id) {
                expand(v1_Node)
            }
        }
    }
}
""").strip()

DGRAPH_FLOATING_OBJECT_QUERY_TWO_CATEGORIES = dedent("""
{
    q0_node_n0(func: eq(id, "NCBIGene:3778")) @cascade(id, out_edges_e0) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "causes")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n1: object @filter(eq(category, "Disease")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_QUALIFIER_SET = dedent("""
{
  q0_node_n0(func: eq(id, ["MONDO:0030010", "MONDO:0011766", "MONDO:0009890"])) @cascade(id, out_edges_e0) {
    expand(Node)
    out_edges_e0: ~subject @filter(eq(predicate_ancestors, "has_phenotype") AND
      (eq(frequency_qualifier, "HP:0040280") AND eq(onset_qualifier, "ANY_VALUE") AND eq(sex_qualifier, "ANY_OTHER_VALUE"))) @cascade(predicate, object) {
      expand(Edge) { sources expand(Source) }
      node_n1: object @cascade(id) {
        expand(Node)
      }
    }
  }
}
""").strip()

EXP_QUALIFIER_SETS_AND = dedent("""
{
  q0_node_n0(func: eq(id, "X")) @cascade(id, out_edges_e0) {
    expand(Node)
    out_edges_e0: ~subject @filter(eq(predicate_ancestors, "R") AND
      ((eq(frequency_qualifier, "F1") AND eq(onset_qualifier, "O1")) OR eq(sex_qualifier, "S1"))) @cascade(predicate, object) {
      expand(Edge) { sources expand(Source) }
      node_n1: object @cascade(id) { expand(Node) }
    }
  }
}
""").strip()


# -----------------------
# Case pairs
# -----------------------

CASES: list[QueryCase] = [
    QueryCase("simple-one", SIMPLE_QGRAPH, EXP_SIMPLE),
    QueryCase("simple-gt-not", SIMPLE_QGRAPH_GT_NOT, EXP_SIMPLE_GT_NOT),
    QueryCase("simple-multiple-ids", SIMPLE_QGRAPH_MULTIPLE_IDS, EXP_SIMPLE_MULTIPLE_IDS),
    QueryCase("simple-reverse", SIMPLE_REVERSE_QGRAPH, EXP_SIMPLE_REVERSE),
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
    QueryCase("floating-object-query", TRAPI_FLOATING_OBJECT_QUERY, DGRAPH_FLOATING_OBJECT_QUERY),
    QueryCase("floating-object-query-two-categories", TRAPI_FLOATING_OBJECT_QUERY_TWO_CATEGORIES, DGRAPH_FLOATING_OBJECT_QUERY_TWO_CATEGORIES),
    QueryCase("qualifier-set", QUALIFIER_SET_QGRAPH, EXP_QUALIFIER_SET),
    QueryCase("qualifier-sets-and", QUALIFIER_SETS_AND_QGRAPH, EXP_QUALIFIER_SETS_AND),
]

CASES_VERSIONED: list[QueryCase] = [
    QueryCase("simple-one-versioned", SIMPLE_QGRAPH, EXP_SIMPLE_WITH_VERSION),
    QueryCase("two-hop-versioned", TWO_HOP_QGRAPH, EXP_TWO_HOP_WITH_VERSION),
    QueryCase("floating-object-query-versioned", TRAPI_FLOATING_OBJECT_QUERY, DGRAPH_FLOATING_OBJECT_QUERY_WITH_VERSION),
]

BATCH_CASES: list[BatchCase] = [
    BatchCase("batch-qgraphs", BATCH_QGRAPHS, EXP_BATCH_QGRAPHS),
    BatchCase("batch-qgraph-multi-hop", BATCH_QGRAPHS_MULTI_HOP, EXP_BATCH_QGRAPHS_MULTI_HOP),
    BatchCase("batch-multi-ids-single-graph", BATCH_MULTI_IDS_SINGLE_GRAPH, EXP_BATCH_MULTI_IDS_SINGLE),
    BatchCase("batch-no-ids-single-graph", BATCH_NO_IDS_SINGLE_GRAPH, EXP_BATCH_NO_IDS_SINGLE),
]


# -----------------------
# Tests
# -----------------------

@pytest.fixture
def transpiler() -> _TestDgraphTranspiler:
    return _TestDgraphTranspiler()

@pytest.fixture
def transpiler_no_subclassing() -> _TestDgraphTranspiler:
    return _TestDgraphTranspiler(subclassing_enabled=False)

@pytest.fixture
def transpiler_with_subclassing() -> _TestDgraphTranspiler:
    return _TestDgraphTranspiler(subclassing_enabled=True)

@pytest.fixture
def transpiler_no_subclassing_versioned() -> _TestDgraphTranspiler:
    return _TestDgraphTranspiler(version="v1", subclassing_enabled=False)

@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_convert_multihop_pairs(transpiler_no_subclassing: _TestDgraphTranspiler, case: QueryCase) -> None:
    actual = transpiler_no_subclassing.convert_multihop_public(case.qgraph)
    assert_query_equals(actual, case.expected)

@pytest.mark.parametrize("case", CASES_VERSIONED, ids=[c.name for c in CASES_VERSIONED])
def test_convert_multihop_pairs_with_version(transpiler_no_subclassing_versioned: _TestDgraphTranspiler, case: QueryCase) -> None:
    actual = transpiler_no_subclassing_versioned.convert_multihop_public(case.qgraph)
    assert_query_equals(actual, case.expected)

@pytest.mark.parametrize("case", BATCH_CASES, ids=[c.name for c in BATCH_CASES])
def test_convert_batch_multihop_pairs(transpiler_no_subclassing: _TestDgraphTranspiler, case: BatchCase) -> None:
    actual = transpiler_no_subclassing.convert_batch_multihop_public(case.qgraphs)
    assert_query_equals(actual, case.expected)


# --- Dgraph result for testing convert_results ---
DGRAPH_RESULT_WITH_SOURCES = {
    "q0_node_n1": [
        {
            "uid": "0x1",
            "id": "NCBIGene:11276",
            "name": "SYNRG",
            "category": ["Gene", "Protein"],
            "out_edges_e0": [
                {
                    "uid": "0x2",
                    "binding": "e0",
                    "direction": "out",
                    "predicate": "located_in",
                    "sources": [
                        {
                            "resource_id": "infores:goa",
                            "resource_role": "primary_knowledge_source",
                            "upstream_resource_ids": ["infores:uniprot"],
                            "source_record_urls": ["http://example.com/record1"],
                        }
                    ],
                    "node_n0": {
                        "uid": "0x3",
                        "id": "GO:0031410",
                        "name": "cytoplasmic vesicle",
                        "binding": "n0",
                        "category": ["CellularComponent"],
                    },
                }
            ],
        }
    ]
}


def test_convert_results_with_full_source_info(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """
    Test that convert_results correctly populates all source fields,
    including upstream_resource_ids and source_record_urls.
    """
    from retriever.data_tiers.tier_0.dgraph import result_models as dg

    # 1. Define the qgraph that produced these results
    qgraph = qg({
        "nodes": {
            "n0": {"categories": ["biolink:CellularComponent"]},
            "n1": {"ids": ["NCBIGene:11276"]},
        },
        "edges": {"e0": {"subject": "n1", "object": "n0", "predicates": ["biolink:located_in"]}},
    })

    # 2. Parse the raw Dgraph JSON into the result_models objects
    parsed_nodes = dg.DgraphResponse.parse(DGRAPH_RESULT_WITH_SOURCES, prefix="q0").data["q0"]

    # 3. Run the conversion
    backend_result = transpiler_no_subclassing.convert_results(qgraph, parsed_nodes)

    # 4. Assertions
    assert len(backend_result["knowledge_graph"]["edges"]) == 1

    # Get the single edge from the knowledge graph to inspect its sources
    trapi_edge = next(iter(backend_result["knowledge_graph"]["edges"].values()))

    assert len(trapi_edge["sources"]) == 2  # expect original source + tier0 aggregator we add
    source = trapi_edge["sources"][0]

    # Assert that the keys exist before accessing them.
    assert "upstream_resource_ids" in source
    assert "source_record_urls" in source

    # Assert that the new fields are correctly populated
    assert source["resource_id"] == "infores:goa"
    assert source["resource_role"] == "primary_knowledge_source"
    assert source["upstream_resource_ids"] == ["infores:uniprot"]
    assert source["source_record_urls"] == ["http://example.com/record1"]


# -----------------------
# Symmetric predicate tests
# -----------------------

def test_symmetric_predicate_generates_bidirectional_queries(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test that symmetric predicates generate queries checking both directions."""

    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "n1": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:related_to"],  # Symmetric predicate
            }
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, out_edges_e0) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) { expand(Node) } }
            in_edges-symmetric_e0: ~object @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)
    # Should have both the normal direction and reverse direction
    assert "out_edges_e0:" in actual
    assert "in_edges-symmetric_e0:" in actual


def test_symmetric_predicate_incoming_edge(transpiler: _TestDgraphTranspiler) -> None:
    """Test that symmetric predicates work for incoming edges."""
    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "n1": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0": {
                "subject": "n1",  # Note: reversed from previous test
                "object": "n0",
                "predicates": ["biolink:correlated_with"],  # Symmetric predicate
            }
        },
    })

    # 2. Act
    actual = transpiler.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, in_edges_e0) {
            expand(Node)
            in_edges_e0: ~object @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
            out_edges-symmetric_e0: ~subject @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
            in_edges-subclassObjB_e0: ~object @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_intermediate: subject @filter(has(id)) @cascade(id, ~subject) {
                    expand(Node)
                    out_edges-subclassObjB-tail_e0: ~subject @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, object) {
                        expand(Edge) { sources expand(Source) }
                        node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                            expand(Node)
                        }
                    }
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)
    # Should have both the incoming direction and its reverse
    assert "in_edges_e0:" in actual
    assert "out_edges-symmetric_e0:" in actual


def test_symmetric_predicate_multi_hop(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test symmetric predicates in a multi-hop query."""

    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "n1": {"categories": ["biolink:Gene"]},
            "n2": {"categories": ["biolink:Pathway"]},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:related_to"],  # Symmetric
            },
            "e1": {
                "subject": "n1",
                "object": "n2",
                "predicates": ["biolink:participates_in"],  # Non-symmetric
            },
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, out_edges_e0) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id, out_edges_e1) {
                    expand(Node)
                    out_edges_e1: ~subject @filter(eq(predicate_ancestors, "participates_in")) @cascade(predicate, object) {
                        expand(Edge) { sources expand(Source) }
                        node_n2: object @filter(eq(category, "Pathway")) @cascade(id) {
                            expand(Node)
                        }
                    }
                }
            }
            in_edges-symmetric_e0: ~object @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id, out_edges_e1) {
                    expand(Node)
                    out_edges_e1: ~subject @filter(eq(predicate_ancestors, "participates_in")) @cascade(predicate, object) {
                        expand(Edge) { sources expand(Source) }
                        node_n2: object @filter(eq(category, "Pathway")) @cascade(id) {
                            expand(Node)
                        }
                    }
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)
    # First edge should have bidirectional queries
    assert "out_edges_e0:" in actual
    assert "in_edges-symmetric_e0:" in actual
    # Second edge should only have one direction
    assert "out_edges_e1:" in actual
    assert "in_edges_e1:" not in actual


def test_multiple_symmetric_predicates_on_edge(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test edge with multiple predicates where all are symmetric."""

    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "n1": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": [
                    "biolink:related_to",  # Symmetric
                    "biolink:associated_with",  # Symmetric
                ],
            }
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, out_edges_e0) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, ["related_to", "associated_with"])) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
            in_edges-symmetric_e0: ~object @filter(eq(predicate_ancestors, ["related_to", "associated_with"])) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)
    # Should have both directions since at least one predicate is symmetric
    assert "out_edges_e0:" in actual
    assert "in_edges-symmetric_e0:" in actual


def test_mixed_predicates_treats_as_symmetric(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test edge with mixed symmetric and non-symmetric predicates."""

    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "n1": {"categories": ["biolink:ChemicalEntity"]},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": [
                    "biolink:related_to",  # Symmetric
                    "biolink:treated_by",  # Non-symmetric
                ],
            }
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, out_edges_e0) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, ["related_to", "treated_by"])) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "ChemicalEntity")) @cascade(id) {
                    expand(Node)
                }
            }
            in_edges-symmetric_e0: ~object @filter(eq(predicate_ancestors, ["related_to", "treated_by"])) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "ChemicalEntity")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)
    # If ANY predicate is symmetric, should check both directions
    assert "out_edges_e0:" in actual
    assert "in_edges-symmetric_e0:" in actual


# -----------------------
# Normalization tests
# -----------------------

def test_normalization_with_underscores_in_ids(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test that node and edge IDs with underscores are normalized to prevent injection."""

    # 1. Arrange - Query with underscores in IDs
    qgraph = qg({
        "nodes": {
            "n0_test": {"ids": ["GO:0031410"], "categories": ["biolink:CellularComponent"]},
            "n1_patient": {"ids": ["NCBIGene:11276"], "categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0_test": {
                "subject": "n1_patient",
                "object": "n0_test",
                "predicates": ["biolink:located_in"],
            }
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)

    # 3. Expected - Should use normalized IDs (n0, n1, e0) not original ones
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "GO:0031410")) @cascade(id, in_edges_e0) {
            expand(Node)
            in_edges_e0: ~object @filter(eq(predicate_ancestors, "located_in")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(id, "NCBIGene:11276")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 4. Assert
    assert_query_equals(actual, expected)

    # Verify normalized IDs are used in the query
    assert "node_n0" in actual, "Should use normalized node ID n0, not n0_test"
    assert "node_n1" in actual, "Should use normalized node ID n1, not n1_patient"
    assert "in_edges_e0:" in actual, "Should use normalized edge ID e0, not e0_test"

    # Verify original IDs are NOT in the query structure
    assert "node_n0_test" not in actual, "Original node ID should not appear in query"
    assert "node_n1_patient" not in actual, "Original node ID should not appear in query"
    assert "in_edges_e0_test:" not in actual, "Original edge ID should not appear in query"


def test_normalization_with_special_characters(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test that IDs with special characters are normalized to safe identifiers."""

    # 1. Arrange - Query with special characters that could cause injection
    qgraph = qg({
        "nodes": {
            "n0-dash": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "n1.dot": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0@special": {
                "subject": "n0-dash",
                "object": "n1.dot",
                "predicates": ["biolink:gene_associated_with_condition"],
            }
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)

    # 3. Expected - Should use normalized IDs
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, out_edges_e0) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, "gene_associated_with_condition")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 4. Assert
    assert_query_equals(actual, expected)

    # Verify safe normalized IDs are used
    assert "node_n0" in actual
    assert "node_n1" in actual
    assert "out_edges_e0:" in actual

    # Verify special characters are NOT in query structure
    assert "n0-dash" not in actual
    assert "n1.dot" not in actual
    assert "e0@special" not in actual


def test_normalization_alphabetical_ordering(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test that normalization uses alphabetical ordering for consistent ID assignment."""

    # 1. Arrange - Query with non-sequential node/edge names
    qgraph = qg({
        "nodes": {
            "z_last": {"ids": ["A"], "categories": ["biolink:Gene"]},
            "a_first": {"ids": ["B"], "categories": ["biolink:Disease"]},
            "m_middle": {"ids": ["C"], "categories": ["biolink:Pathway"]},
        },
        "edges": {
            "e_zulu": {
                "subject": "a_first",
                "object": "z_last",
                "predicates": ["biolink:related_to"],
            },
            "e_alpha": {
                "subject": "z_last",
                "object": "m_middle",
                "predicates": ["biolink:related_to"],
            },
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)

    # 3. Assert - Check alphabetical assignment
    # Nodes: a_first -> n0, m_middle -> n1, z_last -> n2 (alphabetical)
    # Edges: e_alpha -> e0, e_zulu -> e1 (alphabetical)

    # Since transpiler picks start node by pinnedness, we verify the normalized IDs exist
    assert "node_n0" in actual, "a_first should map to n0"
    assert "node_n1" in actual, "m_middle should map to n1"
    assert "node_n2" in actual, "z_last should map to n2"

    # Edges should be e0 and e1
    assert "_e0:" in actual, "e_alpha should map to e0"
    assert "_e1:" in actual, "e_zulu should map to e1"


def test_normalization_batch_queries_independent(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test that each query in a batch gets independent normalization."""

    # 1. Arrange - Batch with different edge/node names
    qgraphs = [
        qg({
            "nodes": {
                "custom_n0": {"ids": ["X"], "categories": ["biolink:Gene"]},
                "custom_n1": {"ids": ["Y"], "categories": ["biolink:Disease"]},
            },
            "edges": {
                "custom_e0": {
                    "subject": "custom_n0",
                    "object": "custom_n1",
                    "predicates": ["biolink:related_to"],
                }
            },
        }),
        qg({
            "nodes": {
                "different_n0": {"ids": ["A"], "categories": ["biolink:Gene"]},
                "different_n1": {"ids": ["B"], "categories": ["biolink:Disease"]},
            },
            "edges": {
                "different_e0": {
                    "subject": "different_n0",
                    "object": "different_n1",
                    "predicates": ["biolink:related_to"],
                }
            },
        }),
    ]

    # 2. Act
    actual = transpiler_no_subclassing.convert_batch_multihop_public(qgraphs)

    # 3. Assert - Both queries should use normalized IDs independently
    # Query 0 should have q0_node_n0 or q0_node_n1
    assert "q0_node_n0" in actual or "q0_node_n1" in actual

    # Query 1 should also have q1_node_n0 or q1_node_n1
    assert "q1_node_n0" in actual or "q1_node_n1" in actual

    # Both should use e0 for their single edge
    assert "q0" in actual and "_e0:" in actual
    assert "q1" in actual and "_e0:" in actual

    # Original names should not appear
    assert "custom_n0" not in actual
    assert "custom_e0" not in actual
    assert "different_n0" not in actual
    assert "different_e0" not in actual


def test_normalization_symmetric_predicate(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test that normalization works correctly with symmetric predicates."""

    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "node_alpha": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
            "node_beta": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "edge_gamma": {
                "subject": "node_alpha",
                "object": "node_beta",
                "predicates": ["biolink:correlated_with"],  # Symmetric
            }
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)

    # 3. Expected - Should use normalized IDs with symmetric edges
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, out_edges_e0) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
            in_edges-symmetric_e0: ~object @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 4. Assert
    assert_query_equals(actual, expected)

    # Verify normalized IDs in both directions
    assert "out_edges_e0:" in actual
    assert "in_edges-symmetric_e0:" in actual
    assert "node_n0" in actual
    assert "node_n1" in actual

    # Verify original names not present
    assert "edge_gamma" not in actual
    assert "node_alpha" not in actual
    assert "node_beta" not in actual


def test_normalization_multihop_query(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """Test normalization in a multi-hop query with various ID formats."""

    # 1. Arrange
    qgraph = qg({
        "nodes": {
            "start_node": {"ids": ["CHEBI:3125"], "categories": ["biolink:SmallMolecule"]},
            "middle_node": {"ids": ["UMLS:C0282090"], "categories": ["biolink:Disease"]},
            "end_node": {"ids": ["UMLS:C0496995"], "categories": ["biolink:Phenotype"]},
        },
        "edges": {
            "first_edge": {
                "subject": "middle_node",
                "object": "start_node",
                "predicates": ["biolink:treats"],
            },
            "second_edge": {
                "subject": "end_node",
                "object": "middle_node",
                "predicates": ["biolink:phenotype_of"],
            },
        },
    })

    # 2. Act
    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)

    # 3. Assert - Should use n0, n1, n2 and e0, e1 (alphabetically sorted)
    # Nodes: end_node->n0, middle_node->n1, start_node->n2
    # Edges: first_edge->e0, second_edge->e1

    assert "node_n0" in actual
    assert "node_n1" in actual
    assert "node_n2" in actual
    assert "_e0:" in actual
    assert "_e1:" in actual

    # Original names should not be in query structure
    assert "start_node" not in actual
    assert "middle_node" not in actual
    assert "end_node" not in actual
    assert "first_edge" not in actual
    assert "second_edge" not in actual


# -----------------------
# Subclassing tests
# -----------------------

def test_subclassing_case1_id_to_id_generates_b_c_d_forms(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """Case 1: ID → R → ID yields subclass forms B, C, and D."""

    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["A"], "constraints": []},
            "n1": {"ids": ["B"], "constraints": []},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:related_to"],  # non-subclass_of
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # Expect primary edge plus three subclass forms
    assert "out_edges_e0:" in actual, "Primary traversal missing"
    assert "in_edges-subclassB_e0:" in actual, "Subclass Form B missing"
    assert "out_edges-subclassC_e0:" in actual, "Subclass Form C missing"
    assert "in_edges-subclassD_e0:" in actual, "Subclass Form D missing"

    # Subclass edges should filter only on subclass_of predicate (no attribute/qualifier constraints)
    # Check that subclass blocks include subclass_of filter
    assert "in_edges-subclassB_e0: ~object @filter(eq(predicate_ancestors, \"subclass_of\"))" in actual
    assert "out_edges-subclassC-tail_e0: ~subject @filter(eq(predicate_ancestors, \"subclass_of\"))" in actual
    assert "in_edges-subclassD_tail_e0" not in actual  # use exact names defined in transpiler
    assert "out_edges-subclassD-tail_e0: ~subject @filter(eq(predicate_ancestors, \"subclass_of\"))" in actual


def test_subclassing_case2_id_to_category_generates_b_only(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """Case 2: ID → R → CAT yields only subclass form B, and category filter is applied to final node."""

    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["A"], "constraints": []},  # source ID
            "n1": {"categories": ["biolink:Disease"], "constraints": []},  # target category only
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:related_to"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # Primary traversal present
    assert "out_edges_e0:" in actual
    # Only Form B should be added
    assert "in_edges-subclassB_e0:" in actual
    assert "in_edges-subclassC_e0:" not in actual
    assert "in_edges-subclassD_e0:" not in actual

    # Final node should retain category filter
    assert "@filter(eq(category, \"Disease\"))" in actual


def test_subclassing_skips_when_predicate_is_subclass_of_case0a_0b(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """Case 0a/0b: If original edge is subclass_of, no subclass expansions should be emitted."""

    # 0a: ID → subclass_of → ID
    qgraph_id_id = qg({
        "nodes": {
            "n0": {"ids": ["A"], "constraints": []},
            "n1": {"ids": ["B"], "constraints": []},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:subclass_of"],
            }
        },
    })
    actual_id_id = transpiler_with_subclassing.convert_multihop_public(qgraph_id_id)
    assert "out_edges_e0:" in actual_id_id
    assert "in_edges-subclassB_e0:" not in actual_id_id
    assert "in_edges-subclassC_e0:" not in actual_id_id
    assert "in_edges-subclassD_e0:" not in actual_id_id

    # 0b: ID → subclass_of → CAT
    qgraph_id_cat = qg({
        "nodes": {
            "n0": {"ids": ["A"], "constraints": []},
            "n1": {"categories": ["biolink:Disease"], "constraints": []},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:subclass_of"],
            }
        },
    })
    actual_id_cat = transpiler_with_subclassing.convert_multihop_public(qgraph_id_cat)
    assert "out_edges_e0:" in actual_id_cat
    assert "in_edges-subclassB_e0:" not in actual_id_cat
    assert "in_edges-subclassC_e0:" not in actual_id_cat
    assert "in_edges-subclassD_e0:" not in actual_id_cat


def test_subclassing_constraints_apply_only_to_predicate_edges_not_to_subclass_edges(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """Constraints on original edge apply to predicate segments in subclass forms, not to subclass_of edges."""

    qgraph = qg({
        "nodes": {"n0": {"ids": ["A"]}, "n1": {"ids": ["B"]}},
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:related_to"],
                "attribute_constraints": [
                    {"id": "knowledge_level", "operator": "==", "value": "prediction"},
                ],
                "qualifier_constraints": [
                    {"qualifier_set": [
                        {"qualifier_type_id": "biolink:frequency_qualifier", "qualifier_value": "QX"},
                    ]},
                ],
            }
        },
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # Subclass blocks should include subclass_of filter without attribute/qualifier constraints
    # Check that subclass_of traversals do NOT show the attribute filter
    assert "in_edges-subclassB_e0: ~object @filter(eq(predicate_ancestors, \"subclass_of\"))" in actual
    assert "eq(knowledge_level, \"prediction\")" not in \
           actual.split("in_edges-subclassB_e0:")[1].split("}")[0], "Subclass edge should not carry attributes"
    assert "frequency_qualifier" not in \
           actual.split("in_edges-subclassB_e0:")[1].split("}")[0], "Subclass edge should not carry qualifiers"

    # Predicate segments within subclass forms should carry the original edge constraints
    # Find the nested predicate traversal under subclassB and assert filters present
    assert "@filter(eq(predicate_ancestors, \"related_to\") AND eq(knowledge_level, \"prediction\")" in actual or \
           "@filter(eq(predicate_ancestors, [\"related_to\"]) AND eq(knowledge_level, \"prediction\")" in actual
    assert "eq(frequency_qualifier, \"QX\")" in actual


def test_subclassing_disabled_emits_no_subclass_blocks(transpiler_no_subclassing: _TestDgraphTranspiler) -> None:
    """If subclassing is disabled via constructor flag, no subclass blocks should be emitted."""

    qgraph = qg({
        "nodes": {"n0": {"ids": ["A"]}, "n1": {"ids": ["B"]}},
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:related_to"]}}
    })

    actual = transpiler_no_subclassing.convert_multihop_public(qgraph)

    assert "out_edges_e0:" in actual
    assert "in_edges-subclassB_e0:" not in actual
    assert "in_edges-subclassC_e0:" not in actual
    assert "in_edges-subclassD_e0:" not in actual


def test_subclassing_case2_does_not_trigger_when_target_has_ids(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """Target has IDs, so this is Case 1 (ID→R→ID), not Case 2. Expect B/C/D forms only."""

    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["A"]},
            "n1": {"categories": ["biolink:Disease"], "ids": ["X"]},  # has IDs → Case 1
        },
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:related_to"]}}
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # Primary traversal present
    assert "out_edges_e0:" in actual

    # Because this is Case 1 (IDs on both ends), we should see B/C/D forms:
    assert "in_edges-subclassB_e0:" in actual
    assert "out_edges-subclassC_e0:" in actual
    assert "in_edges-subclassD_e0:" in actual


def test_subclassing_two_hop_id_to_id_on_both_hops(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """Two-hop query where both hops are ID→R→ID. Expect subclassing forms B/C/D on both hops."""

    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["CHEBI:3125"], "constraints": []},        # object of e0
            "n1": {"ids": ["UMLS:C0282090"], "constraints": []},     # subject of e0, object of e1
            "n2": {"ids": ["UMLS:C0496995"], "constraints": []},     # subject of e1
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:has_phenotype"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
            "e1": {
                "object": "n1",
                "subject": "n2",
                "predicates": ["biolink:has_phenotype"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # Primary traversals (same as two-hop baseline)
    assert "in_edges_e0:" in actual
    assert "in_edges_e1:" in actual

    # Subclassing on first hop (e0): expect B/C/D
    assert "in_edges-subclassB_e0:" in actual
    assert "out_edges-subclassC_e0:" in actual
    assert "in_edges-subclassD_e0:" in actual

    # Subclassing on second hop (e1): expect B/C/D
    assert "in_edges-subclassB_e1:" in actual
    assert "out_edges-subclassC_e1:" in actual
    assert "in_edges-subclassD_e1:" in actual

    # Subclass edges should use only subclass_of filter (no attribute/qualifier constraints)
    assert ' @filter(eq(predicate_ancestors, "subclass_of"))' in actual

    # Predicate segments inside subclass forms should carry the original predicate filter
    assert ' @filter(eq(predicate_ancestors, "has_phenotype"))' in actual


def test_subclassing_two_hop_id_to_category_on_second_hop_generates_b_only(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["CHEBI:3125"], "constraints": []},       # object of e0 (ID)
            "n1": {"ids": ["UMLS:C0282090"], "constraints": []},    # subject of e0, subject of e1 (ID)
            "n2": {"categories": ["biolink:Pathway"], "constraints": []},  # object of e1 (CAT only)
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:has_phenotype"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
            "e1": {
                "object": "n2",              # <- target is categories
                "subject": "n1",             # <- source is ID
                "predicates": ["biolink:participates_in"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # Primary traversals present
    assert "in_edges_e0:" in actual
    assert "out_edges_e1:" in actual  # direction matches subject=n1 → object=n2

    # First hop (ID→R→ID) should have B/C/D
    assert "in_edges-subclassB_e0:" in actual
    assert "out_edges-subclassC_e0:" in actual
    assert "in_edges-subclassD_e0:" in actual

    # Second hop (ID→R→CAT) should have only Form B
    assert "in_edges-subclassB_e1:" in actual or "out_edges-subclassB_e1:" in actual
    assert "subclassC_e1:" not in actual
    assert "subclassD_e1:" not in actual

    # Category filter must be applied on the final node of the second hop
    assert '@filter(eq(category, "Pathway"))' in actual

    # Subclass edges must use only subclass_of filter
    assert ' @filter(eq(predicate_ancestors, "subclass_of"))' in actual

    # Predicate segment inside subclass form on second hop must carry the original predicate filter
    assert ' @filter(eq(predicate_ancestors, "participates_in"))' in actual


# -----------------------
# Pinnedness Algorithm Tests
# -----------------------

def test_pinnedness_empty_graph_raises(transpiler: _TestDgraphTranspiler) -> None:
    qgraph = qg({"nodes": {}, "edges": {}})
    with pytest.raises(ValueError):
        transpiler.convert_multihop_public(qgraph)


def test_pinnedness_single_node_no_edges(transpiler: _TestDgraphTranspiler) -> None:
    qgraph = qg({"nodes": {"n0": {"ids": ["X"]}}, "edges": {}})
    # convert_multihop still works and selects n0
    actual = transpiler.convert_multihop_public(qgraph)
    assert "q0_node_n0" in actual


def test_pinnedness_two_nodes_one_edge_prefers_node_with_ids(transpiler: _TestDgraphTranspiler) -> None:
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["A"]},            # constrained
            "n1": {"categories": ["biolink:Gene"]},  # less constrained than IDs
        },
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:related_to"]}},
    })
    actual = transpiler.convert_multihop_public(qgraph)
    # Should start at the most constrained node, n0
    assert "q0_node_n0" in actual
    assert "out_edges_e0:" in actual

    # And should not use category at root
    assert "func: eq(category" not in actual


def test_pinnedness_multiple_parallel_edges_increase_weight(transpiler: _TestDgraphTranspiler) -> None:
    # Same two nodes but with two parallel edges; pinnedness should still pick the ID node now
    qgraph = qg({
        "nodes": {"n0": {"ids": ["A"]}, "n1": {"categories": ["biolink:Gene"]}},
        "edges": {
            "e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:related_to"]},
            "e1": {"subject": "n0", "object": "n1", "predicates": ["biolink:related_to"]},
        },
    })
    actual = transpiler.convert_multihop_public(qgraph)
    assert "q0_node_n0" in actual
    assert "out_edges_e0:" in actual and "out_edges_e1:" in actual
    # Root filter should be ID-based
    assert 'q0_node_n0(func: eq(id, "A"))' in actual or 'q0_node_n0(func: eq(id, ["A"]))' in actual
    assert "func: eq(category" not in actual


def test_pinnedness_prefers_more_ids_over_fewer(transpiler: _TestDgraphTranspiler) -> None:
    # Fewer IDs (n0 has 1) should be preferred over more IDs (n1 has 3)
    qgraph = qg({
        "nodes": {
            "n0": {"ids": ["A"]},
            "n1": {"ids": ["B1", "B2", "B3"]},
        },
        "edges": {"e0": {"subject": "n1", "object": "n0", "predicates": ["biolink:related_to"]}},
    })
    actual = transpiler.convert_multihop_public(qgraph)
    assert "q0_node_n0" in actual
    assert 'q0_node_n0(func: eq(id, "A"))' in actual or 'q0_node_n0(func: eq(id, ["A"]))' in actual


def test_pinnedness_tie_breaker_uses_node_id(transpiler: _TestDgraphTranspiler) -> None:
    # Both nodes unconstrained (no IDs); tie-breaker should pick max by (score, node_id)
    # With alphabetical normalization, node IDs sorted: a_first -> n0, z_last -> n1
    qgraph = qg({
        "nodes": {
            "a_first": {"categories": ["biolink:Gene"]},
            "z_last": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "e0": {"subject": "a_first", "object": "z_last", "predicates": ["biolink:related_to"]},
        },
    })
    actual = transpiler.convert_multihop_public(qgraph)

    # With equal category-only nodes, tie-breaker picks lexicographically larger id ("z_last"), normalized to n1.
    assert "q0_node_n0" in actual
    # Root should NOT use ID filter since neither node has IDs; it should use category
    assert "func: eq(id," not in actual
    assert 'func: eq(category, "Gene")' in actual


def test_pinnedness_issue(transpiler: _TestDgraphTranspiler) -> None:
    """Test Pinnedness algorithm issue."""
    # 1. Arrange
    qgraph = qg({
      "nodes": {
        "SN": {
          "categories": ["biolink:ChemicalEntity"],
          "set_interpretation": "BATCH",
          "constraints": [],
          "member_ids": []
        },
        "ON": {
          "ids": ["MONDO:0011705"],
          "categories": ["biolink:DiseaseOrPhenotypicFeature"],
          "set_interpretation": "BATCH",
          "constraints": [],
          "member_ids": []
        },
        "f": {
          "categories": ["biolink:Disease"],
          "set_interpretation": "BATCH",
          "constraints": [],
          "member_ids": []
        }
      },
      "edges": {
        "edge_0": {
          "subject": "SN",
          "object": "f",
          "predicates": ["biolink:treats_or_applied_or_studied_to_treat"],
          "attribute_constraints": [],
          "qualifier_constraints": []
        },
        "edge_1": {
          "subject": "f",
          "object": "ON",
          "predicates": ["biolink:has_phenotype"],
          "attribute_constraints": [],
          "qualifier_constraints": []
        },
        "edge_2": {
          "subject": "ON",
          "object": "f",
          "predicates": ["biolink:has_phenotype"],
          "attribute_constraints": [],
          "qualifier_constraints": []
        }
      }
    })

    # 2. Act
    actual = transpiler.convert_multihop_public(qgraph)

    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0011705")) @cascade(id, out_edges_e2, in_edges_e1) {
            expand(Node)
            out_edges_e2: ~subject @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n2: object @filter(eq(category, "Disease")) @cascade(id, in_edges_e0) {
                    expand(Node)
                    in_edges_e0: ~object @filter(eq(predicate_ancestors, "treats_or_applied_or_studied_to_treat")) @cascade(predicate, subject) {
                        expand(Edge) { sources expand(Source) }
                        node_n1: subject @filter(eq(category, "ChemicalEntity")) @cascade(id) {
                            expand(Node)
                        }
                    }
                }
            }
            in_edges-subclassB_e2: ~object @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_intermediate: subject @filter(has(id)) @cascade(id, ~subject) {
                    expand(Node)
                    out_edges-subclassB-mid_e2: ~subject @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, object) {
                        expand(Edge) { sources expand(Source) }
                        node_n2: object @filter(eq(category, "Disease")) @cascade(id, in_edges_e0) {
                            expand(Node)
                        }
                    }
                }
            }
            in_edges_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n2: subject @filter(eq(category, "Disease")) @cascade(id, in_edges_e0) {
                    expand(Node)
                    in_edges_e0: ~object @filter(eq(predicate_ancestors, "treats_or_applied_or_studied_to_treat")) @cascade(predicate, subject) {
                        expand(Edge) { sources expand(Source) }
                        node_n1: subject @filter(eq(category, "ChemicalEntity")) @cascade(id) {
                            expand(Node)
                        }
                    }
                }
            }
            in_edges-subclassObjB_e1: ~object @filter(eq(predicate_ancestors, "has_phenotype")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_intermediate: subject @filter(has(id)) @cascade(id, ~subject) {
                    expand(Node)
                    out_edges-subclassObjB-tail_e1: ~subject @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, object) {
                        expand(Edge) { sources expand(Source) }
                        node_n2: object @filter(eq(category, "Disease")) @cascade(id, in_edges_e0) {
                            expand(Node)
                        }
                    }
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)


# -----------------------
# Subclassing Tests
# -----------------------

def test_subclassing_preserves_constraints_on_predicate_edge(transpiler_with_subclassing: _TestDgraphTranspiler) -> None:
    """
    Unit test to verify that attribute constraints are applied to the main predicate
    edge within subclassing forms, but NOT to the 'subclass_of' edge.
    """
    qgraph = qg({
        "nodes": {"n0": {"ids": ["A"]}, "n1": {"ids": ["B"]}},
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:related_to"],
                "attribute_constraints": [{
                    "id": "knowledge_level",
                    "operator": "==",
                    "value": "prediction",
                }],
            }
        },
    })

    actual = transpiler_with_subclassing.convert_multihop_public(qgraph)

    # 1. The constraint should be on the main traversal blocks
    assert 'out_edges_e0: ~subject @filter(eq(predicate_ancestors, "related_to") AND eq(knowledge_level, "prediction"))' in actual
    assert 'in_edges-symmetric_e0: ~object @filter(eq(predicate_ancestors, "related_to") AND eq(knowledge_level, "prediction"))' in actual

    # 2. The constraint should be on the 'related_to' part of Form B, but not the 'subclass_of' part
    assert 'in_edges-subclassB_e0: ~object @filter(eq(predicate_ancestors, "subclass_of"))' in actual
    assert 'out_edges-subclassB-mid_e0: ~subject @filter(eq(predicate_ancestors, "related_to") AND eq(knowledge_level, "prediction"))' in actual

    # 3. The constraint should be on the 'related_to' part of Form C, but not the 'subclass_of' part
    assert 'out_edges-subclassC_e0: ~subject @filter(eq(predicate_ancestors, "related_to") AND eq(knowledge_level, "prediction"))' in actual
    assert 'out_edges-subclassC-tail_e0: ~subject @filter(eq(predicate_ancestors, "subclass_of"))' in actual

    # 4. The constraint should be on the 'related_to' part of Form D, but not the 'subclass_of' parts
    assert 'in_edges-subclassD_e0: ~object @filter(eq(predicate_ancestors, "subclass_of"))' in actual
    assert 'out_edges-subclassD-mid_e0: ~subject @filter(eq(predicate_ancestors, "related_to") AND eq(knowledge_level, "prediction"))' in actual
    assert 'out_edges-subclassD-tail_e0: ~subject @filter(eq(predicate_ancestors, "subclass_of"))' in actual
