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

SIMPLE_QGRAPH_MULTIPLE_IDS: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125", "CHEBI:53448"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090", "CHEBI:10119"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["interacts_with"],
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
            "predicates": ["biolink:interacts_with"],
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
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["interacts_with"],
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
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["interacts_with"],
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
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e3": {
            "object": "n3",
            "subject": "n4",
            "predicates": ["interacts_with"],
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
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e1": {
            "object": "n1",
            "subject": "n2",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e2": {
            "object": "n2",
            "subject": "n3",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e3": {
            "object": "n3",
            "subject": "n4",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
        "e4": {
            "object": "n4",
            "subject": "n5",
            "predicates": ["interacts_with"],
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
                "interacts_with",
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


# -----------------------
# Expected queries
# -----------------------

EXP_SIMPLE = dedent("""
{
    q0_node_n1(func: eq(id, "UMLS:C1564592")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "CHEBI:4514"))  @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_SIMPLE_REVERSE = dedent("""
{
    q0_node_n1(func: eq(id, "NCBIGene:3778")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
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
    q0_node_n1(func: eq(v1_id, "UMLS:C1564592")) @cascade(v1_id, ~v1_subject) {
        expand(v1_Node)
        out_edges_e0: ~v1_subject @filter(eq(v1_predicate_ancestors, "subclass_of")) @cascade(v1_predicate, v1_object) {
            expand(v1_Edge) { v1_sources expand(v1_Source) }
            node_n0: v1_object @filter(eq(v1_id, "CHEBI:4514")) @cascade(v1_id) {
                expand(v1_Node)
            }
        }
    }
}
"""
).strip()

EXP_SIMPLE_MULTIPLE_IDS = dedent("""
{
    q0_node_n1(func: eq(id, ["UMLS:C0282090", "CHEBI:10119"])) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, ["CHEBI:3125", "CHEBI:53448"])) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_TWO_HOP = dedent("""
{
    q0_node_n1(func: eq(id, "UMLS:C0282090")) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "CHEBI:3125")) @cascade(id) {
                expand(Node)
            }
        }
        in_edges_e1: ~object @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_TWO_HOP_WITH_VERSION = dedent("""
{
    q0_node_n1(func: eq(v1_id, "UMLS:C0282090")) @cascade(v1_id, ~v1_subject, ~v1_object) {
        expand(v1_Node)
        out_edges_e0: ~v1_subject @filter(eq(v1_predicate_ancestors, "interacts_with")) @cascade(v1_predicate, v1_object) {
            expand(v1_Edge) { v1_sources expand(v1_Source) }
            node_n0: v1_object @filter(eq(v1_id, "CHEBI:3125")) @cascade(v1_id) {
                expand(v1_Node)
            }
        }
        in_edges_e1: ~v1_object @filter(eq(v1_predicate_ancestors, "interacts_with")) @cascade(v1_predicate, v1_subject) {
            expand(v1_Edge) { v1_sources expand(v1_Source) }
            node_n2: v1_subject @filter(eq(v1_id, "UMLS:C0496995")) @cascade(v1_id) {
                expand(v1_Node)
            }
        }
    }
}
""").strip()

EXP_THREE_HOP = dedent("""
{
    q0_node_n2(func: eq(id, "UMLS:C0496995")) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e1: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n1: object @filter(eq(id, "UMLS:C0282090")) @cascade(id, ~subject) {
                expand(Node)
                out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
                    expand(Edge) { sources expand(Source) }
                    node_n0: object @filter(eq(id, "CHEBI:3125")) @cascade(id) {
                        expand(Node)
                    }
                }
            }
        }
        in_edges_e2: ~object @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n3: subject @filter(eq(id, "UMLS:C0149720")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip().replace("knowledge level", "knowledge_level")

EXP_FOUR_HOP = dedent("""
{
    q0_node_n3(func: eq(id, "UMLS:C0149720")) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e2: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n2: object @filter(eq(id, "UMLS:C0496995")) @cascade(id, ~subject) {
                expand(Node)
                out_edges_e1: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
                    expand(Edge) { sources expand(Source) }
                    node_n1: object @filter(eq(id, "UMLS:C0282090")) @cascade(id, ~subject) {
                        expand(Node)
                        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
                            expand(Edge) { sources expand(Source) }
                            node_n0: object @filter(eq(id, "CHEBI:3125")) @cascade(id) {
                                expand(Node)
                            }
                        }
                    }
                }
            }
        }
        in_edges_e3: ~object @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n4: subject @filter(eq(id, "UMLS:C0496994")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_FIVE_HOP = dedent("""
{
    q0_node_n4(func: eq(id, "UMLS:C0496994")) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e3: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n3: object @filter(eq(id, "UMLS:C0149720")) @cascade(id, ~subject) {
                expand(Node)
                out_edges_e2: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
                    expand(Edge) { sources expand(Source) }
                    node_n2: object @filter(eq(id, "UMLS:C0496995")) @cascade(id, ~subject) {
                        expand(Node)
                        out_edges_e1: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
                            expand(Edge) { sources expand(Source) }
                            node_n1: object @filter(eq(id, "UMLS:C0282090")) @cascade(id, ~subject) {
                                expand(Node)
                                out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
                                    expand(Edge) { sources expand(Source) }
                                    node_n0: object @filter(eq(id, "CHEBI:3125")) @cascade(id) {
                                        expand(Node)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        in_edges_e4: ~object @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n5: subject @filter(eq(id, "UMLS:C2879715")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_FIVE_HOP_MULTIPLE_IDS = dedent("""
{
    q0_node_n3(func: eq(id, ["Q6", "Q7"])) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e2: ~subject @filter(eq(predicate_ancestors, "P2")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n2: object @filter(eq(id, ["Q4", "Q5"])) @cascade(id, ~subject) {
                expand(Node)
                out_edges_e1: ~subject @filter(eq(predicate_ancestors, "P1")) @cascade(predicate, object) {
                    expand(Edge) { sources expand(Source) }
                    node_n1: object @filter(eq(id, ["Q2", "Q3"])) @cascade(id, ~subject) {
                        expand(Node)
                        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "P0")) @cascade(predicate, object) {
                            expand(Edge) { sources expand(Source) }
                            node_n0: object @filter(eq(id, ["Q0", "Q1"])) @cascade(id) {
                                expand(Node)
                            }
                        }
                    }
                }
            }
        }
        in_edges_e3: ~object @filter(eq(predicate_ancestors, "P3")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n4: subject @filter(eq(id, ["Q8", "Q9"])) @cascade(id, ~object) {
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
""").strip()

EXP_CATEGORY_FILTER = dedent("""
{
    q0_node_n1(func: eq(category, "Disease")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "gene_associated_with_condition")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(category, "Gene")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_MULTIPLE_FILTERS = dedent("""
{
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, ~object) {
        expand(Node)
        in_edges_e0: ~object @filter(eq(predicate_ancestors, ["interacts_with", "contributes_to"]) AND eq(knowledge_level, "prediction")) @cascade(predicate, subject) {
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
    q0_node_n0(func: eq(id, "CHEBI:3125")) @cascade(id, ~object) {
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
    q0_node_n0(func: eq(id, "DOID:14330")) @cascade(id, ~object) {
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
    q0_node_n0(func: eq(id, "DOID:14330")) @cascade(id, ~object) {
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
    q0_node_n1(func: eq(id, "Q2")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "P")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, ["Q0", "Q1"])) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_PREDICATES_SINGLE = dedent("""
{
    q0_node_n1(func: eq(id, "B")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "Ponly")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "A")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_ATTRIBUTES_ONLY = dedent("""
{
    q0_node_n1(func: eq(id, "B")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(knowledge_level, "primary")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "A")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_START_OBJECT_WITH_IDS = dedent("""
{
    q0_node_n1(func: eq(id, "Y")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "rel")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "X")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_BATCH_QGRAPHS = dedent("""
{
    q0_node_n1(func: eq(id, "UMLS:C1564592")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "CHEBI:4514")) @cascade(id) {
                expand(Node)
            }
        }
    }

    q1_node_n1(func: eq(id, ["UMLS:C0282090", "CHEBI:10119"])) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, ["CHEBI:3125", "CHEBI:53448"])) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()

EXP_BATCH_QGRAPHS_MULTI_HOP = dedent("""
{
    q0_node_n1(func: eq(id, "UMLS:C1564592")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "subclass_of")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "CHEBI:4514")) @cascade(id) {
                expand(Node)
            }
        }
    }

    q1_node_n1(func: eq(id, ["UMLS:C0282090", "CHEBI:10119"])) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, ["CHEBI:3125", "CHEBI:53448"])) @cascade(id) {
                expand(Node)
            }
        }
    }

    q2_node_n1(func: eq(id, "UMLS:C0282090")) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n0: object @filter(eq(id, "CHEBI:3125")) @cascade(id) {
                expand(Node)
            }
        }
        in_edges_e1: ~object @filter(eq(predicate_ancestors, "interacts_with")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n2: subject @filter(eq(id, "UMLS:C0496995")) @cascade(id) {
                expand(Node)
            }
        }
    }

    q3_node_n3(func: eq(id, ["Q6", "Q7"])) @cascade(id, ~subject, ~object) {
        expand(Node)
        out_edges_e2: ~subject @filter(eq(predicate_ancestors, "P2")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n2: object @filter(eq(id, ["Q4", "Q5"])) @cascade(id, ~subject) {
                expand(Node)
                out_edges_e1: ~subject @filter(eq(predicate_ancestors, "P1")) @cascade(predicate, object) {
                    expand(Edge) { sources expand(Source) }
                    node_n1: object @filter(eq(id, ["Q2", "Q3"])) @cascade(id, ~subject) {
                        expand(Node)
                        out_edges_e0: ~subject @filter(eq(predicate_ancestors, "P0")) @cascade(predicate, object) {
                            expand(Edge) { sources expand(Source) }
                            node_n0: object @filter(eq(id, ["Q0", "Q1"])) @cascade(id) {
                                expand(Node)
                            }
                        }
                    }
                }
            }
        }
        in_edges_e3: ~object @filter(eq(predicate_ancestors, "P3")) @cascade(predicate, subject) {
            expand(Edge) { sources expand(Source) }
            node_n4: subject @filter(eq(id, ["Q8", "Q9"])) @cascade(id, ~object) {
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
""").strip()

EXP_BATCH_MULTI_IDS_SINGLE = dedent("""
{
    q0_node_n1(func: eq(id, "C")) @cascade(id, ~subject) {
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
    q0_node_n1(func: eq(id, "D")) @cascade(id, ~subject) {
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
    q0_node_n0(func: eq(id, "NCBIGene:3778")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e01: ~subject @filter(eq(predicate_ancestors, "causes")) @cascade(predicate, object) {
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
    q0_node_n0(func: eq(v1_id, "NCBIGene:3778")) @cascade(v1_id, ~v1_subject) {
        expand(v1_Node)
        out_edges_e01: ~v1_subject @filter(eq(v1_predicate_ancestors, "causes")) @cascade(v1_predicate, v1_object) {
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
    q0_node_n0(func: eq(id, "NCBIGene:3778")) @cascade(id, ~subject) {
        expand(Node)
        out_edges_e01: ~subject @filter(eq(predicate_ancestors, "causes")) @cascade(predicate, object) {
            expand(Edge) { sources expand(Source) }
            node_n1: object @filter(eq(category, "Disease")) @cascade(id) {
                expand(Node)
            }
        }
    }
}
""").strip()


# -----------------------
# Case pairs
# -----------------------

CASES: list[QueryCase] = [
    QueryCase("simple-one", SIMPLE_QGRAPH, EXP_SIMPLE),
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

@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_convert_multihop_pairs(transpiler: _TestDgraphTranspiler, case: QueryCase) -> None:
    actual = transpiler.convert_multihop_public(case.qgraph)
    assert_query_equals(actual, case.expected)

@pytest.mark.parametrize("case", CASES_VERSIONED, ids=[c.name for c in CASES_VERSIONED])
def test_convert_multihop_pairs_with_version(transpiler: _TestDgraphTranspiler, case: QueryCase) -> None:
    transpiler = _TestDgraphTranspiler(version="v1")
    actual = transpiler.convert_multihop_public(case.qgraph)
    assert_query_equals(actual, case.expected)

@pytest.mark.parametrize("case", BATCH_CASES, ids=[c.name for c in BATCH_CASES])
def test_convert_batch_multihop_pairs(transpiler: _TestDgraphTranspiler, case: BatchCase) -> None:
    actual = transpiler.convert_batch_multihop_public(case.qgraphs)
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
                    "object": {
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


def test_convert_results_with_full_source_info(transpiler: _TestDgraphTranspiler) -> None:
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
    backend_result = transpiler.convert_results(qgraph, parsed_nodes)

    # 4. Assertions
    assert len(backend_result["knowledge_graph"]["edges"]) == 1

    # Get the single edge from the knowledge graph to inspect its sources
    trapi_edge = next(iter(backend_result["knowledge_graph"]["edges"].values()))

    assert len(trapi_edge["sources"]) == 1
    source = trapi_edge["sources"][0]

    # Assert that the keys exist before accessing them.
    assert "upstream_resource_ids" in source
    assert "source_record_urls" in source

    # Assert that the new fields are correctly populated
    assert source["resource_id"] == "infores:goa"
    assert source["resource_role"] == "primary_knowledge_source"
    assert source["upstream_resource_ids"] == ["infores:uniprot"]
    assert source["source_record_urls"] == ["http://example.com/record1"]


def test_symmetric_predicate_generates_bidirectional_queries(transpiler: _TestDgraphTranspiler) -> None:
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
    actual = transpiler.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, ~subject) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) { expand(Node) } }
            out_edges_e0_reverse: ~object @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, subject) {
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
    assert "out_edges_e0_reverse:" in actual


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
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, ~object) {
            expand(Node)
            in_edges_e0: ~object @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
            in_edges_e0_reverse: ~subject @filter(eq(predicate_ancestors, "correlated_with")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
        }
    }
    """).strip()

    # 3. Assert
    assert normalize(actual) == normalize(expected)
    # Should have both the incoming direction and its reverse
    assert "in_edges_e0:" in actual
    assert "in_edges_e0_reverse:" in actual


def test_symmetric_predicate_multi_hop(transpiler: _TestDgraphTranspiler) -> None:
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
    actual = transpiler.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, ~subject) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id, ~subject) {
                    expand(Node)
                    out_edges_e1: ~subject @filter(eq(predicate_ancestors, "participates_in")) @cascade(predicate, object) {
                        expand(Edge) { sources expand(Source) }
                        node_n2: object @filter(eq(category, "Pathway")) @cascade(id) {
                            expand(Node)
                        }
                    }
                }
            }
            out_edges_e0_reverse: ~object @filter(eq(predicate_ancestors, "related_to")) @cascade(predicate, subject) {
                expand(Edge) { sources expand(Source) }
                node_n1: subject @filter(eq(category, "Gene")) @cascade(id, ~subject) {
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
    assert "out_edges_e0_reverse:" in actual
    # Second edge should only have one direction
    assert "out_edges_e1:" in actual
    assert "out_edges_e1_reverse:" not in actual


def test_multiple_symmetric_predicates_on_edge(transpiler: _TestDgraphTranspiler) -> None:
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
    actual = transpiler.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, ~subject) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, ["related_to", "associated_with"])) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "Gene")) @cascade(id) {
                    expand(Node)
                }
            }
            out_edges_e0_reverse: ~object @filter(eq(predicate_ancestors, ["related_to", "associated_with"])) @cascade(predicate, subject) {
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
    assert "out_edges_e0_reverse:" in actual


def test_mixed_predicates_treats_as_symmetric(transpiler: _TestDgraphTranspiler) -> None:
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
    actual = transpiler.convert_multihop_public(qgraph)
    expected = dedent("""
    {
        q0_node_n0(func: eq(id, "MONDO:0005148")) @cascade(id, ~subject) {
            expand(Node)
            out_edges_e0: ~subject @filter(eq(predicate_ancestors, ["related_to", "treated_by"])) @cascade(predicate, object) {
                expand(Edge) { sources expand(Source) }
                node_n1: object @filter(eq(category, "ChemicalEntity")) @cascade(id) {
                    expand(Node)
                }
            }
            out_edges_e0_reverse: ~object @filter(eq(predicate_ancestors, ["related_to", "treated_by"])) @cascade(predicate, subject) {
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
    assert "out_edges_e0_reverse:" in actual
