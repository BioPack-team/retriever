import json
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import pytest

from retriever.data_tiers.tier_0.dgraph import result_models as dg
from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.types.trapi import QueryGraphDict

# -----------------------
# Helpers
# -----------------------

@dataclass(frozen=True)
class QueryCase:
    """Pair of input TRAPI qgraph and expected Dgraph query."""
    name: str
    qgraph: QueryGraphDict
    expected_dgraph_response: str
    expected_after_cascade_or: str


class _TestDgraphTranspiler(DgraphTranspiler):
    """Expose protected methods for testing without modifying production code."""

    def filter_cascaded_with_or_public(
        self,
        nodes: list[Any],
        qgraph: QueryGraphDict,
    ) -> list[Any]:
        self._normalize_qgraph_ids(qgraph)
        return self._filter_cascaded_with_or(nodes, qgraph)


def qg(d: dict[str, Any]) -> QueryGraphDict:
    return cast(QueryGraphDict, cast(object, d))


@pytest.fixture
def transpiler() -> _TestDgraphTranspiler:
    return _TestDgraphTranspiler()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def assert_query_equals(actual: str, expected: str) -> None:
    assert normalize(actual) == normalize(expected)


def _parse_filter_cascade_response(
    raw_response: dict[str, Any] | str,
) -> list[dg.Node]:
    """Parse a Dgraph response fixture into dg.Node objects."""
    payload = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
    response = dg.DgraphResponse.parse(payload["data"], prefix="vL_")
    return response.data.get("q0", [])


# -----------------------
# Query graph inputs
# -----------------------

# Sample 00: A 2-hop query with a symmetric edge and qualifiers on both edges. The Dgraph response contains 4 candidate e0 branches, but none forms a complete path to n1, so everything should be pruned.
# The Dgraph response from the example query:
#   CHEBI:6801 -[e0:affects]-> n2 (Gene) -[e1:related_to, symmetric]-> MONDO:0005147
#
# Four candidate e0 branches are returned by Dgraph:
#   branch 1         — missing node_n2 entirely                    -> incomplete path
#   NCBIGene:3162    — has out_edges_e1, but no node_n1            -> incomplete path
#   NCBIGene:7852    — no out_edges_e1 or in_edges-symmetric_e1    -> incomplete path
#   NCBIGene:9451    — has out_edges_e1, but no node_n1            -> incomplete path
#
# Intended pruning logic:
#   node_n0 AND out_edges_e0 AND node_n2
#     AND ( (out_edges_e1 AND node_n1) OR (in_edges-symmetric_e1 AND node_n1) )

SAMPLE_00_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:6801"]},
        "n1": {"ids": ["MONDO:0005147"]},
        "n2": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "decreased"},
                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity_or_abundance"},
                    ]
                }
            ],
        },
        "e1": {
            "subject": "n2",
            "object": "n1",
            "predicates": ["biolink:related_to"],  # symmetric predicate
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                        {"qualifier_type_id": "biolink:subject_form_or_variant_qualifier", "qualifier_value": "loss_of_function_variant_form"},
                    ]
                }
            ],
        },
    },
})

SAMPLE_00_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "CHEBI:6801",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects"
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:3162",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects"
                                }
                            ]
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:7852"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:9451",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects"
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_00_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
    }
}
""").strip()

# Sample 01: A 2-hop query with a symmetric edge and qualifiers on both edges. The Dgraph response contains 3 candidate n2 nodes, but only one has the e1 hop to n1. After cascading with OR, only the complete path should remain.
# The Dgraph response from the example query:
#   CHEBI:6801 -[e0:affects]-> n2 (Gene) -[e1:related_to, symmetric]-> MONDO:0005147
#
# Three n2 candidates are returned by Dgraph:
#   NCBIGene:7004  — no out_edges_e1 or in_edges-symmetric_e1 → incomplete path
#   NCBIGene:3162  — has out_edges_e1 → MONDO:0005147         → complete path ✓
#   NCBIGene:7852  — no out_edges_e1 or in_edges-symmetric_e1 → incomplete path
#
# Intended pruning logic:
#   node_n0 AND out_edges_e0 AND node_n2
#     AND ( (out_edges_e1 AND node_n1) OR (in_edges-symmetric_e1 AND node_n1) )

SAMPLE_01_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:6801"]},
        "n1": {"ids": ["MONDO:0005147"]},
        "n2": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "decreased"},
                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity_or_abundance"},
                    ]
                }
            ],
        },
        "e1": {
            "subject": "n2",
            "object": "n1",
            "predicates": ["biolink:related_to"],  # symmetric predicate
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                        {"qualifier_type_id": "biolink:subject_form_or_variant_qualifier", "qualifier_value": "loss_of_function_variant_form"},
                    ]
                }
            ],
        },
    },
})

SAMPLE_01_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "CHEBI:6801",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:7004"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:3162",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n1": {
                                        "vL_id": "MONDO:0005147"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:7852"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:9451",
                                "out_edges_e1": [
                                    {
                                        "vL_predicate": "affects",
                                        "node_n1": {
                                            "vL_id": "MONDO:0005147"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_01_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "CHEBI:6801",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:3162",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n1": {
                                        "vL_id": "MONDO:0005147"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "NCBIGene:9451",
                                "out_edges_e1": [
                                    {
                                        "vL_predicate": "affects",
                                        "node_n1": {
                                            "vL_id": "MONDO:0005147"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

# Sample 02
# Two-hop query with one symmetric hop:
#   n0 -[e0]-> n2 -[e1 symmetric]-> n1
#
# Intended pruning logic:
#   node_n0 AND out_edges_e0 AND node_n2
#     AND ( (out_edges_e1 AND node_n1) OR (in_edges-symmetric_e1 AND node_n1) )

SAMPLE_02_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["A"]},
        "n1": {"ids": ["B"]},
        "n2": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
        },
        "e1": {
            "subject": "n2",
            "object": "n1",
            "predicates": ["biolink:related_to"],  # symmetric predicate
        },
    },
})

SAMPLE_02_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ],
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X"
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_02_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ],
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

# Sample 03
# Five-hop query:
#   n0 -[e0]-> n2 -[e1 symmetric]-> n3 -[e2 symmetric]-> n4
#      -[e3 symmetric]-> n5 -[e4]-> n1
#
# Intended pruning logic:
#   node_n0 AND out_edges_e0 AND node_n2
#     AND ( (out_edges_e1 AND node_n3) OR (in_edges-symmetric_e1 AND node_n3) )
#     AND ( (out_edges_e2 AND node_n4) OR (in_edges-symmetric_e2 AND node_n4) )
#     AND ( (out_edges_e3 AND node_n5) OR (in_edges-symmetric_e3 AND node_n5) )
#     AND ( out_edges_e4 AND node_n1 )

SAMPLE_03_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["A"]},
        "n1": {"ids": ["B"]},
        "n2": {"categories": ["biolink:Gene"]},
        "n3": {"categories": ["biolink:Gene"]},
        "n4": {"categories": ["biolink:Gene"]},
        "n5": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
        },
        "e1": {
            "subject": "n2",
            "object": "n3",
            "predicates": ["biolink:related_to"],
        },
        "e2": {
            "subject": "n3",
            "object": "n4",
            "predicates": ["biolink:related_to"],
        },
        "e3": {
            "subject": "n4",
            "object": "n5",
            "predicates": ["biolink:related_to"],
        },
        "e4": {
            "subject": "n5",
            "object": "n1",
            "predicates": ["biolink:affects"],
        },
    },
})

SAMPLE_03_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n3": {
                                        "vL_id": "X3"
                                    }
                                }
                            ],
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n3": {
                                        "vL_id": "X3",
                                        "out_edges_e2": [
                                            {
                                                "vL_predicate": "affects",
                                                "node_n4": {
                                                    "vL_id": "X4"
                                                }
                                            }
                                        ],
                                        "in_edges-symmetric_e2": [
                                            {
                                                "vL_predicate": "affects",
                                                "node_n4": {
                                                    "vL_id": "X4",
                                                    "out_edges_e3": [
                                                        {
                                                            "vL_predicate": "affects",
                                                            "node_n5": {
                                                                "vL_id": "X5"
                                                            }
                                                        }
                                                    ],
                                                    "in_edges-symmetric_e3": [
                                                        {
                                                            "vL_predicate": "affects",
                                                            "node_n5": {
                                                                "vL_id": "X5",
                                                                "out_edges_e4": [
                                                                    {
                                                                        "vL_predicate": "affects",
                                                                        "node_n1": {
                                                                            "vL_id": "B"
                                                                        }
                                                                    }
                                                                ]
                                                            }
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_03_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "affects",
                                    "node_n3": {
                                        "vL_id": "X3",
                                        "in_edges-symmetric_e2": [
                                            {
                                                "vL_predicate": "affects",
                                                "node_n4": {
                                                    "vL_id": "X4",
                                                    "in_edges-symmetric_e3": [
                                                        {
                                                            "vL_predicate": "affects",
                                                            "node_n5": {
                                                                "vL_id": "X5",
                                                                "out_edges_e4": [
                                                                    {
                                                                        "vL_predicate": "affects",
                                                                        "node_n1": {
                                                                            "vL_id": "B"
                                                                        }
                                                                    }
                                                                ]
                                                            }
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

# Sample 04
# Reverse-order variant:
#   n0 <-[e0]- n2 -[e1 symmetric]-> n1
#
# Intended pruning logic:
#   node_n0 AND in_edges_e0 AND node_n2
#     AND ( (out_edges_e1 AND node_n1) OR (in_edges-symmetric_e1 AND node_n1) )

SAMPLE_04_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["A"]},
        "n1": {"ids": ["B"]},
        "n2": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n2",
            "object": "n0",
            "predicates": ["biolink:affects"],
        },
        "e1": {
            "subject": "n2",
            "object": "n1",
            "predicates": ["biolink:related_to"],  # symmetric predicate
        },
    },
})

SAMPLE_04_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "in_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X1"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ],
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X3"
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_04_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "in_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ],
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

# Sample 05
# Reverse-only symmetric branch:
#   n0 -[e0]-> n2 -[e1 symmetric]-> n1
#
# Only the reverse symmetric branch is present for the valid candidate.
# Intended pruning logic:
#   node_n0 AND out_edges_e0 AND node_n2
#     AND ( (out_edges_e1 AND node_n1) OR (in_edges-symmetric_e1 AND node_n1) )

SAMPLE_05_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["A"]},
        "n1": {"ids": ["B"]},
        "n2": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
        },
        "e1": {
            "subject": "n2",
            "object": "n1",
            "predicates": ["biolink:related_to"],
        },
    },
})

SAMPLE_05_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X1"
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X3"
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_05_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()


# Sample 06
# Multiple top-level roots:
#   n0 -[e0]-> n2 -[e1 symmetric]-> n1
#
# CHEBI:1 has a complete path.
# CHEBI:2 has no complete downstream path and should be removed.

SAMPLE_06_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"categories": ["biolink:ChemicalEntity"]},
        "n1": {"ids": ["B"]},
        "n2": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
        },
        "e1": {
            "subject": "n2",
            "object": "n1",
            "predicates": ["biolink:related_to"],
        },
    },
})

SAMPLE_06_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "CHEBI:1",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X1",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "vL_id": "CHEBI:2",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2"
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_06_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "CHEBI:1",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X1",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n1": {
                                        "vL_id": "B"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()


# Sample 07
# One symmetric qedge group with mixed valid/invalid branches:
#   n0 -[e0]-> n2 -[e1 symmetric]-> n3 -[e2]-> n1
#
# At n2, both out_edges_e1 and in_edges-symmetric_e1 are present.
# The out_edges_e1 branch is incomplete because its n3 has no e2 to n1.
# The in_edges-symmetric_e1 branch is complete and should be kept.

SAMPLE_07_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["A"]},
        "n1": {"ids": ["B"]},
        "n2": {"categories": ["biolink:Gene"]},
        "n3": {"categories": ["biolink:Gene"]},
    },
    "edges": {
        "e0": {
            "subject": "n0",
            "object": "n2",
            "predicates": ["biolink:affects"],
        },
        "e1": {
            "subject": "n2",
            "object": "n3",
            "predicates": ["biolink:related_to"],
        },
        "e2": {
            "subject": "n3",
            "object": "n1",
            "predicates": ["biolink:affects"],
        },
    },
})

SAMPLE_07_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "out_edges_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n3": {
                                        "vL_id": "X3_BAD"
                                    }
                                }
                            ],
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n3": {
                                        "vL_id": "X3_GOOD",
                                        "out_edges_e2": [
                                            {
                                                "vL_predicate": "affects",
                                                "node_n1": {
                                                    "vL_id": "B"
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()

SAMPLE_07_EXPECTED_AFTER_CASCADE_OR = dedent("""
{
    "data": {
        "q0_node_n0": [
            {
                "vL_id": "A",
                "out_edges_e0": [
                    {
                        "vL_predicate": "affects",
                        "node_n2": {
                            "vL_id": "X2",
                            "in_edges-symmetric_e1": [
                                {
                                    "vL_predicate": "related_to",
                                    "node_n3": {
                                        "vL_id": "X3_GOOD",
                                        "out_edges_e2": [
                                            {
                                                "vL_predicate": "affects",
                                                "node_n1": {
                                                    "vL_id": "B"
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
""").strip()


# -----------------------
# Case pairs
# -----------------------

CASES: list[QueryCase] = [
    QueryCase("sample00-no-complete-paths", SAMPLE_00_QGRAPH, SAMPLE_00_EXPECTED_DGRAPH_RESPONSE, SAMPLE_00_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample01-keep-complete-symmetric-paths", SAMPLE_01_QGRAPH, SAMPLE_01_EXPECTED_DGRAPH_RESPONSE, SAMPLE_01_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample02-keep-branch-with-both-symmetric-directions", SAMPLE_02_QGRAPH, SAMPLE_02_EXPECTED_DGRAPH_RESPONSE, SAMPLE_02_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample03-prune-deep-symmetric-chain", SAMPLE_03_QGRAPH, SAMPLE_03_EXPECTED_DGRAPH_RESPONSE, SAMPLE_03_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample04-handle-reverse-root-edge", SAMPLE_04_QGRAPH, SAMPLE_04_EXPECTED_DGRAPH_RESPONSE, SAMPLE_04_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample05-keep-reverse-only-symmetric-branch", SAMPLE_05_QGRAPH, SAMPLE_05_EXPECTED_DGRAPH_RESPONSE, SAMPLE_05_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample06-prune-invalid-top-level-roots", SAMPLE_06_QGRAPH, SAMPLE_06_EXPECTED_DGRAPH_RESPONSE, SAMPLE_06_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample07-keep-only-valid-symmetric-branch", SAMPLE_07_QGRAPH, SAMPLE_07_EXPECTED_DGRAPH_RESPONSE, SAMPLE_07_EXPECTED_AFTER_CASCADE_OR),
]


# -------------------------------------------
# Tests: Parametrized regression test
# -------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_cascade_with_or(transpiler: _TestDgraphTranspiler, case: QueryCase) -> None:
    nodes = _parse_filter_cascade_response(case.expected_dgraph_response)

    expected_nodes = _parse_filter_cascade_response(
        case.expected_after_cascade_or
    )

    actual_after_cascade = transpiler.filter_cascaded_with_or_public(
        nodes, case.qgraph
    )

    assert actual_after_cascade == expected_nodes
