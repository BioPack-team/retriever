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

# Sample 01: A single-edge Drug -> Disease query from CHEBI:6801 to MONDO:0005015.
# The Dgraph fixture returns one q0 source node with four OR-connected ways to reach node_n1:
#   direct: out_edges_e0 -> node_n1
#   subclass B: in_edges-subclassB_e0 -> node_intermediate_n0 -> out_edges-subclassB-mid_e0 -> node_n1
#   subclass C: out_edges-subclassC_e0 -> node_intermediate_n1 -> out_edges-subclassC-tail_e0 -> node_n1
#   subclass D: in_edges-subclassD_e0 -> node_intermediateA_n0 -> out_edges-subclassD-mid_e0 -> node_intermediateB_n1 -> out_edges-subclassD-tail_e0 -> node_n1
#
# Several entries in each branch are intentionally incomplete, such as missing node_n1 or missing one of the intermediate hops.
# After cascading with OR, only structurally complete branch instances should remain, while incomplete partial paths are pruned.
#
# Intended pruning logic:
# q0_node_n0 
# AND (
#         (out_edges_e0 AND node_n1)
#         OR (in_edges-subclassB_e0 AND node_intermediate_n0 AND out_edges-subclassB-mid_e0 AND node_n1)
#         OR (out_edges-subclassC_e0 AND node_intermediate_n1 AND out_edges-subclassC-tail_e0 AND node_n1)
#         OR (in_edges-subclassD_e0 AND node_intermediateA_n0 AND out_edges-subclassD-mid_e0 AND node_intermediateB_n1 AND out_edges-subclassD-tail_e0 AND node_n1)
# )

SAMPLE_01_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "nB": {
            "categories": ["biolink:Disease"],
            "ids": ["MONDO:0005015"]
        },
        "nA": {
            "categories": ["biolink:Drug"],
            "ids": ["CHEBI:6801"]
        }
    },
    "edges": {
        "e1": {
            "subject": "nA",
            "object": "nB",
            "predicates": ["biolink:treats_or_applied_or_studied_to_treat"]
        }
    }
})

SAMPLE_01_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
  "data": {
    "q0_node_n0": [
      {
        "vL_id": "CHEBI:6801",
        "out_edges_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for"
          }
        ],


        "in_edges-subclassB_e0": [
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeX",
                    "out_edges-subclassB-mid_e0": [{
                        "vL_predicate": "related_to",
                        "node_n1": {
                            "vL_id": "MONDO:0005015"
                        }
                    }]
                }
            },
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeY",
                    "out_edges-subclassB-mid_e0": [{
                        "vL_predicate": "related_to"
                    }]
                }
            },
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeY"
                }
            },
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat"
            }
        ],


        "out_edges-subclassC_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005406",
              "out_edges-subclassC-tail_e0": [
                {
                  "vL_predicate": "subclass_of",
                  "node_n1": {
                    "vL_id": "MONDO:0005015"
                  }
                }
              ]
            }
          },
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005148",
              "out_edges-subclassC-tail_e0": [
                {
                  "vL_predicate": "subclass_of"
                }
              ]
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005147"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for"
          }
        ],

        "in_edges-subclassD_e0": [
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW",
                            "out_edges-subclassD-tail_e0": [{
                                "vL_predicate": "subclass_of",
                                "node_n1": {
                                    "vL_id": "MONDO:0005015"
                                }
                            }]
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW",
                            "out_edges-subclassD-tail_e0": [{
                                "vL_predicate": "subclass_of"
                            }]
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW"
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat"
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ"
                }
            },
            {
                "vL_predicate": "subclass_of"
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
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          }
        ],

        "in_edges-subclassB_e0": [
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeX",
                    "out_edges-subclassB-mid_e0": [{
                        "vL_predicate": "related_to",
                        "node_n1": {
                            "vL_id": "MONDO:0005015"
                        }
                    }]
                }
            }
        ],

        "out_edges-subclassC_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005406",
              "out_edges-subclassC-tail_e0": [
                {
                  "vL_predicate": "subclass_of",
                  "node_n1": {
                    "vL_id": "MONDO:0005015"
                  }
                }
              ]
            }
          }
        ],

        "in_edges-subclassD_e0": [
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW",
                            "out_edges-subclassD-tail_e0": [{
                                "vL_predicate": "subclass_of",
                                "node_n1": {
                                    "vL_id": "MONDO:0005015"
                                }
                            }]
                        }
                    }]
                }
            }
        ]

      }
    ]
  }
}
""").strip()

SAMPLE_02_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "nB": {
            "categories": ["biolink:Disease"],
            "ids": ["MONDO:0005015"]
        },
        "nA": {
            "categories": ["biolink:Drug"],
            "ids": ["CHEBI:6801"]
        }
    },
    "edges": {
        "e1": {
            "subject": "nA",
            "object": "nB",
            "predicates": ["biolink:treats_or_applied_or_studied_to_treat"]
        }
    }
})

SAMPLE_02_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
  "data": {
    "q0_node_n0": [
      {
        "vL_id": "CHEBI:6801",
        "out_edges_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for"
          }
        ],

        "in_edges-subclassB_e0": [
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeX",
                    "out_edges-subclassB-mid_e0": [{
                        "vL_predicate": "related_to"
                    }]
                }
            },
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeY",
                    "out_edges-subclassB-mid_e0": [{
                        "vL_predicate": "related_to"
                    }]
                }
            },
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat",
                "node_intermediate_n0": {
                    "vL_id": "NodeY"
                }
            },
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat"
            }
        ],


        "out_edges-subclassC_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005406",
              "out_edges-subclassC-tail_e0": [
                {
                  "vL_predicate": "subclass_of"
                }
              ]
            }
          },
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005148",
              "out_edges-subclassC-tail_e0": [
                {
                  "vL_predicate": "subclass_of"
                }
              ]
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005147"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for"
          }
        ],

        "in_edges-subclassD_e0": [
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW",
                            "out_edges-subclassD-tail_e0": [{
                                "vL_predicate": "subclass_of"
                            }]
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW",
                            "out_edges-subclassD-tail_e0": [{
                                "vL_predicate": "subclass_of"
                            }]
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW"
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat"
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ"
                }
            },
            {
                "vL_predicate": "subclass_of"
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
        "vL_id": "CHEBI:6801",
        "out_edges_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          },
          {
            "vL_predicate": "in_clinical_trials_for",
            "node_n1": {
              "vL_id": "MONDO:0005015"
            }
          }
        ]

      }
    ]
  }
}
""").strip()

SAMPLE_03_QGRAPH: QueryGraphDict = qg({
    "nodes": {
        "nB": {
            "categories": ["biolink:Disease"],
            "ids": ["MONDO:0005015"]
        },
        "nA": {
            "categories": ["biolink:Drug"],
            "ids": ["CHEBI:6801"]
        }
    },
    "edges": {
        "e1": {
            "subject": "nA",
            "object": "nB",
            "predicates": ["biolink:treats_or_applied_or_studied_to_treat"]
        }
    }
})

SAMPLE_03_EXPECTED_DGRAPH_RESPONSE = dedent("""
{
  "data": {
    "q0_node_n0": [
      {
        "vL_id": "CHEBI:6801",
        "out_edges_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat"
          },
          {
            "vL_predicate": "in_clinical_trials_for"
          }
        ],


        "in_edges-subclassB_e0": [
            {
                "vL_predicate": "treats_or_applied_or_studied_to_treat"
            }
        ],


        "out_edges-subclassC_e0": [
          {
            "vL_predicate": "treats_or_applied_or_studied_to_treat",
            "node_intermediate_n1": {
              "vL_id": "MONDO:0005406",
              "out_edges-subclassC-tail_e0": [
                {
                  "vL_predicate": "subclass_of"
                }
              ]
            }
          }
        ],

        "in_edges-subclassD_e0": [
            {
                "vL_predicate": "subclass_of",
                "node_intermediateA_n0": {
                    "vL_id": "NodeZ",
                    "out_edges-subclassD-mid_e0": [{
                        "vL_predicate": "treats_or_applied_or_studied_to_treat",
                        "node_intermediateB_n1": {
                            "vL_id": "NodeW",
                            "out_edges-subclassD-tail_e0": [{
                                "vL_predicate": "subclass_of"
                            }]
                        }
                    }]
                }
            },
            {
                "vL_predicate": "subclass_of"
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
  }
}
""").strip()

# # Sample 02: A 4-edge Drug -> Gene -> BiologicalProcess -> PhenotypicFeature -> Disease query
# # from CHEBI:6801 to MONDO:0005015, where deeper hops exercise multiple subclass-expansion forms.
# #
# # The fixture keeps one complete path and mixes in incomplete alternatives at several depths:
# #   e0 (nA -> nC): subclass B
# #     in_edges-subclassB_e0 -> node_intermediate_n0 -> out_edges-subclassB-mid_e0 -> node_n2
# #   e1 (nC -> nD): subclass object form B
# #     in_edges-subclassObjB_e1 -> node_intermediate_n2 -> out_edges-subclassObjB-tail_e1 -> node_n3
# #   e2 (nD -> nE): subclass C
# #     out_edges-subclassC_e2 -> node_intermediate_n4 -> out_edges-subclassC-tail_e2 -> node_n4
# #   e3 (nE -> nB): subclass D
# #     in_edges-subclassD_e3 -> node_intermediateA_n4 -> out_edges-subclassD-mid_e3
# #       -> node_intermediateB_n1 -> out_edges-subclassD-tail_e3 -> node_n1
# #
# # Several branch instances are intentionally incomplete, such as:
# #   missing node_n2 at e0,
# #   missing node_n3 at e1,
# #   missing node_n4 or subclass tail at e2,
# #   missing intermediateB_n1 or final node_n1 at e3.
# #
# # After cascading with OR, only the single structurally complete multi-hop branch should remain,
# # and all partial branches should be pruned from their respective edge groups.
# #
# # Intended pruning logic:
# # q0_node_n0
# # AND (
# #         in_edges-subclassB_e0
# #         AND node_intermediate_n0
# #         AND out_edges-subclassB-mid_e0
# #         AND node_n2
# # )
# # AND (
# #         in_edges-subclassObjB_e1
# #         AND node_intermediate_n2
# #         AND out_edges-subclassObjB-tail_e1
# #         AND node_n3
# # )
# # AND (
# #         out_edges-subclassC_e2
# #         AND node_intermediate_n4
# #         AND out_edges-subclassC-tail_e2
# #         AND node_n4
# # )
# # AND (
# #         in_edges-subclassD_e3
# #         AND node_intermediateA_n4
# #         AND out_edges-subclassD-mid_e3
# #         AND node_intermediateB_n1
# #         AND out_edges-subclassD-tail_e3
# #         AND node_n1
# # )

# SAMPLE_02_QGRAPH: QueryGraphDict = qg({
#     "nodes": {
#         "nA": {
#             "categories": ["biolink:Drug"],
#             "ids": ["CHEBI:6801"],
#         },
#         "nB": {
#             "categories": ["biolink:Disease"],
#             "ids": ["MONDO:0005015"],
#         },
#         "nC": {
#             "categories": ["biolink:Gene"],
#         },
#         "nD": {
#             "categories": ["biolink:BiologicalProcessOrActivity"],
#             "ids": ["GO:0008150"],
#         },
#         "nE": {
#             "categories": ["biolink:PhenotypicFeature"],
#             "ids": ["HP:0000118"],
#         },
#     },
#     "edges": {
#         "e1": {
#             "subject": "nA",
#             "object": "nC",
#             "predicates": ["biolink:treats_or_applied_or_studied_to_treat"],
#         },
#         "e2": {
#             "subject": "nC",
#             "object": "nD",
#             "predicates": ["biolink:affects"],
#         },
#         "e3": {
#             "subject": "nD",
#             "object": "nE",
#             "predicates": ["biolink:affects"],
#         },
#         "e4": {
#             "subject": "nE",
#             "object": "nB",
#             "predicates": ["biolink:causes"],
#         },
#     },
# })

# SAMPLE_02_EXPECTED_DGRAPH_RESPONSE = dedent("""
# {
#   "data": {
#     "q0_node_n0": [
#       {
#         "vL_id": "CHEBI:6801",
#         "out_edges_e0": [
#           {
#             "vL_predicate": "treats_or_applied_or_studied_to_treat",
#             "node_n2": {
#               "vL_id": "NCBIGene:7004"
#             }
#           },
#           {
#             "vL_predicate": "treats_or_applied_or_studied_to_treat"
#           }
#         ],
#         "in_edges-subclassB_e0": [
#           {
#             "vL_predicate": "subclass_of",
#             "node_intermediate_n0": {
#               "vL_id": "CHEBI:6801-child",
#               "out_edges-subclassB-mid_e0": [
#                 {
#                   "vL_predicate": "treats_or_applied_or_studied_to_treat",
#                   "node_n2": {
#                     "vL_id": "NCBIGene:3162",
#                     "out_edges_e1": [
#                       {
#                         "vL_predicate": "affects"
#                       }
#                     ],
#                     "in_edges-subclassObjB_e1": [
#                       {
#                         "vL_predicate": "affects",
#                         "node_intermediate_n2": {
#                           "vL_id": "NCBIGene:3162-child",
#                           "out_edges-subclassObjB-tail_e1": [
#                             {
#                               "vL_predicate": "subclass_of",
#                               "node_n3": {
#                                 "vL_id": "GO:0008150",
#                                 "out_edges_e2": [
#                                   {
#                                     "vL_predicate": "affects",
#                                     "node_n4": {
#                                       "vL_id": "HP:9999999"
#                                     }
#                                   },
#                                   {
#                                     "vL_predicate": "affects"
#                                   }
#                                 ],
#                                 "out_edges-subclassC_e2": [
#                                   {
#                                     "vL_predicate": "affects",
#                                     "node_intermediate_n4": {
#                                       "vL_id": "HP:0000118-child",
#                                       "out_edges-subclassC-tail_e2": [
#                                         {
#                                           "vL_predicate": "subclass_of",
#                                           "node_n4": {
#                                             "vL_id": "HP:0000118",
#                                             "out_edges_e3": [
#                                               {
#                                                 "vL_predicate": "causes"
#                                               }
#                                             ],
#                                             "in_edges-subclassD_e3": [
#                                               {
#                                                 "vL_predicate": "subclass_of",
#                                                 "node_intermediateA_n4": {
#                                                   "vL_id": "HP:0000118-parent",
#                                                   "out_edges-subclassD-mid_e3": [
#                                                     {
#                                                       "vL_predicate": "causes",
#                                                       "node_intermediateB_n1": {
#                                                         "vL_id": "MONDO:0005015-child",
#                                                         "out_edges-subclassD-tail_e3": [
#                                                           {
#                                                             "vL_predicate": "subclass_of",
#                                                             "node_n1": {
#                                                               "vL_id": "MONDO:0005015"
#                                                             }
#                                                           },
#                                                           {
#                                                             "vL_predicate": "subclass_of"
#                                                           }
#                                                         ]
#                                                       }
#                                                     },
#                                                     {
#                                                       "vL_predicate": "causes"
#                                                     }
#                                                   ]
#                                                 }
#                                               },
#                                               {
#                                                 "vL_predicate": "subclass_of",
#                                                 "node_intermediateA_n4": {
#                                                   "vL_id": "HP:0000118-parent"
#                                                 }
#                                               },
#                                               {
#                                                 "vL_predicate": "subclass_of"
#                                               }
#                                             ]
#                                           }
#                                         },
#                                         {
#                                           "vL_predicate": "subclass_of"
#                                         }
#                                       ]
#                                     }
#                                   },
#                                   {
#                                     "vL_predicate": "affects",
#                                     "node_intermediate_n4": {
#                                       "vL_id": "HP:bad-child"
#                                     }
#                                   },
#                                   {
#                                     "vL_predicate": "affects"
#                                   }
#                                 ]
#                               }
#                             },
#                             {
#                               "vL_predicate": "subclass_of"
#                             }
#                           ]
#                         }
#                       },
#                       {
#                         "vL_predicate": "affects",
#                         "node_intermediate_n2": {
#                           "vL_id": "NCBIGene:3162-child"
#                         }
#                       },
#                       {
#                         "vL_predicate": "affects"
#                       }
#                     ]
#                   }
#                 },
#                 {
#                   "vL_predicate": "treats_or_applied_or_studied_to_treat",
#                   "node_n2": {
#                     "vL_id": "NCBIGene:9999"
#                   }
#                 },
#                 {
#                   "vL_predicate": "treats_or_applied_or_studied_to_treat"
#                 }
#               ]
#             }
#           },
#           {
#             "vL_predicate": "subclass_of",
#             "node_intermediate_n0": {
#               "vL_id": "CHEBI:6801-child"
#             }
#           },
#           {
#             "vL_predicate": "subclass_of"
#           }
#         ]
#       }
#     ]
#   }
# }
# """).strip()

# SAMPLE_02_EXPECTED_AFTER_CASCADE_OR = dedent("""
# {
#   "data": {
#     "q0_node_n0": [
#       {
#         "vL_id": "CHEBI:6801",
#         "in_edges-subclassB_e0": [
#           {
#             "vL_predicate": "subclass_of",
#             "node_intermediate_n0": {
#               "vL_id": "CHEBI:6801-child",
#               "out_edges-subclassB-mid_e0": [
#                 {
#                   "vL_predicate": "treats_or_applied_or_studied_to_treat",
#                   "node_n2": {
#                     "vL_id": "NCBIGene:3162",
#                     "in_edges-subclassObjB_e1": [
#                       {
#                         "vL_predicate": "affects",
#                         "node_intermediate_n2": {
#                           "vL_id": "NCBIGene:3162-child",
#                           "out_edges-subclassObjB-tail_e1": [
#                             {
#                               "vL_predicate": "subclass_of",
#                               "node_n3": {
#                                 "vL_id": "GO:0008150",
#                                 "out_edges-subclassC_e2": [
#                                   {
#                                     "vL_predicate": "affects",
#                                     "node_intermediate_n4": {
#                                       "vL_id": "HP:0000118-child",
#                                       "out_edges-subclassC-tail_e2": [
#                                         {
#                                           "vL_predicate": "subclass_of",
#                                           "node_n4": {
#                                             "vL_id": "HP:0000118",
#                                             "in_edges-subclassD_e3": [
#                                               {
#                                                 "vL_predicate": "subclass_of",
#                                                 "node_intermediateA_n4": {
#                                                   "vL_id": "HP:0000118-parent",
#                                                   "out_edges-subclassD-mid_e3": [
#                                                     {
#                                                       "vL_predicate": "causes",
#                                                       "node_intermediateB_n1": {
#                                                         "vL_id": "MONDO:0005015-child",
#                                                         "out_edges-subclassD-tail_e3": [
#                                                           {
#                                                             "vL_predicate": "subclass_of",
#                                                             "node_n1": {
#                                                               "vL_id": "MONDO:0005015"
#                                                             }
#                                                           }
#                                                         ]
#                                                       }
#                                                     }
#                                                   ]
#                                                 }
#                                               }
#                                             ]
#                                           }
#                                         }
#                                       ]
#                                     }
#                                   }
#                                 ]
#                               }
#                             }
#                           ]
#                         }
#                       }
#                     ]
#                   }
#                 }
#               ]
#             }
#           }
#         ]
#       }
#     ]
#   }
# }
# """).strip()


# -----------------------
# Case pairs
# -----------------------

CASES: list[QueryCase] = [
    QueryCase("sample-one", SAMPLE_01_QGRAPH, SAMPLE_01_EXPECTED_DGRAPH_RESPONSE, SAMPLE_01_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample-two", SAMPLE_02_QGRAPH, SAMPLE_02_EXPECTED_DGRAPH_RESPONSE, SAMPLE_02_EXPECTED_AFTER_CASCADE_OR),
    QueryCase("sample-three", SAMPLE_03_QGRAPH, SAMPLE_03_EXPECTED_DGRAPH_RESPONSE, SAMPLE_03_EXPECTED_AFTER_CASCADE_OR),
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
