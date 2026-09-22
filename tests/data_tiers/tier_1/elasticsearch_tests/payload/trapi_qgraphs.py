import copy
from typing import Any, cast

from translator_tom.v2_0 import (
    QEdgeID,
    QNodeID,
)
from translator_tom.v2_0.model_dicts import (
    AttributeConstraintDict,
    QualifierSetConstraint,
    QueryGraphDict,
)

from .trapi_attributes import (
    ATTRIBUTE_CONSTRAINTS,
    INVALID_REGEX_CONSTRAINTS,
    NODE_CONSTRAINTS,
    VALID_REGEX_CONSTRAINTS,
    base_constraint,
    base_negation_constraint,
)
from .trapi_qualifiers import (
    frequency_qualifier_constraint,
    multiple_qualifier_constraints,
    sex_qualifier_constraint,
    single_qualifier_constraint,
    single_qualifier_constraint_with_single_qualifier_entry,
)


def qg(d: dict[str, Any]) -> QueryGraphDict:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(QueryGraphDict, cast(object, d))


E0 = QEdgeID("e0")
N0 = QNodeID("n0")
N1 = QNodeID("n1")

BASE_QGRAPH = qg(
    {
        "nodes": {
            N0: {"ids": ["CHEBI:4514"]},
            N1: {"ids": ["UMLS:C1564592"]},
        },
        "edges": {
            E0: {
                "object": "n0",
                "subject": "n1",
                "predicates": ["subclass_of"],
            },
        },
    }
)

DINGO_QGRAPH = qg(
    {
        "nodes": {
            N0: {"ids": ["MONDO:0030010", "MONDO:0011766", "MONDO:0009890"]},
            N1: {},
        },
        "edges": {
            E0: {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:has_phenotype"],
                "constraints": {
                    "qualifiers": [
                        sex_qualifier_constraint,
                        frequency_qualifier_constraint,
                    ],
                    "attributes": [base_constraint, base_negation_constraint],
                },
            }
        },
    }
)

QGRAPH_MULTIPLE_IDS = qg(
    {
        "nodes": {
            "n0": {"ids": ["CHEBI:3125", "CHEBI:53448"]},
            "n1": {"ids": ["UMLS:C0282090", "CHEBI:10119"]},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["interacts_with"],
            },
        },
    }
)


def generate_qgraph_with_qualifier_constraints(
    qualifier_constraints: list[QualifierSetConstraint],
):
    """Generate a QGraph with qualifier constraints."""
    _q_graph = copy.deepcopy(BASE_QGRAPH)
    _q_graph["edges"][E0]["constraints"] = {"qualifiers": qualifier_constraints}

    return _q_graph


Q_GRAPH_WITH_ATTRIBUTE_CONSTRAINTS = copy.deepcopy(BASE_QGRAPH)
Q_GRAPH_WITH_ATTRIBUTE_CONSTRAINTS["edges"][E0]["constraints"] = {
    "attributes": ATTRIBUTE_CONSTRAINTS
}


Q_GRAPH_WITH_SUBJECT_NODE_CONSTRAINTS = copy.deepcopy(BASE_QGRAPH)
Q_GRAPH_WITH_SUBJECT_NODE_CONSTRAINTS["nodes"][N1]["constraints"] = NODE_CONSTRAINTS

Q_GRAPH_WITH_OBJECT_NODE_CONSTRAINTS = copy.deepcopy(BASE_QGRAPH)
Q_GRAPH_WITH_OBJECT_NODE_CONSTRAINTS["nodes"][N0]["constraints"] = NODE_CONSTRAINTS

qualifier_constraints_variants = [
    multiple_qualifier_constraints,
    single_qualifier_constraint,
    single_qualifier_constraint_with_single_qualifier_entry,
]

Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS: list[QueryGraphDict] = [
    generate_qgraph_with_qualifier_constraints(variant)
    for variant in qualifier_constraints_variants
]

COMPREHENSIVE_QGRAPH = copy.deepcopy(Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS[0])
COMPREHENSIVE_QGRAPH["edges"][E0]["constraints"]["attributes"] = ATTRIBUTE_CONSTRAINTS


ALT_BASE_GRAPH = qg(
    {
        "nodes": {
            "n0": {"categories": ["biolink:NamedThing"]},
            "n1": {"ids": ["NCBIGene:3778"]},
        },
        "edges": {
            E0: {"object": "n1", "subject": "n0", "predicates": ["biolink:related_to"]}
        },
    }
)


ID_BYPASS_PAYLOAD = qg(
    {
        "nodes": {
            "sn": {
                "categories": ["biolink:Gene"],
                "ids": ["CHEBI:45783"],
                "is_set": False,
            },
            "n1": {
                "categories": ["biolink:NamedThing"],
            },
        },
        "edges": {
            "e0": {
                "subject": "sn",
                "object": "n1",
                "predicates": ["biolink:related_to"],
            }
        },
    }
)

SINGLE_EXPANDED_QUALIFIER_QGRAPH = qg(
    {
        "nodes": {
            "ON": {
                "ids": ["HGNC:3870"],
                "categories": ["biolink:Gene"],
            },
            "SN": {
                "ids": ["CHEBI:59173"],
                "categories": ["biolink:ChemicalEntity"],
            },
        },
        "edges": {
            "e0": {
                "subject": "SN",
                "object": "ON",
                "predicates": ["biolink:affects"],
                "constraints": {
                    "qualifiers": [
                        {"biolink:qualified_predicate": "biolink:causes"},
                    ],
                },
            },
        },
    }
)


EXPANDED_QUALIFIER_QGRAPH = qg(
    {
        "nodes": {
            "ON": {
                "ids": ["HGNC:3870"],
                "categories": ["biolink:Gene"],
            },
            "SN": {
                "ids": ["CHEBI:59173"],
                "categories": ["biolink:ChemicalEntity"],
            },
        },
        "edges": {
            "e0": {
                "subject": "SN",
                "object": "ON",
                "predicates": ["biolink:affects"],
                "constraints": {
                    "qualifiers": [
                        {
                            "biolink:qualified_predicate": "biolink:causes",
                            "biolink:object_aspect_qualifier": "activity_or_abundance",
                            "biolink:object_direction_qualifier": "decreased",
                        }
                    ],
                },
            },
        },
    }
)


HYDRATION_QGRAPH = qg(
    {
        "nodes": {
            "on": {
                "categories": ["biolink:Gene", "biolink:Protein"],
                "ids": ["NCBIGene:4314"],
                "is_set": False,
            },
            "sn": {
                "categories": ["biolink:ChemicalEntity"],
                "ids": ["CHEBI:48927"],
                "is_set": False,
            },
        },
        "edges": {
            "e00": {
                "object": "on",
                "predicates": ["biolink:affects"],
                "subject": "sn",
            }
        },
    }
)


def generate_qgraph_with_attribute_constraints(
    constraints: list[AttributeConstraintDict],
):
    """Generate a QGraph with attribute constraints."""
    _q_graph = copy.deepcopy(ALT_BASE_GRAPH)
    _q_graph["edges"][E0]["constraints"] = {"attributes": constraints}

    return _q_graph


VALID_REGEX_QGRAPHS: list[QueryGraphDict] = [
    generate_qgraph_with_attribute_constraints([constraint])
    for constraint in VALID_REGEX_CONSTRAINTS
]

INVALID_REGEX_QGRAPHS: list[QueryGraphDict] = [
    generate_qgraph_with_attribute_constraints([constraint])
    for constraint in INVALID_REGEX_CONSTRAINTS
]
