import copy

from translator_tom import (
    AttributeConstraint,
    Biolink,
    QEdge,
    QEdgeID,
    QNode,
    QNodeID,
    Qualifier,
    QualifierConstraint,
    QueryGraph,
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

E0 = QEdgeID("e0")
N0 = QNodeID("n0")
N1 = QNodeID("n1")
SN = QNodeID("sn")
SUBJECT_NODE = QNodeID("SN")
OBJECT_NODE = QNodeID("ON")

BASE_QGRAPH = QueryGraph(
    nodes={N0: QNode(ids=["CHEBI:4514"]), N1: QNode(ids=["UMLS:C1564592"])},
    edges={E0: QEdge(subject=N0, object=N1, predicates=[Biolink("subclass_of")])},
)

DINGO_QGRAPH = QueryGraph(
    nodes={
        N0: QNode(ids=["MONDO:0030010", "MONDO:0011766", "MONDO:0009890"]),
        N1: QNode(),
    },
    edges={
        E0: QEdge(
            subject=N0,
            object=N1,
            predicates=[Biolink("has_phenotype")],
            qualifier_constraints=[
                sex_qualifier_constraint,
                frequency_qualifier_constraint,
            ],
            attribute_constraints=[base_constraint, base_negation_constraint],
        )
    },
)

QGRAPH_MULTIPLE_IDS = QueryGraph(
    nodes={
        N0: QNode(ids=["CHEBI:3125", "CHEBI:53448"]),
        N1: QNode(ids=["UMLS:C0282090", "CHEBI:10119"]),
    },
    edges={E0: QEdge(subject=N0, object=N1, predicates=[Biolink("interacts_with")])},
)


def generate_qgraph_with_qualifier_constraints(
    qualifier_constraints: list[QualifierConstraint],
):
    """Generate a QGraph with qualifier constraints."""
    _q_graph = copy.deepcopy(BASE_QGRAPH)
    _q_graph.edges[E0].qualifier_constraints = qualifier_constraints

    return _q_graph


Q_GRAPH_WITH_ATTRIBUTE_CONSTRAINTS = copy.deepcopy(BASE_QGRAPH)
Q_GRAPH_WITH_ATTRIBUTE_CONSTRAINTS.edges[
    E0
].attribute_constraints = ATTRIBUTE_CONSTRAINTS


Q_GRAPH_WITH_SUBJECT_NODE_CONSTRAINTS = copy.deepcopy(BASE_QGRAPH)
Q_GRAPH_WITH_SUBJECT_NODE_CONSTRAINTS.nodes[N1].constraints = NODE_CONSTRAINTS

Q_GRAPH_WITH_OBJECT_NODE_CONSTRAINTS = copy.deepcopy(BASE_QGRAPH)
Q_GRAPH_WITH_OBJECT_NODE_CONSTRAINTS.nodes[N0].constraints = NODE_CONSTRAINTS

qualifier_constraints_variants = [
    multiple_qualifier_constraints,
    single_qualifier_constraint,
    single_qualifier_constraint_with_single_qualifier_entry,
]

Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS: list[QueryGraph] = [
    generate_qgraph_with_qualifier_constraints(variant)
    for variant in qualifier_constraints_variants
]

COMPREHENSIVE_QGRAPH = copy.deepcopy(Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS[0])
COMPREHENSIVE_QGRAPH.edges[E0].attribute_constraints = ATTRIBUTE_CONSTRAINTS


ALT_BASE_GRAPH = QueryGraph(
    nodes={
        N0: QNode(categories=[Biolink("NamedThing")]),
        N1: QNode(ids=["NCBIGene:3778"]),
    },
    edges={E0: QEdge(subject=N0, object=N1, predicates=[Biolink("related_to")])},
)


ID_BYPASS_PAYLOAD = QueryGraph(
    nodes={
        SN: QNode(
            categories=[Biolink("Gene")],
            constraints=[],
            ids=["CHEBI:45783"],
        ),
        N1: QNode(
            categories=[Biolink("NamedThing")],
            constraints=[],
        ),
    },
    edges={E0: QEdge(subject=SN, object=N1, predicates=[Biolink("related_to")])},
)

SINGLE_EXPANDED_QUALIFIER_QGRAPH = QueryGraph(
    nodes={
        OBJECT_NODE: QNode(
            ids=["HGNC:3870"],
            categories=[Biolink("Gene")],
        ),
        SUBJECT_NODE: QNode(
            ids=["CHEBI:59173"],
            categories=[Biolink("ChemicalEntity")],
        ),
    },
    edges={
        E0: QEdge(
            subject=SUBJECT_NODE,
            object=OBJECT_NODE,
            predicates=[Biolink("affects")],
            qualifier_constraints=[
                QualifierConstraint(
                    qualifier_set=[
                        Qualifier(
                            qualifier_type_id=Biolink("qualified_predicate"),
                            qualifier_value=Biolink("causes"),
                        ),
                    ]
                )
            ],
        ),
    },
)


EXPANDED_QUALIFIER_QGRAPH = QueryGraph(
    nodes={
        OBJECT_NODE: QNode(
            ids=["HGNC:3870"],
            categories=[Biolink("Gene")],
        ),
        SUBJECT_NODE: QNode(
            ids=["CHEBI:59173"],
            categories=[Biolink("ChemicalEntity")],
        ),
    },
    edges={
        E0: QEdge(
            subject=SUBJECT_NODE,
            object=OBJECT_NODE,
            predicates=[Biolink("affects")],
            qualifier_constraints=[
                QualifierConstraint(
                    qualifier_set=[
                        Qualifier(
                            qualifier_type_id=Biolink("qualified_predicate"),
                            qualifier_value=Biolink("causes"),
                        ),
                        Qualifier(
                            qualifier_type_id=Biolink("object_aspect_qualifier"),
                            qualifier_value="activity_or_abundance",
                        ),
                        Qualifier(
                            qualifier_type_id=Biolink("object_direction_qualifier"),
                            qualifier_value="decreased",
                        ),
                    ]
                )
            ],
        ),
    },
)


HYDRATION_QGRAPH = QueryGraph(
    nodes={
        OBJECT_NODE: QNode(
            categories=[Biolink("Gene"), Biolink("Protein")],
            ids=["NCBIGene:4314"],
        ),
        SUBJECT_NODE: QNode(
            categories=[Biolink("ChemicalEntity")],
            ids=["CHEBI:48927"],
        ),
    },
    edges={
        QEdgeID("e00"): QEdge(
            object=OBJECT_NODE,
            predicates=[Biolink("affects")],
            subject=SUBJECT_NODE,
        )
    },
)


def generate_qgraph_with_attribute_constraints(constraints: list[AttributeConstraint]):
    """Generate a QGraph with attribute constraints."""
    _q_graph = copy.deepcopy(ALT_BASE_GRAPH)
    _q_graph.edges[E0].attribute_constraints = constraints

    return _q_graph


VALID_REGEX_QGRAPHS: list[QueryGraph] = [
    generate_qgraph_with_attribute_constraints([constraint])
    for constraint in VALID_REGEX_CONSTRAINTS
]

INVALID_REGEX_QGRAPHS: list[QueryGraph] = [
    generate_qgraph_with_attribute_constraints([constraint])
    for constraint in INVALID_REGEX_CONSTRAINTS
]
