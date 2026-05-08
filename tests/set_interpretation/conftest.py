"""
Collection of fixtures for testing the set_interpretation handling
"""

import dataclasses
import uuid

import pytest
from translator_tom import (
    Analysis,
    Biolink,
    EdgeBinding,
    Message,
    NodeBinding,
    QEdge,
    QEdgeID,
    QNode,
    QNodeID,
    QueryGraph,
    Result,
    infores,
)

from retriever.types.trapi_overrides import Parameters, Query


@dataclasses.dataclass
class MockQuery:
    query: QueryGraph
    prefilter_results: list[Result]
    postfilter_results: list[Result]


# --- BATCH SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_batch_query() -> MockQuery:
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=["NCBIGene:3778"],
                        categories=[Biolink("Gene")],
                        set_interpretation="BATCH",
                        constraints=[],
                        member_ids=[],
                    ),
                    QNodeID("n1"): QNode(
                        ids=None,
                        categories=[Biolink("Disease")],
                        set_interpretation="BATCH",
                        constraints=[],
                        member_ids=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("causes")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )
    query = Query.model_validate(query)

    results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0012276", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="NCBIGene:3778", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e01"): [EdgeBinding(id="ce645c286b2f", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0060551", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="NCBIGene:3778", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e01"): [EdgeBinding(id="b882c438207a", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0032886", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="NCBIGene:3778", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e01"): [EdgeBinding(id="3a3380222bd5", attributes=[])]
                    },
                )
            ],
        ),
    ]
    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, results, results)


# --- MIXED SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_mixed_query0() -> MockQuery:
    """Represents a fully connected set of results.

    Node n0 has two identifiers   | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: ALL

    We expect 6 results returned prior to filtering
    We expect 2 results returned post filtering
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=["MONDO:0008903", "MONDO:0000001"],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)


@pytest.fixture(scope="session")
def mock_mixed_query1() -> MockQuery:
    """Represents a partially connected set of results.

    Node n0 has three identifiers | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: ALL

    We expect 8 results returned prior to filtering
    We expect 2 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1 and thus requires pruning
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=[
                            "MONDO:0008903",
                            "MONDO:0000001",
                            "MONDO:0004993",
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0004993", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0004993", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)


@pytest.fixture(scope="session")
def mock_mixed_query2() -> MockQuery:
    """Represents a partially connected set of results.

    Node n0 has three identifiers | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: MANY

    We expect 8 results returned prior to filtering
    We expect 4 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1, but will not be pruned
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=[
                            "MONDO:0008903",
                            "MONDO:0000001",
                            "MONDO:0004993",
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="MANY",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0004993", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0004993", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0004993", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0004993", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)


@pytest.fixture(scope="session")
def mock_mixed_query3() -> MockQuery:
    """Represents a fully connected set of results.

    An inversion of mock_mixed_query0

    Node n0 has three identifiers | set interpretation: ALL
    Node n1 has two identifiers   | set interpretation: BATCH

    We expect 6 results returned prior to filtering
    We expect 2 results returned post filtering
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                    QNodeID("n1"): QNode(
                        ids=["MONDO:0008903", "MONDO:0000001"],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0020644", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0020644", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n0"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)


@pytest.fixture(scope="session")
def mock_mixed_query4() -> MockQuery:
    """Represents a partially connected set of results.

    An inversion of mock_mixed_query1

    Node n0 has three identifiers | set interpretation: ALL
    Node n1 has three identifiers | set interpretation: BATCH

    We expect 8 results returned prior to filtering
    We expect 2 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1 and thus requires pruning
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                    QNodeID("n1"): QNode(
                        ids=[
                            "MONDO:0008903",
                            "MONDO:0000001",
                            "MONDO:0004993",
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0004993", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0004993", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0020644", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n0"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)


@pytest.fixture(scope="session")
def mock_mixed_query5() -> MockQuery:
    """Represents a partially connected set of results.

    An inversion of mock_mixed_query2

    Node n0 has three identifiers | set interpretation: MANY
    Node n1 has three identifiers | set interpretation: BATCH

    We expect 8 results returned prior to filtering
    We expect 4 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1, but will not be pruned
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="MANY",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                    QNodeID("n1"): QNode(
                        ids=[
                            "MONDO:0008903",
                            "MONDO:0000001",
                            "MONDO:0004993",
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0020644", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0004993", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0004993", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0020644", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0004993", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c378398684b2", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0004993", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000532", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="f75710790a99", attributes=[])]
                    },
                ),
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
                QNodeID("n1"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
                QNodeID("n1"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)


# --- MALFORMED SET INTERPRETATION QUERIES ---


@pytest.fixture
def mock_malformed_query() -> MockQuery:
    """Represents a fully connected set of results.

    This query is manipulated at runtime for tests where
    we wish to verify how we handle malformed queries.
    Function-scoped so each test gets a fresh copy and mutations don't leak.
    """
    query = Query(
        parameters=Parameters(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=Message(
            query_graph=QueryGraph(
                nodes={
                    QNodeID("n0"): QNode(
                        ids=["MONDO:0008903", "MONDO:0000001"],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNode(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            "MONDO:0000532",
                            "UMLS:C2983716",
                            "MONDO:0020644",
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdge(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink("subclass_of")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7f730935b4f8", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7c2a3a2bb437", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="7b6641969611", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0020644", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="0b6c704dfd94", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="UMLS:C2983716", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="c8255a314650", attributes=[])]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n1"): [NodeBinding(id="MONDO:0000532", attributes=[])],
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [EdgeBinding(id="6b2e82827546", attributes=[])]
                    },
                )
            ],
        ),
    ]

    postfilter_results = [
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0008903", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="6b2e82827546", attributes=[]),
                            EdgeBinding(id="0b6c704dfd94", attributes=[]),
                            EdgeBinding(id="c8255a314650", attributes=[]),
                        ]
                    },
                )
            ],
        ),
        Result(
            node_bindings={
                QNodeID("n0"): [NodeBinding(id="MONDO:0000001", attributes=[])],
                QNodeID("n1"): [
                    NodeBinding(
                        id="7c40623f-9da9-5aeb-985d-0d7428dd76ae", attributes=[]
                    )
                ],
            },
            analyses=[
                Analysis(
                    resource_id=infores("retriever"),
                    edge_bindings={
                        QEdgeID("e0"): [
                            EdgeBinding(id="7f730935b4f8", attributes=[]),
                            EdgeBinding(id="7c2a3a2bb437", attributes=[]),
                            EdgeBinding(id="7b6641969611", attributes=[]),
                        ]
                    },
                )
            ],
        ),
    ]

    query = Query.model_validate(query)

    query_graph = query.message.query_graph
    assert isinstance(query_graph, QueryGraph)
    return MockQuery(query_graph, prefilter_results, postfilter_results)
