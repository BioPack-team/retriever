"""
Collection of fixtures for testing the set_interpretation handling
"""

import dataclasses
import uuid

import pytest

from retriever.types.trapi import (
    CURIE,
    BiolinkEntity,
    BiolinkPredicate,
    MessageDict,
    ParametersDict,
    QEdgeDict,
    QEdgeID,
    QNodeDict,
    QNodeID,
    QueryDict,
    QueryGraphDict,
)
from reasoner_pydantic import Query


@dataclasses.dataclass
class MockQuery:
    query: dict
    prefilter_results: dict
    postfilter_results: dict


# --- BATCH SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_batch_query() -> MockQuery:
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[CURIE("NCBIGene:3778")],
                        categories=[BiolinkEntity("biolink:Gene")],
                        set_interpretation="BATCH",
                        constraints=[],
                        member_ids=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=None,
                        categories=[BiolinkEntity("biolink:Disease")],
                        set_interpretation="BATCH",
                        constraints=[],
                        member_ids=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:causes")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )
    query = Query.model_validate(query)

    results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0012276", "attributes": []}],
                "n0": [{"id": "NCBIGene:3778", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e01": [{"id": "ce645c286b2f", "attributes": []}]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0060551", "attributes": []}],
                "n0": [{"id": "NCBIGene:3778", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e01": [{"id": "b882c438207a", "attributes": []}]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0032886", "attributes": []}],
                "n0": [{"id": "NCBIGene:3778", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e01": [{"id": "3a3380222bd5", "attributes": []}]
                    },
                }
            ],
        },
    ]
    return MockQuery(query.message.query_graph.model_dump(), results, results)


# --- MIXED SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_mixed_query0() -> dict:
    """Represents a fully connected set of results.

    Node n0 has two identifiers   | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: ALL

    We expect 6 results returned prior to filtering
    We expect 2 results returned post filtering
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[CURIE("MONDO:0008903"), CURIE("MONDO:0000001")],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )


@pytest.fixture(scope="session")
def mock_mixed_query1() -> dict:
    """Represents a partially connected set of results.

    Node n0 has three identifiers | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: ALL

    We expect 8 results returned prior to filtering
    We expect 2 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1 and thus requires pruning
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[
                            CURIE("MONDO:0008903"),
                            CURIE("MONDO:0000001"),
                            CURIE("MONDO:0004993"),
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0004993", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0004993", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )


@pytest.fixture(scope="session")
def mock_mixed_query2() -> dict:
    """Represents a partially connected set of results.

    Node n0 has three identifiers | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: MANY

    We expect 8 results returned prior to filtering
    We expect 4 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1, but will not be pruned
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[
                            CURIE("MONDO:0008903"),
                            CURIE("MONDO:0000001"),
                            CURIE("MONDO:0004993"),
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="MANY",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0004993", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0004993", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0004993", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0004993", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )


@pytest.fixture(scope="session")
def mock_mixed_query3() -> dict:
    """Represents a fully connected set of results.

    An inversion of mock_mixed_query0

    Node n0 has three identifiers | set interpretation: ALL
    Node n1 has two identifiers   | set interpretation: BATCH

    We expect 6 results returned prior to filtering
    We expect 2 results returned post filtering
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[CURIE("MONDO:0008903"), CURIE("MONDO:0000001")],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "MONDO:0020644", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "MONDO:0020644", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n0": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )


@pytest.fixture(scope="session")
def mock_mixed_query4() -> dict:
    """Represents a partially connected set of results.

    An inversion of mock_mixed_query1

    Node n0 has three identifiers | set interpretation: ALL
    Node n1 has three identifiers | set interpretation: BATCH

    We expect 8 results returned prior to filtering
    We expect 2 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1 and thus requires pruning
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[
                            CURIE("MONDO:0008903"),
                            CURIE("MONDO:0000001"),
                            CURIE("MONDO:0004993"),
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0004993", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0004993", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "MONDO:0020644", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n0": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )


@pytest.fixture(scope="session")
def mock_mixed_query5() -> dict:
    """Represents a partially connected set of results.

    An inversion of mock_mixed_query2

    Node n0 has three identifiers | set interpretation: MANY
    Node n1 has three identifiers | set interpretation: BATCH

    We expect 8 results returned prior to filtering
    We expect 4 results returned post filtering

    The third identifier in node n0 should not fully connect
    to node n1, but will not be pruned
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="MANY",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[
                            CURIE("MONDO:0008903"),
                            CURIE("MONDO:0000001"),
                            CURIE("MONDO:0004993"),
                        ],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "MONDO:0020644", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0004993", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0004993", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "MONDO:0020644", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0004993", "attributes": []}],
                "n0": [{"id": "UMLS:C2983716", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c378398684b2", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0004993", "attributes": []}],
                "n0": [{"id": "MONDO:0000532", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "f75710790a99", "attributes": []}]},
                },
            ],
        },
        {
            "node_bindings": {
                "n0": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
                "n1": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
                "n1": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )


# --- MALFORMED SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_malformed_query() -> dict:
    """Represents a fully connected set of results.

    This query is manipulated at runtime for tests where
    we wish to verify how we handle malformed queries
    """
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[CURIE("MONDO:0008903"), CURIE("MONDO:0000001")],
                        set_interpretation="BATCH",
                        constraints=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=[
                            str(uuid.UUID("uuid:7c40623f-9da9-5aeb-985d-0d7428dd76ae"))
                        ],
                        set_interpretation="ALL",
                        constraints=[],
                        member_ids=[
                            CURIE("MONDO:0000532"),
                            CURIE("UMLS:C2983716"),
                            CURIE("MONDO:0020644"),
                        ],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[BiolinkPredicate("biolink:subclass_of")],
                        attribute_contraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )

    prefilter_results = [
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7f730935b4f8", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7c2a3a2bb437", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "7b6641969611", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0020644", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "0b6c704dfd94", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "UMLS:C2983716", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "c8255a314650", "attributes": []}]},
                }
            ],
        },
        {
            "node_bindings": {
                "n1": [{"id": "MONDO:0000532", "attributes": []}],
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {"e0": [{"id": "6b2e82827546", "attributes": []}]},
                }
            ],
        },
    ]

    postfilter_results = [
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0008903", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "6b2e82827546", "attributes": []},
                            {"id": "0b6c704dfd94", "attributes": []},
                            {"id": "c8255a314650", "attributes": []},
                        ]
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "MONDO:0000001", "attributes": []}],
                "n1": [
                    {"id": ["7c40623f-9da9-5aeb-985d-0d7428dd76ae"], "attributes": []}
                ],
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        "e0": [
                            {"id": "7f730935b4f8", "attributes": []},
                            {"id": "7c2a3a2bb437", "attributes": []},
                            {"id": "7b6641969611", "attributes": []},
                        ]
                    },
                }
            ],
        },
    ]

    query = Query.model_validate(query)

    return MockQuery(
        query.message.query_graph.model_dump(), prefilter_results, postfilter_results
    )
