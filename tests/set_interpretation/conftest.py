"""
Collection of fixtures for testing the set_interpretation handling
"""

import dataclasses
import uuid
from typing import Any, cast

import pytest
from translator_tom.v1_6 import (
    CURIE,
    Biolink,
    QEdgeID,
    QNodeID,
    Query,
)
from translator_tom.v1_6.model_dicts import (
    MessageDict,
    QEdgeDict,
    QNodeDict,
    QueryDict,
    QueryGraphDict,
    ResultDict,
)


@dataclasses.dataclass
class MockQuery:
    query: QueryGraphDict
    prefilter_results: list[ResultDict]
    postfilter_results: list[ResultDict]


def _validated_qgraph(query: QueryDict) -> QueryGraphDict:
    """Validate the query and return its normalized query graph as a dict."""
    query_graph = Query.model_validate(query).message.query_graph
    assert query_graph is not None
    dumped: Any = query_graph.model_dump()
    return dumped


# --- BATCH SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_batch_query() -> MockQuery:
    query = QueryDict(
        submitter="setinterp-automated-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=[CURIE("NCBIGene:3778")],
                        categories=[Biolink.Entity("biolink:Gene")],
                        set_interpretation="BATCH",
                        constraints=[],
                        member_ids=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=None,
                        categories=[Biolink.Entity("biolink:Disease")],
                        set_interpretation="BATCH",
                        constraints=[],
                        member_ids=[],
                    ),
                },
                edges={
                    QEdgeID("e01"): QEdgeDict(
                        subject=QNodeID("n0"),
                        object=QNodeID("n1"),
                        predicates=[Biolink.Predicate("biolink:causes")],
                        attribute_constraints=[],
                        qualifier_constraints=[],
                    ),
                },
            )
        ),
    )
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
    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", results),
        cast("list[ResultDict]", results),
    )


# --- MIXED SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_mixed_query0() -> MockQuery:
    """Represents a fully connected set of results.

    Node n0 has two identifiers   | set interpretation: BATCH
    Node n1 has three identifiers | set interpretation: ALL

    We expect 6 results returned prior to filtering
    We expect 2 results returned post filtering
    """
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )


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
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )


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
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )


@pytest.fixture(scope="session")
def mock_mixed_query3() -> MockQuery:
    """Represents a fully connected set of results.

    An inversion of mock_mixed_query0

    Node n0 has three identifiers | set interpretation: ALL
    Node n1 has two identifiers   | set interpretation: BATCH

    We expect 6 results returned prior to filtering
    We expect 2 results returned post filtering
    """
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )


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
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )


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
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )


# --- MALFORMED SET INTERPRETATION QUERIES ---


@pytest.fixture(scope="session")
def mock_malformed_query() -> MockQuery:
    """Represents a fully connected set of results.

    This query is manipulated at runtime for tests where
    we wish to verify how we handle malformed queries
    """
    query = QueryDict(
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
                        predicates=[Biolink.Predicate("biolink:subclass_of")],
                        attribute_constraints=[],
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

    return MockQuery(
        _validated_qgraph(query),
        cast("list[ResultDict]", prefilter_results),
        cast("list[ResultDict]", postfilter_results),
    )
