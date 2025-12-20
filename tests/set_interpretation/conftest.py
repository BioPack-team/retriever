"""
Collection of fixtures for testing the set_interpretation handling
"""

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


@pytest.fixture(scope="session")
def set_interpretation_batch_query() -> dict:
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-batch-testing",
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
    return query


@pytest.fixture(scope="session")
def set_interpretation_all_query() -> dict:
    query = QueryDict(
        parameters=ParametersDict(tiers=[0, 1]),
        submitter="setinterp-automated-batch-testing",
        message=MessageDict(
            query_graph=QueryGraphDict(
                nodes={
                    QNodeID("n0"): QNodeDict(
                        ids=["uuid:90040029-d631-58fe-bf60-df7a8555c6db"],
                        categories=[BiolinkEntity("biolink:Gene")],
                        set_interpretation="ALL",
                        member_ids=["MONDO:0060551", "NCBIGene:3778"],
                        constraints=[],
                    ),
                    QNodeID("n1"): QNodeDict(
                        ids=["uuid:90040029-d631-58fe-bf60-df7a8555c6db"],
                        categories=[BiolinkEntity("biolink:Disease")],
                        set_interpretation="ALL",
                        member_ids=["MONDO:0060551", "NCBIGene:3778"],
                        constraints=[],
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
    return query


@pytest.fixture(scope="session")
def mock_query_result():
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
    return results
