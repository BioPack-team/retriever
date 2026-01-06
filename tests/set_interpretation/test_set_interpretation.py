"""
Effectively a standard query as the BATCH mode is the default operation
for set_interpretation. We just wish to ensure that things work as expected
here
"""

import random

import pytest
import _pytest

from retriever.utils.trapi import evaluate_set_interpretation
from retriever.utils.logs import TRAPILogger

from .conftest import MockQuery


@pytest.mark.asyncio
@pytest.mark.parametrize("mock_query", ["mock_batch_query"])
async def test_set_interpretation_batch_handling(
    mock_query: str, request: _pytest.fixtures.TopRequest
):
    """Tests default set_interpretation BATCH value. Should be a no-opt"""
    mock_batch_query: MockQuery = request.getfixturevalue(mock_query)

    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    empirical_results = evaluate_set_interpretation(
        qgraph=mock_batch_query.query,
        results=mock_batch_query.prefilter_results,
        job_log=job_log,
    )
    assert empirical_results == mock_batch_query.postfilter_results

    for expected_result in mock_batch_query.postfilter_results:
        discovered_result_match = False
        for empirical_result in empirical_results:
            if (
                empirical_result["node_bindings"]["n0"][0]["id"]
                == expected_result["node_bindings"]["n0"][0]["id"]
                and empirical_result["node_bindings"]["n1"][0]["id"]
                == expected_result["node_bindings"]["n1"][0]["id"]
            ):
                assert empirical_result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_query",
    [
        "mock_mixed_query0",
        "mock_mixed_query1",
        "mock_mixed_query2",
        "mock_mixed_query3",
        "mock_mixed_query4",
        "mock_mixed_query5",
    ],
)
async def test_mixed_set_interpretation_values(
    mock_query: str, request: _pytest.fixtures.TopRequest
):
    """Test case(s) where user supplied multiple different set_interpretation values.

    Primarily used for evaluating the logic when set interpreation is set to either
    ALL or MANY
    """
    mock_mixed_query: MockQuery = request.getfixturevalue(mock_query)

    # Ensure that the nodes have two different values for set_interpretation
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    empirical_results = evaluate_set_interpretation(
        qgraph=mock_mixed_query.query,
        results=mock_mixed_query.prefilter_results,
        job_log=job_log,
    )
    assert len(empirical_results) == len(mock_mixed_query.postfilter_results)

    for expected_result in mock_mixed_query.postfilter_results:
        discovered_result_match = False
        for empirical_result in empirical_results:
            if (
                empirical_result["node_bindings"]["n0"][0]["id"]
                == expected_result["node_bindings"]["n0"][0]["id"]
                and empirical_result["node_bindings"]["n1"][0]["id"]
                == expected_result["node_bindings"]["n1"][0]["id"]
            ):
                assert empirical_result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_query",
    [
        "mock_malformed_query",
    ],
)
async def test_malformed_set_identifier(
    mock_query: str, request: _pytest.fixtures.TopRequest
):
    """Test case where the set identifier isn't a valid UUID."""
    mock_malformed_query: MockQuery = request.getfixturevalue(mock_query)

    # Ensure that the nodes have two different values for set_interpretation
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    # Set the ids field for node n1 to something other than a UUID
    mock_malformed_query.query["nodes"]["n1"]["ids"] = ["woodcarving"]

    empirical_results0 = evaluate_set_interpretation(
        qgraph=mock_malformed_query.query,
        results=mock_malformed_query.prefilter_results,
        job_log=job_log,
    )

    assert len(empirical_results0) != len(mock_malformed_query.postfilter_results)

    # Set the ids field for node n1 to an empty list
    mock_malformed_query.query["nodes"]["n1"]["ids"] = []

    empirical_results1 = evaluate_set_interpretation(
        qgraph=mock_malformed_query.query,
        results=mock_malformed_query.prefilter_results,
        job_log=job_log,
    )

    assert len(empirical_results1) != len(mock_malformed_query.postfilter_results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_query",
    [
        "mock_malformed_query",
    ],
)
async def test_malformed_set_interpretation_value(
    mock_query: str, request: _pytest.fixtures.TopRequest
):
    """Test case where the set_interpretation value isn't BATCH|ALL|MANY."""
    mock_malformed_query: MockQuery = request.getfixturevalue(mock_query)

    # Ensure that the nodes have two different values for set_interpretation
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    # Set the set_interpretation value to an invalid value
    mock_malformed_query.query["nodes"]["n0"]["set_interpretation"] = "SIGNAL"
    mock_malformed_query.query["nodes"]["n1"]["set_interpretation"] = "IRONCLAD"

    empirical_results = evaluate_set_interpretation(
        qgraph=mock_malformed_query.query,
        results=mock_malformed_query.prefilter_results,
        job_log=job_log,
    )

    assert len(empirical_results) != len(mock_malformed_query.postfilter_results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_query",
    [
        "mock_malformed_query",
    ],
)
async def test_malformed_member_identifiers(
    mock_query: str, request: _pytest.fixtures.TopRequest
):
    """Test case where the member_ids attribute is malformed."""
    mock_malformed_query: MockQuery = request.getfixturevalue(mock_query)

    # Ensure that the nodes have two different values for set_interpretation
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    # Set the member_ids value to a malformed entry
    mock_malformed_query.query["nodes"]["n1"]["set_interpretation"] = "ALL"
    mock_malformed_query.query["nodes"]["n1"]["member_ids"] = []

    empirical_results = evaluate_set_interpretation(
        qgraph=mock_malformed_query.query,
        results=mock_malformed_query.prefilter_results,
        job_log=job_log,
    )

    assert len(empirical_results) != len(mock_malformed_query.postfilter_results)

    # Set the member_ids value to a malformed entry
    mock_malformed_query.query["nodes"]["n1"]["set_interpretation"] = "MANY"
    mock_malformed_query.query["nodes"]["n1"]["member_ids"] = None

    empirical_results = evaluate_set_interpretation(
        qgraph=mock_malformed_query.query,
        results=mock_malformed_query.prefilter_results,
        job_log=job_log,
    )

    assert len(empirical_results) != len(mock_malformed_query.postfilter_results)
