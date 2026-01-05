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

    response = evaluate_set_interpretation(
        qgraph=mock_batch_query.query,
        results=mock_batch_query.prefilter_results,
        job_log=job_log,
    )
    assert response == mock_batch_query.postfilter_results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_query", ["mock_mixed_query0", "mock_mixed_query1", "mock_mixed_query2"]
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

    response = evaluate_set_interpretation(
        qgraph=mock_mixed_query.query,
        results=mock_mixed_query.prefilter_results,
        job_log=job_log,
    )
    assert len(response) == len(mock_mixed_query.postfilter_results)
