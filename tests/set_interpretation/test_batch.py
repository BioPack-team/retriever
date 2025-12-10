"""
Effectively a standard query as the BATCH mode is the default operation
for set_interpretation. We just wish to ensure that things work as expected
here
"""

import random

import pytest

from retriever.utils.trapi import evaluate_set_interpretation
from retriever.utils.logs import TRAPILogger


@pytest.mark.asyncio
async def test_set_interpretation_batch_handling(
    set_interpretation_batch_query: dict, mock_query_result: dict
):
    """Tests default set_interpretation BATCH value. Should be a non-opt"""
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    response = evaluate_set_interpretation(
        qgraph=set_interpretation_batch_query,
        results=mock_query_result,
        kgraph=None,
        aux_graphs=None,
        job_log=job_log,
    )
    assert response == mock_query_result


@pytest.mark.asyncio
async def test_multiple_specified_set_interpretation_values(
    set_interpretation_batch_query: dict, mock_query_result: dict
):
    """Test edge case where user supplied multiple set_interpretation values."""

    # Ensure that the nodes have two different values for set_interpretation
    query_nodes = set_interpretation_batch_query.message.query_graph.nodes
    query_nodes.root["n0"].set_interpretation = "BATCH"
    query_nodes.root["n1"].set_interpretation = "ALL"
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    response = evaluate_set_interpretation(
        qgraph=set_interpretation_batch_query,
        results=mock_query_result,
        kgraph=None,
        aux_graphs=None,
        job_log=job_log,
    )
    assert response == mock_query_result
