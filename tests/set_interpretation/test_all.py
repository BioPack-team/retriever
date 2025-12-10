"""
Handles cases associated with enabling set_interpretation : ALL
"""

import random

import pytest

from retriever.utils.trapi import evaluate_set_interpretation
from retriever.utils.logs import TRAPILogger


@pytest.mark.asyncio
async def test_set_interpretation_all_handling(
    set_interpretation_all_query: dict, mock_query_result: dict
):
    """Tests set_interpretation ALL value."""
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    response = evaluate_set_interpretation(
        qgraph=set_interpretation_all_query,
        results=mock_query_result,
        kgraph=None,
        aux_graphs=None,
        job_log=job_log,
    )
    assert response != mock_query_result


@pytest.mark.asyncio
async def test_multiple_specified_set_interpretation_values(
    set_interpretation_all_query: dict, mock_query_result: dict
):
    """Test edge case where user supplied multiple set_interpretation values."""

    # Ensure that the nodes have two different values for set_interpretation
    query_nodes = set_interpretation_all_query.message.query_graph.nodes
    query_nodes.root["n0"].set_interpretation = "BATCH"
    query_nodes.root["n1"].set_interpretation = "ALL"
    job_log = TRAPILogger(job_id=random.randint(0, 10000))

    response = evaluate_set_interpretation(
        qgraph=set_interpretation_all_query,
        results=mock_query_result,
        kgraph=None,
        aux_graphs=None,
        job_log=job_log,
    )
    assert response == mock_query_result
