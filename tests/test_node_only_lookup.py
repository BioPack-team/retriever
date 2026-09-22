"""Node-only (edge-less) query support.

Covers the relaxed validation rule and QGX's direct node-fetch path, which
bypasses branch traversal and pruning to return node_bindings-only results.
"""

import asyncio
import contextlib

import pytest
from fastapi.datastructures import Headers
from translator_tom.v2_0 import QueryGraph

from retriever.lookup.qgx import QueryGraphExecutor
from retriever.lookup.validate import validate
from retriever.types.general import QueryInfo
from retriever.types.trapi import Query


def _query_info(tier: int = 1) -> QueryInfo:
    return QueryInfo(
        endpoint="/query",
        method="POST",
        headers=Headers({}),
        body=None,
        job_id="node-only-test",
        tier=tier,
        timeout=-1,
    )


def _node_only_query(node: dict) -> Query:
    return Query.model_validate(
        {"message": {"query_graph": {"nodes": {"n0": node}}}}
    )


def test_node_only_query_passes_validation():
    """An edge-less query with an ID'd node is valid (no >=1-edge requirement)."""
    query = _node_only_query({"ids": ["MONDO:1"], "categories": ["biolink:Disease"]})

    _warnings, problems = validate(query)

    assert problems == []


def test_edgeless_query_without_ids_still_rejected():
    """An edge-less query with no ID'd node fails validation."""
    query = _node_only_query({"categories": ["biolink:Disease"]})

    _warnings, problems = validate(query)

    assert any("ID" in problem for problem in problems)


@pytest.mark.asyncio
async def test_node_only_execute_binds_nodes_without_analyses(
    monkeypatch: pytest.MonkeyPatch,
):
    """Node-only execution yields one node_bindings-only Result per hydrated id."""
    qgraph = QueryGraph.model_validate(
        {
            "nodes": {
                "n0": {"ids": ["MONDO:1", "MONDO:2"], "categories": ["biolink:Disease"]}
            },
        }
    )
    qgx = QueryGraphExecutor(qgraph, _query_info())

    async def fake_hydrate() -> None:
        for curie in qgx.kgraph["nodes"]:
            qgx.kgraph["nodes"][curie]["categories"] = ["biolink:Disease"]

    monkeypatch.setattr(qgx, "hydrate_missing_nodes", fake_hydrate)

    task = asyncio.create_task(asyncio.sleep(60))
    try:
        artifacts = await qgx._execute_node_only(task)
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert artifacts.status == "Success"
    assert len(artifacts.results) == 2
    for result in artifacts.results:
        assert set(result["node_bindings"]) == {"n0"}
        assert "analyses" not in result
    assert set(artifacts.kgraph["nodes"]) == {"MONDO:1", "MONDO:2"}


@pytest.mark.asyncio
async def test_node_only_drops_unhydrated_nodes(monkeypatch: pytest.MonkeyPatch):
    """A node with no canonical match is dropped: no skeletal KG node, no result."""
    qgraph = QueryGraph.model_validate(
        {
            "nodes": {
                "n0": {"ids": ["MONDO:1", "MONDO:404"], "categories": ["biolink:Disease"]}
            },
        }
    )
    qgx = QueryGraphExecutor(qgraph, _query_info())

    async def fake_hydrate() -> None:
        # Only MONDO:1 resolves; MONDO:404 stays skeletal (empty categories).
        qgx.kgraph["nodes"]["MONDO:1"]["categories"] = ["biolink:Disease"]

    monkeypatch.setattr(qgx, "hydrate_missing_nodes", fake_hydrate)

    task = asyncio.create_task(asyncio.sleep(60))
    try:
        artifacts = await qgx._execute_node_only(task)
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert set(artifacts.kgraph["nodes"]) == {"MONDO:1"}
    assert len(artifacts.results) == 1
    assert artifacts.results[0]["node_bindings"]["n0"]["ids"] == ["MONDO:1"]
