import importlib
import json
import re
import time
from collections.abc import Iterator
from typing import Any, cast, final, override
from textwrap import dedent
from unittest.mock import MagicMock, patch, AsyncMock

import asyncio
import aiohttp
import grpc
import pytest

import retriever.config.general as general_mod
import retriever.data_tiers.tier_0.dgraph.driver as driver_mod
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models
from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.types.trapi import QueryGraphDict


# Test-only subclass exposing the client property
class _TestDgraphHttpDriver(driver_mod.DgraphHttpDriver):
    def __init__(self, *, version: str | None = None) -> None:
        """Initialize and pass version to the parent class."""
        super().__init__(version=version)

    @property
    def http_session(self) -> aiohttp.ClientSession | None:
        return self._http_session


# Test-only subclass exposing the client property
class _TestDgraphGrpcDriver(driver_mod.DgraphGrpcDriver):
    def __init__(self, *, version: str | None = None) -> None:
        """Initialize and pass version to the parent class."""
        super().__init__(version=version)

    _client: driver_mod.DgraphClientProtocol | None = None

    @property
    def client(self) -> driver_mod.DgraphClientProtocol | None:
        return self._client

    @client.setter
    def client(self, value: driver_mod.DgraphClientProtocol | None) -> None:
        """Allow setting the client for testing purposes."""
        self._client = value


def new_http_driver(version: str | None = None) -> _TestDgraphHttpDriver:
    return _TestDgraphHttpDriver(version=version)


def new_grpc_driver(version: str | None = None) -> _TestDgraphGrpcDriver:
    return _TestDgraphGrpcDriver(version=version)


# Test-only subclass exposing the client property
class _TestDgraphTranspiler(DgraphTranspiler):
    """Expose protected methods for testing without modifying production code."""
    def convert_multihop_public(self, qgraph: QueryGraphDict) -> str:
        return self.convert_multihop(qgraph)

    def convert_batch_multihop_public(self, qgraphs: list[QueryGraphDict]) -> str:
        return self.convert_batch_multihop(qgraphs)


def qg(d: dict[str, Any]) -> QueryGraphDict:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(QueryGraphDict, cast(object, d))


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def assert_query_equals(actual: str, expected: str) -> None:
    assert normalize(actual) == normalize(expected)


@pytest.fixture
def mock_dgraph_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TIER0__DGRAPH__HOST", "localhost")
    monkeypatch.setenv("TIER0__DGRAPH__HTTP_PORT", "8080")
    monkeypatch.setenv("TIER0__DGRAPH__GRPC_PORT", "9080")
    monkeypatch.setenv("TIER0__DGRAPH__PREFERRED_VERSION", "vSubClass")
    monkeypatch.setenv("TIER0__DGRAPH__USE_TLS", "false")
    monkeypatch.setenv("TIER0__DGRAPH__QUERY_TIMEOUT", "10")
    monkeypatch.setenv("TIER0__DGRAPH__CONNECT_RETRIES", "0")

    # Rebuild CONFIG from env and reload driver so classes bind to the new CONFIG
    importlib.reload(general_mod)
    importlib.reload(driver_mod)
    # Ensure the driver module uses the same CONFIG instance
    monkeypatch.setattr(driver_mod, "CONFIG", general_mod.CONFIG, raising=False)
    yield


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclassing_case0_no_expansion_live() -> None:
    """
    Live test: Case 0 (predicate is 'subclass_of').
    The transpiler should NOT generate any subclassing expansion forms.
    """
    driver = new_http_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()
    transpiler = _TestDgraphTranspiler(version=dgraph_schema_version, subclassing_enabled=True)

    qgraph_query: QueryGraphDict = qg({
        "nodes": {"n0": {"ids": ["TEST:A_case0"]}, "n1": {"ids": ["TEST:B_case0"]}},
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["subclass_of"]}},
    })

    dgraph_query = transpiler.convert_multihop_public(qgraph_query)

    # Assert that no subclassing forms were generated
    assert "subclassB" not in dgraph_query
    assert "subclassC" not in dgraph_query
    assert "subclassD" not in dgraph_query

    result = await driver.run_query(dgraph_query, transpiler=transpiler)
    assert "q0" in result.data and result.data["q0"], "Query should find the direct path"
    assert len(result.data["q0"]) == 1
    root_node = result.data["q0"][0]
    assert root_node.id == "TEST:A_case0"
    assert len(root_node.edges) == 1
    assert root_node.edges[0].node.id == "TEST:B_case0"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclassing_case1_form_b_live() -> None:
    """Live test: Case 1, Form B (A' -> subclass_of -> A; A' -> R -> B)."""
    driver = new_http_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()
    transpiler = _TestDgraphTranspiler(version=dgraph_schema_version, subclassing_enabled=True)

    qgraph_query: QueryGraphDict = qg({
        "nodes": {"n0": {"ids": ["TEST:A_case1"]}, "n1": {"ids": ["TEST:B_case1"]}},
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["related_to"]}},
    })

    dgraph_query = transpiler.convert_multihop_public(qgraph_query)
    result = await driver.run_query(dgraph_query, transpiler=transpiler)

    assert "q0" in result.data and result.data["q0"], "Query should find the Form B path"
    assert len(result.data["q0"]) == 1
    print("### result.data =", result.data)
    root_node = result.data["q0"][0]
    assert root_node.id == "TEST:A_case1"

    # Find the specific edge from the Form B expansion by looking for the
    # intermediate node's unique binding, set in the transpiler.
    form_b_edge = next(
        (
            edge
            for edge in root_node.edges
            if edge.node.binding == "intermediate"
        ),
        None,
    )
    assert form_b_edge is not None, "Form B path not found in results"
    assert form_b_edge.node.edges, "Intermediate node for Form B should have an outgoing edge"
    assert len(form_b_edge.node.edges) == 1, "Form B path should have one final edge"

    # Now, assert the final node ID by traversing the nested structure
    final_node = form_b_edge.node.edges[0].node
    assert final_node.id == "TEST:B_case1"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclassing_case1_form_c_live() -> None:
    """Live test: Case 1, Form C (A -> R -> B'; B' -> subclass_of -> B)."""
    driver = new_http_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()
    transpiler = _TestDgraphTranspiler(version=dgraph_schema_version, subclassing_enabled=True)

    qgraph_query: QueryGraphDict = qg({
        "nodes": {"n0": {"ids": ["TEST:A_case1c"]}, "n1": {"ids": ["TEST:B_case1c"]}},
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["related_to"]}},
    })

    dgraph_query = transpiler.convert_multihop_public(qgraph_query)
    result = await driver.run_query(dgraph_query, transpiler=transpiler)

    assert "q0" in result.data and result.data["q0"], "Query should find a result"
    assert len(result.data["q0"]) == 1
    root_node = result.data["q0"][0]
    assert root_node.id == "TEST:A_case1c"

    # Find the specific edge from the Form C expansion by looking for the
    # intermediate node's unique binding.
    form_c_edge = next(
        (
            edge
            for edge in root_node.edges
            if edge.node.binding == "intermediate"
        ),
        None,
    )
    assert form_c_edge is not None, "Form C path not found in results"
    assert form_c_edge.node.edges, "Intermediate node for Form C should have an outgoing edge"
    assert len(form_c_edge.node.edges) == 1, "Form C path should have one final edge"

    # Traverse the nested structure to get the final node
    final_node = form_c_edge.node.edges[0].node
    assert final_node.id == "TEST:B_case1c"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclassing_case1_form_d_live() -> None:
    """Live test: Case 1, Form D (A' -> sub -> A; A' -> R -> B'; B' -> sub -> B)."""
    driver = new_http_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()
    transpiler = _TestDgraphTranspiler(version=dgraph_schema_version, subclassing_enabled=True)

    qgraph_query: QueryGraphDict = qg({
        "nodes": {"n0": {"ids": ["TEST:A_case1d"]}, "n1": {"ids": ["TEST:B_case1d"]}},
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["related_to"]}},
    })

    dgraph_query = transpiler.convert_multihop_public(qgraph_query)
    result = await driver.run_query(dgraph_query, transpiler=transpiler)

    assert "q0" in result.data and result.data["q0"], "Query should find a result"
    assert len(result.data["q0"]) == 1
    root_node = result.data["q0"][0]
    assert root_node.id == "TEST:A_case1d"

    # Find the specific edge from the Form D expansion by looking for the
    # first intermediate node's unique binding.
    form_d_edge = next(
        (
            edge
            for edge in root_node.edges
            if edge.node.binding == "intermediate_A"
        ),
        None,
    )
    assert form_d_edge is not None, "Form D path not found in results"
    assert form_d_edge.node.edges, "First intermediate node for Form D should have an outgoing edge"
    
    # Traverse to the second intermediate node
    second_intermediate_edge = form_d_edge.node.edges[0]
    assert second_intermediate_edge.node.binding == "intermediate_B"
    assert second_intermediate_edge.node.edges, "Second intermediate node for Form D should have an outgoing edge"

    # Traverse to the final node
    final_node = second_intermediate_edge.node.edges[0].node
    assert final_node.id == "TEST:B_case1d"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclassing_case2_id_to_cat_live() -> None:
    """Live test: Case 2 (ID -> R -> CAT-only) should find a path via Form B."""
    driver = new_http_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()
    transpiler = _TestDgraphTranspiler(version=dgraph_schema_version, subclassing_enabled=True)

    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0": {"ids": ["TEST:A_case2"]},
            "n1": {"categories": ["TestCategory"]},
        },
        "edges": {"e0": {"subject": "n0", "object": "n1", "predicates": ["related_to"]}},
    })

    dgraph_query = transpiler.convert_multihop_public(qgraph_query)

    # Verify only Form B is generated for this case
    assert "subclassB" in dgraph_query
    assert "subclassC" not in dgraph_query
    assert "subclassD" not in dgraph_query

    result = await driver.run_query(dgraph_query, transpiler=transpiler)

    assert "q0" in result.data and result.data["q0"], "Query should find the Case 2 path"
    assert len(result.data["q0"]) == 1
    root_node = result.data["q0"][0]
    assert root_node.id == "TEST:A_case2"

    # Find the specific edge from the Form B expansion
    form_b_edge = next(
        (
            edge
            for edge in root_node.edges
            if edge.node.binding == "intermediate"
        ),
        None,
    )
    assert form_b_edge is not None, "Form B path for Case 2 not found"
    assert form_b_edge.node.edges, "Intermediate node for Case 2 should have an outgoing edge"

    # The result should be the specific node found, not just the category
    final_node = form_b_edge.node.edges[0].node
    assert final_node.id == "TEST:B_case2"
    assert "TestCategory" in final_node.category

    await driver.close()
