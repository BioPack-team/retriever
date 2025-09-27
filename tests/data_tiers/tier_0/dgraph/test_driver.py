from typing import Any, cast, final, override

import pydgraph
import pytest
import aiohttp

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.tier_0.dgraph.driver import DgraphHttpDriver, DgraphGrpcDriver
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


# Test-only subclass exposing the client property
class _TestDgraphHttpDriver(DgraphHttpDriver):
    @property
    def http_session(self) -> aiohttp.ClientSession | None:
        return self._http_session


# Test-only subclass exposing the client property
class _TestDgraphGrpcDriver(DgraphGrpcDriver):
    @property
    def client(self) -> pydgraph.DgraphClient | None:
        return self._client


@pytest.mark.asyncio
async def test_dgraph_http_live_with_settings(monkeypatch: pytest.MonkeyPatch):
    settings = DgraphSettings(host="transltr.biothings.io", http_port=8080, grpc_port=9080, use_tls=False, query_timeout=3, connect_retries=0)
    monkeypatch.setattr(CONFIG.tier0, "dgraph", settings, raising=False)

    driver = _TestDgraphHttpDriver()

    try:
        await driver.connect()
        assert driver.http_session is not None

        result: dg_models.DgraphResponse = await driver.run_query(
            "{ node(func: has(id), first: 1) { id name category } }"
        )
        assert isinstance(result, dg_models.DgraphResponse)
        for nodes in result.data.values():
            if nodes:
                assert nodes[0].id
                break
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph HTTP test (cannot connect or query): {e}")
    finally:
        await driver.close()
        assert driver.http_session is None


@pytest.mark.asyncio
async def test_dgraph_grpc_live_with_settings(monkeypatch: pytest.MonkeyPatch):
    settings = DgraphSettings(host="transltr.biothings.io", http_port=8080, grpc_port=9080, use_tls=False, query_timeout=3, connect_retries=0)
    monkeypatch.setattr(CONFIG.tier0, "dgraph", settings, raising=False)

    driver = _TestDgraphGrpcDriver()

    try:
        await driver.connect()
        assert driver.client is not None
        result: dg_models.DgraphResponse = await driver.run_query(
            "{ node(func: has(id), first: 1) { id name category } }"
        )
        assert isinstance(result, dg_models.DgraphResponse)
        for nodes in result.data.values():
            if nodes:
                assert nodes[0].id
                break
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph gRPC test (cannot connect or query): {e}")
    finally:
        await driver.close()
        assert driver.client is None


@pytest.mark.asyncio
async def test_dgraph_mock():
    @final
    class MockDgraphDriver(_TestDgraphGrpcDriver):  # Use gRPC version for mock
        @override
        async def connect(self, retries: int = 0) -> None:
            self._client = cast(Any, object())

        @override
        async def run_query(self, query: str, *args: Any, **kwargs: Any) -> dg_models.DgraphResponse:
            # Return a parsed DgraphResponse consistent with the real driver
            return dg_models.parse_response(
                {"data": {"node": [{"id": "test_id", "name": "name", "category": "cat"}]}}
            )

        @override
        async def close(self) -> None:
            self._client = None

    driver = MockDgraphDriver()
    await driver.connect()
    assert driver.client is not None

    result = await driver.run_query("test query")
    assert isinstance(result, dg_models.DgraphResponse)
    # Arbitrary query key "node"
    nodes = result.data.get("node", [])
    assert nodes and nodes[0].id == "test_id"

    await driver.close()
    assert driver.client is None
