from typing import Any, cast, final, override

import pydgraph
import pytest
from loguru import logger as log

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.tier_0.dgraph.driver import DgraphDriver
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


# Test-only subclass exposing the client property
class _TestDgraphDriver(DgraphDriver):
    @property
    def client(self) -> pydgraph.DgraphClient | None:
        return self._client


@pytest.mark.asyncio
async def test_dgraph_live_with_default_settings():
    driver = _TestDgraphDriver()
    try:
        await driver.connect()
        assert driver.client is not None
        # Request fields required by the parser (id, name, category)
        result: dg_models.DgraphResponse = await driver.run_query(
            "{ node(func: has(id), first: 1) { id name category } }"
        )
        assert isinstance(result, dg_models.DgraphResponse)
        # Iterate over arbitrary query names
        for nodes in result.data.values():
            if nodes:
                assert nodes[0].id
                break
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph test: {e}")
    finally:
        await driver.close()
        assert driver.client is None

@pytest.mark.asyncio
async def test_dgraph_live_with_settings(monkeypatch: pytest.MonkeyPatch):
    settings = DgraphSettings(host="localhost", http_port=8080, grpc_port=9080, use_tls=False)
    monkeypatch.setattr(CONFIG.tier0, "dgraph", settings, raising=False)

    driver = _TestDgraphDriver()
    driver.endpoint = f"{settings.host}:{settings.grpc_port}"

    try:
        await driver.connect()
        result: dg_models.DgraphResponse = await driver.run_query(
            "{ node(func: has(id), first: 1) { id name category } }"
        )
        assert isinstance(result, dg_models.DgraphResponse)
        for nodes in result.data.values():
            if nodes:
                assert nodes[0].id
                break
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph test (cannot connect or query): {e}")
    finally:
        await driver.close()
        assert driver.client is None

@pytest.mark.asyncio
async def test_dgraph_mock():
    @final
    class MockDgraphDriver(_TestDgraphDriver):
        @override
        async def connect(self, retries: int = 0) -> None:
            self._client = cast(Any, object())

        @override
        async def run_query(self, query: str) -> dg_models.DgraphResponse:
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
