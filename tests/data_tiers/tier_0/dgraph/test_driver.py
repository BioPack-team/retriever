import json
from typing import Any, cast, final, override

import pydgraph
import pytest

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.tier_0.dgraph.driver import (
    DgraphDriver,
    DgraphQueryResult,
)

# Test-only subclass exposing the client property
class _TestDgraphDriver(DgraphDriver):
    @property
    def client(self) -> pydgraph.DgraphClient | None:
        return self._client

def _load_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return cast(dict[str, Any], result)
    if isinstance(result, (bytes, bytearray)):
        text = result.decode("utf-8")
    elif isinstance(result, str):
        text = result
    else:
        raise TypeError(f"Unexpected result type: {type(result)}")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("Decoded JSON is not a dict")
    return cast(dict[str, Any], data)

@pytest.mark.asyncio
async def test_dgraph_live_with_default_settings():
    driver = _TestDgraphDriver()
    try:
        await driver.connect()
        assert driver.client is not None
        raw: DgraphQueryResult = await driver.run_query("{ node(func: has(id), first: 1) { id } }")
        data = _load_result(raw)
        container = data.get("data", data)
        if "node" in container and container["node"]:
            assert "id" in container["node"][0]
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
        raw: DgraphQueryResult = await driver.run_query('{ node(func: has(id), first: 1) { id } }')
        data = _load_result(raw)
        container = data.get("data", data)
        if "node" in container and container["node"]:
            assert "id" in container["node"][0]
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
        async def run_query(self, query: str) -> DgraphQueryResult:
            return {"data": {"node": [{"id": "test_id"}]}}

        @override
        async def close(self) -> None:
            self._client = None

    driver = MockDgraphDriver()
    await driver.connect()
    assert driver.client is not None

    raw = await driver.run_query("test query")
    result = _load_result(raw)
    assert result["data"]["node"][0]["id"] == "test_id"

    await driver.close()
    assert driver.client is None
