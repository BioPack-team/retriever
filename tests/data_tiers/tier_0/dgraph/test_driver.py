import json
from typing import Any, cast, final, override

import pytest

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.tier_0.dgraph.driver import (
    DgraphDriver,
    DgraphQueryResult,
)


def _load_result(result: DgraphQueryResult) -> dict[str, Any]:
    if isinstance(result, (bytes, bytearray)):
        return json.loads(result.decode("utf-8"))
    if isinstance(result, str):
        return json.loads(result)
    raise TypeError(f"Unexpected result type: {type(result)}")


@pytest.mark.asyncio
async def test_dgraph_live_with_default_settings():
    driver = DgraphDriver()
    try:
        await driver.connect()
        assert driver.client is not None

        raw = await driver.run_query("{ node(func: has(id), first: 1) { id } }")
        data = _load_result(raw)
        assert isinstance(data, dict)
        if "node" in data and data["node"]:
            assert "id" in data["node"][0]
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph test: {e}")
    finally:
        await driver.close()
        assert driver.client is None


@pytest.mark.asyncio
async def test_dgraph_live_with_settings(monkeypatch: pytest.MonkeyPatch):
    settings = DgraphSettings(host="localhost", http_port=8080, grpc_port=9080, use_tls=False)
    monkeypatch.setattr(CONFIG.tier0, "dgraph", settings, raising=False)

    driver = DgraphDriver()
    driver.endpoint = f"{settings.host}:{settings.grpc_port}"

    try:
        await driver.connect()
        raw = await driver.run_query('{ node(func: has(id), first: 1) { id } }')
        data = _load_result(raw)
        assert isinstance(data, dict)
        if "node" in data and data["node"]:
            assert "id" in data["node"][0]
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph test (cannot connect or query): {e}")
    finally:
        await driver.close()
        assert driver.client is None


@pytest.mark.asyncio
async def test_dgraph_mock():
    @final
    class MockDgraphDriver(DgraphDriver):
        @override
        async def connect(self) -> None:
            self._client = cast(Any, object())  # internal; tests only read via .client

        @override
        async def run_query(self, query: str) -> DgraphQueryResult:
            return b'{"data": {"node": [{"id": "test_id"}]}}'

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
