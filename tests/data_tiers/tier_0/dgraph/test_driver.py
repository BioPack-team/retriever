import importlib
from collections.abc import Iterator
from typing import Any, cast, final, override

import aiohttp
import pytest

import retriever.config.general as general_mod
import retriever.data_tiers.tier_0.dgraph.driver as driver_mod
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


# Test-only subclass exposing the client property
class _TestDgraphHttpDriver(driver_mod.DgraphHttpDriver):
    @property
    def http_session(self) -> aiohttp.ClientSession | None:
        return self._http_session


# Test-only subclass exposing the client property
class _TestDgraphGrpcDriver(driver_mod.DgraphGrpcDriver):
    @property
    def client(self) -> driver_mod.DgraphClientProtocol | None:
        return self._client


def new_http_driver() -> _TestDgraphHttpDriver:
    return _TestDgraphHttpDriver()


def new_grpc_driver() -> _TestDgraphGrpcDriver:
    return _TestDgraphGrpcDriver()


@pytest.fixture
def mock_dgraph_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # These tests should use localhost and SKIP
    monkeypatch.setenv("TIER0__DGRAPH__HOST", "localhost")
    monkeypatch.setenv("TIER0__DGRAPH__HTTP_PORT", "8080")
    monkeypatch.setenv("TIER0__DGRAPH__GRPC_PORT", "9080")
    monkeypatch.setenv("TIER0__DGRAPH__USE_TLS", "false")
    monkeypatch.setenv("TIER0__DGRAPH__QUERY_TIMEOUT", "3")
    monkeypatch.setenv("TIER0__DGRAPH__CONNECT_RETRIES", "0")

    # Rebuild CONFIG from env and reload driver so classes bind to the new CONFIG
    importlib.reload(general_mod)
    importlib.reload(driver_mod)
    # Ensure the driver module uses the same CONFIG instance
    monkeypatch.setattr(driver_mod, "CONFIG", general_mod.CONFIG, raising=False)
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_dgraph_live_with_http_settings_from_config() -> None:
    driver = new_http_driver()
    try:
        await driver.connect()
        assert driver.http_session is not None
        print(f"HTTP host={driver.settings.host}, endpoint={driver.endpoint}")
        result: dg_models.DgraphResponse = await driver.run_query(
            "{ node(func: has(id), first: 1) { id name category } }"
        )
        assert isinstance(result, dg_models.DgraphResponse)
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph HTTP test (cannot connect or query): {e}")
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_dgraph_live_with_grpc_settings_from_config() -> None:
    driver = new_grpc_driver()
    try:
        await driver.connect()
        assert driver.client is not None
        print(f"gRPC host={driver.settings.host}, endpoint={driver.endpoint}")
        result: dg_models.DgraphResponse = await driver.run_query(
            "{ node(func: has(id), first: 1) { id name category } }"
        )
        assert isinstance(result, dg_models.DgraphResponse)
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph gRPC test (cannot connect or query): {e}")
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_dgraph_mock() -> None:
    @final
    class MockDgraphDriver(driver_mod.DgraphGrpcDriver):
        @override
        async def connect(self, retries: int = 0) -> None:
            self._client = cast(Any, object())

        @override
        async def run_query(self, query: str, *args: Any, **kwargs: Any) -> dg_models.DgraphResponse:
            return dg_models.DgraphResponse.parse(
                {"data": {"node": [{"id": "test_id", "name": "name", "category": "cat"}]}}
            )

        @override
        async def close(self) -> None:
            self._client = None

    driver = MockDgraphDriver()
    await driver.connect()

    result = await driver.run_query("test query")
    assert isinstance(result, dg_models.DgraphResponse)
    nodes = result.data.get("node", [])
    assert nodes and nodes[0].id == "test_id"

    await driver.close()
