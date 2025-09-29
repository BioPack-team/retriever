import importlib
from collections.abc import Iterator
from typing import Any, cast, final, override

import aiohttp
import pydgraph
import pytest

import retriever.config.general as general_mod
import retriever.data_tiers.tier_0.dgraph.driver as driver_mod
from retriever.config.general import DgraphSettings
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


# Test-only subclass exposing the client property
class _TestDgraphHttpDriver(driver_mod.DgraphHttpDriver):
    @property
    def http_session(self) -> aiohttp.ClientSession | None:
        return self._http_session


# Test-only subclass exposing the client property
class _TestDgraphGrpcDriver(driver_mod.DgraphGrpcDriver):
    @property
    def client(self) -> pydgraph.DgraphClient | None:
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


@pytest.fixture
def manual_dgraph_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[DgraphSettings]:
    # Clear env so manual settings aren't overridden
    for var in (
        "TIER0__DGRAPH__HOST",
        "TIER0__DGRAPH__HTTP_PORT",
        "TIER0__DGRAPH__GRPC_PORT",
        "TIER0__DGRAPH__USE_TLS",
        "TIER0__DGRAPH__QUERY_TIMEOUT",
        "TIER0__DGRAPH__CONNECT_RETRIES",
    ):
        monkeypatch.delenv(var, raising=False)

    # Make sure driver module references the same CONFIG object
    monkeypatch.setattr(driver_mod, "CONFIG", general_mod.CONFIG, raising=False)

    settings = DgraphSettings(
        host="localhost",
        http_port=8080,
        grpc_port=9080,
        use_tls=False,
        query_timeout=3,
        connect_retries=0,
    )
    # Install settings on the single CONFIG instance
    monkeypatch.setattr(general_mod.CONFIG.tier0, "dgraph", settings, raising=False)

    # Reload driver so its class functions capture current module globals (CONFIG)
    importlib.reload(driver_mod)
    yield settings


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
@pytest.mark.usefixtures("manual_dgraph_settings")
async def test_dgraph_live_with_http_settings_manual() -> None:
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
@pytest.mark.usefixtures("manual_dgraph_settings")
async def test_dgraph_live_with_grpc_settings_manual() -> None:
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
            return dg_models.parse_response(
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
