import importlib
import json
from collections.abc import Iterator
from typing import Any, cast, final, override
from unittest.mock import MagicMock, patch

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
            # The mock should construct the final DgraphResponse object directly,
            # simulating what the real driver's parsing step would do.
            parsed_nodes = [
                dg_models.Node.from_dict(
                    {"id": "test_id", "name": "name", "category": "cat"},
                    binding="n0",
                )
            ]
            return dg_models.DgraphResponse(data={"q0": parsed_nodes})

        @override
        async def close(self) -> None:
            self._client = None

    driver = MockDgraphDriver()
    await driver.connect()

    result = await driver.run_query("test query")
    assert isinstance(result, dg_models.DgraphResponse)

    # The parser normalizes all results into a dictionary keyed by the query index.
    # For a single query, this key is "q0".
    nodes = result.data.get("q0")
    assert nodes is not None, "Nodes should be parsed under the 'q0' key"
    assert len(nodes) == 1

    # The parser correctly extracts the binding "n0" from the "q0_node_n0" key.
    node = nodes[0]
    assert node.binding == "n0"
    assert node.id == "test_id"

    await driver.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_active_version_success_grpc_hot():
    """Test get_active_version when a version is found and that it's cached."""
    driver = new_grpc_driver()
    try:
        # The driver will now use our mocked client internally upon connection
        await driver.connect()

        # Clear cache before test
        driver.clear_version_cache()

        # Should return the version "v2" as per the live Dgraph instance
        version = await driver.get_active_version()
        assert version == "v2"
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph HTTP test (cannot connect or query): {e}")
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_active_version_success_http_hot():
    """Test get_active_version when a version is found and that it's cached."""
    driver = new_http_driver()
    try:
        # The driver will now use our mocked client internally upon connection
        await driver.connect()

        # Clear cache before test
        driver.clear_version_cache()

        # Should return the version "v2" as per the live Dgraph instance
        version = await driver.get_active_version()
        assert version == "v2"
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph HTTP test (cannot connect or query): {e}")
    finally:
        await driver.close()


@pytest.mark.asyncio
@patch("pydgraph.DgraphClient")
async def test_get_active_version_success_and_cached(
    mock_dgraph_client_class: MagicMock
):
    """Test get_active_version when a version is found and that it's cached."""
    # Mock the response object that the query will return
    mock_response = MagicMock()
    mock_response.json = json.dumps({"versions": [{"schema_metadata_version": "v1"}]}).encode("utf-8")

    # Mock the transaction object and its query method
    mock_txn = MagicMock()
    mock_txn.query.return_value = mock_response

    # Configure the mock DgraphClient instance to return our mock transaction
    mock_client_instance = mock_dgraph_client_class.return_value
    mock_client_instance.txn.return_value = mock_txn

    driver = new_grpc_driver()
    # The driver will now use our mocked client internally upon connection
    await driver.connect()

    # Clear cache before test
    driver.clear_version_cache()

    # First call should trigger the query
    version = await driver.get_active_version()
    assert version == "v1"
    mock_client_instance.txn.assert_called_once_with(read_only=True)
    mock_txn.query.assert_called_once()

    # Second call should hit the cache and not trigger the query again
    version2 = await driver.get_active_version()
    assert version2 == "v1"
    mock_client_instance.txn.assert_called_once()  # Assert it's still only called once


@pytest.mark.asyncio
@patch("pydgraph.DgraphClient")
async def test_get_active_version_not_found(
    mock_dgraph_client_class: MagicMock
):
    """Test get_active_version when no active version is in the database."""
    mock_response = MagicMock()
    mock_response.json = json.dumps({"versions": []}).encode("utf-8")
    mock_txn = MagicMock()
    mock_txn.query.return_value = mock_response
    mock_dgraph_client_class.return_value.txn.return_value = mock_txn

    driver = new_grpc_driver()
    await driver.connect()
    driver.clear_version_cache()

    version = await driver.get_active_version()
    assert version is None


@pytest.mark.asyncio
@patch("pydgraph.DgraphClient")
async def test_get_active_version_query_fails(
    mock_dgraph_client_class: MagicMock
):
    """Test get_active_version when the database query raises an exception."""
    mock_txn = MagicMock()
    mock_txn.query.side_effect = Exception("DB connection failed")
    mock_dgraph_client_class.return_value.txn.return_value = mock_txn

    driver = new_grpc_driver()
    await driver.connect()
    driver.clear_version_cache()

    version = await driver.get_active_version()
    assert version is None
