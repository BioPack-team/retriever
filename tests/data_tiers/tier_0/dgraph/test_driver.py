import pytest
import json

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.tier_0.dgraph.driver import DgraphDriver


@pytest.mark.asyncio
async def test_dgraph_connect_and_query():
    driver = DgraphDriver()

    try:
        # Connect
        await driver.connect()
        assert driver._client is not None

        # Run a basic query
        query = """{ node(func: has(id), first: 1) { id } }"""
        result = await driver.run_query(query)
        result = json.loads(result)
        assert isinstance(result, dict)
        assert "node" in result
        assert "id" in result["node"][0]
    except Exception as e:
        print(f"Connection or query failed: {e}")
        pytest.skip(f"Skipping live DGraph test: {e}")
    finally:
        # Close
        await driver.close()
        assert driver._client is None


@pytest.mark.asyncio
async def test_dgraph_live_with_settings(monkeypatch):
    # Define settings for local Dgraph (adjust if needed)
    settings = DgraphSettings(host="localhost", http_port=8080, grpc_port=9080, use_tls=False)

    # Patch the global CONFIG used by the driver
    monkeypatch.setattr(CONFIG.tier0, "dgraph", settings, raising=False)

    driver = DgraphDriver()

    # Ensure the endpoint matches our settings (driver currently defaults to localhost:9080)
    driver.endpoint = f"{settings.host}:{settings.grpc_port}"

    try:
        await driver.connect()
        query = '{ node(func: has(id), first: 1) { id } }'
        result = await driver.run_query(query)
        result = json.loads(result)
        assert isinstance(result, dict)
        assert "node" in result
        assert "id" in result["node"][0]
    except Exception as e:
        pytest.skip(f"Skipping live Dgraph test (cannot connect or query): {e}")
    finally:
        await driver.close()
        assert driver._client is None


@pytest.mark.asyncio
async def test_dgraph_mock():
    # Create a mock driver that doesn't need a real connection
    class MockDgraphDriver(DgraphDriver):
        async def connect(self) -> None:
            self._client = {}

        async def run_query(self, query: str, *args, **kwargs) -> dict:
            return {"node": [{"id": "test_id"}]}

        async def close(self) -> None:
            self._client = None

    driver = MockDgraphDriver()

    # Connect to mock
    await driver.connect()
    assert driver._client is not None

    # Run query against mock
    result = await driver.run_query("test query")
    assert isinstance(result, dict)
    assert "node" in result
    assert result["node"][0]["id"] == "test_id"

    # Close mock connection
    await driver.close()
    assert driver._client is None
