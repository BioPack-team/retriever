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
    monkeypatch.setenv("TIER0__DGRAPH__PREFERRED_VERSION", "vG")
    monkeypatch.setenv("TIER0__DGRAPH__USE_TLS", "false")
    monkeypatch.setenv("TIER0__DGRAPH__QUERY_TIMEOUT", "5")
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
async def test_dgraph_live_with_http_settings_from_config() -> None:
    driver = new_http_driver()
    await driver.connect()
    assert driver.http_session is not None

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)

    result: dg_models.DgraphResponse = await driver.run_query(
        "{ node(func: has(id), first: 1) { id name category } }", transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_dgraph_live_with_grpc_settings_from_config() -> None:
    driver = new_grpc_driver()
    await driver.connect()
    assert driver.client is not None
    dgraph_schema_version = await driver.get_active_version()
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)
    result: dg_models.DgraphResponse = await driver.run_query(
        "{ node(func: has(id), first: 1) { id name category } }", transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
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


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_active_version_success_grpc_live():
    """Test get_active_version when a version is found and that it's cached."""
    driver = new_grpc_driver()
    await driver.connect()

    # Clear cache before test
    driver.clear_version_cache()

    # Should return the version "v2" as per the live Dgraph instance
    version = await driver.get_active_version()
    assert version == "vG"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_active_version_success_http_live():
    """Test get_active_version when a version is found and that it's cached."""
    driver = new_http_driver()
    await driver.connect()

    # Clear cache before test
    driver.clear_version_cache()

    # Should return the version "v2" as per the live Dgraph instance
    version = await driver.get_active_version()
    assert version == "vG"

    await driver.close()


@pytest.mark.asyncio
@patch("pydgraph.DgraphClient")
async def test_get_active_version_prefers_manual_version(
    mock_dgraph_client_class: MagicMock,
):
    """
    Test that get_active_version returns the manually provided version
    without hitting the database.
    """
    # Initialize driver with a manual version
    driver = new_grpc_driver(version="manual_v1")
    await driver.connect()

    # The mock client should NOT be used
    mock_client_instance = mock_dgraph_client_class.return_value

    # First call
    version = await driver.get_active_version()
    assert version == "manual_v1"
    mock_client_instance.txn.assert_not_called()

    # Second call (should also not query)
    version2 = await driver.get_active_version()
    assert version2 == "manual_v1"
    mock_client_instance.txn.assert_not_called()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_schema_metadata_mapping_success_grpc_live():
    """Test get_schema_metadata_mapping when a version is found and that it's cached."""
    driver = new_grpc_driver()
    await driver.connect()

    # Should return the mapping from the live Dgraph instance
    mapping = await driver.get_metadata()
    assert mapping is not None, "Mapping should not be None"

    # Verify top-level schema metadata structure
    assert "@id" in mapping, "Mapping should contain @id field"
    assert "@type" in mapping, "Mapping should contain @type field"
    assert mapping["@type"] == "sc:Dataset", "Type should be sc:Dataset"

    assert "name" in mapping, "Mapping should contain name field"
    assert mapping["name"] == "translator_kg", "Name should be translator_kg"

    assert "description" in mapping, "Mapping should contain description field"
    assert "Translator" in mapping["description"], "Description should mention Translator"

    assert "license" in mapping, "Mapping should contain license field"

    assert "url" in mapping, "Mapping should contain url field"
    assert "version" in mapping, "Mapping should contain version field"
    assert "dateCreated" in mapping, "Mapping should contain dateCreated field"

    assert "biolinkVersion" in mapping, "Mapping should contain biolinkVersion"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_schema_metadata_mapping_success_http_live():
    """Test get_schema_metadata_mapping when a version is found and that it's cached."""
    driver = new_http_driver()
    await driver.connect()

    # Should return the mapping from the live Dgraph instance
    mapping = await driver.get_metadata()
    assert mapping is not None, "Mapping should not be None"

    # Verify top-level schema metadata structure
    assert "@id" in mapping, "Mapping should contain @id field"
    assert "@type" in mapping, "Mapping should contain @type field"
    assert mapping["@type"] == "sc:Dataset", "Type should be sc:Dataset"

    assert "name" in mapping, "Mapping should contain name field"
    assert mapping["name"] == "translator_kg", "Name should be translator_kg"

    assert "description" in mapping, "Mapping should contain description field"
    assert "Translator" in mapping["description"], "Description should mention Translator"

    assert "license" in mapping, "Mapping should contain license field"

    assert "url" in mapping, "Mapping should contain url field"
    assert "version" in mapping, "Mapping should contain version field"
    assert "dateCreated" in mapping, "Mapping should contain dateCreated field"

    assert "biolinkVersion" in mapping, "Mapping should contain biolinkVersion"

    await driver.close()


@pytest.mark.parametrize(
    "protocol, mock_path",
    [
        (driver_mod.DgraphProtocol.GRPC, "pydgraph.DgraphClient"),
        (driver_mod.DgraphProtocol.HTTP, "aiohttp.ClientSession"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_schema_metadata_mapping_caching(
    protocol: driver_mod.DgraphProtocol, mock_path: str
):
    """
    Tests that get_schema_metadata_mapping properly caches results per-version.
    """
    import base64
    import msgpack

    # Create a sample mapping
    mapping_data = {
        "@id": "https://example.org/test_kg",
        "name": "test_kg",
        "version": "vTest",
        "biolinkVersion": "4.3.4",
    }

    packed = msgpack.packb(mapping_data)
    encoded = base64.b64encode(packed).decode("utf-8")

    with patch(mock_path) as mock_class:
        if protocol == driver_mod.DgraphProtocol.GRPC:
            mock_response = MagicMock()
            mock_response.json = json.dumps(
                {"metadata": [{"schema_metadata_mapping": encoded}]}
            ).encode("utf-8")
            mock_txn = MagicMock()
            mock_txn.query.return_value = mock_response
            mock_class.return_value.txn.return_value = mock_txn
        else:  # HTTP
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "data": {"metadata": [{"schema_metadata_mapping": encoded}]}
                }
            )
            mock_post = MagicMock()
            mock_post.__aenter__.return_value = mock_response
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session = mock_class.return_value
            mock_session.post.return_value = mock_post
            mock_session.close = AsyncMock()

        driver = (
            new_grpc_driver()
            if protocol == driver_mod.DgraphProtocol.GRPC
            else new_http_driver()
        )

        # Clear cache
        driver.clear_mapping_cache()
        driver.clear_version_cache()

        # Mock version to return "vTest"
        with patch.object(driver, "get_active_version", new=AsyncMock(return_value="vTest")):
            await driver.connect()

            # First call - should query DB
            mapping1 = await driver.get_metadata()
            assert mapping1 is not None
            assert mapping1["name"] == "test_kg"
            assert mapping1["version"] == "vTest"

            # Get call count after first query
            if protocol == driver_mod.DgraphProtocol.GRPC:
                call_count_1 = mock_class.return_value.txn.return_value.query.call_count
            else:
                call_count_1 = mock_class.return_value.post.call_count

            # Second call - should use cache (same version)
            mapping2 = await driver.get_metadata()
            assert mapping2 is not None
            assert mapping2["name"] == "test_kg"

            # Verify same object (cached)
            assert mapping1 is mapping2

            # Get call count after second query
            if protocol == driver_mod.DgraphProtocol.GRPC:
                call_count_2 = mock_class.return_value.txn.return_value.query.call_count
            else:
                call_count_2 = mock_class.return_value.post.call_count

            # Should be same count (cache hit, no new query)
            assert call_count_2 == call_count_1

            # Clear cache and fetch again
            driver.clear_mapping_cache()

            mapping3 = await driver.get_metadata()
            assert mapping3 is not None
            assert mapping3["name"] == "test_kg"

            # Should be different object (new fetch)
            assert mapping3 is not mapping1

            # Get call count after third query
            if protocol == driver_mod.DgraphProtocol.GRPC:
                call_count_3 = mock_class.return_value.txn.return_value.query.call_count
            else:
                call_count_3 = mock_class.return_value.post.call_count

            # Should be one more call (cache was cleared)
            assert call_count_3 == call_count_1 + 1

        await driver.close()


@pytest.mark.parametrize(
    "protocol, mock_path",
    [
        (driver_mod.DgraphProtocol.GRPC, "pydgraph.DgraphClient"),
        (driver_mod.DgraphProtocol.HTTP, "aiohttp.ClientSession"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_get_active_version_from_db_mocked(
    protocol: driver_mod.DgraphProtocol, mock_path: str
):
    """
    Tests get_active_version for DB query, caching, not-found, and failure scenarios
    for both gRPC and HTTP protocols using mocks.
    """

    def setup_mock(mock_class: MagicMock, response_data: Any, *, fails: bool = False):
        """Helper to configure the mock for either gRPC or HTTP."""
        if protocol == driver_mod.DgraphProtocol.GRPC:
            mock_response = MagicMock()
            mock_response.json = json.dumps(response_data).encode("utf-8")
            mock_txn = MagicMock()
            if fails:
                mock_txn.query.side_effect = Exception("DB connection failed")
            else:
                mock_txn.query.return_value = mock_response
            mock_class.return_value.txn.return_value = mock_txn
            return mock_class.return_value
        else:  # HTTP
            mock_response = MagicMock()
            mock_response.status = 200
            if fails:
                mock_response.json = AsyncMock(side_effect=Exception("HTTP connection failed"))
            else:
                mock_response.json = AsyncMock(return_value={"data": response_data})

            mock_post = MagicMock()
            mock_post.__aenter__.return_value = mock_response

            mock_session = mock_class.return_value
            mock_session.post.return_value = mock_post
            mock_session.close = AsyncMock()  # <-- make awaitable
            return mock_session

    # Force DB query path by removing preferred_version during these tests
    with patch.object(general_mod.CONFIG.tier0.dgraph, "preferred_version", None):

        # --- Test Case 1: Success ---
        with patch(mock_path) as mock_class:
            mock_instance = setup_mock(
                mock_class, {"versions": [{"schema_metadata_version": "v_db"}]}
            )

            driver = (
                new_grpc_driver(version=None)
                if protocol == driver_mod.DgraphProtocol.GRPC
                else new_http_driver(version=None)
            )

            # Bypass any prefetch during connect
            with patch.object(
                driver_mod.DgraphDriver, "get_active_version", new=AsyncMock(return_value=None)
            ):
                await driver.connect()

            # Ensure clean slate
            driver.clear_version_cache()
            driver.version = None

            version = await driver.get_active_version()
            assert version == "v_db"

            # Just assert the underlying query was used at least once
            if protocol == driver_mod.DgraphProtocol.GRPC:
                assert mock_instance.txn.return_value.query.called
            else:
                assert mock_instance.post.called

            await driver.close()

        # --- Test Case 2: Version Not Found ---
        with patch(mock_path) as mock_class:
            setup_mock(mock_class, {"versions": []})

            driver = (
                new_grpc_driver(version=None)
                if protocol == driver_mod.DgraphProtocol.GRPC
                else new_http_driver(version=None)
            )

            with patch.object(
                driver_mod.DgraphDriver, "get_active_version", new=AsyncMock(return_value=None)
            ):
                await driver.connect()

            driver.clear_version_cache()
            driver.version = None

            version = await driver.get_active_version()
            assert version is None

            await driver.close()

        # --- Test Case 3: Query Fails ---
        with patch(mock_path) as mock_class:
            setup_mock(mock_class, {}, fails=True)

            driver = (
                new_grpc_driver(version=None)
                if protocol == driver_mod.DgraphProtocol.GRPC
                else new_http_driver(version=None)
            )

            with patch.object(
                driver_mod.DgraphDriver, "get_active_version", new=AsyncMock(return_value=None)
            ):
                await driver.connect()

            driver.clear_version_cache()
            driver.version = None

            version = await driver.get_active_version()
            assert version is None

            await driver.close()


@pytest.mark.parametrize(
    "protocol, mock_path",
    [
        (driver_mod.DgraphProtocol.GRPC, "pydgraph.DgraphClient"),
        (driver_mod.DgraphProtocol.HTTP, "aiohttp.ClientSession"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_fetch_mapping_from_db_mocked(
    protocol: driver_mod.DgraphProtocol, mock_path: str
):
    """
    Tests _fetch_mapping_from_db for success, not-found, and failure scenarios
    for both gRPC and HTTP protocols using mocks.
    """
    import base64
    import msgpack

    # Create a sample mapping to use in tests
    sample_mapping = {
        "@id": "https://example.org/test_kg",
        "@type": "sc:Dataset",
        "name": "test_kg",
        "version": "2025_test",
        "biolinkVersion": "4.3.4",
        "schema": {
            "nodes": [
                {
                    "category": ["biolink:SmallMolecule"],
                    "count": 100,
                    "id_prefixes": {"CHEBI": 90, "PUBCHEM.COMPOUND": 10},
                }
            ]
        },
    }

    # Encode the mapping as msgpack + base64 (as stored in Dgraph)
    packed_mapping = msgpack.packb(sample_mapping)
    encoded_mapping = base64.b64encode(packed_mapping).decode("utf-8")

    def setup_mock(mock_class: MagicMock, response_data: Any, *, fails: bool = False):
        """Helper to configure the mock for either gRPC or HTTP."""
        if protocol == driver_mod.DgraphProtocol.GRPC:
            mock_response = MagicMock()
            mock_response.json = json.dumps(response_data).encode("utf-8")
            mock_txn = MagicMock()
            if fails:
                mock_txn.query.side_effect = Exception("DB connection failed")
            else:
                mock_txn.query.return_value = mock_response
            mock_class.return_value.txn.return_value = mock_txn
            return mock_class.return_value
        else:  # HTTP
            mock_response = MagicMock()
            mock_response.status = 200
            if fails:
                mock_response.json = AsyncMock(side_effect=Exception("HTTP connection failed"))
            else:
                mock_response.json = AsyncMock(return_value={"data": response_data})

            mock_post = MagicMock()
            mock_post.__aenter__.return_value = mock_response
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session = mock_class.return_value
            mock_session.post.return_value = mock_post
            mock_session.close = AsyncMock()
            return mock_session

    # --- Test Case 1: Success ---
    with patch(mock_path) as mock_class:
        mock_instance = setup_mock(
            mock_class,
            {"metadata": [{"schema_metadata_mapping": encoded_mapping}]},
        )

        driver = (
            new_grpc_driver()
            if protocol == driver_mod.DgraphProtocol.GRPC
            else new_http_driver()
        )

        await driver.connect()

        # Call the helper method directly with a version
        mapping = await driver.fetch_mapping_from_db("vTest")
        assert mapping is not None, "Mapping should not be None"
        assert mapping["@id"] == "https://example.org/test_kg"
        assert mapping["@type"] == "sc:Dataset"
        assert mapping["name"] == "test_kg"
        assert mapping["version"] == "2025_test"
        assert mapping["biolinkVersion"] == "4.3.4"
        assert "schema" in mapping
        assert "nodes" in mapping["schema"]
        assert len(mapping["schema"]["nodes"]) == 1
        assert mapping["schema"]["nodes"][0]["category"] == ["biolink:SmallMolecule"]
        assert mapping["schema"]["nodes"][0]["count"] == 100

        # Verify the underlying query was called
        if protocol == driver_mod.DgraphProtocol.GRPC:
            assert mock_instance.txn.return_value.query.called
        else:
            assert mock_instance.post.called

        await driver.close()

    # --- Test Case 2: Mapping Not Found (no metadata) ---
    with patch(mock_path) as mock_class:
        setup_mock(mock_class, {"metadata": []})

        driver = (
            new_grpc_driver()
            if protocol == driver_mod.DgraphProtocol.GRPC
            else new_http_driver()
        )

        await driver.connect()

        mapping = await driver.fetch_mapping_from_db("vTest")
        assert mapping is None, "Mapping should be None when no metadata found"

        await driver.close()

    # --- Test Case 3: Mapping Field Empty ---
    with patch(mock_path) as mock_class:
        setup_mock(
            mock_class,
            {"metadata": [{"schema_metadata_mapping": None}]},
        )

        driver = (
            new_grpc_driver()
            if protocol == driver_mod.DgraphProtocol.GRPC
            else new_http_driver()
        )

        await driver.connect()

        mapping = await driver.fetch_mapping_from_db("vTest")
        assert mapping is None, "Mapping should be None when field is empty"

        await driver.close()

    # --- Test Case 4: Query Fails ---
    with patch(mock_path) as mock_class:
        setup_mock(mock_class, {}, fails=True)

        driver = (
            new_grpc_driver()
            if protocol == driver_mod.DgraphProtocol.GRPC
            else new_http_driver()
        )

        await driver.connect()

        mapping = await driver.fetch_mapping_from_db("vTest")
        assert mapping is None, "Mapping should be None when query fails"

        await driver.close()

    # --- Test Case 5: Invalid msgpack data ---
    with patch(mock_path) as mock_class:
        # Send invalid base64/msgpack data
        invalid_data = base64.b64encode(b"not valid msgpack").decode("utf-8")
        setup_mock(
            mock_class,
            {"metadata": [{"schema_metadata_mapping": invalid_data}]},
        )

        driver = (
            new_grpc_driver()
            if protocol == driver_mod.DgraphProtocol.GRPC
            else new_http_driver()
        )

        await driver.connect()

        mapping = await driver.fetch_mapping_from_db("vTest")
        assert mapping is None, "Mapping should be None when msgpack deserialization fails"

        await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_one_query_live_http() -> None:
    """
    Integration test: Run the 'simple-one' query against a live Dgraph HTTP instance.
    """

    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0": {"ids": ["GO:0031410"], "constraints": []},
            "n1": {"ids": ["NCBIGene:11276"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["located_in"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    })

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vG_id, "NCBIGene:11276")) @cascade(vG_id, ~vG_subject) {
            expand(vG_Node)
            out_edges_e0: ~vG_subject @filter(eq(vG_predicate_ancestors, "located_in")) @cascade(vG_predicate, vG_object) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }
                node_n0: vG_object @filter(eq(vG_id, "GO:0031410")) @cascade(vG_id) {
                    expand(vG_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_http_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)
    assert transpiler.version == "vG"
    assert transpiler.prefix == "vG_"

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(dgraph_query, transpiler=transpiler)
    assert isinstance(result, dg_models.DgraphResponse)

    # Minimal existence checks
    assert result.data, "No data returned from Dgraph for simple-one query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # Root node (NCBIGene:11276)
    root_node = result.data["q0"][0]

    # 1. Validate Root Node (NCBIGene:11276)
    assert root_node.binding == "n1"
    assert root_node.id == "NCBIGene:11276"
    assert root_node.name == "SYNRG"

    expected_root_cats = sorted([
        "MacromolecularMachineMixin", "NamedThing", "Gene",
        "ChemicalEntityOrProteinOrPolypeptide", "PhysicalEssence",
        "PhysicalEssenceOrOccurrent", "OntologyClass",
        "ChemicalEntityOrGeneOrGeneProduct", "GeneOrGeneProduct", "Polypeptide",
        "ThingWithTaxon", "GenomicEntity", "GeneProductMixin", "Protein",
        "BiologicalEntity"
    ])
    assert sorted(root_node.category) == expected_root_cats

    # Validate attributes dictionary
    root_attrs = root_node.attributes
    assert root_attrs.get("description") == "Synergin gamma"
    assert root_attrs.get("full_name") == "synergin gamma"
    assert root_attrs.get("symbol") == "SYNRG"
    assert root_attrs.get("taxon") == "NCBITaxon:9606"
    assert root_attrs.get("in_taxon") == ["NCBITaxon:9606"]
    assert root_attrs.get("information_content") == pytest.approx(83.1)

    expected_equiv_ids = sorted([
        "PR:Q9UMZ2", "OMIM:607291", "UniProtKB:Q9UMZ2", "ENSEMBL:ENSG00000275066",
        "UMLS:C1412437", "UMLS:C0893518", "MESH:C121510", "HGNC:557", "NCBIGene:11276"
    ])
    assert sorted(root_attrs.get("equivalent_identifiers", [])) == expected_equiv_ids

    # 2. Validate Edge
    assert len(root_node.edges) == 1
    edge = root_node.edges[0]
    assert edge.binding == "e0"
    assert edge.direction == "out"
    assert edge.predicate == "located_in"
    assert isinstance(edge.id, str) and "urn:uuid:" in edge.id
    assert edge.qualifiers == {}
    assert edge.source_inforeses == []

    # Validate edge attributes
    edge_attrs = edge.attributes
    assert edge_attrs.get("original_object") == "GO:0031410"
    assert edge_attrs.get("original_subject") == "UniProtKB:Q9UMZ2"
    assert edge_attrs.get("agent_type") == "automated_agent"
    assert edge_attrs.get("has_evidence") == ["ECO:IEA"]
    assert edge_attrs.get("category") == ["Association"]
    assert edge_attrs.get("knowledge_level") == "prediction"

    # Validate sources (order-independent)
    source_ids = {s.resource_id for s in edge.sources}
    assert source_ids == {"infores:goa", "infores:biolink"}

    # 3. Validate Connected Node (GO:0031410)
    connected_node = edge.node
    assert connected_node.binding == "n0"
    assert connected_node.id == "GO:0031410"
    assert connected_node.name == "cytoplasmic vesicle"
    assert connected_node.edges == []

    expected_go_cats = sorted([
        "NamedThing", "OrganismalEntity", "PhysicalEssence",
        "PhysicalEssenceOrOccurrent", "CellularComponent", "ThingWithTaxon",
        "SubjectOfInvestigation", "AnatomicalEntity", "BiologicalEntity"
    ])
    assert sorted(connected_node.category) == expected_go_cats

    # Validate connected node attributes
    go_attrs = connected_node.attributes
    assert go_attrs.get("information_content") == 56.8
    assert go_attrs.get("equivalent_identifiers") == ["GO:0031410"]

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_one_query_live_grpc() -> None:
    """
    Integration test: Run the 'simple-one' query against a live Dgraph HTTP instance.
    """

    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0_test": {"ids": ["GO:0031410"], "constraints": []},
            "n1": {"ids": ["NCBIGene:11276"], "constraints": []},
        },
        "edges": {
            "e0_test": {
                "object": "n0_test",
                "subject": "n1",
                "predicates": ["located_in"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    })

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vG_id, "NCBIGene:11276")) @cascade(vG_id, ~vG_subject) {
            expand(vG_Node)
            out_edges_e0: ~vG_subject @filter(eq(vG_predicate_ancestors, "located_in")) @cascade(vG_predicate, vG_object) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }
                node_n0: vG_object @filter(eq(vG_id, "GO:0031410")) @cascade(vG_id) {
                    expand(vG_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)
    assert transpiler.version == "vG"
    assert transpiler.prefix == "vG_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(dgraph_query, transpiler=transpiler)
    assert isinstance(result, dg_models.DgraphResponse)

    # Assertions to check that some data is returned
    assert result.data, "No data returned from Dgraph for simple-one query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # 2. Assertions for the root node (n0)
    root_node = result.data["q0"][0]
    assert root_node.binding == "n1"
    assert root_node.id == "NCBIGene:11276"
    assert len(root_node.edges) == 1

    # 3. Assertions for the incoming edge (e0)
    in_edge = root_node.edges[0]
    assert in_edge.binding == "e0_test"
    assert in_edge.direction == "out"
    assert in_edge.predicate == "located_in"

    # 4. Assertions for the connected node (n1)
    connected_node = in_edge.node
    assert connected_node.binding == "n0_test"
    assert connected_node.id == "GO:0031410"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_reverse_query_live_grpc() -> None:
    """
    Integration test: Run the 'simple-one' query against a live Dgraph HTTP instance.
    """

    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
            "n1": {"ids": ["NCBIGene:3778"], "constraints": []}
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:has_phenotype"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        }
    })

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vG_id, "NCBIGene:3778")) @cascade(vG_id, ~vG_subject) {
            expand(vG_Node)
            out_edges_e0: ~vG_subject @filter(eq(vG_predicate_ancestors, "has_phenotype")) @cascade(vG_predicate, vG_object) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }
                node_n0: vG_object @filter(eq(vG_category, "NamedThing")) @cascade(vG_id) {
                    expand(vG_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)
    assert transpiler.version == "vG"
    assert transpiler.prefix == "vG_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(dgraph_query, transpiler=transpiler)
    assert isinstance(result, dg_models.DgraphResponse)

    # Assertions to check that some data is returned
    assert result.data, "No data returned from Dgraph for simple-one query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # 2. Assertions for the root node (n0)
    root_node = result.data["q0"][0]
    assert root_node.binding == "n1"
    assert root_node.id == "NCBIGene:3778"
    root_node_edges_count = len(root_node.edges)
    assert root_node_edges_count == 65

    # 3. Assertions for the outgoing edge (e0)
    out_edge = root_node.edges[0]
    assert out_edge.binding == "e0"
    assert out_edge.direction == "out"
    assert isinstance(out_edge.predicate, str) and out_edge.predicate

    # 4. Assertions for the connected node (n1)
    connected_node = out_edge.node
    assert connected_node.binding == "n0"
    assert isinstance(connected_node.id, str) and connected_node.id

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_query_with_symmetric_predicate_live_grpc() -> None:
    """
    Integration test: Run the 'simple-one' query with symmetric predicate against a live Dgraph HTTP instance.
    """

    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
            "n1": {"ids": ["NCBIGene:3778"], "constraints": []}
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:related_to"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        }
    })

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vG_id, "NCBIGene:3778")) @cascade(vG_id, ~vG_subject) {
            expand(vG_Node)

            out_edges_e0: ~vG_subject
            @filter(eq(vG_predicate_ancestors, "related_to"))
            @cascade(vG_predicate, vG_object) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }

                node_n0: vG_object
                @filter(eq(vG_category, "NamedThing"))
                @cascade(vG_id) {
                    expand(vG_Node)
                }
            }

            in_edges-symmetric_e0: ~vG_object
            @filter(eq(vG_predicate_ancestors, "related_to"))
            @cascade(vG_predicate, vG_subject) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }

                node_n0: vG_subject
                @filter(eq(vG_category, "NamedThing"))
                @cascade(vG_id) {
                    expand(vG_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)
    assert transpiler.version == "vG"
    assert transpiler.prefix == "vG_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(dgraph_query_match, transpiler=transpiler)
    assert isinstance(result, dg_models.DgraphResponse)

    # Assertions to check that some data is returned
    assert result.data, "No data returned from Dgraph for simple-one query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # 2. Assertions for the root node (n1)
    root_node = result.data["q0"][0]
    assert root_node.binding == "n1"
    assert root_node.id == "NCBIGene:3778"
    root_node_edges_count = len(root_node.edges)
    assert root_node_edges_count == 744

    # Both out_edges_e0 and in_edges-symmetric_e0 are merged under binding "e0"
    e0_edges = [e for e in root_node.edges if e.binding == "e0"]
    e0_edges_count = len(e0_edges)
    assert e0_edges_count == 744, "Expected 744 total edges for binding 'e0' (merged)"

    # Separate by direction instead of binding
    out_edges = [e for e in e0_edges if e.direction == "out"]
    in_edges = [e for e in e0_edges if e.direction == "in"]

    # Both forward (out) and reverse (in) edge groups must be present and non-empty
    assert out_edges, "Expected at least one outgoing edge (from out_edges_e0)"
    assert in_edges, "Expected at least one incoming edge (from in_edges_e0 - including symmetric predicate)"

    # TODO: Double check this assertion
    # Predicate/ancestors should reflect the symmetric predicate filter ("related_to")
    # assert all(
    #     ("related_to" in e.predicate_ancestors) or (e.predicate == "related_to")
    #     for e in out_edges
    # ), "All outgoing edges should have 'related_to' in predicate/ancestors"
    # assert all(
    #     ("related_to" in e.predicate_ancestors) or (e.predicate == "related_to")
    #     for e in in_edges
    # ), "All incoming edges should have 'related_to' in predicate/ancestors"

    # Connected nodes should be parsed and have the expected binding for the query (n0)
    assert all(e.node.binding == "n0" for e in e0_edges)
    assert all(isinstance(e.node.id, str) and e.node.id for e in e0_edges)

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_one_query_grpc_parallel_live_nonblocking() -> None:
    """
    Test that two gRPC queries run in parallel and do not block each other.
    """
    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0": {"ids": ["CHEBI:4514"], "constraints": []},
            "n1": {"ids": ["UMLS:C1564592"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["subclass_of"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    })

    driver = new_grpc_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)

    async def run_query_with_delay():
        # Add an artificial delay to simulate a slow query
        await asyncio.sleep(1)
        return await driver.run_query(dgraph_query, transpiler=transpiler)

    start = time.perf_counter()
    # Run queries concurrently. Calling run_query_with_delay three times to increase chance of blocking.
    results = await asyncio.gather(
        run_query_with_delay(),
        run_query_with_delay(),
        run_query_with_delay(),
    )
    elapsed = time.perf_counter() - start

    # Both should succeed
    for result in results:
        assert isinstance(result, dg_models.DgraphResponse)
        assert result.data, "No data returned from Dgraph for simple-one query"

    # If queries are non-blocking, elapsed should be just over 1 second, not 2+
    assert elapsed < 2, f"Queries are blocking each other! Elapsed: {elapsed:.2f}s"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_normalization_with_special_edge_id_live_grpc() -> None:
    """
    Integration test: Verify that edge IDs with special characters (e.g., 'e0_bad$%^')
    are normalized to safe identifiers ('e0') in the query but restored in results.
    """

    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0_test!@#": {"categories": ["biolink:NamedThing"], "constraints": []},
            "n1": {"ids": ["NCBIGene:3778"], "constraints": []}
        },
        "edges": {
            "e0_bad$%^": {
                "object": "n0_test!@#",
                "subject": "n1",
                "predicates": ["biolink:related_to"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        }
    })

    # Expected query should use normalized edge ID 'e0', not 'e0_bad$%^'
    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vG_id, "NCBIGene:3778")) @cascade(vG_id, ~vG_subject) {
            expand(vG_Node)

            out_edges_e0: ~vG_subject
            @filter(eq(vG_predicate_ancestors, "related_to"))
            @cascade(vG_predicate, vG_object) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }

                node_n0: vG_object
                @filter(eq(vG_category, "NamedThing"))
                @cascade(vG_id) {
                    expand(vG_Node)
                }
            }

            in_edges-symmetric_e0: ~vG_object
            @filter(eq(vG_predicate_ancestors, "related_to"))
            @cascade(vG_predicate, vG_subject) {
                expand(vG_Edge) { vG_sources expand(vG_Source) }

                node_n0: vG_subject
                @filter(eq(vG_category, "NamedThing"))
                @cascade(vG_id) {
                    expand(vG_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)
    assert transpiler.version == "vG"
    assert transpiler.prefix == "vG_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)

    # Verify the query uses normalized edge ID 'e0', not 'e0_bad$%^'
    assert_query_equals(dgraph_query, dgraph_query_match)
    assert "out_edges_e0:" in dgraph_query, "Query should use normalized edge ID 'e0'"
    assert "in_edges-symmetric_e0:" in dgraph_query, "Symmetric edge should use normalized ID 'e0'"
    assert "e0_bad$%^" not in dgraph_query, "Original edge ID 'e0_bad$%^' should not appear in query"

    # Run the query against the live Dgraph instance, passing transpiler for ID mapping
    result: dg_models.DgraphResponse = await driver.run_query(dgraph_query, transpiler=transpiler)
    assert isinstance(result, dg_models.DgraphResponse)

    # Assertions to check that some data is returned
    assert result.data, "No data returned from Dgraph for normalization test query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # Verify the root node
    root_node = result.data["q0"][0]
    assert root_node.binding == "n1"
    assert root_node.id == "NCBIGene:3778"
    assert len(root_node.edges) > 0, "Expected at least one edge"

    # CRITICAL: Verify that edges have the ORIGINAL binding 'e0_bad', not 'e0'
    e0_bad_edges = [e for e in root_node.edges if e.binding == "e0_bad$%^"]
    assert len(e0_bad_edges) > 0, "Edges should have original binding 'e0_bad$%^' restored from normalization"

    # Verify no edges have the normalized binding 'e0'
    e0_edges = [e for e in root_node.edges if e.binding == "e0"]
    assert len(e0_edges) == 0, "No edges should have normalized binding 'e0' in results"

    # Both forward (out) and reverse (in) edge groups should be present
    out_edges = [e for e in e0_bad_edges if e.direction == "out"]
    in_edges = [e for e in e0_bad_edges if e.direction == "in"]

    assert out_edges, "Expected at least one outgoing edge with binding 'e0_bad$%^'"
    assert in_edges, "Expected at least one incoming edge with binding 'e0_bad$%^' (symmetric)"

    # Verify connected nodes have correct binding
    assert all(e.node.binding == "n0_test!@#" for e in e0_bad_edges)
    assert all(isinstance(e.node.id, str) and e.node.id for e in e0_bad_edges)

    # TODO: Double check this assertion
    # Verify predicates match the query
    # assert all(
    #     ("related_to" in e.predicate_ancestors) or (e.predicate == "related_to")
    #     for e in e0_bad_edges
    # ), "All edges should have 'related_to' in predicate/ancestors"

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_one_query_http_parallel_live_nonblocking() -> None:
    """
    Test that two HTTP queries run in parallel and do not block each other.
    """
    qgraph_query: QueryGraphDict = qg({
        "nodes": {
            "n0": {"ids": ["CHEBI:4514"], "constraints": []},
            "n1": {"ids": ["UMLS:C1564592"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["subclass_of"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    })

    driver = new_http_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(version=dgraph_schema_version)

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)

    async def run_query_with_delay():
        # Add an artificial delay to simulate a slow query
        await asyncio.sleep(1)
        return await driver.run_query(dgraph_query, transpiler=transpiler)

    start = time.perf_counter()
    # Run queries concurrently. Calling run_query_with_delay three times to increase chance of blocking.
    results = await asyncio.gather(
        run_query_with_delay(),
        run_query_with_delay(),
        run_query_with_delay(),
    )
    elapsed = time.perf_counter() - start

    # Both should succeed
    for result in results:
        assert isinstance(result, dg_models.DgraphResponse)
        assert result.data, "No data returned from Dgraph for simple-one query"

    # If queries are non-blocking, elapsed should be just over 1 second, not 2+
    assert elapsed < 2, f"Queries are blocking each other! Elapsed: {elapsed:.2f}s"

    await driver.close()

@pytest.mark.asyncio
@patch.object(driver_mod.DgraphGrpcDriver, "_connect_grpc", new_callable=AsyncMock)
@patch("pydgraph.Txn.handle_query_future")
async def test_run_grpc_query_raises_timeout_on_deadline_exceeded(
    mock_handle_query: MagicMock,
    _mock_connect_grpc: AsyncMock,
) -> None:
    """Test that a grpc.RpcError with DEADLINE_EXCEEDED raises a TimeoutError."""
    # 1. Arrange
    driver = new_grpc_driver()
    await driver.connect()

    # Manually set up the mock client that connect() would have created
    driver.client = MagicMock()
    driver.client.txn.return_value.handle_query_future = mock_handle_query

    # Mock the RpcError
    mock_rpc_error = grpc.RpcError()
    mock_rpc_error.code = MagicMock(return_value=grpc.StatusCode.DEADLINE_EXCEEDED)
    mock_rpc_error.details = MagicMock(return_value="Deadline exceeded")
    mock_handle_query.side_effect = mock_rpc_error

    # Create a mock transpiler
    mock_transpiler = MagicMock()
    mock_transpiler._reverse_node_map = {}
    mock_transpiler._reverse_edge_map = {}

    # 2. Act & Assert
    with pytest.raises(TimeoutError, match="Dgraph query exceeded (.*) timeout"):
        await driver.run_query("any query", transpiler=mock_transpiler)

    await driver.close()


@pytest.mark.asyncio
@patch.object(driver_mod.DgraphGrpcDriver, "_connect_grpc", new_callable=AsyncMock)
@patch("pydgraph.Txn.handle_query_future")
async def test_run_grpc_query_raises_connection_error_on_generic_rpc_error(
    mock_handle_query: MagicMock,
    _mock_connect_grpc: AsyncMock,
) -> None:
    """Test that a generic grpc.RpcError raises a ConnectionError."""
    # 1. Arrange
    driver = new_grpc_driver()
    await driver.connect()

    driver.client = MagicMock()
    driver.client.txn.return_value.handle_query_future = mock_handle_query

    # Mock the RpcError
    mock_rpc_error = grpc.RpcError()
    mock_rpc_error.code = MagicMock(return_value=grpc.StatusCode.UNKNOWN)
    mock_rpc_error.details = MagicMock(return_value="Some other gRPC error")
    mock_handle_query.side_effect = mock_rpc_error

    # Create a mock transpiler
    mock_transpiler = MagicMock()
    mock_transpiler._reverse_node_map = {}
    mock_transpiler._reverse_edge_map = {}

    # 2. Act & Assert
    with pytest.raises(ConnectionError, match="Dgraph gRPC query failed: Some other gRPC error"):
        await driver.run_query("any query", transpiler=mock_transpiler)

    await driver.close()


@pytest.mark.asyncio
@patch.object(driver_mod.DgraphGrpcDriver, "_connect_grpc", new_callable=AsyncMock)
@patch("pydgraph.Txn.handle_query_future")
async def test_run_grpc_query_name_error_workaround(
    mock_handle_query: MagicMock,
    _mock_connect_grpc: AsyncMock,
) -> None:
    """Test the workaround for pydgraph raising NameError instead of RpcError."""
    # 1. Arrange
    driver = new_grpc_driver()
    await driver.connect()

    driver.client = MagicMock()
    driver.client.txn.return_value.handle_query_future = mock_handle_query

    # Mock the underlying RpcError
    mock_rpc_error = grpc.RpcError()
    mock_rpc_error.code = MagicMock(return_value=grpc.StatusCode.UNKNOWN)
    mock_rpc_error.details = MagicMock(return_value="while running ToJson")

    # Create a NameError and set its __context__ to our mock RpcError
    name_error = NameError("name 'txn' is not defined")
    name_error.__context__ = mock_rpc_error
    mock_handle_query.side_effect = name_error

    # Create a mock transpiler
    mock_transpiler = MagicMock()
    mock_transpiler._reverse_node_map = {}
    mock_transpiler._reverse_edge_map = {}

    # 2. Act & Assert
    with pytest.raises(ConnectionError, match="Dgraph gRPC query failed: while running ToJson"):
        await driver.run_query("any query", transpiler=mock_transpiler)

    await driver.close()
