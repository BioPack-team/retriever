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
    monkeypatch.setenv("TIER0__DGRAPH__PREFERRED_VERSION", "vI")
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
async def test_dgraph_live_with_http_settings_from_config() -> None:
    driver = new_http_driver()
    await driver.connect()
    assert driver.http_session is not None

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version
    )

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
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version
    )
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
        async def run_query(
            self, query: str, *args: Any, **kwargs: Any
        ) -> dg_models.DgraphResponse:
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
    assert version == "vI"

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
    assert version == "vI"

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
    assert (
        "Translator" in mapping["description"]
    ), "Description should mention Translator"

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
    assert (
        "Translator" in mapping["description"]
    ), "Description should mention Translator"

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
        with patch.object(
            driver, "get_active_version", new=AsyncMock(return_value="vTest")
        ):
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
                mock_response.json = AsyncMock(
                    side_effect=Exception("HTTP connection failed")
                )
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
                driver_mod.DgraphDriver,
                "get_active_version",
                new=AsyncMock(return_value=None),
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
                driver_mod.DgraphDriver,
                "get_active_version",
                new=AsyncMock(return_value=None),
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
                driver_mod.DgraphDriver,
                "get_active_version",
                new=AsyncMock(return_value=None),
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
                mock_response.json = AsyncMock(
                    side_effect=Exception("HTTP connection failed")
                )
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
        assert (
            mapping is None
        ), "Mapping should be None when msgpack deserialization fails"

        await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_one_query_live_http() -> None:
    """
    Integration test: Run the 'simple-one' query against a live Dgraph HTTP instance.
    """

    qgraph_query: QueryGraphDict = qg(
        {
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
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "GO:0031410")) @cascade(vI_id, ~vI_object) {
            expand(vI_Node)
            in_edges_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "located_in")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_subject @filter(eq(vI_id, "NCBIGene:11276")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
        }
    }
    """).strip()

    # driver = new_http_driver(version="vI")
    driver = new_http_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=True,
        enable_subclass_edges=False,
    )
    assert transpiler.version == "vI"
    assert transpiler.prefix == "vI_"

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)

    # Minimal existence checks
    assert result.data, "No data returned from Dgraph for simple-one query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # Root node (NCBIGene:11276)
    root_node = result.data["q0"][0]

    # 1. Validate Root Node (NCBIGene:11276)
    assert root_node.binding == "n0"
    assert root_node.id == "GO:0031410"
    assert root_node.name == "cytoplasmic vesicle"

    expected_root_cats = sorted(
        [
            "NamedThing",
            "OrganismalEntity",
            "PhysicalEssence",
            "PhysicalEssenceOrOccurrent",
            "CellularComponent",
            "ThingWithTaxon",
            "SubjectOfInvestigation",
            "AnatomicalEntity",
            "BiologicalEntity",
        ]
    )
    assert sorted(root_node.category) == expected_root_cats

    # Validate attributes dictionary
    root_attrs = root_node.attributes
    assert root_attrs.get("information_content") == 56.8
    assert root_attrs.get("equivalent_identifiers") == ["GO:0031410"]

    # 2. Validate Edge
    assert len(root_node.edges) == 1
    edge = root_node.edges[0]
    assert edge.binding == "e0"
    assert edge.direction == "in"
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
    assert connected_node.binding == "n1"
    assert connected_node.id == "NCBIGene:11276"
    assert connected_node.name == "SYNRG"
    assert connected_node.edges == []

    expected_go_cats = sorted(
        [
            "MacromolecularMachineMixin",
            "NamedThing",
            "Gene",
            "ChemicalEntityOrProteinOrPolypeptide",
            "PhysicalEssence",
            "PhysicalEssenceOrOccurrent",
            "OntologyClass",
            "ChemicalEntityOrGeneOrGeneProduct",
            "GeneOrGeneProduct",
            "Polypeptide",
            "ThingWithTaxon",
            "GenomicEntity",
            "GeneProductMixin",
            "Protein",
            "BiologicalEntity",
        ]
    )
    assert sorted(connected_node.category) == expected_go_cats

    # Validate connected node attributes
    go_attrs = connected_node.attributes
    assert go_attrs.get("description") == "synergin gamma"
    assert go_attrs.get("full_name") == "synergin gamma"
    assert go_attrs.get("symbol") == "SYNRG"
    assert go_attrs.get("taxon") == "NCBITaxon:9606"
    assert go_attrs.get("in_taxon") == ["NCBITaxon:9606"]
    assert go_attrs.get("information_content") == 83.1

    expected_equiv_ids = sorted(
        [
            "PR:Q9UMZ2",
            "OMIM:607291",
            "UniProtKB:Q9UMZ2",
            "ENSEMBL:ENSG00000275066",
            "UMLS:C1412437",
            "UMLS:C0893518",
            "MESH:C121510",
            "HGNC:557",
            "NCBIGene:11276",
        ]
    )
    assert sorted(go_attrs.get("equivalent_identifiers", [])) == expected_equiv_ids

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_simple_one_query_live_grpc() -> None:
    """
    Integration test: Run the 'simple-one' query against a live Dgraph HTTP instance.
    """

    qgraph_query: QueryGraphDict = qg(
        {
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
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vI_id, "NCBIGene:11276")) @cascade(vI_id, ~vI_subject) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "located_in")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n0: vI_object @filter(eq(vI_id, "GO:0031410")) @cascade(vI_id) {
                    expand(vI_Node)
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
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=True,
        enable_subclass_edges=False,
    )
    assert transpiler.version == "vI"
    assert transpiler.prefix == "vI_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
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

    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
                "n1": {"ids": ["DOID:0070271"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "object": "n0",
                    "subject": "n1",
                    "predicates": ["biolink:has_phenotype"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vI_id, "DOID:0070271")) @cascade(vI_id, ~vI_subject) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "has_phenotype")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n0: vI_object @filter(eq(vI_category, "NamedThing")) @cascade(vI_id) {
                    expand(vI_Node)
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
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=True,
        enable_subclass_edges=False,
    )
    assert transpiler.version == "vI"
    assert transpiler.prefix == "vI_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)

    # Assertions to check that some data is returned
    assert result.data, "No data returned from Dgraph for simple-one query"
    assert "q0" in result.data
    assert len(result.data["q0"]) == 1

    # 2. Assertions for the root node (n0)
    root_node = result.data["q0"][0]
    assert root_node.binding == "n1"
    assert root_node.id == "DOID:0070271"
    root_node_edges_count = len(root_node.edges)
    assert root_node_edges_count == 1

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

    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
                "n1": {"ids": ["NCBIGene:3778"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "object": "n0",
                    "subject": "n1",
                    "predicates": ["biolink:related_to"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vI_id, "NCBIGene:3778")) @cascade(vI_id) {
            expand(vI_Node)

            out_edges_e0: ~vI_subject
            @filter(eq(vI_predicate_ancestors, "related_to"))
            @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }

                node_n0: vI_object
                @filter(eq(vI_category, "NamedThing"))
                @cascade(vI_id) {
                    expand(vI_Node)
                }
            }

            in_edges-symmetric_e0: ~vI_object
            @filter(eq(vI_predicate_ancestors, "related_to"))
            @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }

                node_n0: vI_subject
                @filter(eq(vI_category, "NamedThing"))
                @cascade(vI_id) {
                    expand(vI_Node)
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
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=True,
        enable_subclass_edges=False,
    )
    assert transpiler.version == "vI"
    assert transpiler.prefix == "vI_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Run the query against the live Dgraph instance
    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query_match, transpiler=transpiler
    )
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
    assert root_node_edges_count == 809

    # Both out_edges_e0 and in_edges-symmetric_e0 are merged under binding "e0"
    e0_edges = [e for e in root_node.edges if e.binding == "e0"]
    e0_edges_count = len(e0_edges)
    assert e0_edges_count == 809, "Expected 809 total edges for binding 'e0' (merged)"

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
    qgraph_query: QueryGraphDict = qg(
        {
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
        }
    )

    driver = new_grpc_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version
    )

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

    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0_test!@#": {"categories": ["biolink:NamedThing"], "constraints": []},
                "n1": {"ids": ["NCBIGene:3778"], "constraints": []},
            },
            "edges": {
                "e0_bad$%^": {
                    "object": "n0_test!@#",
                    "subject": "n1",
                    "predicates": ["biolink:related_to"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    # Expected query should use normalized edge ID 'e0', not 'e0_bad$%^'
    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vI_id, "NCBIGene:3778")) @cascade(vI_id) {
            expand(vI_Node)

            out_edges_e0: ~vI_subject
            @filter(eq(vI_predicate_ancestors, "related_to"))
            @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }

                node_n0: vI_object
                @filter(eq(vI_category, "NamedThing"))
                @cascade(vI_id) {
                    expand(vI_Node)
                }
            }

            in_edges-symmetric_e0: ~vI_object
            @filter(eq(vI_predicate_ancestors, "related_to"))
            @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }

                node_n0: vI_subject
                @filter(eq(vI_category, "NamedThing"))
                @cascade(vI_id) {
                    expand(vI_Node)
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
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=True,
        enable_subclass_edges=False,
    )
    assert transpiler.version == "vI"
    assert transpiler.prefix == "vI_"

    # Use the transpiler to generate the Dgraph query
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)

    # Verify the query uses normalized edge ID 'e0', not 'e0_bad$%^'
    assert_query_equals(dgraph_query, dgraph_query_match)
    assert "out_edges_e0:" in dgraph_query, "Query should use normalized edge ID 'e0'"
    assert (
        "in_edges-symmetric_e0:" in dgraph_query
    ), "Symmetric edge should use normalized ID 'e0'"
    assert (
        "e0_bad$%^" not in dgraph_query
    ), "Original edge ID 'e0_bad$%^' should not appear in query"

    # Run the query against the live Dgraph instance, passing transpiler for ID mapping
    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
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
    assert (
        len(e0_bad_edges) > 0
    ), "Edges should have original binding 'e0_bad$%^' restored from normalization"

    # Verify no edges have the normalized binding 'e0'
    e0_edges = [e for e in root_node.edges if e.binding == "e0"]
    assert len(e0_edges) == 0, "No edges should have normalized binding 'e0' in results"

    # Both forward (out) and reverse (in) edge groups should be present
    out_edges = [e for e in e0_bad_edges if e.direction == "out"]
    in_edges = [e for e in e0_bad_edges if e.direction == "in"]

    assert out_edges, "Expected at least one outgoing edge with binding 'e0_bad$%^'"
    assert (
        in_edges
    ), "Expected at least one incoming edge with binding 'e0_bad$%^' (symmetric)"

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
    qgraph_query: QueryGraphDict = qg(
        {
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
        }
    )

    driver = new_http_driver()
    await driver.connect()

    # Get the active Dgraph schema version
    dgraph_schema_version = await driver.get_active_version()

    # Initialize the transpiler with the detected version
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version
    )

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
    with pytest.raises(
        ConnectionError, match="Dgraph gRPC query failed: Some other gRPC error"
    ):
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
    with pytest.raises(
        ConnectionError, match="Dgraph gRPC query failed: while running ToJson"
    ):
        await driver.run_query("any query", transpiler=mock_transpiler)

    await driver.close()


# ---------------------------------------------------------------------------
# Subclassing live tests
#
# ── Case 1, Form B (ID→P→ID, source subclass) ───────────────────────────
#   A  = GO:0051055  (negative regulation of lipid biosynthetic process)
#   A' = GO:0031393  (negative regulation of prostaglandin biosynthetic process)
#          A' subclass_of A;  A' → genetic_association → B
#   B  = EFO:0004528 (mean corpuscular hemoglobin concentration)
#   No direct A → genetic_association → B edge exists.
#
# ── Case 1, Form C (ID→P→ID, target subclass) ───────────────────────────
#   A  = CHEBI:4042  (Cypermethrin, SmallMolecule)
#   B' = GO:0031393  (negative regulation of prostaglandin biosynthetic process)
#          A → affects → B' (direct);  B' subclass_of B
#   B  = GO:0051055  (negative regulation of lipid biosynthetic process)
#   No direct A → affects → B edge exists.
#
# ── Case 2 (ID→P→CAT, source subclass) ──────────────────────────────────
#   A   = UMLS:C3273258 (Congenital Systemic Disorder)  — no direct non-subclass edges
#   A'  = MONDO:0018551 (patent urachus)  subclass_of A
#          A' → has_phenotype → HP:0034267 (PhenotypicFeature)
#   CAT = biolink:PhenotypicFeature
#
# ── Case 3 (CAT→P→ID, target subclass) ──────────────────────────────────
#   INTERMEDIATE = GO:0031393 (has IDs)
#          → genetic_association → EFO:0004528 (B)
#          → subclass_of         → GO:0051055  (RESULT_N0, BiologicalProcess)
#   n0 = {categories: ["biolink:BiologicalProcess"]}
#   n1 = {ids: ["EFO:0004528"]}
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case1_direct_path_absent_live_grpc() -> None:
    """
    Baseline: confirm GO:0051055 has no direct genetic_association to EFO:0004528.

    This establishes the precondition for the subclass Form B test: without
    subclass expansion the query must return zero results.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"ids": ["GO:0051055"], "constraints": []},
                "n1": {"ids": ["EFO:0004528"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:genetic_association"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "GO:0051055")) @cascade(vI_id, ~vI_subject) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_object @filter(eq(vI_id, "EFO:0004528")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=False,
    )

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Sanity-check: no subclass aliases in the generated DQL
    assert "subclassB" not in dgraph_query
    assert "subclassC" not in dgraph_query
    assert "subclassD" not in dgraph_query

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    assert result.data is not None

    # No direct edge exists → @cascade removes the root node → empty result
    nodes = result.data.get("q0", [])
    assert len(nodes) == 0, (
        "Expected 0 results: GO:0051055 has no direct genetic_association "
        "to EFO:0004528 — this is the baseline for the subclass Form B test"
    )

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case1_form_b_live_grpc() -> None:
    """
    Integration test: Case 1, Form B subclass expansion (gRPC).

    Query:  GO:0051055 → genetic_association → EFO:0004528
    There is no direct edge between these nodes (see baseline test above).

    GO:0031393 (negative regulation of prostaglandin biosynthetic process) is a
    subclass_of GO:0051055 AND has a genetic_association edge → EFO:0004528.

    With subclass expansion enabled (Form B):
        A' ← subclass_of ← A ; A' → P → B
    the transpiler should generate the following additional traversal:
        GO:0051055 ← subclass_of subject ← GO:0031393 → genetic_association → EFO:0004528

    The result must surface GO:0051055 as a root node with the intermediate
    subclass edge (binding "e0", predicate "subclass_of") whose nested node
    GO:0031393 carries the actual genetic_association edge to EFO:0004528.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"ids": ["GO:0051055"], "constraints": []},
                "n1": {"ids": ["EFO:0004528"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:genetic_association"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "GO:0051055")) @cascade(vI_id) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_object @filter(eq(vI_id, "EFO:0004528")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
            in_edges-subclassB_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate: vI_subject @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassB-mid_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_n1: vI_object @filter(eq(vI_id, "EFO:0004528")) @cascade(vI_id) {
                            expand(vI_Node)
                        }
                    }
                }
            }
            out_edges-subclassC_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate: vI_object @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassC-tail_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_n1: vI_object @filter(eq(vI_id, "EFO:0004528")) @cascade(vI_id) {
                            expand(vI_Node)
                        }
                    }
                }
            }
            in_edges-subclassD_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate_A: vI_subject @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassD-mid_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_intermediate_B: vI_object @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                            expand(vI_Node)
                            out_edges-subclassD-tail_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_object) {
                                expand(vI_Edge) { vI_sources expand(vI_Source) }
                                node_n1: vI_object @filter(eq(vI_id, "EFO:0004528")) @cascade(vI_id) {
                                    expand(vI_Node)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=True,
    )

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)

    # Query must include all three Case 1 subclass form aliases (no version prefix on aliases)
    assert "in_edges-subclassB_e0" in dgraph_query, "Missing Form B alias"
    assert "out_edges-subclassC_e0" in dgraph_query, "Missing Form C alias"
    assert "in_edges-subclassD_e0" in dgraph_query, "Missing Form D alias"

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    assert result.data, "No data returned from Dgraph for subclass Form B query"
    assert "q0" in result.data

    nodes = result.data["q0"]
    assert len(nodes) == 1, "Expected 1 root node (GO:0051055)"

    root_node = nodes[0]
    assert root_node.binding == "n0"
    assert root_node.id == "GO:0051055"
    assert root_node.name == "negative regulation of lipid biosynthetic process"

    # Form B produces an incoming subclass_of edge on the root, binding "e0"
    subclass_b_edges = [
        e
        for e in root_node.edges
        if e.binding == "e0" and e.direction == "in" and e.predicate == "subclass_of"
    ]
    assert subclass_b_edges, (
        "Expected at least one incoming subclass_of edge (Form B) with binding 'e0'"
    )

    # The intermediate node A' (GO:0031393) must appear as the connected node
    # of the subclass_of edge, and must itself carry the genetic_association → EFO:0004528
    aprime_ids = {e.node.id for e in subclass_b_edges}
    assert "GO:0031393" in aprime_ids, (
        "Expected GO:0031393 (negative regulation of prostaglandin biosynthetic process) "
        "as the intermediate subclass node A'"
    )

    aprime_node = next(e.node for e in subclass_b_edges if e.node.id == "GO:0031393")

    # The genetic_association edge from A' → B must be present on the intermediate node
    ga_edges = [
        e
        for e in aprime_node.edges
        if e.binding == "e0"
        and e.direction == "out"
        and e.predicate == "genetic_association"
    ]
    assert ga_edges, (
        "Expected at least one genetic_association edge (binding 'e0', direction 'out') "
        "on the intermediate node GO:0031393"
    )

    target_ids = {e.node.id for e in ga_edges}
    assert "EFO:0004528" in target_ids, (
        "Expected EFO:0004528 (mean corpuscular hemoglobin concentration) as the "
        "target node reached via Form B subclass expansion"
    )

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case1_form_c_baseline_live_grpc() -> None:
    """
    Baseline: confirm CHEBI:4042 has no direct affects edge to GO:0051055.

    CHEBI:4042 (Cypermethrin) has a direct affects edge to GO:0031393 (B'),
    and GO:0031393 is a subclass_of GO:0051055 (B). Without subclass
    expansion the query must return zero results.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"ids": ["CHEBI:4042"], "constraints": []},
                "n1": {"ids": ["GO:0051055"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "CHEBI:4042")) @cascade(vI_id, ~vI_subject) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "affects")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_object @filter(eq(vI_id, "GO:0051055")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=False,
    )

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)
    assert "subclassC" not in dgraph_query

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    nodes = result.data.get("q0", []) if result.data else []
    assert len(nodes) == 0, (
        "Expected 0 results: CHEBI:4042 has no direct affects edge to GO:0051055"
    )

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case1_form_c_live_grpc() -> None:
    """
    Integration test: Case 1, Form C subclass expansion (gRPC).

    Query:  CHEBI:4042 → affects → GO:0051055
    There is no direct edge (see baseline test above).

    GO:0031393 (negative regulation of prostaglandin biosynthetic process) is
    a subclass_of GO:0051055 AND CHEBI:4042 has a direct affects edge to it.

    With subclass expansion enabled (Form C):
        A → P → B' ; B' → subclass_of → B
    the transpiler generates an additional traversal:
        CHEBI:4042 → ~subject → affects → object → GO:0031393 (B')
        GO:0031393 → ~subject → subclass_of → object → GO:0051055 (B)

    The result must surface CHEBI:4042 as the root node, with GO:0031393 as
    the intermediate B' node (reached via the affects edge) carrying the
    subclass_of tail edge that resolves to GO:0051055.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"ids": ["CHEBI:4042"], "constraints": []},
                "n1": {"ids": ["GO:0051055"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "CHEBI:4042")) @cascade(vI_id) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "affects")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_object @filter(eq(vI_id, "GO:0051055")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
            in_edges-subclassB_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate: vI_subject @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassB-mid_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "affects")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_n1: vI_object @filter(eq(vI_id, "GO:0051055")) @cascade(vI_id) {
                            expand(vI_Node)
                        }
                    }
                }
            }
            out_edges-subclassC_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "affects")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate: vI_object @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassC-tail_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_n1: vI_object @filter(eq(vI_id, "GO:0051055")) @cascade(vI_id) {
                            expand(vI_Node)
                        }
                    }
                }
            }
            in_edges-subclassD_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate_A: vI_subject @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassD-mid_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "affects")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_intermediate_B: vI_object @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                            expand(vI_Node)
                            out_edges-subclassD-tail_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_object) {
                                expand(vI_Edge) { vI_sources expand(vI_Source) }
                                node_n1: vI_object @filter(eq(vI_id, "GO:0051055")) @cascade(vI_id) {
                                    expand(vI_Node)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=True,
    )

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)
    assert "out_edges-subclassC_e0" in dgraph_query, "Missing Form C alias"
    assert "in_edges-subclassB_e0" in dgraph_query, "Missing Form B alias"
    assert "in_edges-subclassD_e0" in dgraph_query, "Missing Form D alias"

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    assert result.data, "No data returned"
    assert "q0" in result.data

    nodes = result.data["q0"]
    assert len(nodes) == 1, "Expected 1 root node (CHEBI:4042)"

    root_node = nodes[0]
    assert root_node.binding == "n0"
    assert root_node.id == "CHEBI:4042"
    assert root_node.name == "Cypermethrin"

    # Form C: A → affects → B' (intermediate); B' → subclass_of → B
    # The intermediate node is B' = GO:0031393, reached via outgoing affects edge
    affects_edges = [
        e
        for e in root_node.edges
        if e.binding == "e0" and e.direction == "out" and e.predicate == "affects"
    ]
    assert affects_edges, "Expected outgoing affects edges (Form C) with binding 'e0'"

    intermediate_ids = {e.node.id for e in affects_edges}
    assert "GO:0031393" in intermediate_ids, (
        "Expected GO:0031393 as intermediate B' node in Form C affects traversal"
    )

    bprime_node = next(e.node for e in affects_edges if e.node.id == "GO:0031393")

    # B' must carry the subclass_of tail edge → GO:0051055
    subclass_tail_edges = [
        e
        for e in bprime_node.edges
        if e.binding == "e0" and e.direction == "out" and e.predicate == "subclass_of"
    ]
    assert subclass_tail_edges, (
        "Expected outgoing subclass_of edge on GO:0031393 (B') with binding 'e0'"
    )

    tail_ids = {e.node.id for e in subclass_tail_edges}
    assert "GO:0051055" in tail_ids, (
        "Expected GO:0051055 (negative regulation of lipid biosynthetic process) "
        "as the target B reached via Form C subclass expansion"
    )

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case2_id_to_cat_baseline_live_grpc() -> None:
    """
    Baseline: confirm UMLS:C3273258 has no direct has_phenotype edges.

    UMLS:C3273258 (Congenital Systemic Disorder) has no direct non-subclass
    edges. Without subclass expansion, querying it for PhenotypicFeature
    phenotypes must return zero results.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"ids": ["UMLS:C3273258"], "constraints": []},
                "n1": {"categories": ["biolink:PhenotypicFeature"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:has_phenotype"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "UMLS:C3273258")) @cascade(vI_id, ~vI_subject) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "has_phenotype")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_object @filter(eq(vI_category, "PhenotypicFeature")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=False,
    )

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)
    assert "subclassB" not in dgraph_query

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    nodes = result.data.get("q0", []) if result.data else []
    assert len(nodes) == 0, (
        "Expected 0 results: UMLS:C3273258 has no direct has_phenotype edges"
    )

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case2_id_to_cat_form_b_live_grpc() -> None:
    """
    Integration test: Case 2, Form B subclass expansion — ID → P → CAT (gRPC).

    Query:  UMLS:C3273258 → has_phenotype → CAT:PhenotypicFeature
    UMLS:C3273258 (Congenital Systemic Disorder) has zero direct non-subclass
    edges (see baseline test). However, its subclass child MONDO:0018551
    (patent urachus) has has_phenotype edges to PhenotypicFeature nodes
    (e.g. HP:0034267, HP:0000010).

    With subclass expansion enabled (Form B, Case 2):
        A' ← subclass_of ← A ; A' → P → CAT
    the transpiler adds:
        UMLS:C3273258 ← subclass_of ← MONDO:0018551 → has_phenotype → PhenotypicFeature

    The result must surface UMLS:C3273258 as the root with the incoming
    subclass_of edge leading to MONDO:0018551, which in turn carries the
    has_phenotype edges to PhenotypicFeature nodes.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"ids": ["UMLS:C3273258"], "constraints": []},
                "n1": {"categories": ["biolink:PhenotypicFeature"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:has_phenotype"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    dgraph_query_match: str = dedent("""
    {
        q0_node_n0(func: eq(vI_id, "UMLS:C3273258")) @cascade(vI_id) {
            expand(vI_Node)
            out_edges_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "has_phenotype")) @cascade(vI_predicate, vI_object) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n1: vI_object @filter(eq(vI_category, "PhenotypicFeature")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
            in_edges-subclassB_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate: vI_subject @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassB-mid_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "has_phenotype")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_n1: vI_object @filter(eq(vI_category, "PhenotypicFeature")) @cascade(vI_id) {
                            expand(vI_Node)
                        }
                    }
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=True,
    )

    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)
    # Case 2 generates Form B only (no Form C / D since target has no IDs)
    assert "in_edges-subclassB_e0" in dgraph_query, "Missing Form B alias"
    assert "out_edges-subclassC_e0" not in dgraph_query, "Unexpected Form C alias"
    assert "in_edges-subclassD_e0" not in dgraph_query, "Unexpected Form D alias"

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    assert result.data, "No data returned"
    assert "q0" in result.data

    nodes = result.data["q0"]
    assert len(nodes) == 1, "Expected 1 root node (UMLS:C3273258)"

    root_node = nodes[0]
    assert root_node.binding == "n0"
    assert root_node.id == "UMLS:C3273258"
    assert root_node.name == "Congenital Systemic Disorder"

    # Form B: A' ← subclass_of ← A ; root receives incoming subclass_of edges
    subclass_b_edges = [
        e
        for e in root_node.edges
        if e.binding == "e0" and e.direction == "in" and e.predicate == "subclass_of"
    ]
    assert subclass_b_edges, (
        "Expected incoming subclass_of edges (Form B) on UMLS:C3273258 with binding 'e0'"
    )

    # MONDO:0018551 (patent urachus) must be among the A' subclass children
    aprime_ids = {e.node.id for e in subclass_b_edges}
    assert "MONDO:0018551" in aprime_ids, (
        "Expected MONDO:0018551 (patent urachus) as a subclass child A'"
    )

    aprime_node = next(e.node for e in subclass_b_edges if e.node.id == "MONDO:0018551")

    # A' must carry has_phenotype edges reaching PhenotypicFeature nodes
    hp_edges = [
        e
        for e in aprime_node.edges
        if e.binding == "e0"
        and e.direction == "out"
        and e.predicate == "has_phenotype"
    ]
    assert hp_edges, (
        "Expected has_phenotype edges on MONDO:0018551 with binding 'e0'"
    )

    phenotype_ids = {e.node.id for e in hp_edges}
    # HP:0034267 and HP:0000010 are confirmed PhenotypicFeature nodes
    assert phenotype_ids & {"HP:0034267", "HP:0000010"}, (
        "Expected at least one of the known phenotype nodes (HP:0034267 or HP:0000010) "
        "to be reached via Form B subclass expansion from MONDO:0018551"
    )

    await driver.close()


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dgraph_config")
async def test_subclass_case3_cat_to_id_live_grpc() -> None:
    """
    Integration test: Case 3, Mirrored Form B subclass expansion — CAT → P → ID (gRPC).

    Query:  CAT:BiologicalProcess → genetic_association → EFO:0004528
    EFO:0004528 (mean corpuscular hemoglobin concentration) receives direct
    genetic_association edges from several BiologicalProcess nodes including
    GO:0031393 (found via Form A baseline). The Case 3 expansion additionally
    surfaces GO:0051055 (negative regulation of lipid biosynthetic process)
    because the intermediate GO:0031393 is a subclass_of GO:0051055.

    With subclass expansion enabled (Case 3 Mirrored Form B):
        INTERMEDIATE → P → B ; INTERMEDIATE → subclass_of → RESULT_N0(CAT)
    the transpiler adds:
        EFO:0004528 ← genetic_association ← GO:0031393
        GO:0031393 → subclass_of → GO:0051055 (BiologicalProcess)

    The result must surface EFO:0004528 as root (n1) with GO:0031393 as an
    intermediate and GO:0051055 discoverable as the n0 node via the subclass
    tail on GO:0031393. GO:0051055 must NOT appear in a query without
    subclass expansion as it has no direct genetic_association to EFO:0004528.
    """
    qgraph_query: QueryGraphDict = qg(
        {
            "nodes": {
                "n0": {"categories": ["biolink:BiologicalProcess"], "constraints": []},
                "n1": {"ids": ["EFO:0004528"], "constraints": []},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:genetic_association"],
                    "attribute_constraints": [],
                    "qualifier_constraints": [],
                }
            },
        }
    )

    baseline_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vI_id, "EFO:0004528")) @cascade(vI_id, ~vI_object) {
            expand(vI_Node)
            in_edges_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n0: vI_subject @filter(eq(vI_category, "BiologicalProcess")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
        }
    }
    """).strip()

    dgraph_query_match: str = dedent("""
    {
        q0_node_n1(func: eq(vI_id, "EFO:0004528")) @cascade(vI_id) {
            expand(vI_Node)
            in_edges_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_n0: vI_subject @filter(eq(vI_category, "BiologicalProcess")) @cascade(vI_id) {
                    expand(vI_Node)
                }
            }
            in_edges-subclassObjB_e0: ~vI_object @filter(eq(vI_predicate_ancestors, "genetic_association")) @cascade(vI_predicate, vI_subject) {
                expand(vI_Edge) { vI_sources expand(vI_Source) }
                node_intermediate: vI_subject @filter(has(vI_id)) @cascade(vI_id, ~vI_subject) {
                    expand(vI_Node)
                    out_edges-subclassObjB-tail_e0: ~vI_subject @filter(eq(vI_predicate_ancestors, "subclass_of")) @cascade(vI_predicate, vI_object) {
                        expand(vI_Edge) { vI_sources expand(vI_Source) }
                        node_n0: vI_object @filter(eq(vI_category, "BiologicalProcess")) @cascade(vI_id) {
                            expand(vI_Node)
                        }
                    }
                }
            }
        }
    }
    """).strip()

    driver = new_grpc_driver()
    await driver.connect()
    dgraph_schema_version = await driver.get_active_version()

    # ── 1. Baseline: confirm GO:0051055 absent without subclass expansion ──
    baseline_transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=False,
    )
    baseline_query: str = baseline_transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(baseline_query, baseline_query_match)
    assert "subclassObjB" not in baseline_query

    baseline_result: dg_models.DgraphResponse = await driver.run_query(
        baseline_query, transpiler=baseline_transpiler
    )
    assert isinstance(baseline_result, dg_models.DgraphResponse)
    baseline_n1_nodes = baseline_result.data.get("q0", []) if baseline_result.data else []
    # GO:0051055 must NOT be reachable directly
    baseline_all_n0_ids: set[str] = set()
    for root in baseline_n1_nodes:
        for e in root.edges:
            if e.binding == "e0":
                baseline_all_n0_ids.add(e.node.id)
    assert "GO:0051055" not in baseline_all_n0_ids, (
        "GO:0051055 should not appear as a direct n0 result (it has no direct "
        "genetic_association edge to EFO:0004528)"
    )

    # ── 2. Subclass expansion: GO:0051055 appears via Case 3 Mirrored Form B ──
    transpiler: _TestDgraphTranspiler = _TestDgraphTranspiler(
        version=dgraph_schema_version,
        enable_symmetric_edges=False,
        enable_subclass_edges=True,
    )
    dgraph_query: str = transpiler.convert_multihop_public(qgraph_query)
    assert_query_equals(dgraph_query, dgraph_query_match)
    assert "in_edges-subclassObjB_e0" in dgraph_query, "Missing Case 3 ObjB alias"
    assert "out_edges-subclassObjB-tail_e0" in dgraph_query, "Missing Case 3 ObjB tail alias"

    result: dg_models.DgraphResponse = await driver.run_query(
        dgraph_query, transpiler=transpiler
    )
    assert isinstance(result, dg_models.DgraphResponse)
    assert result.data, "No data returned"
    assert "q0" in result.data

    n1_nodes = result.data["q0"]
    assert len(n1_nodes) == 1, "Expected 1 root node (EFO:0004528)"

    root_node = n1_nodes[0]
    assert root_node.binding == "n1"
    assert root_node.id == "EFO:0004528"
    assert root_node.name == "mean corpuscular hemoglobin concentration"

    # Case 3 Mirrored Form B: root receives an incoming P edge whose subject is
    # INTERMEDIATE (GO:0031393), and INTERMEDIATE carries a subclass_of tail
    # edge whose object is RESULT_N0 = GO:0051055.
    intermediate_edges = [
        e
        for e in root_node.edges
        if e.binding == "e0"
        and e.direction == "in"
        and e.predicate == "genetic_association"
        and e.node.binding == "intermediate"
    ]
    assert intermediate_edges, (
        "Expected incoming genetic_association edges (Case 3 ObjB) with an "
        "intermediate node on the root EFO:0004528"
    )

    intermediate_ids = {e.node.id for e in intermediate_edges}
    assert "GO:0031393" in intermediate_ids, (
        "Expected GO:0031393 as the intermediate node in Case 3 ObjB expansion"
    )

    go031393_node = next(e.node for e in intermediate_edges if e.node.id == "GO:0031393")

    # The intermediate must carry the subclass_of tail edge → GO:0051055
    subclass_tail_edges = [
        e
        for e in go031393_node.edges
        if e.binding == "e0" and e.direction == "out" and e.predicate == "subclass_of"
    ]
    assert subclass_tail_edges, (
        "Expected outgoing subclass_of edge (Case 3 ObjB tail) on GO:0031393"
    )

    tail_n0_ids = {e.node.id for e in subclass_tail_edges}
    assert "GO:0051055" in tail_n0_ids, (
        "Expected GO:0051055 (negative regulation of lipid biosynthetic process) "
        "as RESULT_N0 reached via Case 3 Mirrored Form B subclass expansion"
    )

    await driver.close()
