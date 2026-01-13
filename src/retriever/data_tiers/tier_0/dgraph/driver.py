import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from enum import Enum
from http import HTTPStatus
from typing import Any, Protocol, TypedDict, cast, override
from urllib.parse import urljoin

import aiohttp
import grpc
import msgpack
import pydgraph
from aiohttp import ClientTimeout
from cachetools import TTLCache
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models
from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.types.dingo import DINGOMetadata
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity, Infores


class DgraphProtocol(str, Enum):
    """Enum for Dgraph connection protocols."""

    GRPC = "grpc"
    HTTP = "http"


class PydgraphResponse(Protocol):
    """Protocol for Dgraph response object returned by pydgraph."""

    @property
    def json(self) -> Any:
        """Return the JSON content of the response (bytes/str/dict depending on client)."""
        ...


class _GrpcFuture(Protocol):
    """Minimal protocol for the future returned by pydgraph async_query."""

    def result(self, timeout: float | None = None) -> Any:
        """Get the result of the future, with optional timeout."""
        ...


class DgraphClientStubProtocol(Protocol):
    """Protocol for Dgraph client stub."""

    def close(self) -> None:
        """Close the client stub."""
        ...


class DgraphTxnProtocol(Protocol):
    """Protocol for Dgraph transaction object."""

    def query(
        self,
        query: str,
        variables: Any = None,
        resp_format: str = "JSON",
    ) -> PydgraphResponse:
        """Run a synchronous query and return the response."""
        ...

    def async_query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        timeout: float | None = None,
        credentials: Any | None = None,
        resp_format: str = "JSON",
    ) -> _GrpcFuture:
        """Run an async query and return a future."""
        ...

    def discard(self) -> None:
        """Discard the transaction."""
        ...

    def handle_query_future(self, future: _GrpcFuture) -> PydgraphResponse:
        """Handle the future returned by an async query."""
        ...


class DgraphQueryResult(TypedDict, total=False):
    """TypedDict for legacy Dgraph query result (kept for backward compat where needed)."""

    data: Any
    errors: Any


class DgraphClientProtocol(Protocol):
    """Protocol for the pydgraph.DgraphClient."""

    def txn(
        self, read_only: bool = False, best_effort: bool = False
    ) -> DgraphTxnProtocol:
        """Create a new Dgraph transaction."""
        ...

    def check_version(self, timeout: float | None = None) -> Any:
        """Check the version of the Dgraph server."""
        ...

    def login_into_namespace(
        self, userid: str, password: str, namespace: int | None
    ) -> None:
        """Logs into a Dgraph namespace."""
        ...


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph supporting both gRPC and HTTP protocols."""

    settings: DgraphSettings
    protocol: DgraphProtocol
    endpoint: str
    query_timeout: float
    connect_retries: int
    version: str | None = None
    _client_stub: DgraphClientStubProtocol | None = None
    _client: DgraphClientProtocol | None = None
    _http_session: aiohttp.ClientSession | None = None
    _access_token: str | None = None
    _token_expiry: datetime | None = None
    _failed: bool = False
    _version_cache: TTLCache[str, str | None] = TTLCache(maxsize=1, ttl=60)
    _mapping_cache: TTLCache[str, dict[str, Any] | None] = TTLCache(maxsize=1, ttl=300)

    def __init__(
        self,
        protocol: DgraphProtocol = DgraphProtocol.GRPC,
        *,
        version: str | None = None,
    ) -> None:
        """Initialize the Dgraph driver with connection settings.

        Args:
            protocol: The protocol to use (gRPC or HTTP)
            version: An optional, fixed schema version to use, bypassing auto-detection.
                     This parameter has the highest precedence.
        """
        self.settings = CONFIG.tier0.dgraph
        self.protocol = protocol
        self.query_timeout = self.settings.query_timeout
        self.connect_retries = self.settings.connect_retries

        # Version precedence: constructor arg > env var > auto-detect from DB
        if version is not None:
            self.version = version
            log.debug(f"Using schema version from constructor parameter: {version}")
        elif self.settings.preferred_version:
            self.version = self.settings.preferred_version
            log.debug(
                "Using schema version from TIER0__DGRAPH__PREFERRED_VERSION env var: "
                + f"{self.version}"
            )
        else:
            self.version = None

        # Set endpoint based on protocol
        if protocol == DgraphProtocol.GRPC:
            self.endpoint = self.settings.grpc_endpoint
        else:  # HTTP
            self.endpoint = self.settings.http_endpoint

    def clear_version_cache(self) -> None:
        """Clears the internal schema version cache. Primarily for testing."""
        log.debug("Clearing Dgraph schema version cache.")
        self._version_cache.clear()

    def clear_mapping_cache(self) -> None:
        """Clears the internal schema mapping cache. Primarily for testing."""
        log.debug("Clearing Dgraph schema mapping cache.")
        self._mapping_cache.clear()

    async def get_active_version(self) -> str | None:
        """Queries Dgraph for the active schema version and caches the result.

        If a version was provided at initialization, it will be returned directly.

        This method implements manual caching to be async-safe.
        """
        # If a version was manually set on the driver, always use it.
        if self.version:
            log.debug(f"Using manually specified Dgraph schema version: {self.version}")
            return self.version

        # Manually check the cache first
        try:
            cached_version = self._version_cache["active_version"]
            log.debug(f"Returning cached Dgraph schema version: {cached_version}")
            return cached_version
        except KeyError:
            log.debug("Querying for active Dgraph schema version (cache miss)...")
            # Not in cache, proceed to fetch

        query = """
            query active_version() {
              versions(func: type(SchemaMetadata)) @filter(eq(schema_metadata_is_active, true)) {
                schema_metadata_version
              }
            }
        """
        txn: DgraphTxnProtocol | None = None
        try:
            versions = []
            if self.protocol == DgraphProtocol.GRPC:
                assert self._client is not None, "gRPC client not initialized"
                txn = self._client.txn(read_only=True)
                response = await asyncio.to_thread(txn.query, query)
                raw_data = json.loads(response.json)
                log.debug(f"Dgraph version query raw data: {raw_data}")
                versions = raw_data.get("versions", [])
            else:  # HTTP
                assert self._http_session is not None, "HTTP session not initialized"
                async with self._http_session.post(
                    urljoin(self.endpoint, "/query"),
                    json={"query": query},
                    timeout=ClientTimeout(total=self.query_timeout),
                ) as response:
                    raw_data = await response.json()
                    versions = raw_data.get("data").get("versions", [])

            version = versions[0].get("schema_metadata_version") if versions else None

            if version:
                log.info(f"Found and cached active Dgraph schema version: {version}")
            else:
                log.warning(
                    "No active Dgraph schema version found. Caching null result."
                )

            # Manually store the result in the cache before returning
            self._version_cache["active_version"] = version
            return version
        except Exception as e:
            log.error(f"Failed to query for active Dgraph schema version: {e}")
            # Cache the failure as None to prevent retrying on every call
            self._version_cache["active_version"] = None
            return None
        finally:
            # Only discard the transaction if it was successfully created
            if self.protocol == DgraphProtocol.GRPC and txn:
                await asyncio.to_thread(txn.discard)

    async def fetch_mapping_from_db(self, version: str) -> dict[str, Any] | None:
        """Helper to fetch and deserialize mapping from Dgraph.

        Args:
            version: The schema version to fetch mapping for

        Returns:
            Deserialized mapping dictionary, or None if not found or on error.
        """
        query = f"""
            query schema_mapping() {{
                metadata(func: type(SchemaMetadata))
                    @filter(eq(schema_metadata_version, "{version}")) {{
                    schema_metadata_mapping
                }}
            }}
        """
        txn: DgraphTxnProtocol | None = None
        try:
            metadata_list = []
            if self.protocol == DgraphProtocol.GRPC:
                assert self._client is not None, "gRPC client not initialized"
                txn = self._client.txn(read_only=True)
                response = await asyncio.to_thread(txn.query, query)
                raw_data = json.loads(response.json)
                metadata_list = raw_data.get("metadata", [])
            else:  # HTTP
                assert self._http_session is not None, "HTTP session not initialized"
                async with self._http_session.post(
                    urljoin(self.endpoint, "/query"),
                    json={"query": query},
                    timeout=ClientTimeout(total=self.query_timeout),
                ) as response:
                    raw_data = await response.json()
                    metadata_list = raw_data.get("data", {}).get("metadata", [])

            if not metadata_list:
                log.warning(f"No schema metadata found for version '{version}'")
                return None

            # Get the msgpack-encoded mapping
            mapping_blob = metadata_list[0].get("schema_metadata_mapping")
            if not mapping_blob:
                log.warning(
                    f"schema_metadata_mapping field is empty for version '{version}'"
                )
                return None

            # Dgraph may return the blob as a base64-encoded string or raw bytes
            # depending on how it was stored. Try to handle both cases.
            mapping_bytes: bytes
            if isinstance(mapping_blob, str):
                # If it's a string, it might be base64-encoded

                try:
                    mapping_bytes = base64.b64decode(mapping_blob)
                except Exception:
                    # If base64 decode fails, assume it's UTF-8 encoded msgpack
                    mapping_bytes = mapping_blob.encode("utf-8")
            else:
                mapping_bytes = mapping_blob

            # Deserialize msgpack - explicitly type the result
            mapping = cast(dict[str, Any], msgpack.unpackb(mapping_bytes, raw=False))
            log.info(
                f"Successfully retrieved schema metadata mapping for version '{version}'"
            )
            return mapping

        except Exception as e:
            log.error(
                f"Failed to retrieve schema metadata mapping for version '{version}': {e}"
            )
            return None
        finally:
            if self.protocol == DgraphProtocol.GRPC and txn:
                await asyncio.to_thread(txn.discard)

    @override
    async def get_metadata(self) -> dict[str, Any] | None:
        """Queries Dgraph for the active schema's metadata mapping.

        The mapping is stored as a msgpack-serialized JSON blob in the
        schema_metadata_mapping field. This method retrieves and deserializes it
        for the active schema version.

        The result is cached per-version with a 5-minute TTL.

        Returns:
            Deserialized mapping dictionary, or None if not found or on error.
        """
        # Get the active version (respects manual version, env var, or DB query)
        version = await self.get_active_version()
        if not version:
            log.warning(
                "Cannot retrieve schema metadata mapping: no active version found"
            )
            return None

        # Check cache first (keyed by version)
        cache_key = f"mapping_{version}"
        try:
            cached_mapping = self._mapping_cache[cache_key]
            log.debug(
                f"Returning cached schema metadata mapping for version '{version}'"
            )
            return cached_mapping
        except KeyError:
            log.debug(
                f"Fetching schema metadata mapping for version '{version}' (cache miss)..."
            )

        # Fetch from database
        mapping = await self.fetch_mapping_from_db(version)

        # Cache the result (whether successful or None)
        self._mapping_cache[cache_key] = mapping
        return mapping

    @override
    async def get_operations(
        self,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        metadata = await self.get_metadata()
        if metadata is None:
            raise ValueError(
                "Unable to obtain metadata from backend, cannot parse operations."
            )
        infores = Infores(CONFIG.tier0.backend_infores)
        operations, nodes = parse_dingo_metadata(DINGOMetadata(**metadata), 0, infores)
        log.success(f"Parsed {infores} as a Tier 0 resource.")
        return operations, nodes

    @override
    async def connect(self, retries: int = 0) -> None:
        """Connect to Dgraph using selected protocol."""
        log.info("Checking Dgraph connection...")
        try:
            if self.protocol == DgraphProtocol.GRPC:
                await self._connect_grpc()
            else:
                await self._connect_http()
            log.success(f"Dgraph {self.protocol} connection successful!")

            # Populate version cache after successful connect so subsequent
            # code can use get_active_version without triggering a query.
            try:
                await self.get_active_version()
            except Exception as e:
                log.warning(f"Unable to fetch active schema version after connect: {e}")

        except Exception as e:
            await self._cleanup_connections()
            if retries < self.connect_retries:
                await asyncio.sleep(1)
                log.error(f"""
                    Could not establish connection to Dgraph via {self.protocol},
                    trying again... retry {retries + 1}
                """)
                await self.connect(retries + 1)
            else:
                log.error(f"Could not establish connection to Dgraph, error: {e}")
                self._failed = True
                raise e

    async def _connect_grpc(self) -> None:
        """Establish gRPC connection to Dgraph."""
        grpc_options = [
            (
                "grpc.max_send_message_length",
                self.settings.grpc_max_send_message_length,
            ),
            (
                "grpc.max_receive_message_length",
                self.settings.grpc_max_receive_message_length,
            ),
        ]
        self._client_stub = pydgraph.DgraphClientStub(
            self.endpoint, options=grpc_options
        )
        self._client = cast(
            DgraphClientProtocol, cast(object, pydgraph.DgraphClient(self._client_stub))
        )

        try:
            assert self._client is not None

            # If username and password are provided, attempt to log in.
            if self.settings.username and self.settings.password:
                username = self.settings.username
                password = self.settings.password.get_secret_value()
                namespace = self.settings.namespace

                log.info(
                    f"Attempting Dgraph login for user '{username}' "
                    + (f"in namespace {namespace}" if namespace is not None else "")
                )

                # pydgraph login is synchronous, so run it in a separate thread.
                await asyncio.to_thread(
                    self._client.login_into_namespace,
                    username,
                    password,
                    namespace,
                )
                log.success(f"Dgraph login successful for user '{username}'.")

            # The pydgraph client is synchronous, so `check_version()` is a blocking
            # network call. We use `asyncio.to_thread()` to run it in a separate
            # thread, which prevents it from freezing the main application's event
            # loop. `asyncio.wait_for()` is used to enforce a timeout.
            await asyncio.wait_for(
                asyncio.to_thread(self._client.check_version),
                timeout=self.query_timeout,
            )
        except Exception as e:
            # If verification fails, clean up immediately and raise a clear error.
            await self._cleanup_connections()
            raise ConnectionError(
                f"Failed to verify gRPC connection to {self.endpoint}"
            ) from e

    async def _http_login(self) -> None:
        """Logs into Dgraph over HTTP and stores the access token."""
        if not (self.settings.username and self.settings.password):
            return

        assert self._http_session is not None, "HTTP session not initialized for login"

        username = self.settings.username
        password = self.settings.password.get_secret_value()
        namespace = self.settings.namespace

        log.info(
            f"Attempting Dgraph HTTP login for user '{username}'"
            + (f" in namespace {namespace}" if namespace is not None else "")
        )

        login_mutation = {
            "query": f"""
                mutation Login {{
                    login(userId: "{username}", password: "{password}"{f", namespace: {namespace}" if namespace else ""}) {{
                        response {{
                            accessJWT
                        }}
                    }}
                }}
            """
        }

        try:
            async with self._http_session.post(
                urljoin(self.endpoint, "/admin"),
                json=login_mutation,
                timeout=ClientTimeout(total=self.query_timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if "errors" in data:
                    raise RuntimeError(f"Dgraph login failed: {data['errors']}")

                self._access_token = data["data"]["login"]["response"]["accessJWT"]
                assert self._access_token is not None, "Access token should not be None after successful login"

                # Decode JWT payload to get expiration time
                # A JWT is three parts: header.payload.signature
                payload_b64 = self._access_token.split(".")[1]
                # Add padding for correct base64 decoding
                payload_b64 += "=" * (-len(payload_b64) % 4)
                payload = json.loads(base64.b64decode(payload_b64))

                # Set expiry with a 60-second buffer to be safe
                self._token_expiry = datetime.fromtimestamp(
                    payload["exp"], tz=UTC
                ) - timedelta(seconds=60)

                # Update the session headers for subsequent requests
                self._http_session.headers[
                    "X-Dgraph-AccessToken"
                ] = self._access_token
                log.success(f"Dgraph HTTP login successful for user '{username}'.")

        except Exception as e:
            self._access_token = None
            self._token_expiry = None
            raise ConnectionError("Dgraph HTTP login failed.") from e

    async def _connect_http(self) -> None:
        """Establish HTTP connection to Dgraph."""
        self._http_session = aiohttp.ClientSession()

        # If auth is configured, perform login
        if self.settings.username and self.settings.password:
            await self._http_login()

        # Test connection with a simple query
        query = "{ health { status } }"
        async with self._http_session.post(
            urljoin(self.endpoint, "/graphql"),
            json={"query": query},
            timeout=ClientTimeout(total=self.query_timeout),

        ) as response:
            if response.status != HTTPStatus.OK:
                text = await response.text()
                raise ConnectionError(
                    f"HTTP connection failed with status {response.status}: {text}"
                )

    @override
    async def run_query(
        self, query: str, *args: Any, **kwargs: Any
    ) -> dg_models.DgraphResponse:
        """Execute a query against the Dgraph database and parse into dataclasses.

        Args:
            query: The Dgraph query string to execute
            *args: Variable positional arguments (unused, for protocol compatibility)
            **kwargs: Additional arguments:
                - transpiler: DgraphTranspiler instance for ID mapping (REQUIRED)

        Returns:
            Parsed DgraphResponse with bindings converted back to original IDs

        Raises:
            ValueError: If transpiler is not provided in kwargs
        """
        if self.protocol == DgraphProtocol.GRPC and not self._client:
            raise RuntimeError(
                "DgraphDriver (gRPC) not connected. Call connect() first."
            )
        if self.protocol == DgraphProtocol.HTTP and not self._http_session:
            raise RuntimeError(
                "DgraphDriver (HTTP) not connected. Call connect() first."
            )

        # Extract transpiler and validate it's provided
        transpiler = kwargs.get("transpiler")
        if not transpiler:
            raise ValueError("""
                transpiler is required in run_query() to properly map normalized IDs back to original IDs. "
                Pass transpiler=<DgraphTranspiler instance> in kwargs."
            """)

        otel_span = trace.get_current_span()
        if otel_span and otel_span.is_recording():
            otel_span.add_event(
                "dgraph_query_start", attributes={"dgraph_query": query}
            )
        else:
            otel_span = None

        # Get the version to build the prefix for parsing the response
        version = await self.get_active_version()
        prefix = f"{version}_" if version else None

        # Extract ID mappings from transpiler
        node_id_map = transpiler._reverse_node_map
        edge_id_map = transpiler._reverse_edge_map

        try:
            if self.protocol == DgraphProtocol.GRPC:
                result = await self._run_grpc_query(
                    query,
                    prefix=prefix,
                    node_id_map=node_id_map,
                    edge_id_map=edge_id_map,
                )
            else:
                result = await self._run_http_query(
                    query,
                    prefix=prefix,
                    node_id_map=node_id_map,
                    edge_id_map=edge_id_map,
                )
        except TimeoutError as e:
            if otel_span is not None:
                otel_span.add_event("dgraph_query_timeout")
            raise TimeoutError(
                f"Dgraph query exceeded {self.query_timeout}s timeout"
            ) from e

        if otel_span is not None:
            otel_span.add_event("dgraph_query_end")

        return result

    async def _run_grpc_query(
        self,
        query: str,
        *,
        prefix: str | None,
        node_id_map: dict[str, str] | None = None,
        edge_id_map: dict[str, str] | None = None,
    ) -> dg_models.DgraphResponse:
        """Execute query using gRPC protocol.

        Args:
            query: The Dgraph query string
            prefix: Schema version prefix
            node_id_map: Optional mapping from normalized to original node IDs
            edge_id_map: Optional mapping from normalized to original edge IDs

        Returns:
            Parsed response with original bindings restored
        """
        assert self._client is not None, "gRPC client not initialized"

        txn: DgraphTxnProtocol | None = None
        try:
            txn = self._client.txn(read_only=True)
            txn_protocol: DgraphTxnProtocol = cast(DgraphTxnProtocol, cast(object, txn))

            future: _GrpcFuture = txn_protocol.async_query(
                query=query,
                variables=None,
                timeout=self.query_timeout,
                credentials=None,
                resp_format="JSON",
            )

            response: PydgraphResponse = await asyncio.to_thread(
                txn_protocol.handle_query_future, future
            )

            raw: Any = response.json

            return dg_models.DgraphResponse.parse(
                raw,
                prefix=prefix,
                node_id_map=node_id_map,
                edge_id_map=edge_id_map,
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(f"Dgraph gRPC query timed out: {e.details()}") from e
            raise ConnectionError(f"Dgraph gRPC query failed: {e.details()}") from e
        except (pydgraph.errors.AbortedError, NameError) as e:
            original_error = e.__context__
            if isinstance(original_error, grpc.RpcError):
                if original_error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        f"Dgraph gRPC query timed out: {original_error.details()}"
                    ) from original_error
                else:
                    raise ConnectionError(
                        f"Dgraph gRPC query failed: {original_error.details()}"
                    ) from original_error
            raise ConnectionError(f"Dgraph transaction was aborted: {e}") from e
        finally:
            if txn:
                await asyncio.to_thread(txn.discard)

    async def _run_http_query(
        self,
        query: str,
        *,
        prefix: str | None,
        node_id_map: dict[str, str] | None = None,
        edge_id_map: dict[str, str] | None = None,
    ) -> dg_models.DgraphResponse:
        """Execute query using HTTP protocol with DQL.

        Args:
            query: The Dgraph query string
            prefix: Schema version prefix
            node_id_map: Optional mapping from normalized to original node IDs
            edge_id_map: Optional mapping from normalized to original edge IDs

        Returns:
            Parsed response with original bindings restored
        """
        assert self._http_session is not None, "HTTP session not initialized"

        # If a token exists and is expired, refresh it before making the query.
        if self._access_token and self._token_expiry and datetime.now(UTC) >= self._token_expiry:
            log.info("Dgraph HTTP access token expired. Refreshing...")
            try:
                await self._http_login()
            except Exception as e:
                log.error(f"Failed to refresh Dgraph HTTP token: {e}")
                # Propagate the error as the query will likely fail anyway
                raise

        clean_query = query.strip()

        log.debug(f"HTTP DQL query: {clean_query}")

        async with self._http_session.post(
            urljoin(self.endpoint, "/query"),
            json={"query": clean_query},
            timeout=ClientTimeout(total=self.query_timeout),
        ) as response:
            if response.status != HTTPStatus.OK:
                text = await response.text()
                raise RuntimeError(
                    f"Dgraph HTTP query failed with status {response.status}: {text}"
                )

            raw_data = await response.json()

            if "errors" in raw_data:
                raise RuntimeError(
                    f"Dgraph query returned errors: {raw_data['errors']}"
                )

            if "data" not in raw_data:
                raise RuntimeError("Dgraph query returned no data field.")

            return dg_models.DgraphResponse.parse(
                raw_data.get("data"),
                prefix=prefix,
                node_id_map=node_id_map,
                edge_id_map=edge_id_map,
            )

    async def _cleanup_connections(self) -> None:
        """Clean up any open connections."""
        # Close gRPC connection if open
        if self._client_stub is not None:
            self._client_stub.close()
            self._client_stub = None
            self._client = None

        # Close HTTP session if open
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

    @override
    async def close(self) -> None:
        """Close the connection to Dgraph and clean up resources."""
        await self._cleanup_connections()


class DgraphHttpDriver(DgraphDriver):
    """Convenience class for HTTP-specific Dgraph driver."""

    def __init__(self, *, version: str | None = None) -> None:
        """Initialize with HTTP protocol."""
        super().__init__(protocol=DgraphProtocol.HTTP, version=version)


class DgraphGrpcDriver(DgraphDriver):
    """Convenience class for gRPC-specific Dgraph driver."""

    def __init__(self, *, version: str | None = None) -> None:
        """Initialize with gRPC protocol."""
        super().__init__(protocol=DgraphProtocol.GRPC, version=version)
