import asyncio
import base64
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ActiveSchemaInfo:
    """Holds all information about the active schema."""

    version: str
    namespace: int | None
    mapping: dict[str, Any]


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph supporting both gRPC and HTTP protocols."""

    settings: DgraphSettings
    protocol: DgraphProtocol
    endpoint: str
    query_timeout: float
    connect_retries: int
    version: str | None = None
    namespace: int | None = None
    _client_stub: DgraphClientStubProtocol | None = None
    _client: DgraphClientProtocol | None = None
    _http_session: aiohttp.ClientSession | None = None
    _access_token: str | None = None
    _token_namespace: int | None = None
    _token_expiry: datetime | None = None
    _failed: bool = False
    _schema_info_cache: TTLCache[str, ActiveSchemaInfo | None] = TTLCache(
        maxsize=1, ttl=300
    )

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

        # Set preferred namespace from settings, if provided.
        if self.settings.preferred_namespace is not None:
            self.namespace = self.settings.preferred_namespace
            log.debug(
                "Using namespace from TIER0__DGRAPH__PREFERRED_NAMESPACE env var: "
                + f"{self.namespace}"
            )
        else:
            self.namespace = None

        # Set endpoint based on protocol
        if protocol == DgraphProtocol.GRPC:
            self.endpoint = self.settings.grpc_endpoint
        else:  # HTTP
            self.endpoint = self.settings.http_endpoint

    def is_connected(self) -> bool:
        """Checks if the driver has an active connection client."""
        if self.protocol == DgraphProtocol.GRPC:
            return self._client is not None
        else:  # HTTP
            return self._http_session is not None and not self._http_session.closed

    def clear_schema_info_cache(self) -> None:
        """Clears the internal active schema info cache. Primarily for testing."""
        log.debug("Clearing Dgraph active schema info cache.")
        self._schema_info_cache.clear()

    async def get_active_schema_info(self) -> ActiveSchemaInfo | None:
        """Fetches and caches all information for the active schema version.

        This is the single source of truth for the active version, namespace, and mapping.
        It respects manual overrides for version and namespace from the config.

        Returns:
            An ActiveSchemaInfo object, or None if no active schema is found.
        """
        # Ensure the driver is connected before proceeding.
        if not self.is_connected():
            log.debug("Driver not connected. Establishing connection now...")
            await self.connect()

        # 1. Check cache first
        try:
            cached_info = self._schema_info_cache["active_info"]
            if cached_info:
                log.debug(
                    f"Returning cached schema info for version '{cached_info.version}'"
                )
            return cached_info
        except KeyError:
            log.debug("Querying for active Dgraph schema info (cache miss)...")

        # 2. Fetch from database
        try:
            info = await self._fetch_and_build_schema_info()
            self._schema_info_cache["active_info"] = info
            if info:
                log.info(
                    f"Successfully fetched and cached active schema info for version '{info.version}'"
                )
            else:
                log.warning("No active schema found. Caching null result.")
            return info
        except Exception as e:
            log.error(f"Failed to fetch active schema info: {e}")
            self._schema_info_cache["active_info"] = None
            return None

    async def _fetch_and_build_schema_info(self) -> ActiveSchemaInfo | None:
        """Performs the actual database query to get version, namespace, and mapping."""
        # Determine the filter to use.
        # If a manual version is set, filter by that version.
        # Otherwise, filter for the active schema.
        if self.version:
            log.debug(f"Querying for manually specified Dgraph schema: {self.version}")
            dgraph_filter = f'@filter(eq(schema_metadata_version, "{self.version}"))'
        else:
            log.debug("Querying for active Dgraph schema.")
            dgraph_filter = "@filter(eq(schema_metadata_is_active, true))"

        # This query now correctly uses `has()` and works for both cases.
        query = f"""
            query get_schema_info {{
              schema_metadata(func: has(schema_metadata_version)) {dgraph_filter} {{
                schema_metadata_version
                schema_metadata_namespace
                schema_metadata_mapping
              }}
            }}
        """
        raw_data = await self._execute_metadata_query(query)
        schema_metadata_list = raw_data.get("schema_metadata", [])

        if not schema_metadata_list:
            return None

        active_schema = schema_metadata_list[0]
        version = active_schema.get("schema_metadata_version")
        mapping_blob = active_schema.get("schema_metadata_mapping")

        if not version or not mapping_blob:
            log.error(
                "Found schema metadata, but it is missing version or mapping fields."
            )
            return None

        # Respect manual namespace override even with auto-detected version
        namespace_val = (
            self.namespace
            if self.namespace is not None
            else active_schema.get("schema_metadata_namespace")
        )
        namespace: int | None = None
        if namespace_val is not None:
            try:
                namespace = int(namespace_val)
            except (ValueError, TypeError):
                log.warning(
                    f"Could not parse namespace '{namespace_val}' as an integer. Defaulting to None."
                )
                namespace = None

        try:
            mapping_bytes = (
                base64.b64decode(mapping_blob)
                if isinstance(mapping_blob, str)
                else mapping_blob
            )
            mapping = cast(dict[str, Any], msgpack.unpackb(mapping_bytes, raw=False))
        except Exception as e:
            log.error(f"Failed to decode mapping for version '{version}': {e}")
            return None

        return ActiveSchemaInfo(version=version, namespace=namespace, mapping=mapping)

    async def _execute_metadata_query(self, query: str) -> dict[str, Any]:
        """Executes a simple, read-only query against the metadata namespace."""
        # With ACLs, metadata is always in namespace 0. We must ensure we are
        # logged into that namespace before executing the query.
        if self.protocol == DgraphProtocol.GRPC:
            # For critical metadata queries with ACLs, we create a temporary, clean
            # client to ensure there's no stale authentication state.
            if self.settings.username and self.settings.password:
                temp_stub = None
                try:
                    log.debug("Creating temporary gRPC client for metadata query.")
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
                    temp_stub = pydgraph.DgraphClientStub(
                        self.endpoint, options=grpc_options
                    )
                    temp_client = cast(
                        DgraphClientProtocol,
                        cast(object, pydgraph.DgraphClient(temp_stub)),
                    )
                    await asyncio.to_thread(
                        temp_client.login_into_namespace,
                        self.settings.username,
                        self.settings.password.get_secret_value(),
                        0,  # Always namespace 0 for metadata
                    )
                    txn = temp_client.txn(read_only=True)
                    response = await asyncio.to_thread(txn.query, query)
                    return json.loads(response.json)
                finally:
                    if temp_stub:
                        temp_stub.close()
            else:
                # No auth, use the existing client
                assert self._client is not None, "gRPC client not initialized"
                txn = self._client.txn(read_only=True)
                try:
                    response = await asyncio.to_thread(txn.query, query)
                    return json.loads(response.json)
                finally:
                    await asyncio.to_thread(txn.discard)
        else:  # HTTP
            assert self._http_session is not None, "HTTP session not initialized"
            async with self._http_session.post(
                urljoin(self.endpoint, "/query"),
                json={"query": query},
                timeout=ClientTimeout(total=self.query_timeout),
            ) as response:
                response.raise_for_status()
                return await response.json()

    @override
    async def get_metadata(self) -> dict[str, Any] | None:
        """Returns the metadata mapping for the active schema."""
        schema_info = await self.get_active_schema_info()
        if not schema_info:
            log.warning("Cannot retrieve schema metadata: no active schema info found.")
            return None
        return schema_info.mapping

    async def get_active_version(self) -> str | None:
        """Convenience method to get only the version string of the active schema."""
        schema_info = await self.get_active_schema_info()
        if not schema_info:
            return None
        return schema_info.version

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

            # Populate version and namespace caches after successful connect so subsequent
            # code can use them without triggering a query.
            try:
                await self.get_active_schema_info()
            except Exception as e:
                log.warning(
                    f"Unable to fetch active schema version/namespace after connect: {e}"
                )
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
                # With ACLs, metadata is always in namespace 0.
                # Data queries will use the dynamic namespace later.
                login_namespace = 0

                log.info(
                    f"Attempting Dgraph login for user '{username}' "
                    + f"in metadata namespace {login_namespace}"
                )

                # pydgraph login is synchronous, so run it in a separate thread.
                await asyncio.to_thread(
                    self._client.login_into_namespace,
                    username,
                    password,
                    login_namespace,
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
        # With ACLs, metadata is always in namespace 0.
        # Data queries will use the dynamic namespace later.
        login_namespace = 0

        log.info(
            f"Attempting Dgraph HTTP login for user '{username}'"
            + f" in metadata namespace {login_namespace}"
        )

        login_mutation = {
            "query": f"""
                mutation Login {{
                    login(userId: "{username}", password: "{password}"{f", namespace: {login_namespace}" if login_namespace else ""}) {{
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
                assert (
                    self._access_token is not None
                ), "Access token should not be None after successful login"

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
                self._http_session.headers["X-Dgraph-AccessToken"] = self._access_token
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

        # Get active schema info
        schema_info = await self.get_active_schema_info()
        if not schema_info:
            raise RuntimeError("Could not determine active Dgraph schema to run query.")

        prefix = f"{schema_info.version}_"

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
            # If auth is enabled, we may need to re-login to the correct data namespace
            if self.settings.username and self.settings.password:
                schema_info = await self.get_active_schema_info()
                data_namespace = schema_info.namespace if schema_info else None

                username = self.settings.username
                password = self.settings.password.get_secret_value()
                log.debug(
                    f"gRPC query: ensuring login to data namespace {data_namespace}"
                )
                await asyncio.to_thread(
                    self._client.login_into_namespace,
                    username,
                    password,
                    data_namespace,
                )

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

        # If auth is enabled, check if we need to re-login.
        if self.settings.username and self.settings.password:
            schema_info = await self.get_active_schema_info()
            data_namespace = schema_info.namespace if schema_info else None

            token_is_expired = (
                self._token_expiry is not None
                and datetime.now(UTC) >= self._token_expiry
            )
            namespace_mismatch = self._token_namespace != data_namespace

            if not self._access_token or token_is_expired or namespace_mismatch:
                if token_is_expired:
                    log.info("Dgraph HTTP access token expired. Refreshing...")
                elif namespace_mismatch:
                    log.info(
                        f"Switching Dgraph HTTP namespace from {self._token_namespace} to {data_namespace}. Re-logging in..."
                    )

                try:
                    await self._http_login_for_namespace(data_namespace)
                except Exception as e:
                    log.error(f"Failed to refresh/acquire Dgraph HTTP token: {e}")
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

    async def _http_login_for_namespace(self, namespace: int | None) -> None:
        """Logs into a specific Dgraph namespace over HTTP.

        This is a helper for _run_http_query to ensure the session token is valid
        for the required data namespace.
        """
        if not (self.settings.username and self.settings.password):
            return

        assert self._http_session is not None, "HTTP session not initialized for login"

        username = self.settings.username
        password = self.settings.password.get_secret_value()

        log.info(
            f"Attempting Dgraph HTTP login for user '{username}'"
            + (f" in namespace {namespace}" if namespace is not None else "")
        )

        login_mutation = {
            "query": f"""
                mutation Login {{
                    login(userId: "{username}", password: "{password}"{f", namespace: {namespace}" if namespace is not None else ""}) {{
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
                assert (
                    self._access_token is not None
                ), "Access token should not be None after successful login"

                # Decode JWT payload to get expiration time
                payload_b64 = self._access_token.split(".")[1]
                payload_b64 += "=" * (-len(payload_b64) % 4)
                payload = json.loads(base64.b64decode(payload_b64))

                # Set expiry with a 60-second buffer to be safe
                self._token_expiry = datetime.fromtimestamp(
                    payload["exp"], tz=UTC
                ) - timedelta(seconds=60)

                # Store the namespace this token is valid for
                self._token_namespace = namespace

                # Update the session headers for subsequent requests
                self._http_session.headers["X-Dgraph-AccessToken"] = self._access_token
                log.success(
                    f"Dgraph HTTP login for user '{username}' in namespace {namespace} successful."
                )

        except Exception as e:
            self._access_token = None
            self._token_expiry = None
            self._token_namespace = None
            raise ConnectionError(
                f"Dgraph HTTP login for namespace {namespace} failed."
            ) from e

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
