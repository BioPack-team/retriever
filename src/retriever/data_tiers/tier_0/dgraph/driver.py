import asyncio
import json
from collections.abc import Callable
from enum import Enum
from http import HTTPStatus
from typing import Any, Protocol, TypedDict, cast, override
from urllib.parse import urljoin

import aiohttp
import pydgraph
from aiohttp import ClientTimeout
from cachetools import TTLCache
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG, DgraphSettings
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


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


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph supporting both gRPC and HTTP protocols."""

    settings: DgraphSettings
    protocol: DgraphProtocol
    endpoint: str
    query_timeout: float
    connect_retries: int
    _client_stub: DgraphClientStubProtocol | None = None
    _client: DgraphClientProtocol | None = None
    _http_session: aiohttp.ClientSession | None = None
    _failed: bool = False
    _version_cache: TTLCache[str, str | None] = TTLCache(maxsize=1, ttl=60)

    def __init__(self, protocol: DgraphProtocol = DgraphProtocol.GRPC) -> None:
        """Initialize the Dgraph driver with connection settings.

        Args:
            protocol: The protocol to use (gRPC or HTTP)
        """
        self.settings = CONFIG.tier0.dgraph
        self.protocol = protocol
        self.query_timeout = self.settings.query_timeout
        self.connect_retries = self.settings.connect_retries

        # Set endpoint based on protocol
        if protocol == DgraphProtocol.GRPC:
            self.endpoint = self.settings.grpc_endpoint
        else:  # HTTP
            self.endpoint = self.settings.http_endpoint

    def clear_version_cache(self) -> None:
        """Clears the internal schema version cache. Primarily for testing."""
        log.debug("Clearing Dgraph schema version cache.")
        self._version_cache.clear()

    async def get_active_version(self) -> str | None:
        """Queries Dgraph for the active schema version and caches the result.

        This method implements manual caching to be async-safe.
        """
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
                log.warning("No active Dgraph schema version found. Caching null result.")

            # Manually store the result in the cache before returning
            self._version_cache["active_version"] = version
            return version
        except Exception as e:
            log.error(f"Failed to query for active Dgraph schema version: {e}")
            # Cache the failure as None to prevent retrying on every call
            self._version_cache["active_version"] = None
            return None

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
                log.error(
                    f"Could not establish connection to Dgraph via {self.protocol}, "
                    + f"trying again... retry {retries + 1}"
                )
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

    async def _connect_http(self) -> None:
        """Establish HTTP connection to Dgraph."""
        self._http_session = aiohttp.ClientSession()
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
        """Execute a query against the Dgraph database and parse into dataclasses."""
        if self.protocol == DgraphProtocol.GRPC and not self._client:
            raise RuntimeError(
                "DgraphDriver (gRPC) not connected. Call connect() first."
            )
        if self.protocol == DgraphProtocol.HTTP and not self._http_session:
            raise RuntimeError(
                "DgraphDriver (HTTP) not connected. Call connect() first."
            )

        otel_span = trace.get_current_span()
        if otel_span and otel_span.is_recording():
            otel_span.add_event(
                "dgraph_query_start", attributes={"dgraph_query": query}
            )
        else:
            otel_span = None

        try:
            if self.protocol == DgraphProtocol.GRPC:
                result = await self._run_grpc_query(query)
            else:
                result = await self._run_http_query(query)
        except TimeoutError as e:
            if otel_span is not None:
                otel_span.add_event("dgraph_query_timeout")
            raise TimeoutError(
                f"Dgraph query exceeded {self.query_timeout}s timeout"
            ) from e

        if otel_span is not None:
            otel_span.add_event("dgraph_query_end")

        return result

    async def _run_grpc_query(self, query: str) -> dg_models.DgraphResponse:
        """Execute query using gRPC protocol."""
        assert self._client is not None, "gRPC client not initialized"

        txn = self._client.txn(read_only=True)
        txn_protocol: DgraphTxnProtocol = cast(DgraphTxnProtocol, cast(object, txn))

        future: _GrpcFuture = txn_protocol.async_query(
            query=query,
            variables=None,
            timeout=self.query_timeout,
            credentials=None,
            resp_format="JSON",
        )

        # Run the blocking handle_query_future in a thread to avoid blocking the event loop
        handle_query_future = cast(
            Callable[[_GrpcFuture], PydgraphResponse], pydgraph.Txn.handle_query_future
        )
        response: PydgraphResponse = await asyncio.to_thread(handle_query_future, future)

        raw: Any = response.json

        return dg_models.DgraphResponse.parse(raw)

    async def _run_http_query(self, query: str) -> dg_models.DgraphResponse:
        """Execute query using HTTP protocol with DQL."""
        assert self._http_session is not None, "HTTP session not initialized"

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

            return dg_models.DgraphResponse.parse(raw_data.get("data"))

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

    def __init__(self) -> None:
        """Initialize with HTTP protocol."""
        super().__init__(protocol=DgraphProtocol.HTTP)


class DgraphGrpcDriver(DgraphDriver):
    """Convenience class for gRPC-specific Dgraph driver."""

    def __init__(self) -> None:
        """Initialize with gRPC protocol."""
        super().__init__(protocol=DgraphProtocol.GRPC)
