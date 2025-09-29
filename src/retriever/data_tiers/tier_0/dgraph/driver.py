import asyncio
from enum import Enum
from http import HTTPStatus
from typing import Any, Protocol, TypedDict, cast, override
from urllib.parse import urljoin

import aiohttp
import pydgraph
from aiohttp import ClientTimeout
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG
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
        """Run a query and return a pydgraph Response-like object."""
        ...


class DgraphQueryResult(TypedDict, total=False):
    """TypedDict for legacy Dgraph query result (kept for backward compat where needed)."""
    data: Any
    errors: Any


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph supporting both gRPC and HTTP protocols."""

    settings: Any
    protocol: DgraphProtocol
    endpoint: str
    query_timeout: float
    connect_retries: int
    _client_stub: DgraphClientStubProtocol | None = None
    _client: pydgraph.DgraphClient | None = None
    _http_session: aiohttp.ClientSession | None = None

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

    @override
    async def connect(self, retries: int = 0) -> None:
        """Connect to Dgraph using selected protocol."""
        try:
            if self.protocol == DgraphProtocol.GRPC:
                await self._connect_grpc()
            else:
                await self._connect_http()
            log.success(f"Dgraph {self.protocol} connection successful!")
        except Exception as e:
            await self._cleanup_connections()
            if retries < self.connect_retries:
                await asyncio.sleep(1)
                log.error(
                    f"Could not establish connection to Dgraph via {self.protocol}, " +
                    f"trying again... retry {retries + 1}"
                )
                await self.connect(retries + 1)
            else:
                log.error(f"Could not establish connection to Dgraph, error: {e}")
                raise e

    async def _connect_grpc(self) -> None:
        """Establish gRPC connection to Dgraph."""
        self._client_stub = pydgraph.DgraphClientStub(self.endpoint)
        self._client = pydgraph.DgraphClient(self._client_stub)

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
                raise ConnectionError(f"HTTP connection failed with status {response.status}: {text}")

    @override
    async def run_query(self, query: str, *args: Any, **kwargs: Any) -> dg_models.DgraphResponse:
        """Execute a query against the Dgraph database and parse into dataclasses."""
        if self.protocol == DgraphProtocol.GRPC and not self._client:
            raise RuntimeError("DgraphDriver (gRPC) not connected. Call connect() first.")
        if self.protocol == DgraphProtocol.HTTP and not self._http_session:
            raise RuntimeError("DgraphDriver (HTTP) not connected. Call connect() first.")

        otel_span = trace.get_current_span()
        if otel_span and otel_span.is_recording():
            otel_span.add_event("dgraph_query_start", attributes={"dgraph_query": query})
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

        loop = asyncio.get_running_loop()
        txn = self._client.txn(read_only=True)
        txn_protocol: DgraphTxnProtocol = cast(DgraphTxnProtocol, cast(object, txn))

        def _query() -> PydgraphResponse:
            return txn_protocol.query(query)

        future = loop.run_in_executor(None, _query)
        resp: PydgraphResponse = await asyncio.wait_for(
            future, timeout=self.query_timeout
        )

        return dg_models.parse_response(resp.json)

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
                raise RuntimeError(f"Dgraph HTTP query failed with status {response.status}: {text}")

            raw_data = await response.json()

            if "errors" in raw_data:
                raise RuntimeError(f"DQL query returned errors: {raw_data['errors']}")

            return dg_models.parse_response(raw_data)

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
