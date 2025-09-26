import asyncio
from typing import Any, Protocol, TypedDict, cast, override

import pydgraph
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.tier_0.dgraph import result_models as dg_models


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
    """Driver for Dgraph using the official gRPC client (pydgraph)."""

    settings: Any
    endpoint: str
    query_timeout: float
    connect_retries: int
    _client_stub: DgraphClientStubProtocol | None
    _client: pydgraph.DgraphClient | None

    def __init__(self) -> None:
        """Initialize the Dgraph driver with connection settings."""
        self.settings = CONFIG.tier0.dgraph
        self.endpoint = self.settings.host
        self.query_timeout = self.settings.query_timeout
        self.connect_retries = self.settings.connect_retries
        self._client_stub = None
        self._client = None

    @override
    async def connect(self, retries: int = 0) -> None:
        """Connect to Dgraph using gRPC (pydgraph client)."""
        try:
            self._client_stub = pydgraph.DgraphClientStub(self.endpoint)
            self._client = pydgraph.DgraphClient(self._client_stub)
            log.success("Dgraph connection successful!")
        except Exception as e:
            # Safely close stub only if it was created
            if self._client_stub is not None:
                self._client_stub.close()
            self._client_stub = None
            self._client = None
            if retries < self.connect_retries:
                await asyncio.sleep(1)
                log.error(
                    f"Could not establish connection to dgraph, trying again... retry {retries + 1}"
                )
                await self.connect(retries + 1)
            else:
                log.error(f"Could not establish connection to dgraph, error: {e}")
                raise e

    @override
    async def run_query(self, query: str, *args: Any, **kwargs: Any) -> dg_models.DgraphResponse:
        """Execute a query against the Dgraph database and parse into dataclasses."""
        if not self._client:
            raise RuntimeError("DgraphDriver not connected. Call connect() first.")

        loop = asyncio.get_running_loop()
        txn = self._client.txn(read_only=True)
        txn_protocol: DgraphTxnProtocol = cast(DgraphTxnProtocol, cast(object, txn))

        def _query() -> PydgraphResponse:
            return txn_protocol.query(query)

        otel_span = trace.get_current_span()
        if otel_span and otel_span.is_recording():
            otel_span.add_event("dgraph_query_start", attributes={"dgraph_query": query})
        else:
            otel_span = None

        future = loop.run_in_executor(None, _query)
        try:
            resp: PydgraphResponse = await asyncio.wait_for(
                future, timeout=self.query_timeout
            )
        except TimeoutError as e:
            if otel_span is not None:
                otel_span.add_event("dgraph_query_timeout")
            raise TimeoutError(
                f"Dgraph query exceeded {self.query_timeout}s timeout"
            ) from e

        if otel_span is not None:
            otel_span.add_event("dgraph_query_end")

        # pydgraph exposes Response.json as a property (bytes/str/dict)
        result = dg_models.parse_response(resp.json)
        return result

    @override
    async def close(self) -> None:
        """Close the connection to Dgraph and clean up resources."""
        if self._client_stub:
            self._client_stub.close()
            self._client_stub = None
            self._client = None
