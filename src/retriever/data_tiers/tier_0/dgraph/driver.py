import asyncio
import logging
from typing import (
    Any,
    Protocol,
    TypedDict,
    cast,
    override,
)

import pydgraph

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver


class DgraphResponse(Protocol):
    """Protocol for Dgraph response object."""

    @property
    def json(self) -> Any:
        """Return the JSON content of the response."""
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
    ) -> DgraphResponse:
        """Run a query and return a DgraphResponse."""
        ...


class DgraphQueryResult(TypedDict, total=False):
    """TypedDict for Dgraph query result."""
    data: Any
    errors: Any


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph using the official gRPC client (pydgraph)."""

    settings: Any
    endpoint: str
    _client_stub: DgraphClientStubProtocol | None
    _client: pydgraph.DgraphClient | None

    def __init__(self) -> None:
        """Initialize the Dgraph driver with connection settings."""
        self.settings = CONFIG.tier0.dgraph
        self.endpoint = self.settings.host
        self.endpoint = "localhost:9080"
        self._client_stub = None
        self._client = None

    @override
    async def connect(self) -> None:
        """Connect to Dgraph using gRPC (pydgraph client)."""
        try:
            self._client_stub = pydgraph.DgraphClientStub(self.endpoint)
            self._client = pydgraph.DgraphClient(self._client_stub)
        except Exception as e:
            logging.error("Failed to connect to Dgraph: %s", e)
            raise

    @override
    async def run_query(self, query: str, *args: Any, **kwargs: Any) -> DgraphQueryResult:
        """Execute a query against the Dgraph database.

        Args:
            query: The Dgraph query to execute
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The JSON response from Dgraph

        Raises:
            RuntimeError: If not connected to Dgraph
        """
        if not self._client:
            raise RuntimeError("DgraphDriver not connected. Call connect() first.")

        loop = asyncio.get_running_loop()
        txn = self._client.txn(read_only=True)
        txn_protocol: DgraphTxnProtocol = cast(DgraphTxnProtocol, cast(object, txn))

        def _query() -> DgraphResponse:
            return txn_protocol.query(query)

        resp: DgraphResponse = await loop.run_in_executor(None, _query)
        return resp.json

    @override
    async def close(self) -> None:
        """Close the connection to Dgraph and clean up resources."""
        if self._client_stub:
            self._client_stub.close()
            self._client_stub = None
            self._client = None
