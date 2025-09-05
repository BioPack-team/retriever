import asyncio
import logging
from typing import Any

import pydgraph

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph using the official gRPC client (pydgraph)."""

    def __init__(self) -> None:
        """Initialize the Dgraph driver with connection settings."""
        self.settings = CONFIG.tier0.dgraph
        self.endpoint = self.settings.host
        self.endpoint = "localhost:9080"
        self._client_stub: pydgraph.DgraphClientStub | None = None
        self._client: pydgraph.DgraphClient | None = None

    async def connect(self) -> None:
        """Connect to Dgraph using gRPC (pydgraph client)."""
        try:
            self._client_stub = pydgraph.DgraphClientStub(self.endpoint)
            self._client = pydgraph.DgraphClient(self._client_stub)
        except Exception as e:
            logging.error("Failed to connect to Dgraph: %s", e)
            raise

    async def run_query(self, query: Any, *args: Any, **kwargs: Any) -> Any:
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

        def _query() -> Any:
            return txn.query(query)

        resp = await loop.run_in_executor(None, _query)
        return resp.json

    async def close(self) -> None:
        """Close the connection to Dgraph and clean up resources."""
        if self._client_stub:
            self._client_stub.close()
            self._client_stub = None
            self._client = None
