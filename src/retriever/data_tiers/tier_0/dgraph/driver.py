from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from typing import Any, Optional
import pydgraph
import asyncio


class DgraphDriver(DatabaseDriver):
    """Driver for Dgraph using the official gRPC client (pydgraph)."""

    def __init__(self):
        self.settings = CONFIG.tier0.dgraph
        self.endpoint = self.settings.host  # should be like 'dgraph-alpha:9080'
        self.endpoint = "localhost:9080"
        self._client_stub: Optional[pydgraph.DgraphClientStub] = None
        self._client: Optional[pydgraph.DgraphClient] = None

    async def connect(self) -> None:
        """Connect to Dgraph using gRPC (pydgraph client)."""
        # pydgraph is synchronous, wrap in threadpool if async is needed
        self._client_stub = pydgraph.DgraphClientStub(self.endpoint)
        self._client = pydgraph.DgraphClient(self._client_stub)

        # Optionally, do a health check
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._client.alter, pydgraph.Operation(drop_all=False))

    async def run_query(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._client:
            raise RuntimeError("DgraphDriver not connected. Call connect() first.")

        loop = asyncio.get_running_loop()
        txn = self._client.txn(read_only=True)

        def _query():
            return txn.query(query)

        resp = await loop.run_in_executor(None, _query)
        return resp.json

    async def close(self) -> None:
        if self._client_stub:
            self._client_stub.close()
            self._client_stub = None
            self._client = None
