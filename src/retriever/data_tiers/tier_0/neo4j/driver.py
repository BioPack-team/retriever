import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal, LiteralString, cast, overload

import bmt
import neo4j
import neo4j.exceptions
from loguru import logger as log
from neo4j import unit_of_work
from opentelemetry import trace
from reasoner_transpiler.cypher import (
    transform_result,  # pyright:ignore[reportUnknownVariableType]
)

from retriever.config.general import CONFIG
from retriever.config.openapi import OPENAPI_CONFIG
from retriever.types.general import TransformedNeo4jResult
from retriever.types.trapi import QueryGraphDict

biolink = bmt.Toolkit()


class Neo4jBoltDriver:
    """Simple Neo4j driver for cypher queries."""

    def __init__(
        self, host: str, port: int, auth: tuple[str, str], database_name: str = "neo4j"
    ) -> None:
        """Initialize a driver instance."""
        self.database_name: str = database_name
        self.database_auth: tuple[str, str] = auth
        self.graph_db_uri: str = f"bolt://{host}:{port}"
        self.neo4j_driver: neo4j.AsyncDriver | None = None
        self._supports_apoc: bool | None = None

    async def connect_to_neo4j(self, retries: int = 0) -> None:
        """Attempt to connect to neo4j."""
        if not self.neo4j_driver:
            self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(
                self.graph_db_uri,
                auth=self.database_auth,
                telemetry_disabled=True,
                max_connection_pool_size=1000,
            )
        try:
            await self.neo4j_driver.verify_connectivity()
        except Exception as e:  # currently the driver says it raises Exception, not something more specific
            await self.neo4j_driver.close()
            if retries <= CONFIG.tier0.neo4j.connect_retries:
                await asyncio.sleep(8)
                log.error(
                    f"Could not establish connection to neo4j, trying again... retry {retries + 1}"
                )
                await self.connect_to_neo4j(retries + 1)
            else:
                log.error(f"Could not establish connection to neo4j, error: {e}")
                raise e

    @staticmethod
    @unit_of_work(timeout=CONFIG.tier0.neo4j.query_timeout)
    async def _async_cypher_tx_function(
        tx: neo4j.AsyncManagedTransaction,
        cypher: LiteralString,
        qgraph: QueryGraphDict,
        query_parameters: dict[str, Any] | None = None,
    ) -> TransformedNeo4jResult:
        """Neo4j transaction function for running cypher queries."""
        if not query_parameters:
            query_parameters = dict[str, Any]()

        neo4j_result: neo4j.AsyncResult = await tx.run(
            cypher, parameters=query_parameters
        )
        neo4j_record = await neo4j_result.single()
        return cast(
            TransformedNeo4jResult,
            cast(
                object,
                transform_result(
                    neo4j_record, cast(dict[str, Any], cast(object, qgraph))
                ),
            ),
        )

    async def run(
        self, query: LiteralString, qgraph: QueryGraphDict
    ) -> TransformedNeo4jResult:
        """Run a given query."""
        if self.neo4j_driver is None:
            raise RuntimeError(
                "Must use Neo4jBoltDriver.connect_to_neo4j() before running queries."
            )
        try:
            async with self.neo4j_driver.session(
                database=self.database_name, default_access_mode=neo4j.READ_ACCESS
            ) as session:
                run_async_result = await session.execute_read(
                    self._async_cypher_tx_function, query, qgraph
                )
        except neo4j.exceptions.ServiceUnavailable as e:
            log.exception(
                f"Session could not establish connection to neo4j ({e}).. trying to connect again"
            )
            await self.connect_to_neo4j()
            return await self.run(query, qgraph)
        except neo4j.exceptions.Neo4jError as e:
            log.exception("Neo4jError encountered.")
            raise e
        except neo4j.exceptions.DriverError as e:
            log.exception("DriverError encountered.")
            raise e
        return run_async_result

    async def close(self) -> None:
        """Close the neo4j connection."""
        if self.neo4j_driver is None:
            return
        await self.neo4j_driver.close()


class GraphInterface:
    """Singleton class for interfacing with the graph."""

    class _GraphInterface:
        def __init__(
            self, host: str, port: int, auth: tuple[str, str], protocol: str = "bolt"
        ) -> None:
            self.protocol: str = protocol
            if protocol == "bolt":
                self.driver: Neo4jBoltDriver = Neo4jBoltDriver(
                    host=host, port=port, auth=auth
                )
            else:
                raise Exception(f"Unsupported graph interface protocol: {protocol}")
            # self.summary = None
            self.toolkit: bmt.Toolkit = biolink
            self.bl_version: str = OPENAPI_CONFIG.version

        async def connect_to_neo4j(self) -> None:
            await self.driver.connect_to_neo4j()

        async def run_cypher(
            self,
            cypher: LiteralString,
            qgraph: QueryGraphDict,
        ) -> TransformedNeo4jResult:
            """Runs cypher directly."""
            # get a reference to the current opentelemetry span
            otel_span = trace.get_current_span()
            if not otel_span or not otel_span.is_recording():
                otel_span = None
            else:
                otel_span.add_event(
                    "neo4j_query_start", attributes={"cypher_query": cypher}
                )
            cypher_results = await self.driver.run(cypher, qgraph)
            if otel_span is not None:
                otel_span.add_event("neo4j_query_end")
            return cypher_results

        async def close(self) -> None:
            await self.driver.close()

    instance: _GraphInterface | None = None

    def __init__(
        self, host: str, port: int, auth: tuple[str, str], protocol: str = "bolt"
    ) -> None:
        """Initialize an instance as a singleton."""
        # create a new instance if not already created.
        if not GraphInterface.instance:
            GraphInterface.instance = GraphInterface._GraphInterface(
                host=host, port=port, auth=auth, protocol=protocol
            )

    @overload
    def __getattr__(
        self, item: Literal["run_cypher"]
    ) -> Callable[
        [LiteralString, QueryGraphDict], Awaitable[TransformedNeo4jResult]
    ]: ...

    @overload
    def __getattr__(self, item: Literal["close"]) -> Callable[[], Awaitable[None]]: ...

    @overload
    def __getattr__(
        self, item: Literal["connect_to_neo4j"]
    ) -> Callable[[], Awaitable[None]]: ...

    def __getattr__(self, item: str) -> ...:
        """Proxy function calls to the inner object."""
        return getattr(self.instance, item)
