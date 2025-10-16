import asyncio
from typing import Any, LiteralString, override

import neo4j
import neo4j.exceptions
from loguru import logger as log
from neo4j import unit_of_work
from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver

tracer = trace.get_tracer("lookup.execution.tracer")


class Neo4jDriver(DatabaseDriver):
    """A Neo4j driver."""

    neo4j_driver: neo4j.AsyncDriver | None = None
    _failed: bool = False

    @override
    async def connect(self, retries: int = 0) -> None:
        """Attempt to connect to neo4j."""
        log.info("Checking Neo4j connection...")
        if not self.neo4j_driver:
            self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(
                f"bolt://{CONFIG.tier0.neo4j.host}:{CONFIG.tier0.neo4j.bolt_port}",
                auth=(
                    CONFIG.tier0.neo4j.username,
                    CONFIG.tier0.neo4j.password.get_secret_value(),
                ),
                telemetry_disabled=True,
                max_connection_pool_size=1000,
            )
        try:
            await self.neo4j_driver.verify_connectivity()
            log.success("Neo4j connection successful!")
        except Exception as e:  # currently the driver says it raises Exception, not something more specific
            await self.neo4j_driver.close()
            self.neo4j_driver = None
            if retries <= CONFIG.tier0.neo4j.connect_retries:
                await asyncio.sleep(1)
                log.error(
                    f"Could not establish connection to neo4j, trying again... retry {retries + 1}"
                )
                await self.connect(retries + 1)
            else:
                log.error(f"Could not establish connection to neo4j, error: {e}")
                self._failed = True
                raise e

    @override
    @tracer.start_as_current_span("neo4j_query")
    async def run_query(
        self,
        cypher: LiteralString,
    ) -> neo4j.Record | None:
        """Run a given query."""
        # get a reference to the current opentelemetry span
        otel_span = trace.get_current_span()
        if not otel_span or not otel_span.is_recording():
            otel_span = None
        else:
            otel_span.add_event(
                "neo4j_query_start", attributes={"cypher_query": cypher}
            )

        cypher_results = await self.run(cypher)

        if otel_span is not None:
            otel_span.add_event("neo4j_query_end")
        return cypher_results

    @override
    async def close(self) -> None:
        """Close the neo4j connection."""
        if self.neo4j_driver is None:
            return
        await self.neo4j_driver.close()

    async def run(self, query: LiteralString) -> neo4j.Record | None:
        """Run a given query."""
        if self.neo4j_driver is None:
            raise RuntimeError(
                "Must use Neo4jBoltDriver.connect() before running queries."
            )
        try:
            async with self.neo4j_driver.session(
                database=CONFIG.tier0.neo4j.database_name,
                default_access_mode=neo4j.READ_ACCESS,
            ) as session:
                run_async_result = await session.execute_read(
                    self._async_cypher_tx_function, query
                )
        except neo4j.exceptions.ServiceUnavailable as e:
            log.exception(
                f"Session could not establish connection to neo4j ({e}).. trying to connect again"
            )
            await self.connect()
            return await self.run(query)
        except neo4j.exceptions.Neo4jError as e:
            log.exception("Neo4jError encountered.")
            raise e
        except neo4j.exceptions.DriverError as e:
            log.exception("DriverError encountered.")
            raise e
        return run_async_result

    @staticmethod
    @unit_of_work(timeout=CONFIG.tier0.neo4j.query_timeout)
    async def _async_cypher_tx_function(
        tx: neo4j.AsyncManagedTransaction,
        cypher: LiteralString,
        query_parameters: dict[str, Any] | None = None,
    ) -> neo4j.Record | None:
        """Neo4j transaction function for running cypher queries."""
        if not query_parameters:
            query_parameters = dict[str, Any]()

        neo4j_result: neo4j.AsyncResult = await tx.run(
            cypher, parameters=query_parameters
        )
        return await neo4j_result.single()
