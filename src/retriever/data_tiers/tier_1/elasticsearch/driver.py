import asyncio
from typing import Any, cast, override

import orjson
from elasticsearch import AsyncElasticsearch
from elasticsearch import exceptions as es_exceptions
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.tier_1.elasticsearch.types import ESHit, ESPayload

tracer = trace.get_tracer("lookup.execution.tracer")


class ElasticSearchDriver(DatabaseDriver):
    """An Elasticsesarch driver."""

    es_connection: AsyncElasticsearch | None = None
    _failed: bool = False

    def setup_es_connection(self) -> None:
        """Setup connection to Elasticsearch instance."""
        # TODO: auth details: token? user/pass?

        es_url = f"http://{CONFIG.tier1.elasticsearch.host}:{CONFIG.tier1.elasticsearch.port}"
        self.es_connection = AsyncElasticsearch(
            es_url, request_timeout=CONFIG.tier1.elasticsearch.query_timeout
        )

    async def check_es_connection(self) -> bool:
        """A thin layer around es.ping() method to check es connection."""
        # es.ping() always resolves to True/False, no try-catch needed
        if self.es_connection is None:
            raise ValueError(
                "ES Connection must be initialized before it can be tested."
            )
        is_connected = await self.es_connection.ping()

        if is_connected:
            log.success("Elasticsearch connection successful!")

        return is_connected

    async def retry_es_connection(self, retries: int) -> None:
        """Retry connection to Elasticsearch and raise exception if retries exceeded."""
        # Keep trying to connect, if allowed
        if retries <= CONFIG.tier1.elasticsearch.connect_retries:
            await self.close()
            await asyncio.sleep(1)
            log.error(
                f"Could not establish connection to elasticsearch, trying again... retry {retries + 1}"
            )
            return await self.connect(retries + 1)

        # Retry limit reached
        try:
            # Connection will always be non-None by this point
            connection_info = await cast(AsyncElasticsearch, self.es_connection).info()
        except Exception as e:
            # Failed to get connection info
            log.error(f"Could not establish connection to elasticsearch, error: {e}")
            self._failed = True
            raise e
        finally:
            await self.close()

        # Theoretical corner case, when ping() failed but connection_info() succeeded
        log.error(
            f"Could not establish connection to elasticsearch, more info: {connection_info}"
        )
        raise Exception(
            f"Could not establish connection to elasticsearch, error: {connection_info}"
        )

    @override
    async def connect(self, retries: int = 0) -> None:
        """Initialize a persistent connection to Elasticsearch instance."""
        log.info("Checking ElasticSearch connection...")

        if self.es_connection is None:
            self.setup_es_connection()

        is_connected = await self.check_es_connection()

        # Evoke retry logic
        if not is_connected:
            await self.retry_es_connection(retries)

    @override
    async def close(self) -> None:
        """Close connection to Elasticsearch instance, if present."""
        if self.es_connection is not None:
            await self.es_connection.close()
        self.es_connection = None

    async def run(self, query: ESPayload) -> list[ESHit] | None:
        """Execute query logic."""
        # Check ES connection instance
        if self.es_connection is None:
            raise RuntimeError(
                "Must use ElasticSearchDriver.connect() before running queries."
            )

        try:
            response = await self.es_connection.search(
                index=CONFIG.tier1.elasticsearch.index_name,
                body=dict(query),
            )
        except es_exceptions.ConnectionError:
            await self.connect()
            return await self.run(query)
        except es_exceptions.ApiError as e:
            log.exception("Elasticsearch query returned non-200 HTTP status")
            raise e
        except es_exceptions.TransportError as e:
            log.exception("Elasticsearch query encountered a transport error")
            raise e
        except Exception as e:
            # TransportError and ApiError covers all Elasticsearch related exceptions.
            # Exceptions caught here are likely caused by connect()
            log.exception("An unexpected exception occurred during Elasticsearch query")
            raise e

        # extract results
        raw_results = response["hits"]["hits"]
        results = [r["_source"] for r in raw_results]

        # empty array
        if not results:
            return None

        return results

    @override
    @tracer.start_as_current_span("elasticsearch_query")
    async def run_query(
        self, query: ESPayload, *args: Any, **kwargs: Any
    ) -> list[ESHit] | None:
        """Use ES async client to execute query via the `_search` endpoint."""
        otel_span = trace.get_current_span()
        if not otel_span or not otel_span.is_recording():
            otel_span = None
        else:
            otel_span.add_event(
                "elasticsearch_query_start",
                attributes={"query_body": orjson.dumps(query).decode()},
            )

        query_result = await self.run(query)
        if otel_span is not None:
            otel_span.add_event("elasticsearch_query_end")

        return query_result
