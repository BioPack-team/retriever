import asyncio
from typing import Optional, Any

from opentelemetry import trace

from elasticsearch import AsyncElasticsearch, exceptions as es_exceptions
from typing_extensions import override

from loguru import logger as log

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver

tracer = trace.get_tracer("lookup.execution.tracer")


class ElasticSearchDriver(DatabaseDriver):
    """An Elasticsesarch driver."""
    es_connection: Optional[AsyncElasticsearch] = None

    def setup_es_connection(self) -> None:
        """setup connection to Elasticsearch instance"""

        # todo auth details: token? user/pass?

        es_url = "http://%s:%s" % (CONFIG.tier1.elasticsearch.host, CONFIG.tier1.elasticsearch.port)
        self.es_connection = AsyncElasticsearch(
            es_url,
            request_timeout=CONFIG.tier1.elasticsearch.request_timeout
        )

    async def check_es_connection(self) -> bool:
        """a thin layer around es.ping() method to check es connection"""

        # es.ping() always resolves to True/False, not try-catch needed
        is_connected = await self.es_connection.ping()

        if is_connected:
            log.success("Elasticsearch connection successful!")

        return is_connected


    async def retry_es_connection(self, retries: int) -> None:
        """retry connection to Elasticsearch and raise exception if retries exceeded"""

        # keep trying to connect, if allowed
        if retries <= CONFIG.tier1.elasticsearch.connect_retries:
            await self.close()
            await asyncio.sleep(8)
            log.error(
                f"Could not establish connection to elasticsearch, trying again... retry {retries + 1}"
            )
            return await self.connect(retries + 1)


        # retry limit reached
        try:
            connection_info = await self.es_connection.info()
        except Exception as e:
            # failed to get connection info
            log.error(f"Could not establish connection to elasticsearch, error: {e}")
            raise e
        finally:
            await self.close()

        # theoretical corner case, when ping() failed but connection_info() succeeded
        log.error(f"Could not establish connection to elasticsearch, more info: {connection_info}")
        raise Exception(f"Could not establish connection to elasticsearch, error: {connection_info}")


    @override
    async def connect(self, retries: int = 0) -> None:
        """Initialize a persistent connection to Elasticsearch instance"""
        log.info("checking ElasticSearch connection")

        if self.es_connection is None:
            self.setup_es_connection()

        is_connected = await self.check_es_connection()

        # evoke retry logic
        if not is_connected:
            await self.retry_es_connection(retries)

    @override
    async def close(self) -> None:
        """Close connection to Elasticsearch instance, if present"""

        if self.es_connection is not None:
            await self.es_connection.close()
        self.es_connection = None


    async def run(self, query:dict) -> list[dict] | None:
        """Execute query logic"""

        # check ES connection instance
        if self.es_connection is None:
            raise RuntimeError(
                "Must use ElasticSearchDriver.connect() before running queries."
            )

        try:
            response = await self.es_connection.search(
                index=CONFIG.tier1.elasticsearch.index_name,
                body=query,
            )
        except es_exceptions.ConnectionError as e:
            await self.connect()
            return await self.run(query)
        except es_exceptions.ApiError as e:
            log.exception("elasticsearch query error encountered.")
            raise e
        except es_exceptions.TransportError as e:
            log.exception("elasticsearch error encountered.")
            raise e
        except Exception as e:
            log.exception("other error encountered")
            raise e

        results = response["hits"]["hits"]

        # empty array
        if not results:
            return None

        return results



    @override
    @tracer.start_as_current_span("elasticsearch_query")
    async def run_query(self, query: dict, *args: Any, **kwargs: Any) -> list[dict] | None:
        """Use ES async client to execute query via the `_search` endpoint"""

        otel_span = trace.get_current_span()
        if not otel_span or not otel_span.is_recording():
            otel_span = None
        else:
            otel_span.add_event(
                "elasticsearch_query_start", attributes={"query_body": query}
            )

        query_result = await self.run(query)
        if otel_span is not None:
            otel_span.add_event("elasticsearch_query_end")

        return query_result




