import asyncio
from typing import Optional

import elasticsearch
from opentelemetry import trace

from elasticsearch import AsyncElasticsearch
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

        if retries <= CONFIG.tier1.elasticsearch.connect_retries:
            await self.close()
            await asyncio.sleep(8)
            log.error(
                f"Could not establish connection to neo4j, trying again... retry {retries + 1}"
            )
            return await self.connect(retries + 1)

        try:
            connection_info = await self.es_connection.info()
            log.error(f"Could not establish connection to neo4j, more info: {connection_info}")
            raise Exception(f"Could not establish connection to neo4j, error: {connection_info}")
        except Exception as e:
            log.error(f"Could not establish connection to neo4j, error: {e}")
            raise e
        finally:
            await self.close()


    @override
    async def connect(self, retries: int = 0) -> None:
        """Initialize a persistent connection to Elasticsearch instance"""
        log.info("checking ElasticSearch connection")

        if self.es_connection is None:
            self.setup_es_connection()

        is_connected = await self.check_es_connection()

        # retry logic
        if not is_connected:
            # reset connection for a clean slate
            await self.es_connection.close()
            await self.retry_es_connection(retries)

    @override
    async def close(self) -> None:
        """Close connection to Elasticsearch instance, if present"""
        if self.es_connection is not None:
            await self.es_connection.close()
        self.es_connection = None





