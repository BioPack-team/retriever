import asyncio
from typing import Any, cast, override

import orjson
import ormsgpack
from elasticsearch import AsyncElasticsearch
from elasticsearch import exceptions as es_exceptions
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.tier_1.elasticsearch.aggregating_querier import (
    run_batch_query,
    run_single_query,
)
from retriever.data_tiers.tier_1.elasticsearch.meta import (
    extract_metadata_entries_from_blob,
    generate_operations,
    get_t1_indices,
    get_t1_metadata,
    get_ubergraph_info,
    merge_operations,
)
from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESEdge,
    ESPayload,
)
from retriever.data_tiers.utils import (
    parse_dingo_metadata_unhashed,
)
from retriever.types.dingo import DINGO_ADAPTER, DINGOMetadata
from retriever.types.general import EntityToEntityMapping
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity, Infores
from retriever.utils.calls import get_metadata_client
from retriever.utils.redis import REDIS_CLIENT
from retriever.utils.trapi import hash_hex

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
            log.info(f"Using ES index: {CONFIG.tier1.elasticsearch.index_name}")

        return is_connected

    async def retry_es_connection(self, retries: int) -> None:
        """Retry connection to Elasticsearch and raise exception if retries exceeded."""
        # Keep trying to connect, if allowed
        if retries <= CONFIG.tier1.elasticsearch.connect_retries:
            await self.close()
            await asyncio.sleep(1)
            log.error(
                f"Could not establish connection to elasticsearch_tests, trying again... retry {retries + 1}"
            )
            return await self.connect(retries + 1)

        # Retry limit reached
        try:
            # Connection will always be non-None by this point
            connection_info = await cast(AsyncElasticsearch, self.es_connection).info()
        except Exception as e:
            # Failed to get connection info
            log.error(
                f"Could not establish connection to elasticsearch_tests, error: {e}"
            )
            self._failed = True
            raise e
        finally:
            await self.close()

        # Theoretical corner case, when ping() failed but connection_info() succeeded
        log.error(
            f"Could not establish connection to elasticsearch_tests, more info: {connection_info}"
        )
        raise Exception(
            f"Could not establish connection to elasticsearch_tests, error: {connection_info}"
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

    async def run(
        self, query: ESPayload | list[ESPayload]
    ) -> list[ESEdge] | list[list[ESEdge]]:
        """Execute query logic."""
        # Check ES connection instance
        if self.es_connection is None:
            raise RuntimeError(
                "Must use ElasticSearchDriver.connect() before running queries."
            )

        try:
            # select query method based on incoming payload
            if isinstance(query, list):
                results = await run_batch_query(
                    es_connection=self.es_connection,
                    index_name=CONFIG.tier1.elasticsearch.index_name,
                    queries=query,
                )
            else:
                results = await run_single_query(
                    es_connection=self.es_connection,
                    index_name=CONFIG.tier1.elasticsearch.index_name,
                    query=query,
                )
        except es_exceptions.ConnectionTimeout as e:
            log.exception(f"query timed out: {e}")
            raise e
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

        return results

    @override
    @tracer.start_as_current_span("elasticsearch_query")
    async def run_query(
        self, query: ESPayload | list[ESPayload], *args: Any, **kwargs: Any
    ) -> list[ESEdge] | list[list[ESEdge]]:
        """Use ES async client to execute query via the `_search/_msearch` endpoints."""
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

    async def _pull_metadata(self, url: str) -> None:
        """Update metadata for a given DINGO ingest."""
        log.info(f"Pulling DINGO Metadata from {url}...")

        client = get_metadata_client()
        response = await client.get(url)
        response.raise_for_status()

        raw_data = response.json()
        metadata = DINGO_ADAPTER.validate_python(raw_data)

        await REDIS_CLIENT.set(
            hash_hex(hash(url)),
            ormsgpack.packb(metadata),
            compress=True,
            ttl=CONFIG.job.metakg.build_time,
        )
        log.success("DINGO Metadata retrieved!")

    async def _get_metadata(self, url: str, retries: int = 0) -> dict[str, Any] | None:
        """Obtain metadata for a given DINGO ingest."""
        metadata_pack = await REDIS_CLIENT.get(hash_hex(hash(url)), compressed=True)

        if metadata_pack is None:
            await self._pull_metadata(url)
            if retries >= 3:  # noqa: PLR2004
                return None
            return await self._get_metadata(url, retries + 1)

        # Don't validate because if we've gotten it at this stage, it's already been validated
        metadata = ormsgpack.unpackb(metadata_pack)
        return metadata

    async def legacy_get_metadata(self) -> dict[str, Any] | None:
        """Legacy method for loading metadata remotely."""
        return await self._get_metadata(CONFIG.tier1.metadata_url)

    async def legacy_get_operations(
        self,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        """Legacy method for getting operations based on unified metadata."""
        metadata = await self.legacy_get_metadata()
        if metadata is None:
            raise ValueError(
                "Unable to obtain metadata from backend, cannot parse operations."
            )
        infores = Infores(CONFIG.tier1.backend_infores)
        # operations, nodes = parse_dingo_metadata(DINGOMetadata(**metadata), 1, infores)
        operations, nodes = parse_dingo_metadata_unhashed(
            DINGOMetadata(**metadata), 1, infores
        )
        operations = merge_operations(operations)
        log.success(f"Parsed {infores} as a Tier 1 resource.")

        return operations, nodes

    @override
    async def get_metadata(self, bypass_cache: bool = False) -> dict[str, Any] | None:
        return await get_t1_metadata(
            es_connection=self.es_connection,
            indices_alias=CONFIG.tier1.elasticsearch.index_name,
            bypass_cache=bypass_cache,
        )

    @override
    async def get_operations(
        self,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        # return await self.legacy_get_metadata()
        return await self.get_t1_operations()

    async def get_valid_metadata(
        self, bypass_cache: bool = False
    ) -> tuple[dict[str, Any], list[str]]:
        """Get valid metadata and raise exception if failed."""
        metadata_blob = await self.get_metadata(bypass_cache)

        if metadata_blob is None:
            raise ValueError(
                "Unable to obtain metadata from backend, cannot parse operations.",
                f"Elasticsearch config: {CONFIG.tier1.elasticsearch}",
            )

        if self.es_connection is None:
            raise ValueError("Elasticsearch connection not configured.")

        indices = await get_t1_indices(self.es_connection)

        # ensure metadata matches indices
        mismatched = any(metadata_blob.get(i) is None for i in indices)

        if mismatched:
            if not bypass_cache:
                log.error("Possibly stale data got from cache. Refetching remotely.")
                return await self.get_valid_metadata(bypass_cache=True)
            else:
                raise ValueError("Invalid metadata retrieved.")

        return metadata_blob, indices

    async def get_t1_operations(
        self,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        """Get tier1 operations based on metadata."""
        metadata_blob, indices = await self.get_valid_metadata()
        metadata_list = extract_metadata_entries_from_blob(metadata_blob, indices)
        operations, nodes = await generate_operations(metadata_list)

        return operations, nodes

    @override
    async def get_subclass_mapping(
        self,
    ) -> EntityToEntityMapping:
        """Get UBERGRAPH nodes mapping/adjacency list."""
        return await get_ubergraph_info(self.es_connection)
