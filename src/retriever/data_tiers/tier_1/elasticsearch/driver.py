import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, override

import orjson
import ormsgpack
from elasticsearch import AsyncElasticsearch
from elasticsearch import exceptions as es_exceptions
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.tier_1.elasticsearch.aggregating_querier import (
    enforce_timestamp,
    run_batch_query,
    run_single_query,
)
from retriever.data_tiers.tier_1.elasticsearch.meta import (
    extract_metadata_entries_from_blob,
    generate_operations,
    get_t1_indices,
    get_t1_metadata,
    merge_operations,
    stream_ubergraph_mapping,
)
from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESEdge,
    ESNode,
    ESPayload,
)
from retriever.data_tiers.utils import (
    parse_dingo_metadata_unhashed,
)
from retriever.types.dingo import DINGO_ADAPTER, DINGOMetadata
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import CURIE, BiolinkEntity, Infores
from retriever.utils.calls import get_metadata_client
from retriever.utils.redis import RedisClient
from retriever.utils.trapi import hash_hex

tracer = trace.get_tracer("lookup.execution.tracer")


class ElasticSearchDriver(DatabaseDriver):
    """An Elasticsearch driver."""

    es_connection: AsyncElasticsearch | None = None

    def __init__(self) -> None:
        """Initialize state. Connection is built lazily by `connect()`."""
        super().__init__()
        self.es_connection = None

    @override
    async def ping(self) -> None:
        """Probe Elasticsearch via the client `ping()`; raises on failure.

        Probes with the short connect timeout rather than the query
        timeout, so an unreachable backend is declared down in seconds
        instead of hanging for a full query timeout.
        """
        await self._ensure_client()
        assert self.es_connection is not None
        probe = self.es_connection.options(
            request_timeout=CONFIG.tier1.elasticsearch.connect_timeout
        )
        if not await probe.ping():
            raise ConnectionError("Elasticsearch ping returned False.")

    @override
    def _handle_ping_failure(self, exc: BaseException) -> None:
        super()._handle_ping_failure(exc)
        if self._is_dead_loop_error(exc):
            # Stale client is bound to the dead loop; can't await close on it.
            # Drop the reference and let the next ping lazy-create a fresh client.
            self.es_connection = None

    @override
    def _client_present(self) -> bool:
        return self.es_connection is not None

    @override
    def _build_client(self) -> None:
        """Build the Elasticsearch client with the query request timeout."""
        # TODO: auth details: token? user/pass?
        es_url = f"http://{CONFIG.tier1.elasticsearch.host}:{CONFIG.tier1.elasticsearch.port}"
        self.es_connection = AsyncElasticsearch(
            es_url, request_timeout=CONFIG.tier1.elasticsearch.query_timeout
        )

    @override
    def _recovery_callback(self) -> Callable[[], Awaitable[None]]:
        return self._refresh_metadata_cache

    async def _refresh_metadata_cache(self) -> None:
        """Repopulate the in-process tier-1 metadata cache from live ES."""
        if self.es_connection is None:
            return
        with contextlib.suppress(Exception):
            _ = await get_t1_metadata(
                self.es_connection,
                CONFIG.tier1.elasticsearch.index_name,
                bypass_cache=True,
            )

    async def _close_connection(self) -> None:
        """Close the ES client connection and drop the reference."""
        if self.es_connection is not None:
            await self.es_connection.close()
        self.es_connection = None

    @override
    async def wrapup(self) -> None:
        """Cancel the health loop first, then close the ES connection."""
        await super().wrapup()
        await self._close_connection()

    async def run(
        self,
        query: ESPayload | list[ESPayload],
        bypass_cache: bool = False,
        retries: int = 0,
    ) -> list[ESEdge] | list[list[ESEdge]]:
        """Execute query logic; one same-client retry on `ConnectionError`."""
        # Check ES connection instance
        if self.es_connection is None:
            raise RuntimeError(
                "Must use ElasticSearchDriver.initialize() before running queries."
            )

        if bypass_cache:
            query = enforce_timestamp(query)

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
            # One same-client retry covers transient network blips without
            # touching es_connection (which would race the health loop).
            if retries < 1:
                log.debug("ES ConnectionError; retrying once on same client.")
                await asyncio.sleep(0.1)
                return await self.run(
                    query, bypass_cache=bypass_cache, retries=retries + 1
                )
            raise
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

    async def fetch_single_node(self, _curie: str) -> ESNode | None:
        """Fetch a single canonical node from the Elasticsearch backend."""
        index_name = "ubergraph_nodes"

        if self.es_connection is None:
            raise RuntimeError(
                "Must use ElasticSearchDriver.initialize() before fetching node metadata."
            )

        with self.nudge_on_failure():
            response = await self.es_connection.search(
                index=index_name,
                size=1,
                query={"term": {"id": _curie}},
            )
        hits = response["hits"]["hits"]
        if len(hits) == 0:
            return None
        total_hits = response["hits"]["total"]["value"]
        if total_hits > 1:
            log.warning(
                f"Found {total_hits} canonical node hits for {_curie} in `ubergraph_nodes`; using the first match."
            )

        return ESNode.from_dict(hits[0]["_source"])

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

        bypass_cache = kwargs.get("bypass_cache", False)
        with self.nudge_on_failure():
            query_result = await self.run(query, bypass_cache)
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

        await RedisClient().set(
            hash_hex(hash(url)),
            ormsgpack.packb(metadata),
            compress=True,
            ttl=CONFIG.job.metakg.build_time,
        )
        log.success("DINGO Metadata retrieved!")

    async def _get_metadata(self, url: str, retries: int = 0) -> dict[str, Any] | None:
        """Obtain metadata for a given DINGO ingest."""
        metadata_pack = await RedisClient().get(hash_hex(hash(url)), compressed=True)

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
        bypass_cache: bool = False,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        # return await self.legacy_get_metadata()
        return await self.get_t1_operations(bypass_cache=bypass_cache)

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
        bypass_cache: bool = False,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        """Get tier1 operations based on metadata."""
        metadata_blob, indices = await self.get_valid_metadata(
            bypass_cache=bypass_cache
        )
        metadata_list = extract_metadata_entries_from_blob(metadata_blob, indices)
        operations, nodes = await generate_operations(metadata_list)

        return operations, nodes

    @override
    def stream_subclass_mapping(
        self, cutoff: int
    ) -> AsyncIterator[tuple[CURIE, list[CURIE]]]:
        """Stream (CURIE, descendants) from ES, dropping over-`cutoff` entries.

        `cutoff <= 0` keeps everything; memory stays bounded.
        """
        if self.es_connection is None:
            raise RuntimeError(
                "Must use ElasticSearchDriver.initialize() before streaming the subclass mapping."
            )
        return stream_ubergraph_mapping(self.es_connection, cutoff)
