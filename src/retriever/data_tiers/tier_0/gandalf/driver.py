import gc
from typing import Any, override

from cachetools import TTLCache
from loguru import logger as log
from opentelemetry import trace
from bmt import Toolkit
from gandalf import CSRGraph, lookup

from retriever.config.general import CONFIG, GandalfSettings
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import BackendResult, EntityToEntityMapping
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity, Infores


class GandalfDriver(DatabaseDriver):
    """Driver for Gandalf."""

    settings: GandalfSettings
    endpoint: str
    query_timeout: float
    connect_retries: int
    version: str | None = None

    _failed: bool = False
    _version_cache: TTLCache[str, str | None] = TTLCache(maxsize=1, ttl=60)
    _mapping_cache: TTLCache[str, dict[str, Any] | None] = TTLCache(maxsize=1, ttl=300)
    _grpc_endpoints: list[str]

    def __init__(
        self,
        *,
        version: str | None = None,
    ) -> None:
        """Initialize the Gandalf driver with connection settings.

        Args:
            version: An optional, fixed schema version to use, bypassing auto-detection.
                     This parameter has the highest precedence.
        """
        self.settings = CONFIG.tier0.gandalf
        self.query_timeout = self.settings.query_timeout
        self.connect_retries = self.settings.connect_retries

        self.bmt = Toolkit()
        self.gandalf = CSRGraph.load_mmap("/app/gandalf_mmap")
        # Freeze all objects allocated so far (graph + BMT) into a permanent
        # generation that the cyclic GC will never scan.  This makes Gen 2
        # collections cheap because they skip the large CSR arrays.
        gc.collect()
        gc.freeze()
        # Raise thresholds so Gen 2 collections are less frequent even for
        # the (now-small) unfrozen query-time object set.
        gc.set_threshold(50_000, 50, 50)

        # Version precedence: constructor arg > env var > auto-detect from DB
        if version is not None:
            self.version = version
            log.debug(f"Using schema version from constructor parameter: {version}")
        elif self.settings.preferred_version:
            self.version = self.settings.preferred_version
            log.debug(
                "Using schema version from TIER0__GANDALF__PREFERRED_VERSION env var: "
                + f"{self.version}"
            )
        else:
            self.version = None

    def remove_none_values(self, d: dict) -> dict:
        """Remove all None values."""
        if not isinstance(d, dict):
            return d
        return {k: self.remove_none_values(v) for k, v in d.items() if v is not None}

    @override
    async def run_query(
        self, query: dict, *args: Any, **kwargs: Any
    ) -> dict:
        """Execute a query against the Dgraph database and parse into dataclasses.

        Args:
            query: The Gandalf query to execute
            *args: Variable positional arguments (unused, for protocol compatibility)
            **kwargs: Additional arguments

        Returns:
            TRAPI Response

        """
        otel_span = trace.get_current_span()
        if otel_span and otel_span.is_recording():
            otel_span.add_event(
                "gandalf_query_start", attributes={"gandalf_query": query}
            )
        else:
            otel_span = None

        try:
            query = self.remove_none_values(query)
            result = lookup(
                self.gandalf,
                query,
                self.bmt,
            )
        except TimeoutError as e:
            if otel_span is not None:
                otel_span.add_event("gandalf_query_timeout")
            raise TimeoutError(
                f"Gandalf query exceeded {self.query_timeout}s timeout"
            ) from e

        if otel_span is not None:
            otel_span.add_event("gandalf_query_end")

        return result

    @override
    async def connect(self, retries: int = 0) -> None:
        """Connect to Gandalf using selected protocol."""

    @override
    async def close(self) -> None:
        """Close the connection to Gandalf and clean up resources."""

    @override
    async def get_metadata(self) -> dict[str, Any] | None:
        """Queries Gandalf for the active schema's metadata mapping.

        The mapping is stored as a msgpack-serialized JSON blob in the
        schema_metadata_mapping field. This method retrieves and deserializes it
        for the active schema version.

        The result is cached per-version with a 5-minute TTL.

        Returns:
            Deserialized mapping dictionary, or None if not found or on error.
        """
        return self.gandalf.graph_metadata

    @override
    async def get_operations(
        self,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        metadata = await self.get_metadata()
        if metadata is None:
            raise ValueError(
                "Unable to obtain metadata from backend, cannot parse operations."
            )
        infores = Infores(CONFIG.tier0.backend_infores)
        operations, nodes = parse_dingo_metadata(DINGOMetadata(**metadata), 0, infores)
        log.success(f"Parsed {infores} as a Tier 0 resource.")
        return operations, nodes

    @override
    async def get_subclass_mapping(self) -> EntityToEntityMapping:
        raise NotImplementedError("Tier 0 does not implement subclass mapping.")

    async def get_active_version(self) -> str | None:
        """Queries Dgraph for the active schema version and caches the result.

        If a version was provided at initialization, it will be returned directly.

        This method implements manual caching to be async-safe.
        """
        # If a version was manually set on the driver, always use it.
        if self.version:
            log.debug(f"Using manually specified Gandalf schema version: {self.version}")
            return self.version

        return None
