import asyncio
from contextlib import suppress
from http import HTTPStatus
from typing import Any, override

import aiohttp
import orjson
from aiohttp import ClientTimeout
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG, GandalfSettings
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import EntityToEntityMapping
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity, Infores, QueryDict, ResponseDict


class GandalfDriver(DatabaseDriver):
    """Driver for Gandalf."""

    metadata: dict[str, Any] | None
    settings: GandalfSettings
    endpoint: str
    query_timeout: float
    connect_retries: int

    _http_session: aiohttp.ClientSession | None = None

    _failed: bool = False

    def __init__(
        self,
    ) -> None:
        """Initialize the Gandalf driver with connection settings.

        Args:
            version: An optional, fixed schema version to use, bypassing auto-detection.
                     This parameter has the highest precedence.
        """
        self.settings = CONFIG.tier0.gandalf
        self.query_timeout = self.settings.query_timeout
        self.connect_retries = self.settings.connect_retries
        self._http_session = None

        self.endpoint = self.settings.http_endpoint
        self.metadata = None

    def remove_none_values(self, d: Any) -> Any:
        """Remove all None values."""
        if not isinstance(d, dict):
            return d
        return {k: self.remove_none_values(v) for k, v in d.items() if v is not None}  # pyright:ignore[reportUnknownVariableType]

    @override
    async def run_query(
        self, query: QueryDict, *args: Any, **kwargs: Any
    ) -> ResponseDict:
        """Execute a query against the Gandalf database and parse into dataclasses.

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
                "gandalf_query_start",
                attributes={"gandalf_query": orjson.dumps(dict(**query)).decode()},
            )
        else:
            otel_span = None

        try:
            query = self.remove_none_values(query)
            result = await self._run_http_query(
                query,
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

    async def _run_http_query(
        self,
        query: QueryDict,
    ) -> ResponseDict:
        """Execute query using HTTP protocol with DQL.

        Args:
            query: The Gandalf TRAPI query

        Returns:
            Parsed response
        """
        assert self._http_session is not None, "HTTP session not initialized"

        async with self._http_session.post(
            f"{self.endpoint}/query",
            json=query,
            timeout=ClientTimeout(total=self.query_timeout),
        ) as response:
            if response.status != HTTPStatus.OK:
                text = await response.text()
                raise RuntimeError(
                    f"Gandalf HTTP query failed with status {response.status}: {text}"
                )

            message = await response.json()

            if "errors" in message:
                raise RuntimeError(
                    f"Gandalf query returned errors: {message['errors']}"
                )

            return message

    @override
    async def connect(self, retries: int = 0) -> None:
        """Connect to Gandalf using selected protocol."""
        log.info("Checking Gandalf connection...")
        try:
            await self._connect_http()
            log.success("Gandalf http connection successful!")

        except Exception as e:
            await self._cleanup_connections()
            if retries < self.connect_retries:
                await asyncio.sleep(1)
                log.error(f"""
                    Could not establish connection to Gandalf via http,
                    trying again... retry {retries + 1}
                """)
                await self.connect(retries + 1)
            else:
                log.error(f"Could not establish connection to Gandalf, error: {e}")
                self._failed = True
                raise e

    async def _connect_http(self) -> None:
        """Establish HTTP connection to Gandalf."""
        self._http_session = aiohttp.ClientSession()
        # Test connection with a simple query
        async with self._http_session.get(
            f"{self.endpoint}/metadata",
            timeout=ClientTimeout(total=self.query_timeout),
        ) as response:
            if response.status != HTTPStatus.OK:
                text = await response.text()
                raise ConnectionError(
                    f"HTTP connection failed with status {response.status}: {text}"
                )
            self.metadata = await response.json()

    async def _cleanup_connections(self) -> None:
        """Clean up any open connections."""
        # Close HTTP session if open
        if self._http_session is not None:
            with suppress(Exception):
                await self._http_session.close()
            self._http_session = None

    @override
    async def close(self) -> None:
        """Close the connection to Gandalf and clean up resources."""
        await self._cleanup_connections()

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
        return self.metadata

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
