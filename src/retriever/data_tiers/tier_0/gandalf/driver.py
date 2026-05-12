import asyncio
import math
import time
from contextlib import suppress
from http import HTTPStatus
from typing import Any, override

import httpx
from loguru import logger as log
from opentelemetry import trace
from translator_tom import Biolink, Infores, Query, Response

from retriever.config.general import CONFIG, GandalfSettings
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import EntityToEntityMapping
from retriever.types.metakg import Operation, OperationNode


class GandalfDriver(DatabaseDriver):
    """Driver for Gandalf."""

    metadata: dict[str, Any] | None
    settings: GandalfSettings
    endpoint: str
    query_timeout: float
    connect_retries: int

    _http_session: httpx.AsyncClient | None = None

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

    @override
    async def run_query(self, query: Query, *args: Any, **kwargs: Any) -> Response:
        """Execute a query against the Gandalf database and parse into dataclasses.

        Args:
            query: The Gandalf query to execute
            *args: Variable positional arguments (unused, for protocol compatibility)
            **kwargs: Additional arguments

        Returns:
            TRAPI Response

        """
        otel_span = trace.get_current_span()  # Serialize once...
        query_json = query.to_json()
        if otel_span and otel_span.is_recording():
            otel_span.add_event(
                "gandalf_query_start",
                attributes={"gandalf_query": query_json},
            )
        else:
            otel_span = None

        try:
            result = await self._run_http_query(
                query_json,
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
        query_json: str,
    ) -> Response:
        """Execute query using HTTP protocol with DQL.

        Args:
            query_json: The Gandalf TRAPI query in JSON format

        Returns:
            Parsed response
        """
        assert self._http_session is not None, "HTTP session not initialized"

        start = time.time()
        log.debug("Querying Gandalf...")
        try:
            response = await self._http_session.post(
                f"{self.endpoint}/query",
                content=query_json,
                headers={"Content-Type": "application/json"},
                timeout=self.query_timeout,
            )
            response.raise_for_status()
            end = time.time()
            trapi_response = Response.from_json(response.text)
            transform = time.time()

            log.debug(
                f"Gandalf query took {math.ceil((end - start) * 1000)}ms. Deserialization took {math.ceil((transform - end) * 1000)}ms"
            )

            if errors := trapi_response.extra_get("errors"):
                raise RuntimeError(f"Gandalf query returned errors: {errors}")

            return trapi_response
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Gandalf query failed with status {error.response.status_code} with body {error.response.text}"
            ) from error
        except httpx.HTTPError as error:
            raise RuntimeError(
                "An unhandled HTTP error occured while querying Gandalf."
            ) from error

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
        self._http_session = httpx.AsyncClient()
        # Test connection with a simple query
        response = await self._http_session.get(
            f"{self.endpoint}/metadata",
            timeout=self.query_timeout,
        )
        if response.status_code != HTTPStatus.OK:
            raise ConnectionError(
                f"HTTP connection failed with status {response.status_code}: {response.text}"
            )
        self.metadata = response.json()

    async def _cleanup_connections(self) -> None:
        """Clean up any open connections."""
        # Close HTTP session if open
        if self._http_session is not None:
            with suppress(Exception):
                await self._http_session.aclose()
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
    ) -> tuple[list[Operation], dict[Biolink.Entity, OperationNode]]:
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
