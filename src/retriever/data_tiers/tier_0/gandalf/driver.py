import math
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from http import HTTPStatus
from typing import Any, override

import httpx
import orjson
import zstandard
from loguru import logger as log
from opentelemetry import trace

from retriever.config.general import CONFIG, GandalfSettings
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import EntityToEntityMapping
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity, Infores, QueryDict, ResponseDict

ZSTD_COMPRESSOR = zstandard.ZstdCompressor()


class GandalfDriver(DatabaseDriver):
    """Driver for Gandalf."""

    metadata: dict[str, Any] | None
    settings: GandalfSettings
    endpoint: str
    query_timeout: float
    connect_timeout: float

    _http_session: httpx.AsyncClient | None = None

    def __init__(
        self,
    ) -> None:
        """Initialize the Gandalf driver with connection settings."""
        super().__init__()
        self.settings = CONFIG.tier0.gandalf
        self.query_timeout = self.settings.query_timeout
        self.connect_timeout = self.settings.connect_timeout
        self._http_session = None

        self.endpoint = self.settings.http_endpoint
        self.metadata = None

    @override
    def _client_present(self) -> bool:
        return self._http_session is not None

    @override
    def _build_client(self) -> None:
        """Build the httpx client with a split connect/read timeout.

        Connects fail fast (`connect_timeout`) so an unreachable backend is
        detected quickly, while reads get the full `query_timeout` for
        long-running queries.
        """
        self._http_session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.query_timeout, connect=self.connect_timeout)
        )

    @override
    async def ping(self) -> None:
        """HEAD the base endpoint; 405 or 2xx means up.

        Bounds the whole probe by `connect_timeout` rather than the query
        timeout, so an unreachable backend is declared down in seconds
        instead of hanging for a full query timeout.
        """
        await self._ensure_client()
        assert self._http_session is not None
        response = await self._http_session.head(
            self.endpoint,
            timeout=self.connect_timeout,
        )
        if response.status_code == HTTPStatus.METHOD_NOT_ALLOWED:
            return
        response.raise_for_status()

    @override
    def _handle_ping_failure(self, exc: BaseException) -> None:
        super()._handle_ping_failure(exc)
        if self._is_dead_loop_error(exc) or isinstance(exc, httpx.PoolTimeout):
            # Loop-closed: session is bound to a dead loop, can't await aclose.
            # PoolTimeout: httpx client is effectively dead
            # see https://github.com/encode/httpx/discussions/2556
            # Either way, drop the reference and let `_ensure_client` rebuild.
            self._http_session = None

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
        otel_span = trace.get_current_span()  # Serialize once...
        query_json = orjson.dumps(query).decode()
        if otel_span and otel_span.is_recording():
            otel_span.add_event(
                "gandalf_query_start",
                attributes={"gandalf_query": query_json},
            )
        else:
            otel_span = None

        try:
            with self.nudge_on_failure():
                await self._ensure_client()
                result = await self._run_http_query(query_json)
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
    ) -> ResponseDict:
        """Execute query using HTTP protocol with DQL.

        Args:
            query_json: The Gandalf TRAPI query in JSON format

        Returns:
            Parsed response
        """
        if self._http_session is None:
            raise RuntimeError("HTTP session not initialized")

        start = time.time()
        log.debug("Querying Gandalf...")
        try:
            payload = (
                ZSTD_COMPRESSOR.compress(query_json.encode())
                if CONFIG.tier0.gandalf.compress_query
                else query_json
            )
            headers = {
                "Content-Type": "application/json",
            }
            if CONFIG.tier0.gandalf.compress_query:
                headers["Content-Encoding"] = "zstd"

            response = await self._http_session.post(
                f"{self.endpoint}/query",
                content=payload,
                timeout=self.query_timeout,
                headers=headers,
            )
            response.raise_for_status()
            end = time.time()
            trapi_response = ResponseDict(**orjson.loads(response.text))
            transform = time.time()

            log.debug(
                f"Gandalf query took {math.ceil((end - start) * 1000)}ms. Deserialization took {math.ceil((transform - end) * 1000)}ms"
            )

            if "errors" in trapi_response:
                raise RuntimeError(
                    f"Gandalf query returned errors: {trapi_response['errors']}"
                )

            return trapi_response
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Gandalf query failed with status {error.response.status_code} with body {error.response.text}"
            ) from error
        except httpx.TimeoutException as error:
            # Surface as TimeoutError so run_query's timeout handler fires;
            # httpx timeouts don't subclass the builtin TimeoutError.
            raise TimeoutError(
                f"Gandalf query timed out after {self.query_timeout}s"
            ) from error
        except httpx.HTTPError as error:
            raise RuntimeError(
                "An unhandled HTTP error occured while querying Gandalf."
            ) from error

    @override
    async def _connect(self) -> None:
        """Build the HTTP client and load fresh metadata."""
        await self._ensure_client()
        await self._fetch_metadata()

    @override
    def _recovery_callback(self) -> Callable[[], Awaitable[None]]:
        return self._fetch_metadata

    async def _fetch_metadata(self) -> None:
        """Pull fresh metadata from Gandalf into the in-process cache."""
        if self._http_session is None:
            raise RuntimeError("HTTP session not initialized")
        response = await self._http_session.get(f"{self.endpoint}/metadata")
        if response.status_code != HTTPStatus.OK:
            raise ConnectionError(
                f"HTTP metadata fetch failed with status {response.status_code}: {response.text}"
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
    async def wrapup(self) -> None:
        """Cancel the health loop first, then tear down the HTTP session."""
        await super().wrapup()
        await self._cleanup_connections()

    @override
    async def get_metadata(self, bypass_cache: bool = False) -> dict[str, Any] | None:
        """Cached metadata; on `bypass_cache=True` refresh first, falling back to cache on failure."""
        if bypass_cache and self._http_session is not None:
            try:
                await self._fetch_metadata()
            except Exception as exc:
                log.warning(
                    f"Gandalf metadata refresh failed; using cached copy. Error: {exc}"
                )
        return self.metadata

    @override
    async def get_operations(
        self,
        bypass_cache: bool = False,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        metadata = await self.get_metadata(bypass_cache=bypass_cache)
        if metadata is None:
            raise ValueError(
                "Unable to obtain metadata from backend, cannot parse operations."
            )
        infores = Infores(CONFIG.tier0.backend_infores)
        operations, nodes = parse_dingo_metadata(DINGOMetadata(**metadata), 0, infores)
        log.success(f"Parsed {infores} as a Tier 0 resource.")
        return operations, nodes

    @override
    async def get_subclass_mapping(
        self, bypass_cache: bool = False
    ) -> EntityToEntityMapping:
        raise NotImplementedError("Tier 0 does not implement subclass mapping.")
