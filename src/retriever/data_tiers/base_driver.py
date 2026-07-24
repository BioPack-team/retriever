from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import CURIE, BiolinkEntity
from retriever.utils.backend_client import BackendClient


class DatabaseDriver(BackendClient, ABC):
    """Driver interface to a database backend; queries are in the backend's own language."""

    healthcheck_interval: float | None = None
    """Tier drivers ping on-demand; query callers nudge `request_health_check()` on failure."""

    @abstractmethod
    async def run_query(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a query against the database backend and return the result."""

    @abstractmethod
    async def get_metadata(self, bypass_cache: bool = False) -> dict[str, Any] | None:
        """Backend metadata; `bypass_cache=True` forces a live fetch."""

    def get_release_version(self) -> str | None:
        """Cached data release version of this backend's knowledge, if known."""
        return None

    @abstractmethod
    async def get_operations(
        self,
        bypass_cache: bool = False,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        """Operations and Nodes from this driver; `bypass_cache=True` forces a live fetch."""

    @abstractmethod
    def stream_subclass_mapping(
        self, cutoff: int
    ) -> AsyncIterator[tuple[CURIE, list[CURIE]]]:
        """Stream (CURIE, descendants) pairs, dropping over-`cutoff` entries.

        `cutoff <= 0` keeps everything. The leader only streams from the tier-1
        driver; other backends raise `NotImplementedError`.
        """
        ...
