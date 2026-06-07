from abc import ABC, abstractmethod
from typing import Any

from retriever.types.general import EntityToEntityMapping
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity
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

    @abstractmethod
    async def get_operations(
        self,
        bypass_cache: bool = False,
    ) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
        """Operations and Nodes from this driver; `bypass_cache=True` forces a live fetch."""

    @abstractmethod
    async def get_subclass_mapping(
        self, bypass_cache: bool = False
    ) -> EntityToEntityMapping:
        """Nodes to ontological descendants; `bypass_cache=True` forces a live fetch."""
