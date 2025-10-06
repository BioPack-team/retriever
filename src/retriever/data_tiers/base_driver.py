from abc import ABC, abstractmethod
from typing import Any

from retriever.utils.general import Singleton


class DatabaseDriver(ABC, metaclass=Singleton):
    """A driver which handles interfacing with a given database backend.

    The driver is completely abstracted from TRAPI, receiving a query in its own language
    and responding in whatever format.
    """

    _failed: bool = False

    @property
    def is_failed(self) -> bool:
        """Returns True if the backend connection has failed unrecoverably."""
        return self._failed

    @abstractmethod
    async def connect(self) -> None:
        """Initialize a persistent connection to the database backend.

        This method should be called in server.py lifespan.
        """

    @abstractmethod
    async def run_query(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a query against the database backend and return the result."""

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection so the server can wrap up.

        This method should be called in server.py lifespan.
        """
