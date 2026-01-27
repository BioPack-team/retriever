import asyncio

from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers import tier_manager
from retriever.types.general import EntityToEntityMapping
from retriever.utils.general import Singleton


class SubclassMapping(metaclass=Singleton):
    """A mapping from primary node ID to its ontological descendents."""

    _timeout_task: asyncio.Task[None] | None = None
    _initialized: bool = False
    _mapping: EntityToEntityMapping | None = None

    async def initialize(self) -> None:
        """Obtain the mapping from the tier 1 driver."""
        logger.info("Initializing subclass mapping...")
        if CONFIG.job.lookup.implicit_subclassing:
            try:
                self._mapping = await tier_manager.get_driver(1).get_subclass_mapping()
                logger.success("Subclass mapping initialized!")
            except Exception:
                logger.error(
                    "Unable to initialize subclass mapping, proceeding without implicit subclassing..."
                )
        else:
            logger.info("Implicit subclassing disabled, skipping initialization.")

        self._initialized = True
        self._timeout_task = asyncio.create_task(
            self.expire(CONFIG.job.metakg.build_time)
        )

    async def get(self) -> EntityToEntityMapping | None:
        """Safely get the mapping."""
        if not self._initialized:
            await self.initialize()
        return self._mapping

    async def expire(self, timeout: float = 0) -> None:
        """Expire the initialization, requiring a new update."""
        try:
            await asyncio.sleep(timeout)
            self._initialized = False
        except asyncio.CancelledError:
            return

    async def wrapup(self) -> None:
        """Stop the expire task."""
        if self._timeout_task:
            self._timeout_task.cancel()
