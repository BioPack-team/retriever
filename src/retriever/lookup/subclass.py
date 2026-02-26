import asyncio
from collections.abc import Callable
from typing import cast, override

import ormsgpack
from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers import tier_manager
from retriever.types.trapi import CURIE
from retriever.utils.general import BatchedAction
from retriever.utils.redis import RedisClient

REDIS_CLIENT = RedisClient()

MAPPING_ID = "SubclassHashMap"


class SubclassMapping(BatchedAction):
    """A redis-backed mapping from primary node CURIE to its ontological descendants."""

    queue_delay: float = 0.05
    # Essentially should flush every interval
    batch_size: int = 1000
    flush_time: float = 0

    is_leader: bool = False
    subscriptions: dict[CURIE, list[Callable[[list[CURIE]], None]]]

    redis_setup_batch_size: int = 5000

    def __init__(self, leader: bool = False) -> None:
        """Initialize an instance."""
        self.is_leader = leader
        self.subscriptions = {}
        super().__init__()

    @override
    async def initialize(self) -> None:
        """Set up the redis hash and initialize the batch handler."""
        if not CONFIG.job.lookup.implicit_subclassing:
            logger.info("Implicit subclassing disabled, skipping initialization.")
            return await super().initialize()

        if not self.is_leader:  # Only need leader to update the redis setup
            return await super().initialize()

        try:
            logger.info("Initializing subclass mapping...")
            mapping = await tier_manager.get_driver(1).get_subclass_mapping()

            # Send to redis in batches to avoid overwhelming it
            packed_mapping = dict[str, bytes]()
            for curie, descendants in mapping.items():
                packed_mapping[curie] = ormsgpack.packb(descendants)

                if len(packed_mapping) >= self.redis_setup_batch_size:
                    await REDIS_CLIENT.hset(
                        MAPPING_ID,
                        mapping=packed_mapping,
                        ttl=CONFIG.job.metakg.build_time,
                    )
                    packed_mapping = {}

            logger.success(
                f"Subclass mapping initialized with {len(mapping)} ancestors."
            )

            self.tasks.append(asyncio.create_task(self.rebuild()))
        except Exception:
            logger.exception(
                "Unable to initialize subclass mapping due to below error, proceeding without implicit subclassing..."
            )

        return await super().initialize()

    async def get(self, curie: CURIE) -> list[CURIE]:
        """Get the descendants of a given CURIE."""
        future = asyncio.Future[list[CURIE]]()

        def return_result(curies: list[CURIE]) -> None:
            self.subscriptions[curie].remove(return_result)
            future.set_result(curies)

        if curie not in self.subscriptions:
            self.subscriptions[curie] = []
        self.subscriptions[curie].append(return_result)

        self.put("batch_expand", curie)

        return await future

    async def batch_expand(self, batch: list[CURIE]) -> None:
        """Get a batch of expanded curies and propagate them to original callers."""
        if len(batch) == 0:
            return
        logger.trace(f"Batch expand batch of {len(batch)}")
        descendants_packed = await REDIS_CLIENT.hmget(MAPPING_ID, *batch)
        for curie, desc_packed in zip(batch, descendants_packed, strict=True):
            descs = []
            if desc_packed is not None:
                descs = cast(list[CURIE], ormsgpack.unpackb(desc_packed))

            for callback in self.subscriptions.get(curie, []):
                callback(descs)

        # return ormsgpack.unpackb(descendants_packed[0])

    async def rebuild(self) -> None:
        """Periodically rebuild the mapping."""
        try:
            timeout = CONFIG.job.metakg.build_time
            if timeout < 0:
                return  # Rebuilding disabled
            await asyncio.sleep(CONFIG.job.metakg.build_time)
            await self.initialize()
        except asyncio.CancelledError:
            return
