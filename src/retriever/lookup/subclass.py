import asyncio
import contextlib
from collections.abc import Callable
from typing import cast, override

import ormsgpack
from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers import tier_manager
from retriever.types.trapi import (
    CURIE,
)
from retriever.utils.general import BatchedAction
from retriever.utils.redis import SUBCLASS_META_KEY, TIER_RECOVERED_CHANNEL, RedisClient

REDIS_CLIENT = RedisClient()

MAPPING_ID = "SubclassHashMap"


class SubclassMapping(BatchedAction):
    """A redis-backed mapping from primary node CURIE to its ontological descendants."""

    queue_delay: float = 0.025
    # Essentially should flush every interval
    batch_size: int = 1000
    flush_time: float = 0
    multibatch: bool = True

    is_leader: bool = False
    subscriptions: dict[CURIE, list[Callable[[list[CURIE] | None], None]]]
    _refresh_lock: asyncio.Lock
    _pending_refresh: bool = False

    redis_setup_batch_size: int = 5000

    def __init__(self) -> None:
        """Initialize without leader role; call `promote_to_leader()` to flip the flag."""
        self.subscriptions = {}
        self._refresh_lock = asyncio.Lock()
        self._pending_refresh = False
        super().__init__()

    def promote_to_leader(self) -> None:
        """Flip this instance to leader mode. Must be called before `initialize()`."""
        self.is_leader = True

    @override
    async def initialize(self) -> None:
        """Set up the redis hash and initialize the batch handler."""
        if not CONFIG.job.lookup.implicit_subclassing:
            logger.info("Implicit subclassing disabled, skipping initialization.")
            return await super().initialize()

        if not self.is_leader:  # Only need leader to update the redis setup
            return await super().initialize()

        if self.initialized:
            return  # rebuild loop already running

        try:
            await self.refresh()
            self.tasks.append(asyncio.create_task(self.rebuild()))
        except Exception:
            logger.exception(
                "Unable to initialize subclass mapping due to below error, proceeding without implicit subclassing..."
            )

        REDIS_CLIENT.on_recover(self.refresh)
        tier_manager.get_driver(1).on_recover(self.refresh)
        # Also listen for worker-detected tier 1 recovery via Redis so
        # the rebuild fires faster than the leader's own periodic ping.
        with contextlib.suppress(Exception):
            await REDIS_CLIENT.subscribe(
                TIER_RECOVERED_CHANNEL, self._on_remote_tier_recover
            )

        return await super().initialize()

    @override
    async def wrapup(self) -> None:
        """Unsubscribe the leader's tier-recovery listener, then cancel the rebuild loop."""
        if self.is_leader:
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.unsubscribe(
                    TIER_RECOVERED_CHANNEL, self._on_remote_tier_recover
                )
        await super().wrapup()

    async def _on_remote_tier_recover(self, message: str) -> None:
        """Refresh when a worker signals tier 1 came back."""
        # The channel is shared with OpTableManager - filter to tier 1.
        if message == "1":
            await self.refresh()

    async def refresh(self) -> None:
        """Rebuild and publish the subclass mapping; concurrent calls collapse to one trailing rebuild."""
        self._pending_refresh = True
        if self._refresh_lock.locked():
            logger.debug(
                "Subclass rebuild already in progress; trailing rebuild queued."
            )
            return
        async with self._refresh_lock:
            while self._pending_refresh:
                self._pending_refresh = False
                await self._reload_mapping()

    async def _reload_mapping(self) -> None:
        """Pull the subclass mapping live from tier 1 and write it to Redis in batches."""
        logger.info("Loading subclass mapping...")
        # bypass_cache: rebuild must reflect live upstream state, not the
        # in-process cache (which `meta._LOCAL_CACHE` returns by reference).
        mapping = await tier_manager.get_driver(1).get_subclass_mapping(
            bypass_cache=True
        )
        total = len(mapping)

        packed_mapping = dict[str, bytes]()
        for curie, descendants in mapping.items():
            packed_mapping[curie] = ormsgpack.packb(descendants)
            if len(packed_mapping) >= self.redis_setup_batch_size:
                await REDIS_CLIENT.hset(MAPPING_ID, mapping=packed_mapping)
                packed_mapping = {}

        if packed_mapping:
            await REDIS_CLIENT.hset(MAPPING_ID, mapping=packed_mapping)

        await REDIS_CLIENT.write_freshness(SUBCLASS_META_KEY, count=total)

        del packed_mapping

        logger.success(f"Subclass mapping refreshed with {total} ancestors.")

    async def get(self, curie: CURIE) -> list[CURIE] | None:
        """Descendants of `curie`; `None` when the mapping is unavailable, `[]` when none exist."""
        future = asyncio.Future[list[CURIE] | None]()

        def return_result(curies: list[CURIE] | None) -> None:
            if not future.done():
                future.set_result(curies)

        self.subscriptions.setdefault(curie, []).append(return_result)
        self.put("batch_expand", curie)

        return await future

    async def batch_expand(self, batch: list[CURIE]) -> None:
        """Expand a batch of curies; signal `None` to callbacks when Redis is unavailable."""
        if len(batch) == 0:
            return
        logger.trace(f"Batch expand batch of {len(batch)}")
        if not REDIS_CLIENT.up:
            self._broadcast(batch, None)
            return
        try:
            descendants_packed = await REDIS_CLIENT.hmget(MAPPING_ID, *batch)
        except Exception:
            logger.debug(
                "Subclass batch_expand failed against Redis; signaling unavailable."
            )
            self._broadcast(batch, None)
            return
        for curie, desc_packed in zip(batch, descendants_packed, strict=True):
            descs: list[CURIE] = []
            if desc_packed is not None:
                descs = cast(list[CURIE], ormsgpack.unpackb(desc_packed))

            # Pop atomically: callbacks can't be skipped by list mutation, and
            # the key doesn't linger forever.
            for callback in self.subscriptions.pop(curie, None) or []:
                callback(descs)

    def _broadcast(self, batch: list[CURIE], value: list[CURIE] | None) -> None:
        """Resolve every pending future for `batch` with the same `value`."""
        for curie in batch:
            for callback in self.subscriptions.pop(curie, None) or []:
                callback(value)

    async def rebuild(self) -> None:
        """Periodically rebuild the mapping."""
        try:
            interval = CONFIG.job.metakg.build_time
            if interval < 0:
                return  # Rebuilding disabled
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.refresh()
                except Exception:
                    logger.exception(
                        f"Subclass mapping rebuild failed, retry in {interval}s."
                    )
        except asyncio.CancelledError:
            return
