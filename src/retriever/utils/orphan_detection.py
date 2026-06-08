"""Orphan-job detection: mark jobs whose worker is gone as Failed.

The background process drives `periodically_mark_orphans` on the interval
configured by `CONFIG.job.orphan_check_interval`. A job is considered
orphaned when its `worker_pid` is no longer in the Redis worker registry,
or when the registered start time at that PID differs from the one
recorded on the job (PID reuse).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from retriever.config.general import CONFIG
from retriever.utils.mongo import MongoClient
from retriever.utils.redis import RedisClient


def _ms_since_epoch(dt: datetime) -> int:
    """Truncate to integer ms since epoch - matches BSON datetime resolution.

    MongoDB stores datetimes at ms precision; Redis preserves the original
    microseconds via ISO-string round-trip. Comparing at ms precision lets
    the two round-trip representations of the same instant match.
    """
    return int(dt.timestamp() * 1000)


async def _mark_orphaned_jobs() -> None:
    """One sweep: find non-terminal jobs whose worker is gone and mark them Failed."""
    if not RedisClient().up:
        logger.debug(
            "Orphan sweep skipped; Redis is down.",
            no_mongo_log=True,
        )
        return
    try:
        # Read Mongo *before* Redis: workers register in Redis before they
        # serve any request, so any job present in the Mongo snapshot was
        # created by a worker that must also be in the later Redis snapshot
        # if it's still alive. Reading the other way around would risk
        # marking a freshly-created job orphaned because its worker
        # registered between the two reads.
        non_terminal = await MongoClient().get_non_terminal_with_worker_info()
        live_workers = await RedisClient().list_workers()
    except Exception:
        logger.exception("Orphan sweep failed to read state; skipping.")
        return

    orphaned_ids: list[str] = []
    for job_id, wpid, wstarted in non_terminal:
        if wpid is None:
            continue  # pre-feature doc; no worker info to check
        live_started = live_workers.get(wpid)
        if live_started is None:
            # Worker PID is no longer in the registry → worker died.
            orphaned_ids.append(job_id)
            continue
        if wstarted is None:
            # Inconsistent state (wpid set but wstarted missing); trust the
            # live PID match rather than orphan based on partial data.
            continue
        if _ms_since_epoch(live_started) != _ms_since_epoch(wstarted):
            # Same PID, different start time → PID was reused by a new worker.
            orphaned_ids.append(job_id)

    if orphaned_ids:
        try:
            count = await MongoClient().mark_jobs_abandoned(orphaned_ids)
            if count:
                logger.warning(
                    f"Orphan sweep: marked {count} job(s) as Failed.",
                    no_mongo_log=True,
                )
        except Exception:
            logger.exception("Orphan sweep failed to write Failed status.")


async def periodically_mark_orphans() -> None:
    """Periodic driver for orphan detection.

    Runs immediately on startup so jobs left behind by a previous run get
    cleaned up without waiting a full interval. There's no startup race:
    workers register in Redis before their lifespan yields, so they cannot
    have unregistered jobs in MongoDB.
    """
    try:
        interval = CONFIG.job.orphan_check_interval
        if interval < 0:
            return
        while True:
            await _mark_orphaned_jobs()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        return
