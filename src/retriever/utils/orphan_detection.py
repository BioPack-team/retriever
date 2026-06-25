"""Orphan-job detection: mark jobs whose worker is gone as Failed.

The background process drives `periodically_mark_orphans` on the interval
configured by `CONFIG.job.orphan_check_interval`. A job is considered
orphaned when its `worker_pid` is no longer in the Redis worker registry,
or when the registered start time at that PID differs from the one
recorded on the job (PID reuse).

As a fallback, any non-terminal job older than `CONFIG.job.orphan_max_age`
is treated as dead regardless of worker info. We can assume no job legitimately runs
that long.
The same loop also reaps GridFS job-doc blobs whose parent `job_docs`
document has been TTL-expired (which Mongo's TTL monitor can't do for us).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

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
        non_terminal = await MongoClient().get_non_terminal_for_orphan_sweep()
        live_workers = await RedisClient().list_workers()
    except Exception:
        logger.exception("Orphan sweep failed to read state; skipping.")
        return

    max_age = CONFIG.job.orphan_max_age
    age_cutoff = (
        datetime.now().astimezone() - timedelta(seconds=max_age)
        if max_age >= 0
        else None
    )

    orphaned_ids: list[str] = []
    for job_id, wpid, wstarted, created in non_terminal:
        if age_cutoff is not None and created is not None and created < age_cutoff:
            # Older than any legitimate job could run -> dead, whatever the
            # worker fields say. This is the only arm that reaches jobs
            # predating worker tracking (no worker info to check otherwise).
            orphaned_ids.append(job_id)
            continue
        if wpid is None:
            continue  # no worker info and not old enough to call dead
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


async def _reap_orphaned_doc_blobs() -> None:
    """One sweep: delete GridFS job-doc blobs whose parent job is gone."""
    if not MongoClient().up:
        return
    try:
        count = await MongoClient().reap_orphaned_doc_blobs()
    except Exception:
        logger.exception("GridFS blob reap failed.")
        return
    if count:
        logger.info(
            f"GridFS reap: deleted {count} orphaned job-doc blob(s).",
            no_mongo_log=True,
        )


async def periodically_mark_orphans() -> None:
    """Periodic driver for orphan detection and GridFS blob reaping.

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
            await _reap_orphaned_doc_blobs()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        return
