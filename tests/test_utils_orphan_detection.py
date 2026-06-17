"""Unit tests for the orphan-job sweep, centered on the age-based fallback.

These mock the Mongo/Redis singletons so the sweep's decision logic can be
exercised without a live backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from retriever.config.general import CONFIG
from retriever.utils import orphan_detection

MAX_AGE_SECONDS = 172_800  # 48h, the configured "certainly dead" threshold


def _aware(hours_ago: float) -> datetime:
    """A timezone-aware timestamp `hours_ago` hours before now."""
    return datetime.now().astimezone() - timedelta(hours=hours_ago)


def _count(ids: list[str]) -> int:
    """Stand-in for mark_jobs_abandoned: report how many ids it received."""
    return len(ids)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    non_terminal: list[tuple[str, int | None, datetime | None, datetime | None]],
    live_workers: dict[int, datetime],
    redis_up: bool = True,
    max_age: int = MAX_AGE_SECONDS,
) -> AsyncMock:
    """Install fake clients into the sweep and return the mark_jobs_abandoned mock."""
    mark = AsyncMock(side_effect=_count)
    mongo = MagicMock()
    mongo.get_non_terminal_for_orphan_sweep = AsyncMock(return_value=non_terminal)
    mongo.mark_jobs_abandoned = mark
    redis = MagicMock()
    redis.up = redis_up
    redis.list_workers = AsyncMock(return_value=live_workers)
    monkeypatch.setattr(orphan_detection, "MongoClient", lambda: mongo)
    monkeypatch.setattr(orphan_detection, "RedisClient", lambda: redis)
    monkeypatch.setattr(CONFIG.job, "orphan_max_age", max_age)
    return mark


def _abandoned(mark: AsyncMock) -> list[str]:
    """The job ids passed to mark_jobs_abandoned, or [] if it was never called."""
    if not mark.await_args_list:
        return []
    ids: list[Any] = mark.await_args_list[-1].args[0]
    return list(ids)


@pytest.mark.asyncio
async def test_age_fallback_marks_pre_feature_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job with no worker info but older than the cutoff is the regression case."""
    mark = _wire(
        monkeypatch,
        non_terminal=[("old-orphan", None, None, _aware(120))],
        live_workers={},
    )
    await orphan_detection._mark_orphaned_jobs()
    assert _abandoned(mark) == ["old-orphan"]


@pytest.mark.asyncio
async def test_age_fallback_overrides_live_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job past the age cutoff is dead even if its worker is still registered."""
    started = _aware(120)
    mark = _wire(
        monkeypatch,
        non_terminal=[("ancient", 4321, started, started)],
        live_workers={4321: started},
    )
    await orphan_detection._mark_orphaned_jobs()
    assert _abandoned(mark) == ["ancient"]


@pytest.mark.asyncio
async def test_young_pre_feature_job_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recent job with no worker info stays untouched; we can't prove it's dead."""
    mark = _wire(
        monkeypatch,
        non_terminal=[("recent-noinfo", None, None, _aware(1))],
        live_workers={},
    )
    await orphan_detection._mark_orphaned_jobs()
    assert mark.await_count == 0


@pytest.mark.asyncio
async def test_worker_death_still_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original PID-based arm keeps working for recent jobs."""
    mark = _wire(
        monkeypatch,
        non_terminal=[("dead-worker", 999, _aware(1), _aware(1))],
        live_workers={},  # PID 999 no longer registered
    )
    await orphan_detection._mark_orphaned_jobs()
    assert _abandoned(mark) == ["dead-worker"]


@pytest.mark.asyncio
async def test_live_recent_job_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recent job whose worker is alive and matches is not swept."""
    started = _aware(1)
    mark = _wire(
        monkeypatch,
        non_terminal=[("healthy", 100, started, started)],
        live_workers={100: started},
    )
    await orphan_detection._mark_orphaned_jobs()
    assert mark.await_count == 0


@pytest.mark.asyncio
async def test_age_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With orphan_max_age == -1 the age arm is off and old no-info jobs survive."""
    mark = _wire(
        monkeypatch,
        non_terminal=[("old-orphan", None, None, _aware(500))],
        live_workers={},
        max_age=-1,
    )
    await orphan_detection._mark_orphaned_jobs()
    assert mark.await_count == 0


@pytest.mark.asyncio
async def test_redis_down_skips_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Mongo read or write happens when Redis is unavailable."""
    mark = _wire(
        monkeypatch,
        non_terminal=[("old-orphan", None, None, _aware(500))],
        live_workers={},
        redis_up=False,
    )
    await orphan_detection._mark_orphaned_jobs()
    assert mark.await_count == 0
