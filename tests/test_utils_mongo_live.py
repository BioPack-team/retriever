"""Live tests for the MongoClient /status-API additions.

Run with: `pytest -m live tests/test_utils_mongo_live.py`. Requires the
docker compose Mongo container (or another Mongo reachable at the
configured host) to be up. Uses an ephemeral test DB and drops it on
teardown so production data isn't affected.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from retriever.utils.job_status import TERMINAL_FAILURE
from retriever.utils.mongo import (
    JobIdentityFilter,
    JobTimeFilter,
    MongoClient,
)
from utils.mongo_fixtures import test_mongo  # noqa: F401  pyright:ignore[reportImplicitRelativeImport]  # fixture import

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_ping_db_storage_count_in_flight(test_mongo: MongoClient) -> None:  # noqa: F811
    """Smoke-test the three connection/inventory methods."""
    assert await test_mongo.ping() is True
    assert await test_mongo.db_storage_bytes() >= 0
    # Two in-flight jobs were seeded (one per tier).
    assert await test_mongo.count_in_flight() == 2


@pytest.mark.asyncio
async def test_count_jobs_with_status_filter(test_mongo: MongoClient) -> None:  # noqa: F811
    """`count_jobs` respects identity filters built via _build_filter_query."""
    completed = await test_mongo.count_jobs(
        identity=JobIdentityFilter(status="Complete")
    )
    # Five recent + one old completed.
    assert completed == 6

    failed = await test_mongo.count_jobs(
        identity=JobIdentityFilter(status=sorted(TERMINAL_FAILURE))
    )
    assert failed == 3


@pytest.mark.asyncio
async def test_compute_durations_basic(test_mongo: MongoClient) -> None:  # noqa: F811
    """`compute_durations` returns sensible stats; percentile fields only when supported."""
    stats = await test_mongo.compute_durations(
        identity=JobIdentityFilter(status="Complete")
    )
    # 5 recent (1..5s) + 1 old (30s) = 6 samples; min 1.0, max 30.0.
    assert stats["sample_size"] == 6
    assert stats["min_seconds"] == pytest.approx(1.0)
    assert stats["max_seconds"] == pytest.approx(30.0)
    assert stats["avg_seconds"] > 0.0

    if test_mongo._supports_percentile:
        # Approximate percentile — assert it's within the data range, not a
        # specific value (Mongo uses an approximation).
        assert "p50_seconds" in stats
        assert "p95_seconds" in stats
        assert "p99_seconds" in stats
        assert stats["min_seconds"] <= stats["p50_seconds"] <= stats["max_seconds"]
    else:
        assert "p50_seconds" not in stats


@pytest.mark.asyncio
async def test_submitter_stats(test_mongo: MongoClient) -> None:  # noqa: F811
    """`submitter_stats` returns per-submitter counts and success_rate."""
    rows = await test_mongo.submitter_stats(top=10)
    by_submitter = {row["submitter"]: row for row in rows}

    # alice: 6 Complete + 1 Running = 7 total; 6 terminal (all success); rate = 1.0.
    assert "alice" in by_submitter
    alice = by_submitter["alice"]
    assert alice["count"] == 7
    assert alice["completed"] == 6
    assert alice["failed"] == 0
    assert alice["success_rate"] == pytest.approx(1.0)

    # bob: 3 Failed* + 1 Running = 4 total; 3 terminal (all failure); rate = 0.0.
    assert "bob" in by_submitter
    bob = by_submitter["bob"]
    assert bob["count"] == 4
    assert bob["completed"] == 0
    assert bob["failed"] == 3
    assert bob["success_rate"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_tier_stats(test_mongo: MongoClient) -> None:  # noqa: F811
    """`tier_stats` aggregates active + completed + failed per tier."""
    rows = await test_mongo.tier_stats()
    by_tier = {row["tier"]: row for row in rows}

    # Tier 0: 6 completed (5 + 1 old) + 1 active
    assert 0 in by_tier
    assert by_tier[0]["active"] == 1
    assert by_tier[0]["completed"] == 6
    assert by_tier[0]["failed"] == 0

    # Tier 1: 0 completed, 3 failed, 1 active
    assert 1 in by_tier
    assert by_tier[1]["active"] == 1
    assert by_tier[1]["completed"] == 0
    assert by_tier[1]["failed"] == 3


@pytest.mark.asyncio
async def test_compute_durations_empty_window(test_mongo: MongoClient) -> None:  # noqa: F811
    """Empty windows return all-zero stats without errors."""
    future = datetime.now().astimezone() + timedelta(days=365)
    stats = await test_mongo.compute_durations(
        times=JobTimeFilter(completed={"after": future})
    )
    assert stats["sample_size"] == 0
    assert stats["min_seconds"] == 0.0
    assert stats["max_seconds"] == 0.0
    assert stats["avg_seconds"] == 0.0
