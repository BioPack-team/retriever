"""Live integration tests for the /status/* route handlers.

Run with: `pytest -m live tests/test_status_live.py`. Requires docker
compose Mongo + Dragonfly. Uses the same seeded test DB pattern as
tests/test_utils_mongo_live.py.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from utils.mongo_fixtures import (
    test_mongo,  # noqa: F401  pyright:ignore[reportImplicitRelativeImport]  # fixture
)

from retriever.status import router as status_router
from retriever.utils.mongo import MongoClient

pytestmark = pytest.mark.live


@pytest_asyncio.fixture
async def status_client(test_mongo: MongoClient):  # noqa: F811
    """Provide an httpx AsyncClient bound to a minimal app mounting just /status.

    The `test_mongo` fixture monkeypatches the MongoClient singleton, so the
    handlers see the seeded test DB instead of production data.
    """
    app = FastAPI()
    app.include_router(status_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_active_returns_in_flight(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/active")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    for row in rows:
        assert row["age_seconds"] >= 0
        # Links should be absolute URLs so they're clickable from a browser.
        assert row["links"]["logs"].startswith("http://test/logs?job_id=")
        assert row["links"]["response"].startswith("http://test/response/")


@pytest.mark.asyncio
async def test_counts_breakdown(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/counts")
    assert resp.status_code == 200
    body = resp.json()
    all_time = body["windows"]["all_time"]
    counts: dict[str, int] = all_time["counts"]
    # Seed has 6 Complete, 3 failure-statuses, 2 Running.
    assert counts.get("Complete") == 6
    assert counts.get("Failed") == 1
    assert counts.get("QueryNotTraversable") == 1
    assert counts.get("UnsupportedConstraint") == 1
    assert counts.get("Running") == 2
    assert all_time["links"]["completed"] == "http://test/status/completed"
    assert all_time["links"]["failed"] == "http://test/status/failed"


@pytest.mark.asyncio
async def test_completed_pagination(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/completed", params={"limit": 2})
    assert resp.status_code == 200
    page1 = resp.json()
    assert len(page1["items"]) == 2
    assert page1["next_cursor"] is not None

    resp2 = await status_client.get(
        "/status/completed", params={"limit": 2, "cursor": page1["next_cursor"]}
    )
    assert resp2.status_code == 200
    page2 = resp2.json()
    assert len(page2["items"]) <= 2

    # First-page job_ids and second-page job_ids should be disjoint.
    page1_ids = {row["job_id"] for row in page1["items"]}
    page2_ids = {row["job_id"] for row in page2["items"]}
    assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio
async def test_failed_with_reason_filter(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/failed", params={"reason": "Failed"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["status"] == "Failed"


@pytest.mark.asyncio
async def test_durations_default_complete(status_client: AsyncClient) -> None:
    # Default status="Complete", default lookback=24h; the older completed
    # job is outside the window. So expect 5 samples, not 6.
    resp = await status_client.get("/status/durations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sample_size"] == 5
    assert body["min_seconds"] > 0
    assert body["max_seconds"] >= body["min_seconds"]


@pytest.mark.asyncio
async def test_timeline_default(status_client: AsyncClient) -> None:
    resp = await status_client.get(
        "/status/timeline",
        params={"field": "completed", "granularity": "hour", "lookback": 24.0},
    )
    assert resp.status_code == 200
    buckets = resp.json()
    assert isinstance(buckets, list)
    assert len(buckets) >= 1
    for b in buckets:
        assert "bucket_start" in b
        assert b["count"] >= 1


@pytest.mark.asyncio
async def test_tiers(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/tiers")
    assert resp.status_code == 200
    rows = resp.json()
    by_tier: dict[int, dict[str, Any]] = {row["tier"]: row for row in rows}
    assert 0 in by_tier
    assert by_tier[0]["active"] >= 1
    assert 1 in by_tier
    assert by_tier[1]["failed"] == 3


@pytest.mark.asyncio
async def test_submitters(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/submitters")
    assert resp.status_code == 200
    rows = resp.json()
    by_submitter: dict[str, dict[str, Any]] = {r["submitter"]: r for r in rows}
    assert "alice" in by_submitter
    assert "bob" in by_submitter
    assert by_submitter["alice"]["success_rate"] == 1.0
    assert by_submitter["bob"]["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_submitter_tier_stats(status_client: AsyncClient) -> None:
    resp = await status_client.get("/status/submitter_tier_stats")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) > 0
    by_cell = {(r["submitter"], r["tier"]): r for r in rows}
    # Seed has alice on tier 0 and bob on tier 1.
    assert ("alice", 0) in by_cell
    assert ("bob", 1) in by_cell
    alice_t0 = by_cell[("alice", 0)]
    assert alice_t0["count"] >= 1
    assert alice_t0["completed"] >= 1
    assert alice_t0["failed"] == 0


@pytest.mark.asyncio
async def test_failure_breakdown_by_submitter(status_client: AsyncClient) -> None:
    resp = await status_client.get(
        "/status/failure_breakdown", params={"by": "submitter"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["by"] == "submitter"
    by_key = {(r["key"], r["status"]): r["count"] for r in body["rows"]}
    # Bob's failed jobs cover the three terminal-failure statuses in the seed.
    assert ("bob", "Failed") in by_key
    assert by_key[("bob", "Failed")] >= 1


@pytest.mark.asyncio
async def test_failure_breakdown_by_tier(status_client: AsyncClient) -> None:
    resp = await status_client.get(
        "/status/failure_breakdown", params={"by": "tier"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["by"] == "tier"
    # Tier keys are stringified ints; failures live on tier 1 in the seed.
    by_key = {(r["key"], r["status"]): r["count"] for r in body["rows"]}
    assert any(k[0] == "1" for k in by_key)
    total_tier_1 = sum(v for k, v in by_key.items() if k[0] == "1")
    assert total_tier_1 == 3


@pytest.mark.asyncio
async def test_completed_sort_by_duration_desc(status_client: AsyncClient) -> None:
    # The /slow endpoint was folded into /completed?sort=duration.
    resp = await status_client.get(
        "/status/completed",
        params={"sort": "duration", "direction": "desc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    durations = [row["duration_seconds"] for row in body["items"]]
    assert durations == sorted(durations, reverse=True)


@pytest.mark.asyncio
async def test_completed_sort_by_submitter_asc(status_client: AsyncClient) -> None:
    resp = await status_client.get(
        "/status/completed",
        params={"sort": "submitter", "direction": "asc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    submitters = [row["submitter"] for row in body["items"]]
    assert submitters == sorted(submitters)


@pytest.mark.asyncio
async def test_completed_sort_by_results_desc(status_client: AsyncClient) -> None:
    resp = await status_client.get(
        "/status/completed",
        params={"sort": "results", "direction": "desc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    results = [row["results"] for row in body["items"]]
    assert results == sorted(results, reverse=True)


@pytest.mark.asyncio
async def test_stuck_with_min_age(status_client: AsyncClient) -> None:
    # Seeded "Running" jobs were created ~2 minutes ago; min_age=0.01 (≈36s)
    # should include both.
    resp = await status_client.get("/status/stuck", params={"min_age": 0.01})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    for row in rows:
        assert row["age_seconds"] >= int(0.01 * 3600)


@pytest.mark.asyncio
async def test_server_logs_flat(status_client: AsyncClient) -> None:
    resp = await status_client.get(
        "/status/server_logs", params={"fmt": "flat", "lookback": 1.0}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
