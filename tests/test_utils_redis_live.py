"""Live tests for the RedisClient /status-API additions.

Run with: `pytest -m live tests/test_utils_redis_live.py`. Requires the
docker compose Dragonfly container (or another Redis-compatible server)
to be up. Each test uses a unique key prefix and cleans up via short
TTLs.
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime

import pytest
import pytest_asyncio

from retriever.utils.redis import (
    PREFIX,
    WORKER_REGISTRY_KEY,
    RedisClient,
)

pytestmark = pytest.mark.live


@pytest_asyncio.fixture
async def redis_client():
    """Provide an initialized RedisClient, scrubbing any test keys on teardown."""
    client = RedisClient()
    await client.initialize()
    try:
        yield client
    finally:
        # Best-effort teardown - TTLs handle anything we miss.
        with contextlib.suppress(Exception):
            await client.client.delete(WORKER_REGISTRY_KEY)


@pytest.mark.asyncio
async def test_ping(redis_client: RedisClient) -> None:
    """ping should return without raising against a healthy server."""
    await redis_client.ping()


@pytest.mark.asyncio
async def test_used_memory_bytes(redis_client: RedisClient) -> None:
    """INFO MEMORY should return a positive used_memory integer."""
    assert await redis_client.used_memory_bytes() > 0


@pytest.mark.asyncio
async def test_freshness_roundtrip(redis_client: RedisClient) -> None:
    """write_freshness then _read_freshness should round-trip the payload."""
    key = f"{PREFIX}test-freshness:{uuid.uuid4().hex[:8]}"
    try:
        await redis_client.write_freshness(key, count=42, ttl=60)
        record = await redis_client._read_freshness(key)
        assert record is not None
        assert record["count"] == 42
        # Refreshed within the last few seconds.
        delta = datetime.now().astimezone() - record["refreshed_at"]
        assert delta.total_seconds() < 5
    finally:
        await redis_client.client.delete(key)


@pytest.mark.asyncio
async def test_worker_registry_roundtrip(redis_client: RedisClient) -> None:
    """register_worker → list_workers → presence."""
    pid = 999999  # unlikely to collide with a real process pid
    started = datetime.now().astimezone()

    await redis_client.register_worker(pid, started, ttl_seconds=60)
    workers = await redis_client.list_workers()
    assert pid in workers
    # Round-trip within ~1s tolerance.
    assert abs((workers[pid] - started).total_seconds()) < 1.0
