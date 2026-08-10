"""Unit tests for the write-aware Redis health probe.

`RedisClient.ping` issues a canary write so a read-only or out-of-memory Redis
(reads/PING succeed, writes rejected) registers as an outage rather than
silently stranding writers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from retriever.utils.redis import HEALTH_CANARY_KEY, RedisClient


@pytest.mark.asyncio
async def test_ping_reads_and_writes_a_canary():
    redis = RedisClient()
    original = redis.client
    try:
        redis.client = AsyncMock()
        await redis.ping()
        redis.client.ping.assert_awaited_once()
        redis.client.set.assert_awaited_once()
        assert redis.client.set.await_args.args[0] == HEALTH_CANARY_KEY
    finally:
        redis.client = original


@pytest.mark.asyncio
async def test_ping_raises_when_writes_are_rejected():
    redis = RedisClient()
    original = redis.client
    try:
        # Reads/PING fine, but the canary write is rejected (read-only / OOM).
        redis.client = AsyncMock()
        redis.client.set.side_effect = RuntimeError("READONLY")
        with pytest.raises(RuntimeError):
            await redis.ping()
    finally:
        redis.client = original
