"""Unit tests for the Redis-lease leader election state machine.

These drive `LeaderElection` directly with a mocked `RedisClient` so the
promote/demote/failure behavior can be exercised without a live Redis.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from retriever.utils import leader
from retriever.utils.leader import LeaderElection


def _fresh() -> LeaderElection:
    """A `LeaderElection` built outside the Singleton cache, for per-test isolation."""
    election = object.__new__(LeaderElection)
    election.__init__()
    return election


def _fake_redis(*, up: bool = True, held: bool = True) -> MagicMock:
    """A stand-in `RedisClient` exposing just the lease/callback surface used here."""
    redis = MagicMock()
    redis.up = up
    redis.renew_or_acquire_lease = AsyncMock(return_value=held)
    redis.release_lease = AsyncMock()
    redis.set = AsyncMock()
    redis.request_health_check = MagicMock()
    redis.on_outage = MagicMock()
    redis.on_recover = MagicMock()
    redis.deregister_callback = MagicMock()
    return redis


async def _drain(event: asyncio.Event) -> None:
    """Wait for a fire-and-forget `on_acquire` callback to run."""
    await asyncio.wait_for(event.wait(), 1)


@pytest.mark.asyncio
async def test_win_on_start_promotes_and_fires_once(monkeypatch: pytest.MonkeyPatch):
    redis = _fake_redis(up=True, held=True)
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    fired = asyncio.Event()
    seen_leader: list[bool] = []

    async def callback() -> None:
        # The flag must already be True when the callback runs (set-before-fire).
        seen_leader.append(election.is_leader)
        fired.set()

    election.on_acquire(callback)

    await election.start()
    await _drain(fired)

    assert election.is_leader is True
    assert seen_leader == [True]
    redis.on_outage.assert_called_once()
    redis.on_recover.assert_called_once()

    await election.stop()
    redis.release_lease.assert_awaited_once()
    assert election.is_leader is False


@pytest.mark.asyncio
async def test_loser_stays_follower(monkeypatch: pytest.MonkeyPatch):
    redis = _fake_redis(up=True, held=False)
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    fired = asyncio.Event()

    async def callback() -> None:
        fired.set()

    election.on_acquire(callback)

    await election.start()
    await asyncio.sleep(0)

    assert election.is_leader is False
    assert not fired.is_set()

    await election.stop()


@pytest.mark.asyncio
async def test_down_at_start_then_recovers(monkeypatch: pytest.MonkeyPatch):
    redis = _fake_redis(up=False, held=True)
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    fired = asyncio.Event()

    async def callback() -> None:
        fired.set()

    election.on_acquire(callback)

    await election.start()
    redis.renew_or_acquire_lease.assert_not_awaited()
    assert election.is_leader is False

    # Redis recovers; the registered on_recover hook drives `_try_acquire`.
    redis.up = True
    await election._try_acquire()
    await _drain(fired)

    assert election.is_leader is True

    await election.stop()


@pytest.mark.asyncio
async def test_outage_demotes(monkeypatch: pytest.MonkeyPatch):
    redis = _fake_redis(up=True, held=True)
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    await election.start()
    assert election.is_leader is True

    await election._demote()

    assert election.is_leader is False
    assert election._elected_at is None

    await election.stop()


@pytest.mark.asyncio
async def test_win_publishes_elected_at(monkeypatch: pytest.MonkeyPatch):
    redis = _fake_redis(up=True, held=True)
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    await election.start()

    assert election.is_leader is True
    assert election._elected_at is not None
    redis.set.assert_awaited()
    key_arg = redis.set.await_args.args[0]
    assert key_arg == leader.LEADER_ELECTED_KEY

    await election.stop()


@pytest.mark.asyncio
async def test_persistent_lease_failure_stays_leaderless(
    monkeypatch: pytest.MonkeyPatch,
):
    redis = _fake_redis(up=True, held=True)
    redis.renew_or_acquire_lease.side_effect = RuntimeError("write rejected")
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    # Reads succeed (up stays True) but every lease write fails: nobody leads.
    await election.start()
    await election._attempt()
    await election._attempt()

    assert election.is_leader is False
    assert election._consecutive_failures >= leader.LEASE_FAILURE_WARN_THRESHOLD
    redis.request_health_check.assert_called()

    # Writes recover: the next attempt promotes and clears the failure count.
    redis.renew_or_acquire_lease.side_effect = None
    redis.renew_or_acquire_lease.return_value = True
    await election._attempt()

    assert election.is_leader is True
    assert election._consecutive_failures == 0

    await election.stop()


@pytest.mark.asyncio
async def test_transient_renew_failure_keeps_leadership(
    monkeypatch: pytest.MonkeyPatch,
):
    redis = _fake_redis(up=True, held=True)
    monkeypatch.setattr(leader, "REDIS_CLIENT", redis)
    election = _fresh()

    await election.start()
    assert election.is_leader is True

    # A transient renew error must not drop leadership - the TTL still protects us.
    redis.renew_or_acquire_lease.side_effect = RuntimeError("boom")
    await election._attempt()

    assert election.is_leader is True
    redis.request_health_check.assert_called()

    await election.stop()
