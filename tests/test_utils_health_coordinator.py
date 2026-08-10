"""Unit tests for HealthCoordinator's publish/apply logic.

The Redis client, tier drivers, and Mongo singleton are mocked so the
propagation decisions can be exercised without a live backend.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from retriever.config.general import CONFIG
from retriever.utils import health_coordinator as hc
from retriever.utils.general import Singleton
from retriever.utils.health_coordinator import HealthCoordinator


def _fresh() -> HealthCoordinator:
    """A fresh coordinator, bypassing the Singleton cache for test isolation."""
    _ = Singleton._instances.pop(HealthCoordinator, None)
    return HealthCoordinator()


def _message(event: str, *, pid: int, backend: str = "GandalfDriver") -> str:
    return json.dumps(
        {
            "backend": backend,
            "event": event,
            "pid": pid,
            "at": datetime.now().astimezone().isoformat(),
            "error": "boom" if event == "outage" else None,
        }
    )


@pytest.mark.asyncio
async def test_on_message_applies_remote_outage() -> None:
    coord = _fresh()
    driver = MagicMock()
    coord._clients = {"GandalfDriver": driver}

    await coord._on_message(_message("outage", pid=os.getpid() + 1))

    driver.note_remote_outage.assert_called_once()
    driver.note_remote_recovery.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_applies_remote_recovery() -> None:
    coord = _fresh()
    driver = MagicMock()
    coord._clients = {"GandalfDriver": driver}

    await coord._on_message(_message("recovery", pid=os.getpid() + 1))

    driver.note_remote_recovery.assert_called_once()
    driver.note_remote_outage.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_ignores_own_echo() -> None:
    coord = _fresh()
    driver = MagicMock()
    coord._clients = {"GandalfDriver": driver}

    await coord._on_message(_message("outage", pid=os.getpid()))

    driver.note_remote_outage.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_ignores_unknown_backend() -> None:
    coord = _fresh()
    driver = MagicMock()
    coord._clients = {"GandalfDriver": driver}

    await coord._on_message(
        _message("outage", pid=os.getpid() + 1, backend="SomethingElse")
    )

    driver.note_remote_outage.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_tolerates_malformed_payload() -> None:
    coord = _fresh()
    coord._clients = {"GandalfDriver": MagicMock()}

    # Should not raise on non-JSON or non-object payloads.
    await coord._on_message("not json{")
    await coord._on_message("42")


@pytest.mark.asyncio
async def test_publish_skips_when_redis_down(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = _fresh()
    redis = MagicMock()
    redis.up = False
    redis.publish = AsyncMock()
    monkeypatch.setattr(hc, "RedisClient", lambda: redis)

    client = MagicMock()
    client.health_key = "MongoClient"
    client.last_error = None
    await coord._publish(client, "outage")

    redis.publish.assert_not_called()


@pytest.mark.asyncio
async def test_publish_sends_when_redis_up(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = _fresh()
    redis = MagicMock()
    redis.up = True
    redis.publish = AsyncMock()
    monkeypatch.setattr(hc, "RedisClient", lambda: redis)

    client = MagicMock()
    client.health_key = "MongoClient"
    client.last_error = "x"
    await coord._publish(client, "recovery")

    redis.publish.assert_awaited_once()
    channel, payload = redis.publish.await_args.args
    assert channel == hc.BACKEND_HEALTH_CHANNEL
    body = json.loads(payload)
    assert body["backend"] == "MongoClient"
    assert body["event"] == "recovery"
    assert body["pid"] == os.getpid()


@pytest.mark.asyncio
async def test_start_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = _fresh()
    monkeypatch.setattr(CONFIG.redis, "propagate_health", False)

    await coord.start()

    assert coord._started is False
    assert coord._clients == {}


@pytest.mark.asyncio
async def test_start_wires_observers_and_subscribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coord = _fresh()
    monkeypatch.setattr(CONFIG.redis, "propagate_health", True)

    redis = MagicMock()
    redis.subscribe = AsyncMock()
    monkeypatch.setattr(hc, "RedisClient", lambda: redis)

    d0 = MagicMock()
    d0.health_key = "GandalfDriver"
    d1 = MagicMock()
    d1.health_key = "ElasticSearchDriver"
    mongo = MagicMock()
    mongo.health_key = "MongoClient"
    monkeypatch.setattr(hc.tier_manager, "get_driver", lambda t: {0: d0, 1: d1}[t])
    monkeypatch.setattr(hc, "MongoClient", lambda: mongo)

    await coord.start()

    assert coord._started is True
    redis.subscribe.assert_awaited_once()
    assert redis.subscribe.await_args.args[0] == hc.BACKEND_HEALTH_CHANNEL
    d0.set_transition_observer.assert_called_once()
    d1.set_transition_observer.assert_called_once()
    mongo.set_transition_observer.assert_called_once()
    assert set(coord._clients) == {
        "GandalfDriver",
        "ElasticSearchDriver",
        "MongoClient",
    }
