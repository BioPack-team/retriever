"""Cross-process backend outage/recovery propagation over Redis pub/sub.

Each process publishes only the transitions it detects itself (wired via the
`BackendClient` transition observer) and applies transitions detected
elsewhere to its own singletons - flipping a peer to fallback the moment any
one process sees a backend go down, instead of each waiting to detect it
independently. Outages are applied immediately (fail-safe); recoveries are
trusted optimistically and self-correct on the next failed local ping.

When Redis is unavailable nothing is published or received, and every process
falls back to its own process-local health loop.
"""

import asyncio
import contextlib
import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import Literal, cast

from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers import tier_manager
from retriever.utils.backend_client import BackendClient
from retriever.utils.general import Singleton
from retriever.utils.mongo import MongoClient
from retriever.utils.redis import (
    BACKEND_HEALTH_CHANNEL,
    BackendHealthMessage,
    RedisClient,
)


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 transition timestamp, tolerating missing/garbage input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class HealthCoordinator(metaclass=Singleton):
    """Propagates tier-driver and Mongo health across processes via Redis.

    One instance per process. `start()` registers a broadcast observer on each
    tracked client and subscribes to `BACKEND_HEALTH_CHANNEL`; `stop()` tears
    both down. Redis itself is intentionally untracked - it can't announce its
    own outage over itself.
    """

    _clients: dict[str, BackendClient]
    """Tracked backends keyed by `health_key`; populated on `start()`."""

    _publish_tasks: set[asyncio.Task[None]]
    """Strong refs to in-flight publish tasks so they aren't GC'd early."""

    _started: bool

    def __init__(self) -> None:
        """Instantiate with no clients tracked; call `start()` to wire up."""
        self._clients = {}
        self._publish_tasks = set()
        self._started = False

    def _tracked_clients(self) -> tuple[BackendClient, ...]:
        """The backends whose health is propagated (tier drivers + Mongo)."""
        return (
            tier_manager.get_driver(0),
            tier_manager.get_driver(1),
            MongoClient(),
        )

    async def start(self) -> None:
        """Register observers and subscribe. No-op if disabled or already started."""
        if self._started or not CONFIG.redis.propagate_health:
            return
        self._clients = {
            client.health_key: client for client in self._tracked_clients()
        }
        for client in self._clients.values():
            client.set_transition_observer(self._make_observer(client))
        with contextlib.suppress(Exception):
            await RedisClient().subscribe(BACKEND_HEALTH_CHANNEL, self._on_message)
        self._started = True
        logger.info("Backend health propagation enabled.")

    async def stop(self) -> None:
        """Unsubscribe, clear observers, and cancel in-flight publishes. Idempotent."""
        if not self._started:
            return
        self._started = False
        with contextlib.suppress(Exception):
            await RedisClient().unsubscribe(BACKEND_HEALTH_CHANNEL, self._on_message)
        for client in self._clients.values():
            client.set_transition_observer(None)
        self._clients = {}
        for task in self._publish_tasks:
            _ = task.cancel()
        for task in list(self._publish_tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _make_observer(
        self, client: BackendClient
    ) -> Callable[[Literal["outage", "recovery"]], None]:
        """Build a per-client observer that broadcasts this process's own transitions."""

        def _observe(event: Literal["outage", "recovery"]) -> None:
            task = asyncio.create_task(self._publish(client, event))
            self._publish_tasks.add(task)
            _ = task.add_done_callback(self._publish_tasks.discard)

        return _observe

    async def _publish(
        self, client: BackendClient, event: Literal["outage", "recovery"]
    ) -> None:
        """Publish one transition; skipped while Redis is down (process-local fallback)."""
        redis_client = RedisClient()
        if not redis_client.up:
            return
        message: BackendHealthMessage = {
            "backend": client.health_key,
            "event": event,
            "pid": os.getpid(),
            "at": datetime.now().astimezone().isoformat(),
            "error": client.last_error,
        }
        try:
            await redis_client.publish(BACKEND_HEALTH_CHANNEL, json.dumps(message))
        except Exception:
            logger.debug(f"Failed to publish {event} for {client.health_key} to Redis.")

    async def _on_message(self, raw: str) -> None:
        """Apply a peer's transition to the matching local client; ignore own echoes."""
        try:
            payload = json.loads(raw)
        except ValueError:
            logger.debug("Discarding malformed backend health message.")
            return
        if not isinstance(payload, dict):
            return
        message = cast(BackendHealthMessage, cast(object, payload))
        if message.get("pid") == os.getpid():
            return
        client = self._clients.get(message.get("backend", ""))
        if client is None:
            return
        occurred_at = _parse_timestamp(message.get("at"))
        event = message.get("event")
        if event == "outage":
            client.note_remote_outage(message.get("error"), occurred_at=occurred_at)
        elif event == "recovery":
            client.note_remote_recovery(recovered_at=occurred_at)
