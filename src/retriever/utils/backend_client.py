from __future__ import annotations

import asyncio
import contextlib
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from typing import Literal, TypedDict, override

from loguru import logger

from retriever.utils.general import AsyncDaemon


class StatusDict(TypedDict):
    """Dict of core status info for a BackendClient."""

    up: bool
    last_outage: datetime | None
    last_recovery: datetime | None
    error: str | None


class BackendClient(AsyncDaemon, ABC):
    """Base for clients of external dependencies.

    Provides lifecycle, periodic healthchecks, error classification, and subclass
    method requirements.
    """

    healthcheck_interval: float | None = 30.0
    """How often to check health, None to disable periodic checks."""

    retry_backoff_start: float = 1.0
    """Starting backoff for reconnecting."""

    retry_backoff_max: float = 60.0
    """Maximum backoff for reconnecting."""

    retry_backoff_jitter: float = 0.2
    """Proportion of jitter in backoff."""

    up: bool = True
    """Status of the service."""

    last_outage: datetime | None
    """Timestamp of last outage, if occurred."""

    last_recovery: datetime | None
    """Timestamp of last outage recovery, if occurred."""

    last_error: str | None
    _recovery_callbacks: list[Callable[[], Awaitable[None]]]
    _outage_callbacks: list[Callable[[], Awaitable[None]]]

    _check_event: asyncio.Event
    """Used to request a background healthcheck."""

    _up_event: asyncio.Event
    """Set while `up` is True; consumers `await` it to park during outages."""

    _callback_tasks: set[asyncio.Task[None]]
    """Callback task storage to prevent early GC."""

    def __init__(self) -> None:
        """Initialize state. Assume up until failure."""
        super().__init__()
        self.up = True
        self.last_outage = None
        self.last_recovery = None
        self.last_error = None
        self._recovery_callbacks = []
        self._outage_callbacks = []
        self._check_event = asyncio.Event()
        self._up_event = asyncio.Event()
        self._up_event.set()
        self._callback_tasks = set()

    @abstractmethod
    async def ping(self) -> None:
        """Probe the backend. Raise on failure; return on success.

        Called by the health loop at `healthcheck_interval` and after
        every `request_health_check()`.
        Implementations should be cheap and idempotent.
        """

    def is_outage_error(self, exc: BaseException) -> bool:
        """Classify whether an exception indicates this backend is down.

        Default biases toward True so callers don't miss real outages.
        `CancelledError` is always treated as a non-outage signal so
        shutdown cancellation isn't mistaken for a backend failure.
        Subclasses override to add more known-benign carve-outs.
        """
        return not isinstance(exc, asyncio.CancelledError)

    def request_health_check(self) -> None:
        """Schedule a one-shot ping ASAP.

        Used for quick status checking by client callers.
        No-op if a check is already pending
        """
        self._check_event.set()

    def on_recover(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a coroutine fired on every recovery; duplicate registrations are ignored.

        Implementers MUST guard the callback body with an `asyncio.Lock`
        (or equivalent) so multiple callbacks do not overlap.
        The base catches and logs callback exceptions but does not serialize them.
        """
        if callback not in self._recovery_callbacks:
            self._recovery_callbacks.append(callback)

    def deregister_callback(
        self,
        event: Literal["outage", "recover"],
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Deregister a previously-registered callback. Idempotent."""
        with contextlib.suppress(ValueError):
            {
                "outage": self._outage_callbacks,
                "recover": self._recovery_callbacks,
            }[event].remove(callback)

    def on_outage(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a coroutine fired on every outage; duplicate registrations are ignored.

        Same self-serialization contract as `on_recover`.
        """
        if callback not in self._outage_callbacks:
            self._outage_callbacks.append(callback)

    def status(self) -> StatusDict:
        """Dict of relevant status fields."""
        return StatusDict(
            up=self.up,
            last_outage=self.last_outage,
            last_recovery=self.last_recovery,
            error=self.last_error,
        )

    # Integrate with AsyncDaemon for Lifecycle
    @override
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        """Append the health loop to whatever the parent class returns."""
        return [*super().get_task_funcs(), self._health_loop]

    async def _health_loop(self) -> None:
        """Driver loop maintaining the health flag.

        Pings periodically (or on demand) while up; jittered
        exponential backoff while down. Transitions update timestamps,
        clear/record `last_error`, and fire callbacks. Failures the
        driver classifies as non-outage (via `is_outage_error`) are
        treated as successful probes.
        """
        backoff = self.retry_backoff_start
        try:
            while True:
                nudged = await self._await_next_tick(backoff)
                self._check_event.clear()
                try:
                    await self.ping()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not self.is_outage_error(exc):
                        self._handle_ping_success()
                        backoff = self.retry_backoff_start
                        continue
                    was_up = self.up
                    self._handle_ping_failure(exc)
                    if was_up:
                        backoff = self.retry_backoff_start
                    elif not nudged:
                        backoff = min(backoff * 2, self.retry_backoff_max)
                else:
                    self._handle_ping_success()
                    backoff = self.retry_backoff_start
        except asyncio.CancelledError:
            return

    async def _await_next_tick(self, backoff: float) -> bool:
        """Sleep until the next ping; returns True if a nudge interrupted the sleep.

        While up: wait for either the steady interval to elapse or
        `request_health_check()` to fire, whichever comes first. If
        polling is disabled, block on the event indefinitely.

        While down: sleep an uninterruptible floor (one
        `retry_backoff_start`) and then a nudge-interruptible tail up
        to the current backoff. The floor caps nudge-driven ping
        frequency under high-traffic nudge sources.
        """
        if self.up:
            if self.healthcheck_interval is None:
                await self._check_event.wait()
                return True
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._check_event.wait(),
                    timeout=self.healthcheck_interval,
                )
                return True
            return False

        jitter = 1.0 + random.uniform(
            -self.retry_backoff_jitter, self.retry_backoff_jitter
        )
        sleep_time = backoff * jitter
        floor = self.retry_backoff_start
        if sleep_time <= floor:
            await asyncio.sleep(sleep_time)
            return False
        await asyncio.sleep(floor)
        try:
            await asyncio.wait_for(
                self._check_event.wait(),
                timeout=sleep_time - floor,
            )
        except TimeoutError:
            return False
        return True

    def _handle_ping_failure(self, exc: BaseException) -> None:
        """Record the failure; on first outage fire callbacks."""
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.up:
            self.up = False
            self._up_event.clear()
            self.last_outage = datetime.now().astimezone()
            logger.warning(f"{type(self).__name__} down: {self.last_error}")
            self._fire_callbacks(self._outage_callbacks, "outage")

    def _handle_ping_success(self) -> None:
        """Record recovery and fire on_recover callbacks. Idempotent."""
        if self.up:
            return
        self.up = True
        self._up_event.set()
        self.last_recovery = datetime.now().astimezone()
        self.last_error = None
        logger.success(f"{type(self).__name__} up: ping succeeded.")
        self._fire_callbacks(self._recovery_callbacks, "recovery")

    def _fire_callbacks(
        self,
        callbacks: list[Callable[[], Awaitable[None]]],
        kind: Literal["outage", "recovery"],
    ) -> None:
        """Spawn each callback as a fire-and-forget task.

        Tasks are kept in `_callback_tasks` as strong refs until they
        complete (they remove themselves via `add_done_callback`).
        Callbacks don't delay health loop. Callbacks must self-manage protection
        against parallel races.
        """
        for cb in callbacks:
            task = asyncio.create_task(self._safe_callback(cb, kind))
            self._callback_tasks.add(task)
            _ = task.add_done_callback(self._callback_tasks.discard)

    async def _safe_callback(
        self,
        callback: Callable[[], Awaitable[None]],
        kind: Literal["outage", "recovery"],
    ) -> None:
        """Run a callback, swallowing exceptions so the health loop is unaffected."""
        try:
            await callback()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"{type(self).__name__} {kind} callback failed.")
