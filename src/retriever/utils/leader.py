import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import datetime
from uuid import uuid4

from loguru import logger

from retriever.config.general import CONFIG
from retriever.utils.general import Singleton
from retriever.utils.redis import LEADER_ELECTED_KEY, LEADER_LEASE_KEY, RedisClient

REDIS_CLIENT = RedisClient()

# Consecutive lease failures (while Redis reads still pass) before escalating from
# a debug line to a warning that the cluster may be leaderless.
LEASE_FAILURE_WARN_THRESHOLD = 3


class LeaderElection(metaclass=Singleton):
    """Elects one instance as the leader across instances via a single Redis lease.

    Each instance's *builder* (background process) attempts to acquire one Redis
    lease. The instance whose builder holds the lease is the *leader*. It
    relinquishes leadership the instant Redis goes down, so a stale build never
    overwrites live state. Only the builder calls `start()`; workers import the
    singleton but leave `is_leader` False.
    """

    is_leader: bool = False
    """Whether this instance is the leader (its builder holds the lease)."""

    def __init__(self) -> None:
        """Set up contention state; asyncio objects are deferred to `start()`."""
        self.token: str = uuid4().hex
        self.is_leader = False
        self._acquire_callbacks: list[Callable[[], Awaitable[None]]] = []
        # When this leadership episode began; published best-effort for /status.
        self._elected_at: datetime | None = None
        # Consecutive lease failures while Redis reads still succeed; drives escalation.
        self._consecutive_failures: int = 0
        self._renew_task: asyncio.Task[None] | None = None
        self._fired_tasks: set[asyncio.Task[None]] = set()

    def on_acquire(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback fired once each time this instance wins leadership."""
        if callback not in self._acquire_callbacks:
            self._acquire_callbacks.append(callback)

    async def start(self) -> None:
        """Begin contending for the lease; call once, from the builder process."""
        REDIS_CLIENT.on_outage(self._demote)
        REDIS_CLIENT.on_recover(self._try_acquire)
        self._renew_task = asyncio.create_task(
            self._renew_loop(), name="leader-lease-renew"
        )
        await self._attempt()

    async def stop(self) -> None:
        """Stop contending, releasing the lease so a peer can take over promptly."""
        if self._renew_task is not None:
            _ = self._renew_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._renew_task
            self._renew_task = None
        REDIS_CLIENT.deregister_callback("outage", self._demote)
        REDIS_CLIENT.deregister_callback("recover", self._try_acquire)
        if REDIS_CLIENT.up:
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.release_lease(LEADER_LEASE_KEY, self.token)
        self._set_leader(False)

    async def _renew_loop(self) -> None:
        """Renew or reclaim the lease each heartbeat interval while Redis is up."""
        interval = CONFIG.redis.heartbeat_interval_seconds
        try:
            while True:
                await asyncio.sleep(interval)
                if REDIS_CLIENT.up:
                    await self._attempt()
        except asyncio.CancelledError:
            return

    async def _attempt(self) -> None:
        """Renew or acquire the lease once, updating leader state from the result."""
        if not REDIS_CLIENT.up:
            return
        try:
            held = await REDIS_CLIENT.renew_or_acquire_lease(
                LEADER_LEASE_KEY,
                self.token,
                CONFIG.redis.leader_lease_ttl_seconds * 1000,
            )
        except Exception:
            # A write-rejecting Redis (memory limit / read-only replica) also fails
            # the health probe's canary write, so an outage demotes us within a
            # probe cycle. This path only catches lease-specific hiccups; warn if
            # they persist since nobody may be leading.
            self._consecutive_failures += 1
            if self._consecutive_failures == LEASE_FAILURE_WARN_THRESHOLD:
                logger.warning(
                    f"Leader lease has failed {self._consecutive_failures} times while Redis reads succeed - the cluster may be leaderless. Check Redis write health (memory limit / read-only replica)."
                )
            else:
                logger.debug("Leader lease renew/acquire failed; will retry.")
            REDIS_CLIENT.request_health_check()
            return
        self._consecutive_failures = 0
        # `held` is True only when our token holds the lease, so promoting while
        # Redis is up always reflects reality; a concurrent outage leaves `up` False.
        if held and REDIS_CLIENT.up:
            self._set_leader(True)
        elif not held:
            self._set_leader(False)
        # Best-effort: refresh when leadership began for /status, so it lives as
        # long as the lease. Failures are non-fatal - it's only observability.
        if self.is_leader and self._elected_at is not None:
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.set(
                    LEADER_ELECTED_KEY,
                    self._elected_at.isoformat().encode(),
                    ttl=CONFIG.redis.leader_lease_ttl_seconds,
                )

    def _set_leader(self, value: bool) -> None:
        """Flip leader state synchronously, firing `on_acquire` on a False->True edge.

        The flag is set before callbacks run so a callback that reads `is_leader`
        (e.g. a gated `refresh`) sees leadership. Callbacks run as tracked
        fire-and-forget tasks so a slow build never stalls the renew loop.
        """
        was_leader = self.is_leader
        self.is_leader = value
        if value and not was_leader:
            self._elected_at = datetime.now().astimezone()
            logger.info("Won leadership; firing initial build.")
            for callback in self._acquire_callbacks:
                task = asyncio.create_task(self._run_callback(callback))
                self._fired_tasks.add(task)
                task.add_done_callback(self._fired_tasks.discard)
        elif was_leader and not value:
            self._elected_at = None
            logger.info("Relinquished leadership.")

    async def _run_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Run an `on_acquire` callback, isolating its failures from the loop."""
        try:
            await callback()
        except Exception:
            logger.exception("Leader on_acquire callback failed.")

    async def _demote(self) -> None:
        """Relinquish leadership immediately when Redis goes down (on_outage hook)."""
        self._set_leader(False)

    async def _try_acquire(self) -> None:
        """Re-contend for the lease when Redis recovers (on_recover hook)."""
        await self._attempt()


LEADER_ELECTION = LeaderElection()
