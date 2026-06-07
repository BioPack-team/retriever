import asyncio
import json
import random
from abc import ABCMeta, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from types import CoroutineType, TracebackType
from typing import (
    Any,
    Protocol,
    Self,
    TypedDict,
    cast,
    overload,
    override,
)

import redis.asyncio as redis
import zstandard
from loguru import logger
from loguru import logger as log
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.typing import AbsExpiryT, ChannelT, EncodableT, ExpiryT, KeyT, ResponseT

from retriever.config.general import CONFIG
from retriever.utils.backend_client import BackendClient

# Required to avoid CROSSSLOT errors: https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/#hash-tags
# Technically not needed as cluster is not supported, but worth keeping in case we need to re-add cluster support
PREFIX = "{Retriever}:"

# Bare names - `RedisClient.set/get/publish/subscribe` add `PREFIX`.
OP_TABLE_KEY = "op_table"
OP_TABLE_UPDATE_CHANNEL = "op_table:update"

# Worker -> leader signal for a tier recovery; payload is the tier index as a string.
TIER_RECOVERED_CHANNEL = "tier:recovered"

# Timestamp keys written alongside published artifacts so /status
# can show freshness without inferring from TTL.
OP_TABLE_META_KEY = f"{PREFIX}op_table:meta"
SUBCLASS_META_KEY = f"{PREFIX}SubclassHashMap:meta"

# Process-registry hashes.
WORKER_REGISTRY_KEY = f"{PREFIX}workers"
BACKGROUND_REGISTRY_KEY = f"{PREFIX}background"
MAIN_REGISTRY_KEY = f"{PREFIX}main"


# For better performance
ZSTD_COMPRESSOR = zstandard.ZstdCompressor()
ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()


class FreshnessRecord(TypedDict):
    """Sidecar metadata for a Redis-published artifact.

    `count` semantics vary by artifact: `op_count` for the MetaKG op
    table, map `size` for the subclass mapping.
    """

    refreshed_at: datetime
    count: int


class PubSubMessage(TypedDict):
    """The message type returned from pubsub.get_message()."""

    pattern: str | None
    type: str
    channel: bytes
    data: Any


class PubSub(Protocol, metaclass=ABCMeta):
    """A protocol surfacing correct type hinting for an async PubSub's methods.

    I'm lazy so I'm only adding what gets used. ~ Willow
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """`async with...` support."""

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """`async with...` support."""

    @abstractmethod
    async def subscribe(
        self,
        *args: ChannelT,
        **kwargs: Callable[[PubSubMessage | None], CoroutineType[None, None, None]],
    ) -> None:
        """Subscribe to channels.

        Channels supplied as keyword arguments expect
        a channel name as the key and a callable as the value. A channel's
        callable will be invoked automatically when a message is received on
        that channel rather than producing a message via ``listen()`` or
        ``get_message()``.
        """

    @abstractmethod
    async def unsubscribe(self, *args: ChannelT) -> None:
        """Unsubscribe from the supplied channels.

        If empty, unsubscribe from all channels.
        """

    @abstractmethod
    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float = 0.0
    ) -> PubSubMessage | None:
        """Get the next message if one is available, otherwise None.

        If timeout is specified, the system will wait for `timeout` seconds
        before returning. Timeout should be specified as a floating point
        number or None to wait indefinitely.
        """


class AsyncRedis(Protocol, metaclass=ABCMeta):
    """A protocol surfacing correct type hinting for Redis's methods.

    I'm lazy so I'm only adding what gets used. ~ Willow
    """

    @abstractmethod
    async def initialize(self) -> Self:
        """Initialize the client."""

    @abstractmethod
    async def ping(self) -> bool:
        """Ping the Redis server to test connectivity.

        Sends a PING command to the Redis server and returns True if the server
        responds with "PONG".

        This command is useful for:
        - Testing whether a connection is still alive
        - Verifying the server's ability to serve data

        For more information on the underlying ping command see https://redis.io/commands/ping
        """

    @abstractmethod
    async def aclose(self, close_connection_pool: bool | None = None) -> None:
        """Closes Redis client connection.

        Args:
            close_connection_pool:
                decides whether to close the connection pool used by this Redis client,
                overriding Redis.auto_close_connection_pool.
                By default, let Redis.auto_close_connection_pool decide
                whether to close the connection pool.
        """

    @abstractmethod
    async def publish(
        self, channel: ChannelT, message: EncodableT, **kwargs: Any
    ) -> ResponseT:
        """Publish ``message`` on ``channel``.

        Returns the number of subscribers the message was delivered to.

        For more information, see https://redis.io/commands/publish
        """

    @abstractmethod
    def pubsub(self, **kwargs: Any) -> PubSub:
        """Return a Publish/Subscribe object.

        With this object, you can
        subscribe to channels and listen for messages that get published to
        them.
        """

    @abstractmethod
    async def set(  # noqa:PLR0913
        self,
        name: KeyT,
        value: EncodableT,
        ex: ExpiryT | None = None,
        px: ExpiryT | None = None,
        nx: bool = False,
        xx: bool = False,
        keepttl: bool = False,
        get: bool = False,
        exat: AbsExpiryT | None = None,
        pxat: AbsExpiryT | None = None,
        ifeq: bytes | str | None = None,
        ifne: bytes | str | None = None,
        ifdeq: str | None = None,  # hex digest of current value
        ifdne: str | None = None,  # hex digest of current value
    ) -> bool:
        """Set the value at key ``name`` to ``value``.

        Warning:
        **Experimental** since 7.1.
        The usage of the arguments ``ifeq``, ``ifne``, ``ifdeq``, and ``ifdne``
        is experimental. The API or returned results when those parameters are used
        may change based on feedback.

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``nx`` if set to True, set the value at key ``name`` to ``value`` only
            if it does not exist.

        ``xx`` if set to True, set the value at key ``name`` to ``value`` only
            if it already exists.

        ``keepttl`` if True, retain the time to live associated with the key.
            (Available since Redis 6.0)

        ``get`` if True, set the value at key ``name`` to ``value`` and return
            the old value stored at key, or None if the key did not exist.
            (Available since Redis 6.2)

        ``exat`` sets an expire flag on key ``name`` for ``ex`` seconds,
            specified in unix time.

        ``pxat`` sets an expire flag on key ``name`` for ``ex`` milliseconds,
            specified in unix time.

        ``ifeq`` set the value at key ``name`` to ``value`` only if the current
            value exactly matches the argument.
            If key doesn't exist - it won't be created.
            (Requires Redis 8.4 or greater)

        ``ifne`` set the value at key ``name`` to ``value`` only if the current
            value does not exactly match the argument.
            If key doesn't exist - it will be created.
            (Requires Redis 8.4 or greater)

        ``ifdeq`` set the value at key ``name`` to ``value`` only if the current
            value XXH3 hex digest exactly matches the argument.
            If key doesn't exist - it won't be created.
            (Requires Redis 8.4 or greater)

        ``ifdne`` set the value at key ``name`` to ``value`` only if the current
            value XXH3 hex digest does not exactly match the argument.
            If key doesn't exist - it will be created.
            (Requires Redis 8.4 or greater)

        For more information, see https://redis.io/commands/set
        """

    @abstractmethod
    async def get(self, name: KeyT) -> bytes | None:
        """Return the value at key ``name``, or None if the key doesn't exist.

        For more information, see https://redis.io/commands/get
        """

    @abstractmethod
    async def delete(self, *names: KeyT) -> bool:
        """Delete one or more keys specified by ``names``."""

    @abstractmethod
    async def hset(
        self,
        name: str,
        key: str | None = None,
        value: str | bytes | None = None,
        mapping: dict[str, bytes] | None = None,
        items: list[tuple[str, bytes]] | None = None,
    ) -> int:
        """Set ``key`` to ``value`` within hash ``name``.

        ``mapping`` accepts a dict of key/value pairs that will be
        added to hash ``name``.
        ``items`` accepts a list of key/value pairs that will be
        added to hash ``name``.
        Returns the number of fields that were added.

        For more information, see https://redis.io/commands/hset
        """

    @abstractmethod
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete ``keys`` from hash ``name``. Returns the number of fields removed.

        For more information, see https://redis.io/commands/hdel
        """

    @abstractmethod
    async def hmget(self, name: str, keys: Iterable[KeyT]) -> list[bytes | None]:
        """Returns a list of values ordered identically to ``keys``.

        For more information, see https://redis.io/commands/hmget
        """

    @abstractmethod
    async def hgetall(self, name: str) -> dict[bytes, bytes]:
        """Return a Python dict of the hash's name/value pairs.

        For more information, see https://redis.io/commands/hgetall
        """

    @abstractmethod
    async def info(self, section: str | None = None) -> dict[str, Any]:
        """Return server INFO (optionally restricted to a section).

        For more information, see https://redis.io/commands/info
        """

    @abstractmethod
    async def hexpire(  # noqa:PLR0913
        self,
        name: KeyT,
        seconds: ExpiryT,
        *fields: str,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> ResponseT:
        """Sets or updates the expiration time for fields within a hash key, using relative time in seconds.

        If a field already has an expiration time, the behavior of the update can be
        controlled using the `nx`, `xx`, `gt`, and `lt` parameters.

        The return value provides detailed information about the outcome for each field.

        For more information, see https://redis.io/commands/hexpire

        Args:
            name: The name of the hash key.
            seconds: Expiration time in seconds, relative. Can be an integer, or a
                     Python `timedelta` object.
            fields: List of fields within the hash to apply the expiration time to.
            nx: Set expiry only when the field has no expiry.
            xx: Set expiry only when the field has an existing expiry.
            gt: Set expiry only when the new expiry is greater than the current one.
            lt: Set expiry only when the new expiry is less than the current one.

        Returns:
            Returns a list which contains for each field in the request:
                - `-2` if the field does not exist, or if the key does not exist.
                - `0` if the specified NX | XX | GT | LT condition was not met.
                - `1` if the expiration time was set or updated.
                - `2` if the field was deleted because the specified expiration time is
                  in the past.
        """


class RedisClient(BackendClient):
    """A client abstraction layer for basic operations."""

    _client: redis.Redis
    client: AsyncRedis

    def __init__(self) -> None:
        """Instantiate a client class without initializing the Redis connection."""
        super().__init__()
        self._build_client()
        self.subscriptions: dict[str, list[Callable[[str], Awaitable[None]]]] = {}

    @override
    async def ping(self) -> None:
        """Probe Redis. Raises on failure."""
        _ = await self.client.ping()

    def _build_client(self) -> None:
        """(Re)build the internal redis-py async client.

        Called from __init__ and again from initialize() when the existing
        client is bound to a now-closed asyncio loop. This keeps the
        RedisClient Singleton usable across event loops (e.g. across
        pytest-asyncio tests, where each test gets a fresh loop).
        """
        retry = Retry(ExponentialBackoff(), CONFIG.redis.attempts)
        self._client = redis.Redis(
            host=CONFIG.redis.host,
            port=CONFIG.redis.port,
            password=CONFIG.redis.password.get_secret_value()
            if CONFIG.redis.password
            else None,
            ssl=CONFIG.redis.ssl_enabled,
            ssl_cert_reqs="none",
            retry=retry,
        )
        self.client = cast(
            AsyncRedis,
            cast(object, self._client),
        )

    @override
    async def initialize(self) -> None:
        """Initialize a connection to the redis server; degraded mode on any startup failure."""
        log.info("Checking redis connection...")
        try:
            await self._connect_with_loop_rebuild()
        except Exception as error:
            log.warning(
                f"Redis startup failed; continuing in degraded mode. Health loop will retry. Error: {error}"
            )
            self._handle_ping_failure(error)
        else:
            log.success("Redis connection successful!")
        return await super().initialize()

    async def _connect_with_loop_rebuild(self) -> None:
        """Run the initial connect, rebuilding once if bound to a closed loop.

        redis-py binds connections to the asyncio loop they were created
        on. When the Singleton survives a loop closure (Python test
        runners, in particular), the first connect raises and we rebuild
        against the current loop.
        """
        try:
            await self.client.initialize()
            await self.client.ping()
        except RuntimeError as error:
            if "Event loop is closed" not in str(error):
                raise
            log.debug("Rebuilding Redis client on current event loop.")
            self._build_client()
            await self.client.initialize()
            await self.client.ping()

    @override
    async def wrapup(self) -> None:
        await super().wrapup()
        await self.client.aclose()

    def start_heartbeat(
        self,
        on_tick: Callable[[], Awaitable[None]],
        role_label: str,
        interval_seconds: int | None = None,
    ) -> None:
        """Spawn a heartbeat task tracked by `self.tasks`; cancelled at wrapup."""
        task = asyncio.create_task(
            self._heartbeat(on_tick, role_label, interval_seconds),
            name=f"{role_label.lower()}-heartbeat",
        )
        self.tasks.append(task)

    async def _heartbeat(
        self,
        on_tick: Callable[[], Awaitable[None]],
        role_label: str,
        interval_seconds: int | None,
    ) -> None:
        """Run `on_tick` on an interval; skip ticks while Redis is down and re-fire on recovery."""
        if interval_seconds is None:
            interval_seconds = CONFIG.redis.heartbeat_interval_seconds
        armed = False

        async def _on_recover_fire() -> None:
            nonlocal armed
            try:
                await on_tick()
            except Exception:
                logger.warning(
                    f"{role_label} heartbeat refresh on recovery failed; will retry next interval.",
                    no_mongo_log=True,
                )
            finally:
                self.deregister_callback("recover", _on_recover_fire)
                armed = False

        def _arm_recovery() -> None:
            nonlocal armed
            if armed:
                return
            self.on_recover(_on_recover_fire)
            armed = True

        try:
            while True:
                await asyncio.sleep(interval_seconds)
                if not self.up:
                    logger.debug(
                        f"{role_label} heartbeat skipped; Redis is down.",
                        no_mongo_log=True,
                    )
                    _arm_recovery()
                    continue
                try:
                    await on_tick()
                except Exception:
                    logger.warning(
                        f"{role_label} heartbeat refresh failed; will retry next interval.",
                        no_mongo_log=True,
                    )
                    self.request_health_check()
                    _arm_recovery()
        except asyncio.CancelledError:
            self.deregister_callback("recover", _on_recover_fire)
            return

    async def publish(self, channel: str, message: Any) -> None:
        """Publish a message to a given channel."""
        await self.client.publish(f"{PREFIX}{channel}", message)

    async def subscribe(
        self, channel: str, callback: Callable[[str], Awaitable[None]]
    ) -> None:
        """Subscribe to a channel, calling callback when a message is received."""
        channel_key = f"{PREFIX}{channel}"
        if channel_key not in self.subscriptions:
            self.subscriptions[channel_key] = []
            self.tasks.append(
                asyncio.create_task(
                    self.subscriber(channel_key),
                    name=f"redis_subscribe_task:{channel}",
                )
            )
            self.subscriptions[channel_key].append(callback)
            return

        self.subscriptions[channel_key].append(callback)

    async def unsubscribe(
        self, channel: str, callback: Callable[[str], Awaitable[None]]
    ) -> bool:
        """Unsubscribe a callback from a channel."""
        channel_key = f"{PREFIX}{channel}"
        if channel_key in self.subscriptions:
            try:
                self.subscriptions[channel_key].remove(callback)
            except ValueError:
                return False
        return True

    async def _pump_messages(self, pubsub: PubSub, channel_key: str) -> None:
        """Dispatch incoming pubsub messages on `channel_key` to its callbacks.

        A callback raising non-cancellation is isolated and logged so one
        misbehaving subscriber can't tear down the connection.
        """
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )
            if message is None:
                continue
            if isinstance(message["data"], str):
                data = message["data"]
            elif isinstance(message["data"], bytes):
                data = message["data"].decode()
            else:
                data = str(message["data"])
            # Snapshot - a callback awaiting can yield to subscribe()/
            # unsubscribe() mutating the underlying list mid-iteration.
            # `.get(..., [])` defends against a future cleanup that
            # `del`s an empty subscriptions list.
            for callback in list(self.subscriptions.get(channel_key, [])):
                try:
                    await callback(data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        f"Pubsub callback for {channel_key} raised; continuing."
                    )

    async def subscriber(self, channel_key: str) -> None:
        """Self-healing subscriber task for a single Redis pub/sub channel.

        Wrapped in an asyncio task (rather than using redis-py's built-in
        handling) so we can unsubscribe per-callback instead of per-channel.

        Resilience model: park on `_up_event` while the client believes
        Redis is down so we don't busy-fail. On a classified outage
        exception, nudge `request_health_check()` and retry with jittered
        exponential backoff using a fresh `pubsub()` (its own connection -
        distinct from the main client's). Non-outage exceptions log and
        continue. `CancelledError` exits cleanly even if cleanup of the
        old `pubsub()` masks it.
        """
        backoff = self.retry_backoff_start
        try:
            while True:
                if not self.up:
                    backoff = self.retry_backoff_start
                    await self._up_event.wait()
                try:
                    async with self.client.pubsub() as pubsub:
                        await pubsub.subscribe(channel_key)
                        logger.trace(f"Subscribed to channel {channel_key}")
                        backoff = self.retry_backoff_start
                        await self._pump_messages(pubsub, channel_key)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # PubSub.__aexit__ can raise a ConnectionError during
                    # cleanup of a dead connection, masking an in-flight
                    # cancellation. Detect that and re-raise so shutdown
                    # exits immediately instead of waiting out the backoff.
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise asyncio.CancelledError from exc
                    if self.is_outage_error(exc):
                        self.request_health_check()
                        jitter = 1.0 + random.uniform(
                            -self.retry_backoff_jitter, self.retry_backoff_jitter
                        )
                        await asyncio.sleep(backoff * jitter)
                        backoff = min(backoff * 2, self.retry_backoff_max)
                    else:
                        logger.exception(
                            f"Pubsub non-outage error on {channel_key}; continuing."
                        )
                        # Short pause to avoid a tight reconnect loop if a
                        # misclassified or transient error keeps recurring.
                        await asyncio.sleep(self.retry_backoff_start)
        except asyncio.CancelledError:
            return

    async def set(
        self,
        key: str,
        value: bytes,
        *,
        compress: bool = False,
        ttl: int = 0,  # Redis requires int
    ) -> None:
        """Generically set a key-value pair."""
        value_to_set = value
        if compress:
            value_to_set = ZSTD_COMPRESSOR.compress(value)
        await self.client.set(
            f"{PREFIX}{key}",
            value_to_set,
            ex=ttl if ttl > 0 else None,
        )

    async def get(self, key: str, *, compressed: bool = False) -> bytes | None:
        """Generically get a key-value pair."""
        data = await self.client.get(f"{PREFIX}{key}")
        if data is None:
            return

        if compressed:
            data = ZSTD_DECOMPRESSOR.decompress(data)
        return data

    async def delete(self, key: str) -> bool:
        """Generically delete a key-value pair."""
        response = await self.client.delete(f"{PREFIX}{key}")
        return response > 0

    async def used_memory_bytes(self) -> int:
        """Return the Dragonfly/Redis `used_memory` field from INFO MEMORY."""
        info = await self.client.info("memory")
        return int(info["used_memory"])

    async def _read_freshness(self, key: str) -> FreshnessRecord | None:
        """Read a JSON sidecar `{refreshed_at, count}` from the given key."""
        data = await self.client.get(key)
        if data is None:
            return None
        try:
            payload = json.loads(data)
            return FreshnessRecord(
                refreshed_at=datetime.fromisoformat(payload["refreshed_at"]),
                count=int(payload["count"]),
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

    async def write_freshness(self, key: str, count: int, ttl: int = 0) -> None:
        """Write a JSON sidecar `{refreshed_at, count}`; `ttl=0` for no expire."""
        payload = json.dumps(
            {"refreshed_at": datetime.now().astimezone().isoformat(), "count": count}
        )
        await self.client.set(key, payload, ex=ttl if ttl > 0 else None)

    async def metakg_freshness(self) -> FreshnessRecord | None:
        """Return the last-refreshed timestamp + op count for the MetaKG."""
        return await self._read_freshness(OP_TABLE_META_KEY)

    async def subclass_freshness(self) -> FreshnessRecord | None:
        """Return the last-refreshed timestamp + map size for the subclass map."""
        return await self._read_freshness(SUBCLASS_META_KEY)

    async def _register_process(
        self,
        registry_key: str,
        pid: int,
        started_at: datetime,
        ttl_seconds: int,
    ) -> None:
        """Write the JSON started-at payload to a hash field with TTL."""
        payload = json.dumps({"started_at": started_at.astimezone().isoformat()})
        _ = await self.client.hset(registry_key, str(pid), payload.encode())
        _ = await self.client.hexpire(registry_key, ttl_seconds, str(pid))

    async def register_worker(
        self, pid: int, started_at: datetime, ttl_seconds: int
    ) -> None:
        """Register this worker's PID + start time with a TTL."""
        await self._register_process(WORKER_REGISTRY_KEY, pid, started_at, ttl_seconds)

    async def unregister_worker(self, pid: int) -> None:
        """Remove this worker's registration entry.

        Called on graceful shutdown so orphan detection picks up the
        worker's leftover jobs immediately rather than waiting on TTL.
        """
        _ = await self.client.hdel(WORKER_REGISTRY_KEY, str(pid))

    async def register_background(
        self, pid: int, started_at: datetime, ttl_seconds: int
    ) -> None:
        """Register the background process's PID + start time with a TTL."""
        await self._register_process(
            BACKGROUND_REGISTRY_KEY, pid, started_at, ttl_seconds
        )

    async def register_main(
        self, pid: int, started_at: datetime, ttl_seconds: int
    ) -> None:
        """Register the main entry-point process's PID + start time with a TTL."""
        await self._register_process(MAIN_REGISTRY_KEY, pid, started_at, ttl_seconds)

    async def _list_processes(self, registry_key: str) -> dict[int, datetime]:
        """Return a mapping of pid -> started_at for the given registry hash."""
        entries = await self.client.hgetall(registry_key)
        out: dict[int, datetime] = {}
        for raw_pid, raw_value in entries.items():
            try:
                payload = json.loads(raw_value)
                out[int(raw_pid)] = datetime.fromisoformat(payload["started_at"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return out

    async def list_workers(self) -> dict[int, datetime]:
        """Return all currently-registered worker PIDs and their start times."""
        return await self._list_processes(WORKER_REGISTRY_KEY)

    async def list_background(self) -> dict[int, datetime]:
        """Return the registered background process PID + start time (0 or 1 entries)."""
        return await self._list_processes(BACKGROUND_REGISTRY_KEY)

    async def list_main(self) -> dict[int, datetime]:
        """Return the registered main process PID + start time (0 or 1 entries)."""
        return await self._list_processes(MAIN_REGISTRY_KEY)

    @overload
    async def hset(self, key: str, field: str, value: bytes) -> None: ...

    @overload
    async def hset(
        self, key: str, field: str, value: bytes, *, compress: bool
    ) -> None: ...

    @overload
    async def hset(self, key: str, field: str, value: bytes, *, ttl: int) -> None: ...

    @overload
    async def hset(
        self, key: str, field: str, value: bytes, *, compress: bool, ttl: int
    ) -> None: ...

    @overload
    async def hset(self, key: str, *, mapping: dict[str, bytes]) -> None: ...

    @overload
    async def hset(
        self, key: str, *, mapping: dict[str, bytes], compress: bool
    ) -> None: ...

    @overload
    async def hset(self, key: str, *, mapping: dict[str, bytes], ttl: int) -> None: ...

    @overload
    async def hset(
        self,
        key: str,
        *,
        mapping: dict[str, bytes],
        compress: bool,
        ttl: int,
    ) -> None: ...

    async def hset(  # noqa: PLR0913
        self,
        key: str,
        field: str | None = None,
        value: bytes | None = None,
        *,
        mapping: dict[str, bytes] | None = None,
        compress: bool = False,
        ttl: int = 0,
    ) -> None:
        """Generically set a mapping or field of a mapping."""
        if value is not None:
            value_to_set = value
        elif mapping is not None:
            value_to_set = mapping
        else:
            raise ValueError("Must provide either value or mapping.")
        if compress:
            value_to_set = (
                ZSTD_COMPRESSOR.compress(value_to_set)
                if isinstance(value_to_set, bytes)
                else {k: ZSTD_COMPRESSOR.compress(v) for k, v in value_to_set.items()}
            )

        if isinstance(value_to_set, bytes):
            if field is None:
                raise ValueError("Must provide a field if setting a specific value.")
            await self.client.hset(
                f"{PREFIX}{key}",
                field,
                value_to_set,
            )
            if ttl > 0:
                await self.client.hexpire(f"{PREFIX}{key}", ttl, field)
        else:
            await self.client.hset(
                f"{PREFIX}{key}",
                mapping=value_to_set,
            )
            if ttl > 0:
                await self._client.hexpire(f"{PREFIX}{key}", ttl, *value_to_set.keys())

    async def hmget(
        self, key: str, *field: str, compressed: bool = False
    ) -> list[bytes | None]:
        """Generically get a field of a mapping."""
        values = await self.client.hmget(f"{PREFIX}{key}", field)
        if compressed:
            return [
                ZSTD_DECOMPRESSOR.decompress(val) if val is not None else val
                for val in values
            ]
        return values
