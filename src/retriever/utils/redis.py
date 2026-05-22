import asyncio
from abc import ABCMeta, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine, Iterable
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
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.typing import AbsExpiryT, ChannelT, EncodableT, ExpiryT, KeyT, ResponseT

from retriever.config.general import CONFIG
from retriever.utils.general import AsyncDaemon

# Required to avoid CROSSSLOT errors: https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/#hash-tags
# Technically not needed as cluster is not supported, but worth keeping in case we need to re-add cluster support
PREFIX = "{Retriever}:"

# For consistency
OP_TABLE_KEY = f"{PREFIX}op_table"
OP_TABLE_UPDATE_CHANNEL = f"{OP_TABLE_KEY}:update"

# For better performance
ZSTD_COMPRESSOR = zstandard.ZstdCompressor()
ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()


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
    async def hmget(self, name: str, keys: Iterable[KeyT]) -> list[bytes | None]:
        """Returns a list of values ordered identically to ``keys``.

        For more information, see https://redis.io/commands/hmget
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


class RedisClient(AsyncDaemon):
    """A client abstraction layer for basic operations."""

    _client: redis.Redis
    client: AsyncRedis

    def __init__(self) -> None:
        """Instantiate a client class without initializing the Redis connection."""
        super().__init__()
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
        self.subscriptions: dict[str, list[Callable[[str], Awaitable[None]]]] = {}

    @override
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        return []

    @override
    async def initialize(self) -> None:
        """Initialize a connection to the redis server."""
        try:
            log.info("Checking redis connection...")
            await self.client.initialize()
            await self.client.ping()
            log.success("Redis connection successful!")
        except RedisConnectionError as error:
            log.critical(
                "Connection to Redis failed. Ensure an instance is running and the connection config is correct."
            )
            raise error
        return await super().initialize()

    @override
    async def wrapup(self) -> None:
        await super().wrapup()
        await self.client.aclose()

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

    async def subscriber(self, channel_key: str) -> None:
        """A subscriber function that should be wrapped in an asyncio task.

        We wrap in an asyncio task rather than using redis-py's built in handling
        specifically so that we can unsubscribe method-wise instead of channel-wise.
        """
        async with self.client.pubsub() as pubsub:
            await pubsub.subscribe(channel_key)
            logger.trace(f"Subscribed to channel {channel_key}")
            while True:
                # if len(self.subscriptions[channel_key]) == 0:
                #     break
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=0.01
                    )
                    if message is not None:
                        if isinstance(message["data"], str):
                            data = message["data"]
                        elif isinstance(message["data"], bytes):
                            data = message["data"].decode()
                        else:
                            data = str(message["data"])

                        for callback in self.subscriptions[channel_key]:
                            await callback(data)

                except (ValueError, asyncio.CancelledError):
                    await pubsub.unsubscribe(channel_key)
                    break

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
                key,
                mapping=value_to_set,
            )
            if ttl > 0:
                await self._client.hexpire(f"{PREFIX}{key}", ttl, *value_to_set.keys())

    async def hmget(
        self, key: str, *field: str, compressed: bool = False
    ) -> list[bytes | None]:
        """Generically get a field of a mapping."""
        values = await self.client.hmget(key, field)
        if compressed:
            return [
                ZSTD_DECOMPRESSOR.decompress(val) if val is not None else val
                for val in values
            ]
        return values
