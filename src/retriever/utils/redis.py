import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import redis.asyncio as redis
import zstandard
from loguru import logger
from loguru import logger as log
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError

from retriever.config.general import CONFIG

# Required to avoid CROSSSLOT errors: https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/#hash-tags
# Technically not needed as cluster is not supported, but worth keeping in case we need to re-add cluster support
PREFIX = "{Retriever}:"

# For consistency
OP_TABLE_KEY = f"{PREFIX}op_table"
OP_TABLE_UPDATE_CHANNEL = f"{OP_TABLE_KEY}:update"

# For better performance
ZSTD_COMPRESSOR = zstandard.ZstdCompressor()
ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()


class RedisClient:
    """A client abstraction layer for basic operations."""

    def __init__(self) -> None:
        """Instantiate a client class without initializing the Redis connection."""
        self.client: redis.Redis

        retry = Retry(ExponentialBackoff(), CONFIG.redis.attempts)
        self.client = redis.Redis(
            host=CONFIG.redis.host,
            port=CONFIG.redis.port,
            password=CONFIG.redis.password.get_secret_value()
            if CONFIG.redis.password
            else None,
            ssl=CONFIG.redis.ssl_enabled,
            ssl_cert_reqs="none",
            retry=retry,
        )
        self.tasks: list[asyncio.Task[None]] = []
        self.subscriptions: dict[str, list[Callable[[str], Awaitable[None]]]] = {}

    async def initialize(self) -> None:
        """Initialize a connection to the redis server."""
        try:
            log.info("Checking redis connection...")
            await self.client.initialize()
            await self.client.ping()  # pyright: ignore[reportUnknownMemberType] redis uses unknowns :/
            log.success("Redis connection successful!")
        except RedisConnectionError as error:
            log.critical(
                "Connection to Redis failed. Ensure an instance is running and the connection config is correct."
            )
            raise error

    async def close(self) -> None:
        """Close redis connections."""
        for task in self.tasks:
            task.cancel()
        await self.client.aclose()

    async def publish(self, channel: str, message: Any) -> None:
        """Publish a message to a given channel."""
        await cast(Awaitable[Any], self.client.publish(f"{PREFIX}{channel}", message))  # pyright:ignore[reportUnknownMemberType] redis-py uses Unknown :(

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
        """A subscriber function that should be wrapped in an asyncio task."""
        async with self.client.pubsub() as pubsub:  # pyright:ignore[reportUnknownMemberType] redis-py uses Unknown :(
            await pubsub.subscribe(channel_key)  # pyright:ignore[reportUnknownMemberType] redis-py uses Unknown :(
            logger.trace(f"Subscribed to channel {channel_key}")
            while True:
                # if len(self.subscriptions[channel_key]) == 0:
                #     break
                try:
                    message = await pubsub.get_message(  # pyright:ignore[reportUnknownVariableType] redis-py uses Unknown :(
                        ignore_subscribe_messages=True, timeout=0.01
                    )
                    if message is not None:
                        if isinstance(message["data"], str):
                            data = message["data"]
                        elif isinstance(message["data"], bytes):
                            data = message["data"].decode()
                        else:
                            data = str(message["data"])  # pyright:ignore[reportUnknownArgumentType] redis-py uses Unknown :(

                        for callback in self.subscriptions[channel_key]:
                            await callback(data)

                except (ValueError, asyncio.CancelledError):
                    await pubsub.unsubscribe(channel_key)  # pyright:ignore[reportUnknownMemberType] redis-py uses Unknown :(
                    break

    async def set(
        self, key: str, value: bytes, compress: bool = False, ttl: int = 0
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

    async def get(self, key: str, compressed: bool = False) -> bytes | None:
        """Generically get a key-value pair."""
        data = await self.client.get(f"{PREFIX}{key}")
        if data is None:
            return

        if compressed:
            data = ZSTD_DECOMPRESSOR.decompress(data)
        return data


REDIS_CLIENT = RedisClient()
