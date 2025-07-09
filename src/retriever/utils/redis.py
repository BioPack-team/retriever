import redis.asyncio as redis
from loguru import logger as log
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException

from retriever.config.general import CONFIG


async def test_redis() -> None:
    """Attempt to open a connection to the Redis server to verify proper functioning."""
    retry = Retry(ExponentialBackoff(), CONFIG.redis.attempts)
    try:
        if CONFIG.redis.cluster:
            log.info("Checking redis connection (cluster mode)...")
            r = RedisCluster(  # pyright:ignore [reportAbstractUsage] this instantiates correctly
                host=CONFIG.redis.host,
                port=CONFIG.redis.port,
                password=str(CONFIG.redis.password),
                ssl=CONFIG.redis.ssl_enabled,
                ssl_cert_reqs="none",
                retry=retry,
            )
            await r.initialize()
        else:
            log.info("Checking redis connection (standard mode)...")
            r = redis.Redis(
                host=CONFIG.redis.host,
                port=CONFIG.redis.port,
                password=str(CONFIG.redis.password),
                ssl=CONFIG.redis.ssl_enabled,
                ssl_cert_reqs="none",
                retry=retry,
            )
            await r.initialize()

        await r.ping()  # pyright: ignore[reportUnknownMemberType] redis uses unknowns :/
        log.success("Redis connection successful!")
        await r.close()
    except (RedisClusterException, RedisConnectionError) as error:
        log.critical(
            "Connection to Redis failed. Ensure an instance is running and the connection config is correct."
        )
        raise error
