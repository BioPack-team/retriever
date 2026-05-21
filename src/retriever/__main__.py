import asyncio
import contextlib
import math
import multiprocessing
import os
from datetime import datetime

import uvicorn
import uvloop
import yaml
from loguru import logger

from retriever.background import BackgroundProcessManager
from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging
from retriever.config.write_configs import write_default_configs
from retriever.utils.logs import add_mongo_sink, cleanup
from retriever.utils.mongo import MongoClient, MongoQueue
from retriever.utils.redis import (
    PROCESS_TTL_SECONDS,
    RedisClient,
    heartbeat,
)
from retriever.utils.uvicorn_multiprocess import AsyncMultiprocess


async def _main_inner() -> None:
    # /// PRE-SERVER SETUP ///
    os.environ["PYTHONHASHSEED"] = "0"  # Deterministic hashing
    multiprocessing.set_start_method("spawn")

    # logging -> loguru intercept needs to be set up early
    logging_config = configure_logging()

    # For main process logging
    main_heartbeat_task: asyncio.Task[None] | None = None
    if not CONFIG.debug:
        await MongoClient().initialize()
        await MongoQueue().initialize()
        add_mongo_sink()
        await RedisClient().initialize()
        main_pid = os.getpid()
        main_started_at = datetime.now().astimezone()
        await RedisClient().register_main(
            main_pid, main_started_at, PROCESS_TTL_SECONDS
        )
        main_heartbeat_task = asyncio.create_task(
            heartbeat(
                lambda: RedisClient().register_main(
                    main_pid, main_started_at, PROCESS_TTL_SECONDS
                ),
                role_label="Main",
            ),
            name="main-heartbeat",
        )

    logger.debug(
        f"Starting with config: \n{yaml.dump(yaml.safe_load(CONFIG.model_dump_json()))}"
    )

    write_default_configs()

    background_manager = BackgroundProcessManager()
    await background_manager.start_process()

    n_workers = CONFIG.workers or math.ceil(
        multiprocessing.cpu_count() * CONFIG.worker_cpu_ratio
    )

    # /// RUN SERVER ///

    config = uvicorn.Config(
        "retriever.server:app",
        host=CONFIG.host,
        port=CONFIG.port,
        log_config=logging_config,
        workers=n_workers,
        loop="uvloop",
        proxy_headers=CONFIG.trust_proxy,
    )
    server = uvicorn.Server(config)

    # Taken from uvicorn.run; run multiple workers
    # Run even 1 worker in separate process for consistency
    try:
        sock = config.bind_socket()
        if not CONFIG.debug:
            await AsyncMultiprocess(config, target=server.run, sockets=[sock]).run()
        else:
            await server.serve([sock])
    except KeyboardInterrupt:
        pass

    # /// POST-SERVER CLEANUP ///

    await background_manager.wrapup()
    if main_heartbeat_task is not None:
        main_heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await main_heartbeat_task
    if not CONFIG.debug:
        await RedisClient().wrapup()
        await MongoQueue().wrapup()
        await MongoClient().close()

    # Wait for loguru to complete
    await cleanup()


def main() -> None:
    """Run the server."""
    uvloop.run(_main_inner())


if __name__ == "__main__":
    main()
