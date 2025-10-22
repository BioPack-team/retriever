import multiprocessing
import os

import uvicorn
import uvloop
import yaml
from loguru import logger

from retriever.background import BackgroundProcessManager
from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging
from retriever.config.write_configs import write_default_configs
from retriever.utils.logs import add_mongo_sink, cleanup
from retriever.utils.mongo import MONGO_CLIENT, MONGO_QUEUE
from retriever.utils.uvicorn_multiprocess import AsyncMultiprocess


async def _main_inner() -> None:
    # /// PRE-SERVER SETUP ///
    os.environ["PYTHONHASHSEED"] = "0"  # Deterministic hashing
    multiprocessing.set_start_method("spawn")

    # logging -> loguru intercept needs to be set up early
    logging_config = configure_logging()

    # For main process logging
    if not CONFIG.debug:
        await MONGO_CLIENT.initialize()
        await MONGO_QUEUE.start_process_task()
        add_mongo_sink()

    logger.debug(
        f"Starting with config: \n{yaml.dump(yaml.safe_load(CONFIG.model_dump_json()))}"
    )

    write_default_configs()

    background_manager = BackgroundProcessManager()
    await background_manager.start_process()

    n_workers = CONFIG.workers or int(multiprocessing.cpu_count() / 2)

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
    if not CONFIG.debug:
        await MONGO_QUEUE.stop_process_task()
        await MONGO_CLIENT.close()

    # Wait for loguru to complete
    await cleanup()


def main() -> None:
    """Run the server."""
    uvloop.run(_main_inner())


if __name__ == "__main__":
    main()
