import asyncio
import multiprocessing

import uvicorn

from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging
from retriever.utils.logs import cleanup


def main() -> None:
    """Run the server."""
    # logging -> loguru intercept needs to be set up early
    logging_config = configure_logging()

    n_workers = CONFIG.workers or int(multiprocessing.cpu_count() / 2)

    uvicorn.run(
        "retriever.server:app",
        host=CONFIG.host,
        port=CONFIG.port,
        log_config=logging_config,
        workers=n_workers,
        loop="uvloop",
    )

    # Wait for loguru to complete
    if n_workers > 1:
        asyncio.get_event_loop().run_until_complete(cleanup())


if __name__ == "__main__":
    main()
