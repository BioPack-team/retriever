import uvicorn

from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging


def main() -> None:
    """Run the server."""
    # logging -> loguru intercept needs to be set up early
    logging_config = configure_logging()

    uvicorn.run(
        "retriever.server:app",
        host=CONFIG.host,
        port=CONFIG.port,
        log_config=logging_config,
    )


if __name__ == "__main__":
    main()
