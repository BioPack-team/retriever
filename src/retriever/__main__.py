import uvicorn

from retriever.config.logging import configure_logging


def main() -> None:
    """Run the server."""
    # logging -> loguru intercept needs to be set up early
    logging_config = configure_logging()

    uvicorn.run(
        "retriever.server:app",
        host="0.0.0.0",
        port=3000,
        log_config=logging_config,
    )


if __name__ == "__main__":
    main()
