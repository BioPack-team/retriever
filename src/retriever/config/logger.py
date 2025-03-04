from __future__ import annotations

import inspect
import logging.config
import sys
import typing
from pathlib import Path
from typing import Any, override

from loguru import logger

from retriever.config.general import CONFIG

if typing.TYPE_CHECKING:
    from loguru import Record


class InterceptHandler(logging.Handler):
    """Logger which forwards to loguru."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Intercept stdlib logging and send it to loguru handling."""
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging() -> dict[str, Any]:
    """Route standardlib logging to loguru and configure loguru."""
    # Reroute standard logging to loguru
    std_log_config = {
        "version": 1,
        "handlers": {
            "loguru": {
                "()": InterceptHandler,
            }
        },
        "loggers": {
            "retriever": {
                "level": "DEBUG",
                "handlers": ["loguru"],
            },
            "uvicorn": {
                "level": "DEBUG",
                "handlers": ["loguru"],
            },
        },
        "incremental": False,
        "disable_existing_loggers": True,
    }
    logging.config.dictConfig(std_log_config)

    def format_stdout(record: Record) -> str:
        header = "<cyan>{time:YYYY-MM-DDTHH:mm:ss.SSSZ}</cyan> <blue>{process.id:4}</blue> <level>{level:8}</level>"
        log = "{message:80} <cyan>{name}:{function}():{line}</cyan>\n{exception}"
        if "job_id" in record["extra"]:
            header += f"<green>{record['extra']['job_id'][:8]}</green> "

        return header + log

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stdout,
        format=format_stdout,
        # format="<cyan>{time:YYYY-MM-DDTHH:mm:ss.SSSZ}</cyan> <blue>{process.id:4}</blue> <level>{level:8}</level> {message:80} <cyan>{name}:{function}():{line}</cyan> {extra}",
        colorize=True,
        backtrace=True,
        diagnose=False,
        enqueue=True,
        level=CONFIG.log_level,
    )
    logger.add(
        Path.cwd() / "logs/retriever.log",
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level:8} | {message:80} | {extra} | {process.id}:{name}:{function}:{line}",
        colorize=False,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        rotation="monthly",
        retention=3,
        compression="tar.gz",
        level=CONFIG.log_level,
    )
    logger.add(
        Path.cwd() / "logs/retriever.log.json",
        format="{message}",
        colorize=False,
        serialize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        rotation="monthly",
        retention=3,
        compression="tar.gz",
        level=CONFIG.log_level,
    )

    return std_log_config
