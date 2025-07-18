from __future__ import annotations

import json
import traceback
from collections import deque
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import loguru
from loguru import logger
from reasoner_pydantic import LogLevel as TRAPILogLevel

from retriever.config.general import CONFIG
from retriever.types.general import LogLevel
from retriever.types.trapi import LogEntryDict
from retriever.utils.mongo import MONGO_QUEUE


def log_level_to_trapi(level: LogLevel) -> TRAPILogLevel:
    """Collapse loguru log levels into TRAPI-spec log levels."""
    level_str = level.upper()
    if level_str in TRAPILogLevel:
        return TRAPILogLevel[level_str]
    match level_str:
        case "TRACE":  # Should never occur since trace logs are left out of TRAPI
            return TRAPILogLevel.DEBUG
        case "SUCCESS":
            return TRAPILogLevel.INFO
        case "CRITICAL":
            return TRAPILogLevel.ERROR
        case _:
            return TRAPILogLevel.DEBUG


def trapi_level_to_int(level: TRAPILogLevel) -> int:
    """Convert a given TRAPI log level to the equivalent loguru level number."""
    return logger.level(level).no


def format_trapi_log(
    level: LogLevel,
    message: str,
    timestamp: datetime | None = None,
    trace: str | None = None,
) -> LogEntryDict:
    """Format a loguru message into a TRAPI-spec LogEntry."""
    if timestamp is None:
        timestamp = datetime.now().astimezone()
    log_entry = LogEntryDict(
        level=log_level_to_trapi(level),
        message=message,
        timestamp=timestamp.isoformat(timespec="milliseconds"),
    )
    if trace:
        log_entry.trace = (  # pyright:ignore[reportAttributeAccessIssue] It's allowed
            trace
        )
    return log_entry


async def structured_log_to_trapi(
    logs: AsyncGenerator[dict[str, Any]],
) -> AsyncGenerator[LogEntryDict]:
    """Take an async generator of structured logs and yield TRAPI logs asynchronously."""
    async for log in logs:
        trace = None
        if log.get("exception"):
            exception = log["exception"]
            trace = f"{exception.get('traceback')}\n{exception.get('type')}: {exception.get('value')}"
        yield format_trapi_log(
            log["level"]["name"],
            log["message"],
            log["time"],
            trace,
        )


async def objs_to_json(generator: AsyncGenerator[Any]) -> AsyncGenerator[str]:
    """Take an async generator of json-dumpable and yield them in a JSON-compliant fasion."""
    yield "["

    first = True
    async for item in generator:
        yield ("" if first else ",") + json.dumps(item)
        first = False
    yield "]"


class TRAPILogger:
    """A logger that logs to both standard loguru outputs and retains TRAPI-format logs for later use."""

    def __init__(self, job_id: str) -> None:
        """Initialize an instance."""
        self.log_deque: deque[LogEntryDict] = deque()
        self.job_id: str = job_id

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        kwargs["job_id"] = self.job_id
        # FIX: currently showing this function rather than originating call

        with logger.contextualize(**kwargs):
            logger.log(level, message)
        # Implicitly drop TRACE logs from TRAPI logs.
        # These should only be used in extensive debugging on a local instance.
        if level.lower() != "trace":
            self.log_deque.append(format_trapi_log(level, message))

    def trace(self, message: str, **kwargs: Any) -> None:
        """Log at trace level."""
        self._log("TRACE", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at debug level."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log("INFO", message, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Log at success level."""
        self._log("SUCCESS", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at warning level."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log at error level."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at critical level."""
        self._log("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Capture an exception as an ERROR-level log."""
        trapi_msg = f"{message}\n{traceback.format_exc()}"
        self.log_deque.append(format_trapi_log("ERROR", trapi_msg))

        kwargs["job_id"] = self.job_id
        with logger.contextualize(**kwargs):
            logger.exception(message)

    def get_logs(self) -> list[LogEntryDict]:
        """Get a generator of stored TRAPI logs."""
        return list(self.log_deque)


def add_mongo_sink() -> None:
    """Add a sink that sends logs to MongoDB via the main process."""
    if not CONFIG.log.log_to_mongo:
        return

    def mongo_sink(message: loguru.Message) -> None:
        # Convert message to serializable dict
        # Replicates loguru's own internal serialization, but simplified
        exception = message.record["exception"]

        if exception is not None:
            exception = {
                "type": None if exception.type is None else exception.type.__name__,
                "value": str(exception.value),
                "traceback": "".join(traceback.format_tb(exception.traceback)),
            }
        log = {
            "elapsed": message.record["elapsed"].total_seconds(),
            "exception": exception,
            "extra": message.record["extra"],
            "file": message.record["file"].path,
            "function": message.record["function"],
            "level": {
                "name": message.record["level"].name,
                "no": message.record["level"].no,
            },
            "line": message.record["line"],
            "message": message.record["message"],
            "module": message.record["module"],
            "name": message.record["name"],
            "process": {
                "id": message.record["process"].id,
                "name": message.record["process"].name,
            },
            "thread": {
                "id": message.record["thread"].id,
                "name": message.record["thread"].name,
            },
            "time": message.record["time"],
        }

        try:
            MONGO_QUEUE.put("log_dump", log)
        except ValueError:
            return

    # Allow an out via no_mongo_log to avoid feedback loops
    logger.add(
        mongo_sink,
        backtrace=True,
        diagnose=True,
        serialize=True,
        enqueue=True,
        filter=lambda record: not record["extra"].get("no_mongo_log", False),
        level=CONFIG.log_level,
    )


async def cleanup() -> None:
    """Finish Loguru operations."""
    await logger.complete()
    logger.remove()
