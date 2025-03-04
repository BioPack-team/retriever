from __future__ import annotations

from collections import deque
from collections.abc import Callable
from multiprocessing.queues import Queue
from typing import Any

import loguru
from loguru import logger
from saq import Job

from retriever.config.general import CONFIG


def format_trapi_log(message: loguru.Message) -> dict[str, str | None]:
    """Format a loguru message into a TRAPI-spec LogEntry."""
    log_entry: dict[str, str | None] = {}
    log_entry["level"] = message.record["level"].name.upper()
    log_entry["message"] = message.record["message"]
    log_entry["timestamp"] = (
        message.record["time"].astimezone().isoformat(timespec="milliseconds")
    )
    log_entry["code"] = None
    return log_entry


def add_context_trapi_sink(
    context_id: str,
) -> tuple[loguru.Logger, deque[dict[str, str | None]], Callable[[], None]]:
    """Add a TRAPI-spec sink to logging which is bound to a specific context by ID."""
    log_deque: deque[dict[str, str | None]] = deque()

    def trapi_sink(message: loguru.Message) -> None:
        log_entry = format_trapi_log(message)
        log_deque.append(log_entry)

    handler_id = logger.add(
        trapi_sink,
        filter=lambda record: record["extra"].get("context_id") == context_id,
        enqueue=True,
    )

    def remove_sink() -> None:
        logger.remove(handler_id)

    return logger.bind(context_id=context_id), log_deque, remove_sink


def add_job_trapi_sink(
    job: Job,
) -> tuple[loguru.Logger, deque[dict[str, str | None]], Callable[[], None]]:
    """Add a TRAPI-spec sink to logging which is held to specific job_id."""
    log_deque: deque[dict[str, str | None]] = deque()

    async def trapi_sink(message: loguru.Message) -> None:
        log_entry = format_trapi_log(message)
        log_deque.append(log_entry)
        job.meta["logs"] = list(log_deque)
        await job.update()
        # await job.update(logs=list(log_deque))

    handler_id = logger.add(
        trapi_sink,
        filter=lambda record: record["extra"].get("job_id") == job.key,
        enqueue=True,
    )

    def remove_sink() -> None:
        logger.remove(handler_id)

    return logger.bind(job_id=job.key), log_deque, remove_sink


def add_mongo_sink(queue: Queue[tuple[str, dict[str, Any]]]) -> None:
    """Add a sink that sends logs to MongoDB via the main process."""
    if not CONFIG.log.log_to_mongo:
        return

    def mongo_sink(message: loguru.Message) -> None:
        # Convert message to serializable dict
        # Replicates loguru's own internal serialization
        # But avoids the overhead of dumping to str, then parsing back for mongo
        exception = message.record["exception"]

        if exception is not None:
            exception = {
                "type": None if exception.type is None else exception.type.__name__,
                "value": str(exception.value),
                "traceback": bool(exception.traceback),
            }
        log = {
            "text": message.record["message"],
            "record": {
                "elapsed": {
                    "repr": repr(message.record["elapsed"]),
                    "seconds": message.record["elapsed"].total_seconds(),
                },
                "exception": exception,
                "extra": message.record["extra"],
                "file": {
                    "name": message.record["file"].name,
                    "path": message.record["file"].path,
                },
                "function": message.record["function"],
                "level": {
                    "icon": message.record["level"].icon,
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
                "time": {
                    "repr": message.record["time"],
                    "timestamp": message.record["time"].timestamp(),
                },
            },
        }

        try:
            queue.put_nowait(("insert", log))
        except ValueError:
            return

    # Allow an out via no_mongo_log to avoid feedback loops
    logger.add(
        mongo_sink,
        backtrace=True,
        diagnose=True,
        serialize=True,
        filter=lambda record: not record["extra"].get("no_mongo_log", False),
    )
