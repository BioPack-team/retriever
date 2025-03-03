import multiprocessing
import os
import sys
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, cast

from loguru import logger as log
from opentelemetry import trace
from saq.worker import import_settings, start

from retriever.config.general import CONFIG
from retriever.config.logging import configure_logging
from retriever.tasks.task_queue import RetrieverQueue
from retriever.utils import telemetry
from retriever.utils.logging import add_mongo_sink

tracer = trace.get_tracer("lookup.execution.tracer")
QUEUES = ["src.retriever.tasks.task_queue.SETTINGS"]


def start_worker(
    queue_settings: str,
    mongo_queue: Queue[tuple[str, dict[str, Any]]],
) -> None:
    """Start up a worker, patching the MongoDB serialize queue into the job queue."""
    settings_obj = import_settings(queue_settings)
    if "queue" not in settings_obj:
        log.critical("Worker did not receive proper queue instance!")
        sys.exit(1)
    cast(RetrieverQueue, settings_obj["queue"]).mongo_queue = mongo_queue

    configure_logging()
    add_mongo_sink(mongo_queue)
    telemetry.configure_telemetry()
    log.info(f"Worker process {os.getpid()} started.")

    start(queue_settings)


def start_workers(
    mongo_queue: Queue[tuple[str, dict[str, Any]]] | None,
) -> list[multiprocessing.Process]:
    """Start SAQ workers for all queues, as one unit."""
    # Allow importlib to find queue (see SAQ's __main__.py)
    sys.path.append(str(Path.cwd()))
    workers: list[multiprocessing.Process] = []

    for queue_settings in QUEUES:
        n_workers = CONFIG.workers or multiprocessing.cpu_count()
        for i in range(n_workers):
            p = multiprocessing.Process(
                target=start_worker,
                args=(queue_settings, mongo_queue),
                name=f"saq-worker-{i + 1}",
            )
            workers.append(p)
            p.start()

    return workers


def stop_workers(workers: list[multiprocessing.Process]) -> None:
    """Stop a list of workers."""
    for worker in workers:
        worker.join(1)
    log.info("All workers stopped.")
