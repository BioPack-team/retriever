from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from datetime import datetime
from threading import Lock
from time import time
from typing import Any

from bson import DEFAULT_CODEC_OPTIONS
from loguru import logger as log
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.operations import InsertOne, ReplaceOne
from pymongo.server_api import ServerApi

from retriever.config.general import CONFIG
from retriever.type_defs import LogLevel

CODEC_OPTIONS = DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)


class MongoClient:
    """A client abstraction layer for basic operations."""

    def __init__(self) -> None:
        """Set up the client."""
        self.client: AsyncIOMotorClient[dict[str, Any]]
        if CONFIG.mongo.authsource is None:
            self.client = AsyncIOMotorClient(
                host=CONFIG.mongo.host,
                port=CONFIG.mongo.port,
                username=CONFIG.mongo.username,
                password=CONFIG.mongo.password,
                server_api=ServerApi("1"),
            )
        else:
            self.client = AsyncIOMotorClient(
                host=CONFIG.mongo.host,
                port=CONFIG.mongo.port,
                username=CONFIG.mongo.username,
                password=CONFIG.mongo.password,
                authsource=CONFIG.mongo.authsource,
                server_api=ServerApi("1"),
            )

    def get_job_collection(self) -> AsyncIOMotorCollection[dict[str, Any]]:
        """Get job_state collection with standard options."""
        db = self.client.retriever_persist
        return db.get_collection("job_state", codec_options=CODEC_OPTIONS)

    def get_log_collection(self) -> AsyncIOMotorCollection[dict[str, Any]]:
        """Get log_dump collection with standard options."""
        db = self.client.retriever_persist
        return db.get_collection("log_dump", codec_options=CODEC_OPTIONS)

    async def initialize(self) -> None:
        """Check and prepare mongo client.

        Pings to check connection, then sets up collection indices.
        """
        log.info("Checking mongodb connection...")
        try:
            await self.client.admin.command("ping")
            log.success("Mongodb connection successful!")
        except Exception as error:
            log.critical(
                "Connection to MongoDB failed. Ensure an instance is running and the connection config is correct."
            )
            raise error

        job_collection = self.get_job_collection()
        await job_collection.create_index("job_id", unique=True, background=True)
        await job_collection.create_index(
            "touched", background=True, expireAfterSeconds=CONFIG.job.ttl
        )
        await job_collection.create_index(
            "completed", background=True, expireAfterSeconds=CONFIG.job.ttl_max
        )

        log_collection = self.get_log_collection()
        await log_collection.create_index(
            {"time": 1}, background=True, expireAfterSeconds=CONFIG.log.ttl
        )

    async def batch_write(
        self,
        operations: list[InsertOne[dict[str, Any]]] | list[ReplaceOne[dict[str, Any]]],
        collection: AsyncIOMotorCollection[dict[str, Any]],
    ) -> None:
        """Do a set of write/replace operations on a given collection.

        Fails silently to avoid runaway log issues.
        """
        for attempt in range(1, CONFIG.mongo.attempts + 1):
            try:
                _ = await collection.bulk_write(  # pyright: ignore[reportUnknownMemberType] Motor uses unknowns :/
                    operations
                )
                break
            except Exception:
                if attempt < CONFIG.mongo.attempts:
                    await asyncio.sleep(min(0.5, 0.025 * 2**attempt))
                    continue

    async def batch_job_state(
        self, operations: list[ReplaceOne[dict[str, Any]]]
    ) -> None:
        """Write a batch of jobs to mongo."""
        collection = self.get_job_collection()
        await self.batch_write(operations, collection)

    def job_state(self, job: dict[str, Any]) -> ReplaceOne[dict[str, Any]]:
        """Create an operation for upserting a job state doc."""
        update_time = datetime.now()
        job["touched"] = update_time
        job["completed"] = update_time
        return ReplaceOne(
            {"job_id": job["job_id"]},
            job,
            upsert=True,
        )

    async def batch_log_dump(self, operations: list[InsertOne[dict[str, Any]]]) -> None:
        """Write a batch of logs to mongo."""
        collection = self.get_log_collection()
        await self.batch_write(operations, collection)

    def log_dump(self, log: dict[str, Any]) -> InsertOne[dict[str, Any]]:
        """Create a write operation for a given log."""
        return InsertOne(log)

    async def get_job_doc(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve a job from the database."""
        collection = self.get_job_collection()

        job = await collection.find_one(
            {"job_id": job_id},
        )
        if job is None:
            return
        await collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"touched": datetime.now()}},
        )

        del job["_id"]
        return job

    async def get_logs(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        level: LogLevel = "DEBUG",
        job_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Return a generator of a filtered set of logs."""
        # Set up query
        levels = {
            "TRACE": 5,
            "DEBUG": 10,
            "INFO": 20,
            "SUCCESS": 25,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        query: dict[str, Any] = {"level.no": {"$gte": levels[level.upper()]}}
        if start or end:
            query["time"] = {}
        if start:
            query["time"]["$gte"] = start
        if end:
            query["time"]["$lte"] = end
        if job_id is not None:
            query["extra.job_id"] = job_id

        logs = self.get_log_collection()
        cursor = logs.find(query).sort("timestamp", 1)
        async for document in cursor:
            del document["_id"]
            document["time"] = (
                document["time"].astimezone().isoformat(timespec="milliseconds")
            )
            yield document

    async def get_flat_logs(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        level: LogLevel = "DEBUG",
        job_id: str | None = None,
    ) -> AsyncGenerator[str]:
        """Return a generator of logs as flat strings, as they would appear in stdout."""
        first = True
        async for log_doc in self.get_logs(start, end, level, job_id):
            message = log_doc["message"]
            if log_doc.get("exception"):
                exception = log_doc["exception"]
                message = f"{message}\n{exception.get('traceback')}\n{exception.get('type')}: {exception.get('value')}"
            yield "{}{} {:4} {:7} {:80} {}".format(
                "\n" if not first else "",
                log_doc["time"],
                log_doc["process"]["id"],
                log_doc["level"]["name"],
                (
                    f"{log_doc['extra']['job_id'][:8]} "
                    if "job_id" in log_doc["extra"]
                    else ""
                )
                + message,
                f"{log_doc['name']}.{log_doc['function']}:{log_doc['line']}",
            )
            first = False

    async def close(self) -> None:
        """Clean up and close MongoDB client threads/connections."""
        self.client.close()


class MongoQueue:
    """A multiprocessing queue for sending items to MongoDB in the main process."""

    def __init__(self, client: MongoClient) -> None:
        """Initialize a mongo client and multiprocessing queue."""
        self.queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self.client: MongoClient = client
        self.process_tasks: list[asyncio.Task[None]] | None = None
        self.task_deques: dict[str, tuple[Lock, deque[Any]]] = {
            "job_state": (Lock(), deque()),
            "log_dump": (Lock(), deque()),
        }

    async def process_queue(self) -> None:
        """Periodically poll the queue for a task and handle it accordingly."""
        while True:
            try:
                target, doc = self.queue.get_nowait()
                if not hasattr(self.client, target):
                    log.error(
                        f"MongoClient has no operation {target}. Operation skipped.",
                        extra={"no_mongo_log": True},
                    )
                try:
                    task = getattr(self.client, target)(doc)
                    lock, task_deque = self.task_deques[target]
                    with lock:
                        task_deque.append(task)
                    self.queue.task_done()
                except Exception:
                    log.exception(
                        f"An exception occurred in the operation {target}",
                        extra={"doc": doc, "no_mongo_log": True},
                    )
            except asyncio.queues.QueueEmpty:
                try:
                    await asyncio.sleep(0.05)
                    continue
                except asyncio.CancelledError:  # likely to be cancelled while waiting
                    break
            except (ValueError, asyncio.CancelledError):  # Queue is closed
                break

    async def process_batch_tasks(self) -> None:
        """Periodically grab a batch of mongo tasks and run them."""
        while True:
            try:
                for target, (lock, task_deque) in self.task_deques.items():
                    with lock:
                        tasks = [
                            task_deque.popleft()
                            for _ in range(
                                min(len(task_deque), CONFIG.mongo.flood_batch_size)
                            )
                        ]
                    if len(tasks) == 0:
                        continue
                    await getattr(self.client, f"batch_{target}")(tasks)

                await asyncio.sleep(0.05)
            except (ValueError, asyncio.CancelledError):
                break

    async def start_process_task(self) -> None:
        """Start a processing loop to serialize documents to MongoDB."""
        if self.process_tasks is not None:
            raise ValueError("Cannot start second MongoDB queue process task.")
        self.process_tasks = [
            asyncio.create_task(self.process_queue(), name="mongo_process_loop"),
            asyncio.create_task(self.process_batch_tasks(), name="mongo_batch_loop"),
        ]
        log.info("Started MongoDB serialize task.")

    async def stop_process_task(self) -> None:
        """Close the queue and stop the process loop task."""
        start = time()
        while not self.queue.empty():
            # Don't wait forever
            if time() - start > CONFIG.mongo.shutdown_timeout:
                break
            await asyncio.sleep(0.1)
        self.queue.shutdown()
        await self.queue.join()

        try:
            if self.process_tasks is None:
                return
            for task in self.process_tasks:
                task.cancel()
                await task
            log.info("Stopped MongoDB serialize task.")
        except Exception:
            log.exception("Exception occurred while stopping MongoDB serialize task.")

    def put(self, task_name: str, doc: dict[str, Any]) -> None:
        """Add a document to the queue."""
        try:
            self.queue.put_nowait((task_name, doc))
        except (asyncio.queues.QueueShutDown, asyncio.queues.QueueFull):
            return


MONGO_CLIENT = MongoClient()
MONGO_QUEUE = MongoQueue(MONGO_CLIENT)
