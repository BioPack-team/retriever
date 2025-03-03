from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue
from collections.abc import AsyncGenerator
from datetime import datetime
from multiprocessing.queues import Queue
from time import time
from typing import Any

from bson import DEFAULT_CODEC_OPTIONS
from loguru import logger as log
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
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
        for attempt in range(CONFIG.setup_attempts):
            try:
                await self.client.admin.command("ping")
                log.success("Mongodb connection successful!")
                break
            except Exception as error:
                if attempt < CONFIG.setup_attempts - 1:
                    await asyncio.sleep(0.1)
                    continue
                log.critical(
                    "Connection to MongoDB failed. Ensure an instance is running and the connection config is correct."
                )
                raise error

        job_collection = self.get_job_collection()
        await job_collection.create_index("key", unique=True, background=True)
        await job_collection.create_index(
            "touched", background=True, expireAfterSeconds=CONFIG.job.ttl
        )
        await job_collection.create_index(
            "completed", background=True, expireAfterSeconds=CONFIG.job.ttl_max
        )

        log_collection = self.get_log_collection()
        await log_collection.create_index(
            {"record.time.repr": 1}, background=True, expireAfterSeconds=CONFIG.log.ttl
        )

    async def insert(
        self,
        document: dict[str, Any],
        collection: str = "log",
        fail_silent: bool = True,
    ) -> None:
        """Insert a document to a collection with no special behavior."""
        working_collection = {
            "log": self.get_log_collection,
            "job": self.get_job_collection,
        }[collection]()

        for attempt in range(1, CONFIG.mongo.attempts + 1):
            try:
                await working_collection.insert_one(document)
                break
            except Exception as e:
                if attempt < CONFIG.mongo.attempts:
                    await asyncio.sleep(min(0.5, 0.025 * 2**attempt))
                    continue
                if not fail_silent:
                    raise e

    async def update_job_doc(self, job: dict[str, Any]) -> None:
        """Upsert a job into the database for persistent storage.

        Fails silently (with log) to avoid job failure.
        """
        collection = self.get_job_collection()

        update_time = datetime.now()
        job["touched"] = update_time
        job["completed"] = update_time

        for attempt in range(1, CONFIG.mongo.attempts + 1):
            try:
                result = await collection.replace_one(
                    {"key": job["key"]},
                    job,
                    upsert=True,
                )
                log.trace(
                    f"Job {job['key']} state serialized to MongoDB.",
                    extra={"mongo_result": result},
                )
                break
            except Exception:
                if attempt < CONFIG.mongo.attempts:
                    await asyncio.sleep(min(0.5, 0.025 * 2**attempt))
                log.exception(f"Job {job['key']} MongoDB serialization failed!")

    async def get_job_doc(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve a job from the database."""
        collection = self.get_job_collection()

        job = await collection.find_one(
            {"key": job_id},
        )
        if job is None:
            return job
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
    ) -> AsyncGenerator[str]:
        """Return a generator of all logs."""
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
        query: dict[str, Any] = {"record.level.no": {"$gte": levels[level.upper()]}}
        if start or end:
            query["record.time.repr"] = {}
        if start:
            query["record.time.repr"]["$gte"] = start
        if end:
            query["record.time.repr"]["$lte"] = end

        logs = self.get_log_collection()
        cursor = logs.find(query).sort("record.time.repr", 1)
        yield "["
        first = True
        async for document in cursor:
            del document["_id"]
            document["record"]["time"]["repr"] = str(document["record"]["time"]["repr"])
            yield ("," if not first else "") + json.dumps(document)
            first = False

        yield "]"

    async def close(self) -> None:
        """Clean up and close MongoDB client threads/connections."""
        self.client.close()


class MongoQueue:
    """A multiprocessing queue for sending items to MongoDB in the main process."""

    def __init__(self, client: MongoClient) -> None:
        """Initialize a mongo client and multiprocessing queue."""
        self.queue: Queue[tuple[str, dict[str, Any]]] = multiprocessing.Queue()
        self.client: MongoClient = client
        self.process_task: asyncio.Task[None] | None = None

    async def process_queue(self) -> None:
        """Periodically poll the queue for a task and handle it accordingly."""
        while True:
            try:
                await asyncio.sleep(0.1)
                target, doc = self.queue.get_nowait()
                if not hasattr(self.client, target):
                    log.error(
                        f"MongoClient has no operation {target}. Operation skipped.",
                        extra={"no_mongo_log": True},
                    )
                try:
                    await getattr(self.client, target)(doc)
                except Exception:
                    print("exception")
                    log.exception(
                        f"An exception occurred in the operation {target}",
                        extra={"doc": doc, "no_mongo_log": True},
                    )
            except queue.Empty:
                continue
            except (ValueError, asyncio.CancelledError):  # Queue is closed
                break

    async def start_process_task(self) -> None:
        """Start a processing loop to serialize documents to MongoDB."""
        if self.process_task is not None:
            raise ValueError("Cannot start second MongoDB queue process task.")
        self.process_task = asyncio.create_task(
            self.process_queue(), name="mongo_process_loop"
        )
        log.info("Started MongoDB serialize task.")

    async def stop_process_task(self) -> None:
        """Close the queue and stop the process loop task."""
        start = time()
        while not self.queue.empty():
            # Don't wait forever
            if time() - start > CONFIG.mongo.shutdown_timeout:
                break
            await asyncio.sleep(0.1)
        self.queue.close()
        self.queue.join_thread()

        try:
            if self.process_task is None:
                return
            self.process_task.cancel()
            await self.process_task
            log.info("Stopped MongoDB serialize task.")
        except Exception:
            log.exception("Exception occurred while stopping MongoDB serialize task.")
