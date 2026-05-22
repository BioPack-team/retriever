from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, ClassVar, Literal, NotRequired, TypedDict

from bson.codec_options import DEFAULT_CODEC_OPTIONS
from loguru import logger as log
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.operations import InsertOne, UpdateOne
from pymongo.server_api import ServerApi

from retriever.config.general import CONFIG
from retriever.types.general import LogLevel
from retriever.utils.general import BatchedAction, Singleton

CODEC_OPTIONS = DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)


class QueryState(TypedDict):
    """Required info about a query for the job state."""

    job_id: str
    query: bytes
    job_timeout: dict[int, float]
    submitter: str
    data_tiers: list[Literal[0, 1, 2]]
    is_async: bool
    qnodes: int
    qedges: int
    qpaths: int
    status: str


class ResponseState(TypedDict):
    """Required info about a response for the job state."""

    job_id: str
    response: bytes
    knodes: int
    kedges: int
    aux_graphs: int
    results: int
    status: str


class JobDocs(TypedDict):
    """Job documents and some lifetime metadata."""

    job_id: str
    doc: NotRequired[bytes]

    # Lifetime tracking
    created: NotRequired[datetime]
    completed: NotRequired[datetime]
    touched: NotRequired[datetime]


class JobStatus(TypedDict):
    """Job status information and metadata, without actual job documents."""

    job_id: str
    status: str

    # Query info
    job_timeout: NotRequired[dict[int, float]]
    submitter: NotRequired[str]
    data_tiers: NotRequired[list[Literal[0, 1, 2]]]
    is_async: NotRequired[bool]
    qnodes: NotRequired[int]
    qedges: NotRequired[int]
    qpaths: NotRequired[int]

    # Response info
    knodes: NotRequired[int]
    kedges: NotRequired[int]
    aux_graphs: NotRequired[int]
    results: NotRequired[int]

    # Lifetime tracking
    created: NotRequired[datetime]
    completed: NotRequired[datetime]
    touched: NotRequired[datetime]


class MongoClient(metaclass=Singleton):
    """A client abstraction layer for basic operations."""

    def __init__(self) -> None:
        """Set up the client."""
        self.client: AsyncIOMotorClient[dict[str, Any]]
        if CONFIG.mongo.authsource is None:
            self.client = AsyncIOMotorClient(
                host=CONFIG.mongo.host,
                port=CONFIG.mongo.port,
                username=CONFIG.mongo.username,
                password=(
                    CONFIG.mongo.password.get_secret_value()
                    if CONFIG.mongo.password
                    else None
                ),
                server_api=ServerApi("1"),
            )
        else:
            self.client = AsyncIOMotorClient(
                host=CONFIG.mongo.host,
                port=CONFIG.mongo.port,
                username=CONFIG.mongo.username,
                password=(
                    CONFIG.mongo.password.get_secret_value()
                    if CONFIG.mongo.password
                    else None
                ),
                authsource=CONFIG.mongo.authsource,
                server_api=ServerApi("1"),
            )
        # Patch client to get the current asyncio loop
        self.client.get_io_loop = asyncio.get_running_loop

    def get_job_collection(
        self,
    ) -> tuple[
        AsyncIOMotorCollection[dict[str, Any]], AsyncIOMotorCollection[dict[str, Any]]
    ]:
        """Get job_state collection with standard options."""
        db = self.client.retriever_persist
        return (
            db.get_collection("job_status", codec_options=CODEC_OPTIONS),
            db.get_collection("job_docs", codec_options=CODEC_OPTIONS),
        )

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

        job_collections = self.get_job_collection()
        for collection in job_collections:
            await collection.create_index("job_id", unique=True, background=True)
            await collection.create_index(
                "touched", background=True, expireAfterSeconds=CONFIG.job.ttl
            )
            await collection.create_index(
                "completed", background=True, expireAfterSeconds=CONFIG.job.ttl_max
            )

        log_collection = self.get_log_collection()
        await log_collection.create_index(
            {"time": 1}, background=True, expireAfterSeconds=CONFIG.log.mongo_ttl
        )

    async def batch_write(
        self,
        operations: list[InsertOne[dict[str, Any]]] | list[UpdateOne],
        collection: AsyncIOMotorCollection[dict[str, Any]],
    ) -> None:
        """Do a set of write/replace operations on a given collection."""
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
                log.exception(
                    f"Failed to write batch after {CONFIG.mongo.attempts} attempts.",
                    no_mongo_log=True,
                )

    async def batch_job_state(
        self, job_status_ops: list[UpdateOne], job_doc_ops: list[UpdateOne]
    ) -> None:
        """Write a batch of jobs to mongo."""
        job_status, job_docs = self.get_job_collection()
        await self.batch_write(job_status_ops, job_status)
        await self.batch_write(job_doc_ops, job_docs)

    def job_state(self, job: QueryState | ResponseState) -> tuple[UpdateOne, UpdateOne]:
        """Create an operation for upserting a job state doc."""
        update_time = datetime.now()
        status_data = {
            "$set": JobStatus(
                **{k: v for k, v in job.items() if k not in ("query", "response")},  # pyright:ignore[reportArgumentType] We know it's valid
                touched=update_time,
            ),
            "$setOnInsert": {"created": update_time},
        }
        doc_type = "query" if "query" in job else "response"
        if doc_type == "response":
            status_data["$set"]["completed"] = update_time

        doc = job.get(doc_type)
        doc_inner: JobDocs = {
            "job_id": job["job_id"],
            "touched": update_time,
        }
        if doc_type == "response":
            doc_inner["completed"] = update_time
        if doc is not None:
            doc_inner["doc"] = doc
        doc_data = {
            "$set": doc_inner,
            "$setOnInsert": {"created": update_time},
        }
        return (
            UpdateOne(
                {"job_id": job["job_id"]},
                status_data,
                upsert=True,
            ),
            UpdateOne(
                {"job_id": job["job_id"]},
                doc_data,
                upsert=True,
            ),
        )

    async def batch_log_dump(self, operations: list[InsertOne[dict[str, Any]]]) -> None:
        """Write a batch of logs to mongo."""
        collection = self.get_log_collection()
        await self.batch_write(operations, collection)

    def log_dump(self, log: dict[str, Any]) -> InsertOne[dict[str, Any]]:
        """Create a write operation for a given log."""
        return InsertOne(log)

    async def get_job_doc(self, job_id: str) -> JobDocs | None:
        """Retrieve a job from the database."""
        status_collection, docs_collection = self.get_job_collection()

        job = await docs_collection.find_one(
            {"job_id": job_id},
        )
        if job is None:
            return
        touch_time = datetime.now()
        await status_collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"touched": touch_time}},
        )
        await docs_collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"touched": touch_time}},
        )

        del job["_id"]
        return JobDocs(**job)

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
            query["extra.job_id"] = {"$regex": job_id}

        logs = self.get_log_collection()
        cursor = logs.find(query).sort("time", 1)
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


class MongoQueue(BatchedAction):
    """An asynchronous queue for sending items to MongoDB."""

    # Essentially should flush every interval
    batch_size: int = 1000
    flush_time: float = 0

    client: ClassVar[MongoClient] = MongoClient()

    async def job_state(self, batch: list[QueryState | ResponseState]) -> None:
        """Send a batch of job states to MongoDB."""
        if len(batch) == 0:
            return
        status_ops, doc_ops = map(
            list, zip(*(self.client.job_state(state) for state in batch), strict=True)
        )
        await self.client.batch_job_state(status_ops, doc_ops)

    async def log_dump(self, batch: list[dict[str, Any]]) -> None:
        """Send a batch of logs to MongoDB."""
        await self.client.batch_log_dump([self.client.log_dump(log) for log in batch])
