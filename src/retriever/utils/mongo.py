from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, ClassVar, Literal, NotRequired, TypedDict, cast, override

from bson import ObjectId
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from loguru import logger as log
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.operations import InsertOne, UpdateOne
from pymongo.server_api import ServerApi

from retriever.config.general import CONFIG
from retriever.types.general import LogLevel
from retriever.utils.general import BatchedAction, Singleton
from retriever.utils.job_status import NON_TERMINAL, TERMINAL_FAILURE, TERMINAL_SUCCESS

CODEC_OPTIONS = DEFAULT_CODEC_OPTIONS.with_options(tz_aware=True)

_PERCENTILE_MIN_MAJOR = 7
"""Mongo major version where `$percentile` became available."""

_LOG_LEVEL_THRESHOLD: dict[str, int] = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
"""Numeric thresholds for loguru levels — used to build `{level.no: {$gte}}` clauses."""


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
    worker_pid: int | None
    worker_started_at: datetime | None


class ResponseState(TypedDict):
    """Required info about a response for the job state."""

    job_id: str
    response: bytes
    knodes: int
    kedges: int
    aux_graphs: int
    results: int
    status: str


class JobDoc(TypedDict):
    """Job documents and some lifetime metadata."""

    job_id: str
    doc: NotRequired[bytes]

    # Lifetime tracking
    created: NotRequired[datetime]
    completed: NotRequired[datetime]
    touched: NotRequired[datetime]

    # Set by the orphan sweep — tells `get_job_response` that `doc`, if
    # present, still holds the original query bytes rather than a real
    # response (the ResponseState write never landed).
    abandoned: NotRequired[bool]


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

    # Worker identity (absent on pre-feature documents)
    worker_pid: NotRequired[int | None]
    worker_started_at: NotRequired[datetime | None]

    # Set by the orphan sweep — distinguishes a Failed-from-orphan job
    # from a Failed-from-lookup one.
    abandoned: NotRequired[bool]


class IntRange(TypedDict, total=False):
    """Inclusive integer bounds. Omit a side for an open-ended range."""

    min: int
    max: int


class TimeRange(TypedDict, total=False):
    """Inclusive datetime bounds. Omit a side for an open-ended range."""

    after: datetime
    before: datetime


class JobIdentityFilter(TypedDict, total=False):
    """Filter on identity / classification fields set at enqueue.

    `job_id` is matched as a regex (mirroring `get_logs`). `status` and
    `submitter` accept either a single value or a list (matched as any-of).
    `data_tiers` matches jobs whose tier list intersects the given tiers.
    """

    job_id: str
    status: str | list[str]
    submitter: str | list[str]
    data_tiers: list[Literal[0, 1, 2]]
    is_async: bool


class JobTimeFilter(TypedDict, total=False):
    """Filter on lifetime timestamps. Each range is independent and ANDed."""

    created: TimeRange
    completed: TimeRange


class QueryGeometryFilter(TypedDict, total=False):
    """Filter on query-graph shape (set at enqueue)."""

    qnodes: IntRange
    qedges: IntRange
    qpaths: IntRange


class ResponseGeometryFilter(TypedDict, total=False):
    """Filter on response-graph shape (set on completion).

    Filtering on any of these implicitly restricts results to completed jobs.
    """

    knodes: IntRange
    kedges: IntRange
    aux_graphs: IntRange
    results: IntRange


class JobSortSpec(TypedDict):
    """How to order job-status results.

    `duration` is a computed field — `completed - created` in seconds —
    and forces the underlying query through an aggregation pipeline.
    All other fields are stored on the document directly.
    """

    field: Literal[
        "created",
        "completed",
        "duration",
        "results",
        "submitter",
        "status",
    ]
    direction: NotRequired[Literal["asc", "desc"]]


class JobStatusPage(TypedDict):
    """A page of job-status results plus an optional continuation cursor.

    `next_cursor` is None when the page is the last one. Otherwise it's an
    opaque token to pass back as the `cursor` arg to fetch the next page.
    Callers must use the same filter and sort args across pages.
    """

    items: list[JobStatus]
    next_cursor: str | None


class JobGroupCount(TypedDict):
    """A single (group-key, count) row from a group-by aggregation."""

    key: Any
    count: int


class JobTimeBucket(TypedDict):
    """A single (bucket-start, count) row from a time-bucketed aggregation."""

    bucket_start: datetime
    count: int


class DurationStats(TypedDict):
    """Aggregate runtime stats (completed - created) over a set of jobs.

    Percentile fields are `NotRequired` so they can be omitted on Mongo
    versions older than 7.0 (which lack `$percentile`). All-zero values
    are returned when the sample is empty.
    """

    sample_size: int
    min_seconds: float
    max_seconds: float
    avg_seconds: float
    p50_seconds: NotRequired[float]
    p95_seconds: NotRequired[float]
    p99_seconds: NotRequired[float]


class SubmitterStats(TypedDict):
    """Per-submitter activity row produced by `submitter_stats`."""

    submitter: str
    count: int
    completed: int
    failed: int
    success_rate: float | None
    p95_duration_seconds: float | None
    last_seen: datetime


class TierDurationStats(TypedDict):
    """Per-tier duration stats; all `None` when no completions in window."""

    avg_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None


class TierStats(TypedDict):
    """Per-tier activity row produced by `tier_stats`."""

    tier: int
    active: int
    completed: int
    failed: int
    durations: TierDurationStats
    last_activity: datetime | None


class SubmitterTierStats(TypedDict):
    """One (submitter, tier) cell produced by `submitter_tier_stats`."""

    submitter: str
    tier: int
    count: int
    completed: int
    failed: int
    avg_seconds: float | None
    p95_seconds: float | None
    last_seen: datetime | None


class FailureBreakdownRow(TypedDict):
    """One {key, status} cell produced by `failure_breakdown_by_*`.

    `key` is the submitter name or the tier index stringified, depending
    on which breakdown was requested. Counts are over jobs whose status
    is in `TERMINAL_FAILURE`.
    """

    key: str
    status: str
    count: int


_DURATION_EXPR: dict[str, Any] = {
    "$divide": [{"$subtract": ["$completed", "$created"]}, 1000]
}
"""Shared aggregation fragment: (completed - created) in seconds."""


def _count_if_status_in(status_list: list[str]) -> dict[str, Any]:
    """Aggregation `$sum` fragment: count of docs whose status matches."""
    return {"$sum": {"$cond": [{"$in": ["$status", status_list]}, 1, 0]}}


def _require_nonnull(query: dict[str, Any], field: str) -> None:
    """Merge `$ne: None` into `query[field]`, preserving any existing clauses."""
    query[field] = {**query.get(field, {}), "$ne": None}


def _push_success_durations(success_list: list[str]) -> dict[str, Any]:
    """`$push` fragment: durations only for success-status docs with both timestamps."""
    return {
        "$push": {
            "$cond": [
                {
                    "$and": [
                        {"$in": ["$status", success_list]},
                        {"$ne": ["$completed", None]},
                        {"$ne": ["$created", None]},
                    ]
                },
                _DURATION_EXPR,
                None,
            ]
        }
    }


def _percentile_first_of(input_field: str, p: float) -> dict[str, Any]:
    """Single approximate percentile over a pushed-duration array, nulls filtered out.

    Used in `$project` stages whose group stage pushed durations via
    `_push_success_durations` (where non-matching docs landed as `None`).
    """
    return {
        "$let": {
            "vars": {
                "filtered": {
                    "$filter": {
                        "input": f"${input_field}",
                        "as": "d",
                        "cond": {"$ne": ["$$d", None]},
                    }
                }
            },
            "in": {
                "$cond": [
                    {"$gt": [{"$size": "$$filtered"}, 0]},
                    {
                        "$first": {
                            "$percentile": {
                                "input": "$$filtered",
                                "p": [p],
                                "method": "approximate",
                            }
                        }
                    },
                    None,
                ]
            },
        }
    }


SortValue = float | int | datetime | str | None
"""All sort dimensions exposed via JobSortSpec resolve to one of these."""


def _encode_cursor(sort_value: SortValue, object_id: ObjectId) -> str:
    """Encode the trailing sort value + _id of a page as an opaque cursor.

    The sort value carries a `t` tag (`null`/`datetime`/`int`/`float`/`bool`/
    `str`) so the decode side can reconstitute the correct Python type
    regardless of which underlying field drove the sort.
    """
    v_part: dict[str, Any]
    if sort_value is None:
        v_part = {"t": "null"}
    elif isinstance(sort_value, datetime):
        v_part = {"t": "datetime", "v": sort_value.isoformat()}
    elif isinstance(sort_value, bool):
        # bool is a subclass of int — check it first.
        v_part = {"t": "bool", "v": sort_value}
    elif isinstance(sort_value, int):
        v_part = {"t": "int", "v": int(sort_value)}
    elif isinstance(sort_value, float):
        v_part = {"t": "float", "v": float(sort_value)}
    else:
        v_part = {"t": "str", "v": str(sort_value)}
    payload: dict[str, Any] = {**v_part, "id": str(object_id)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


class CursorDecodeError(ValueError):
    """Raised when an opaque pagination cursor can't be parsed.

    Callers (FastAPI route handlers) should translate this to HTTP 400
    rather than letting it propagate as 500 — bad cursors are a client
    fault, not a server fault.
    """


def _decode_cursor(cursor: str) -> tuple[SortValue, ObjectId]:
    """Decode a cursor into the trailing sort value + _id it represents.

    Raises:
        CursorDecodeError: if the cursor is malformed (bad base64, bad
            JSON, missing keys, or an ObjectId that won't parse).
    """
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        kind = payload.get("t")
        sort_value: SortValue
        if kind == "null":
            sort_value = None
        elif kind == "datetime":
            sort_value = datetime.fromisoformat(payload["v"])
        elif kind == "int":
            sort_value = int(payload["v"])
        elif kind == "float":
            sort_value = float(payload["v"])
        elif kind == "bool":
            sort_value = bool(payload["v"])
        elif kind == "str":
            sort_value = str(payload["v"])
        else:
            # Backward-compat with cursors emitted before the typed payload —
            # those used `{"v": <iso>|null, "id": ...}` with datetime
            # semantics.
            raw = payload.get("v")
            sort_value = datetime.fromisoformat(raw) if raw is not None else None
        return sort_value, ObjectId(payload["id"])
    except Exception as e:
        raise CursorDecodeError(f"invalid cursor: {e}") from e


def _apply_time_ranges(query: dict[str, Any], spec: JobTimeFilter | None) -> None:
    """Apply TimeRange-typed fields from a filter spec onto a Mongo query dict."""
    if spec is None:
        return
    for field, range_ in spec.items():
        range_ = cast(TimeRange, range_)
        clauses: dict[str, datetime] = {}
        if "after" in range_:
            clauses["$gte"] = range_["after"]
        if "before" in range_:
            clauses["$lte"] = range_["before"]
        if clauses:
            query[field] = clauses


def _apply_int_ranges(
    query: dict[str, Any],
    spec: QueryGeometryFilter | ResponseGeometryFilter | None,
) -> None:
    """Apply IntRange-typed fields from a filter spec onto a Mongo query dict."""
    if spec is None:
        return
    for field, range_ in spec.items():
        range_ = cast(IntRange, range_)
        clauses: dict[str, int] = {}
        if "min" in range_:
            clauses["$gte"] = range_["min"]
        if "max" in range_:
            clauses["$lte"] = range_["max"]
        if clauses:
            query[field] = clauses


async def _format_flat(
    log_stream: AsyncGenerator[dict[str, Any]],
) -> AsyncGenerator[str]:
    """Render a stream of log docs as stdout-style flat lines."""
    first = True
    async for log_doc in log_stream:
        message = log_doc["message"]
        if log_doc.get("exception"):
            exception = log_doc["exception"]
            message = (
                f"{message}\n{exception.get('traceback')}\n"
                f"{exception.get('type')}: {exception.get('value')}"
            )
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


def _build_log_query(
    start: datetime | None,
    end: datetime | None,
    level: LogLevel,
) -> dict[str, Any]:
    """Build the common time/level clause for log queries."""
    query: dict[str, Any] = {"level.no": {"$gte": _LOG_LEVEL_THRESHOLD[level.upper()]}}
    if start or end:
        time_clause: dict[str, datetime] = {}
        if start:
            time_clause["$gte"] = start
        if end:
            time_clause["$lte"] = end
        query["time"] = time_clause
    return query


def _build_filter_query(
    identity: JobIdentityFilter | None = None,
    times: JobTimeFilter | None = None,
    query_geometry: QueryGeometryFilter | None = None,
    response_geometry: ResponseGeometryFilter | None = None,
) -> dict[str, Any]:
    """Translate job-status filter TypedDicts into a Mongo query dict.

    Each group is optional and ANDed into the resulting query. Used by both
    paged retrieval and the count/aggregation methods so the filter
    vocabulary stays in sync across call sites.
    """
    query: dict[str, Any] = {}

    if identity is not None:
        if "job_id" in identity:
            query["job_id"] = {"$regex": identity["job_id"]}
        if "status" in identity:
            status = identity["status"]
            query["status"] = {"$in": status} if isinstance(status, list) else status
        if "submitter" in identity:
            submitter = identity["submitter"]
            query["submitter"] = (
                {"$in": submitter} if isinstance(submitter, list) else submitter
            )
        if "data_tiers" in identity:
            query["data_tiers"] = {"$in": identity["data_tiers"]}
        if "is_async" in identity:
            query["is_async"] = identity["is_async"]

    _apply_time_ranges(query, times)
    _apply_int_ranges(query, query_geometry)
    _apply_int_ranges(query, response_geometry)

    return query


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
        self._supports_percentile: bool = False

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

        # Detect $percentile support once. Branch in aggregation methods on
        # this flag rather than catching MongoCommandError mid-pipeline.
        # buildInfo can be denied on locked-down clusters or return a
        # version string we can't parse; in either case fall back to the
        # no-percentile branch rather than blocking startup.
        try:
            build_info = await self.client.admin.command("buildInfo")
            major = int(str(build_info["version"]).split(".", 1)[0])
            self._supports_percentile = major >= _PERCENTILE_MIN_MAJOR
        except Exception:
            log.warning(
                "Could not read MongoDB buildInfo; assuming no $percentile support."
            )
            self._supports_percentile = False

        job_collections = self.get_job_collection()
        for collection in job_collections:
            await collection.create_index("job_id", unique=True, background=True)
            await collection.create_index(
                "touched", background=True, expireAfterSeconds=CONFIG.job.ttl
            )
            await collection.create_index(
                "completed", background=True, expireAfterSeconds=CONFIG.job.ttl_max
            )
            await collection.create_index("created", background=True)

        log_collection = self.get_log_collection()
        await log_collection.create_index(
            {"time": 1}, background=True, expireAfterSeconds=CONFIG.log.mongo_ttl
        )

    async def ping(self) -> bool:
        """Post-init liveness check. Returns True iff the server responds."""
        try:
            _ = await self.client.admin.command("ping")
        except Exception:
            return False
        return True

    async def db_storage_bytes(self) -> int:
        """Return the on-disk storage size of the retriever_persist database."""
        stats = await self.client.retriever_persist.command("dbStats")
        return int(stats["storageSize"])

    async def count_in_flight(self) -> int:
        """Count jobs whose status is non-terminal (Running / Accepted)."""
        status_collection, _ = self.get_job_collection()
        return await status_collection.count_documents(
            {"status": {"$in": sorted(NON_TERMINAL)}}
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
        # Tz-aware. Naive datetime.now() would be treated as already-UTC by
        # pymongo and shift readers by the local offset.
        update_time = datetime.now().astimezone()
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
        doc_inner: JobDoc = {
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

    async def get_job_doc(self, job_id: str) -> JobDoc | None:
        """Retrieve a job from the database."""
        status_collection, docs_collection = self.get_job_collection()

        job = await docs_collection.find_one(
            {"job_id": job_id},
        )
        if job is None:
            return
        touch_time = datetime.now().astimezone()
        await status_collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"touched": touch_time}},
        )
        await docs_collection.update_one(
            {"_id": job["_id"]},
            {"$set": {"touched": touch_time}},
        )

        del job["_id"]
        return JobDoc(**job)

    async def get_job_status(
        self, job_id: str, *, touch: bool = False
    ) -> JobStatus | None:
        """Return a single job_status doc by job_id, or None if not found.

        Reads only the status collection (no doc bytes). Pass `touch=True`
        to bump the `touched` timestamp on both collections, extending
        TTL — required for TRAPI interactions. Leave it `False` for
        read-only dashboard queries.
        """
        status_collection, docs_collection = self.get_job_collection()
        doc = await status_collection.find_one({"job_id": job_id})
        if doc is None:
            return None
        if touch:
            touch_time = datetime.now().astimezone()
            await status_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"touched": touch_time}},
            )
            await docs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"touched": touch_time}},
            )
        del doc["_id"]
        return JobStatus(**doc)

    async def get_job_statuses(  # noqa: PLR0913 Each arg is a deliberate filter group
        self,
        *,
        identity: JobIdentityFilter | None = None,
        times: JobTimeFilter | None = None,
        query_geometry: QueryGeometryFilter | None = None,
        response_geometry: ResponseGeometryFilter | None = None,
        sort: JobSortSpec | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> JobStatusPage:
        """Return a filtered page of job status docs from the job_status collection.

        Each parameter group is optional and ANDed together. Filtering on
        `response_geometry` implicitly restricts results to completed jobs,
        since those fields are only populated on response upsert.

        Sort defaults to `created` descending. `_id` is always used as a
        tie-breaker so paging is stable across duplicate sort values.
        `sort.field = "duration"` is the one computed dimension and runs
        the query through an aggregation pipeline; every other field is a
        plain `find().sort()`.
        """
        query = _build_filter_query(identity, times, query_geometry, response_geometry)

        sort_field: str = "created"
        sort_direction = -1
        if sort is not None:
            sort_field = sort["field"]
            if sort.get("direction") == "asc":
                sort_direction = 1

        if sort_field == "duration":
            return await self._page_by_duration(query, sort_direction, limit, cursor)

        if cursor is not None:
            cursor_value, cursor_id = _decode_cursor(cursor)
            op = "$gt" if sort_direction == 1 else "$lt"
            cursor_clause: dict[str, Any] = {
                "$or": [
                    {sort_field: {op: cursor_value}},
                    {sort_field: cursor_value, "_id": {op: cursor_id}},
                ]
            }
            query = {"$and": [query, cursor_clause]} if query else cursor_clause

        status_collection, _ = self.get_job_collection()
        found = (
            status_collection.find(query)
            .sort([(sort_field, sort_direction), ("_id", sort_direction)])
            .limit(limit + 1)
        )
        docs = [doc async for doc in found]

        next_cursor: str | None = None
        if len(docs) > limit:
            last = docs[limit - 1]
            next_cursor = _encode_cursor(last.get(sort_field), last["_id"])
            docs = docs[:limit]

        items: list[JobStatus] = [JobStatus(**doc) for doc in docs]

        return JobStatusPage(items=items, next_cursor=next_cursor)

    async def _page_by_duration(
        self,
        match_query: dict[str, Any],
        sort_direction: int,
        limit: int,
        cursor: str | None,
    ) -> JobStatusPage:
        """Aggregation-pipeline branch for sort_field='duration'.

        Computes `duration_seconds = (completed - created) / 1000` via
        `$addFields`, then sorts + paginates on it. Implicitly restricts
        to terminal jobs (both timestamps non-null).
        """
        match_query = dict(match_query)
        _require_nonnull(match_query, "completed")
        _require_nonnull(match_query, "created")

        post_match: dict[str, Any] = {}
        if cursor is not None:
            cursor_value, cursor_id = _decode_cursor(cursor)
            op = "$gt" if sort_direction == 1 else "$lt"
            post_match["$or"] = [
                {"duration_seconds": {op: cursor_value}},
                {"duration_seconds": cursor_value, "_id": {op: cursor_id}},
            ]

        pipeline: list[dict[str, Any]] = [
            {"$match": match_query},
            {"$addFields": {"duration_seconds": _DURATION_EXPR}},
        ]
        if post_match:
            pipeline.append({"$match": post_match})
        pipeline.extend(
            [
                {"$sort": {"duration_seconds": sort_direction, "_id": sort_direction}},
                {"$limit": limit + 1},
            ]
        )

        status_collection, _ = self.get_job_collection()
        docs = [doc async for doc in status_collection.aggregate(pipeline)]

        next_cursor: str | None = None
        if len(docs) > limit:
            last = docs[limit - 1]
            next_cursor = _encode_cursor(last.get("duration_seconds"), last["_id"])
            docs = docs[:limit]

        items: list[JobStatus] = [JobStatus(**doc) for doc in docs]
        return JobStatusPage(items=items, next_cursor=next_cursor)

    async def count_jobs(
        self,
        *,
        identity: JobIdentityFilter | None = None,
        times: JobTimeFilter | None = None,
        query_geometry: QueryGeometryFilter | None = None,
        response_geometry: ResponseGeometryFilter | None = None,
    ) -> int:
        """Return the count of job_status docs matching the given filters."""
        query = _build_filter_query(identity, times, query_geometry, response_geometry)
        status_collection, _ = self.get_job_collection()
        return await status_collection.count_documents(query)

    async def group_jobs(  # noqa: PLR0913 Each arg is a deliberate filter group
        self,
        *,
        group_by: Literal["status", "submitter", "is_async", "data_tiers"],
        identity: JobIdentityFilter | None = None,
        times: JobTimeFilter | None = None,
        query_geometry: QueryGeometryFilter | None = None,
        response_geometry: ResponseGeometryFilter | None = None,
        sort_by: Literal["count", "key"] = "count",
        top: int | None = None,
    ) -> list[JobGroupCount]:
        """Return per-value counts of job_status docs grouped by `group_by`.

        For list-valued fields (currently `data_tiers`), the field is
        `$unwind`ed before grouping — a job that spans multiple tiers
        contributes one to each tier's count.
        """
        match_query = _build_filter_query(
            identity, times, query_geometry, response_geometry
        )
        pipeline: list[dict[str, Any]] = []
        if match_query:
            pipeline.append({"$match": match_query})
        if group_by == "data_tiers":
            pipeline.append({"$unwind": "$data_tiers"})
        pipeline.append({"$group": {"_id": f"${group_by}", "count": {"$sum": 1}}})
        sort_key = "count" if sort_by == "count" else "_id"
        sort_dir = -1 if sort_by == "count" else 1
        pipeline.append({"$sort": {sort_key: sort_dir}})
        if top is not None:
            pipeline.append({"$limit": top})

        status_collection, _ = self.get_job_collection()
        return [
            JobGroupCount(key=doc["_id"], count=doc["count"])
            async for doc in status_collection.aggregate(pipeline)
        ]

    async def bucket_jobs_over_time(  # noqa: PLR0913 Each arg is a deliberate filter group
        self,
        *,
        field: Literal["created", "completed"],
        granularity: Literal["hour", "day", "week", "month"],
        identity: JobIdentityFilter | None = None,
        times: JobTimeFilter | None = None,
        query_geometry: QueryGeometryFilter | None = None,
        response_geometry: ResponseGeometryFilter | None = None,
    ) -> list[JobTimeBucket]:
        """Return counts of job_status docs bucketed by a timestamp field.

        Docs missing the bucketed `field` (e.g. running jobs when bucketing on
        `completed`) are excluded. Empty buckets between populated ones are
        not back-filled — callers that want a continuous series can pad.
        """
        match_query = _build_filter_query(
            identity, times, query_geometry, response_geometry
        )
        _require_nonnull(match_query, field)
        pipeline: list[dict[str, Any]] = [
            {"$match": match_query},
            {
                "$group": {
                    "_id": {"$dateTrunc": {"date": f"${field}", "unit": granularity}},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        status_collection, _ = self.get_job_collection()
        return [
            JobTimeBucket(bucket_start=doc["_id"], count=doc["count"])
            async for doc in status_collection.aggregate(pipeline)
        ]

    async def compute_durations(
        self,
        *,
        identity: JobIdentityFilter | None = None,
        times: JobTimeFilter | None = None,
    ) -> DurationStats:
        """Aggregate runtime stats over terminal jobs in the filter window.

        Returns min/max/avg seconds; on Mongo 7+ also returns p50/p95/p99
        (approximate). When no docs match, all numeric fields are 0 and
        percentile fields are omitted regardless of Mongo version.

        `identity` accepts a `resolve_status_filter` output so callers can
        pass `?status=failed` straight through.
        """
        match_query = _build_filter_query(identity, times)
        _require_nonnull(match_query, "completed")
        _require_nonnull(match_query, "created")

        group_stage: dict[str, Any] = {
            "_id": None,
            "sample_size": {"$sum": 1},
            "min_seconds": {"$min": _DURATION_EXPR},
            "max_seconds": {"$max": _DURATION_EXPR},
            "avg_seconds": {"$avg": _DURATION_EXPR},
        }
        if self._supports_percentile:
            group_stage["percentiles"] = {
                "$percentile": {
                    "input": _DURATION_EXPR,
                    "p": [0.5, 0.95, 0.99],
                    "method": "approximate",
                }
            }

        status_collection, _ = self.get_job_collection()
        cursor = status_collection.aggregate(
            [{"$match": match_query}, {"$group": group_stage}]
        )
        doc = await cursor.to_list(length=1)
        if not doc:
            return DurationStats(
                sample_size=0, min_seconds=0.0, max_seconds=0.0, avg_seconds=0.0
            )

        row = doc[0]
        stats: DurationStats = {
            "sample_size": int(row["sample_size"]),
            "min_seconds": float(row["min_seconds"] or 0.0),
            "max_seconds": float(row["max_seconds"] or 0.0),
            "avg_seconds": float(row["avg_seconds"] or 0.0),
        }
        percentiles = row.get("percentiles")
        if percentiles:
            p50, p95, p99 = percentiles
            stats["p50_seconds"] = float(p50)
            stats["p95_seconds"] = float(p95)
            stats["p99_seconds"] = float(p99)
        return stats

    async def submitter_stats(
        self,
        *,
        times: JobTimeFilter | None = None,
        top: int = 50,
    ) -> list[SubmitterStats]:
        """Per-submitter activity rows for the leaderboard view.

        `times` filters by `created` if supplied. Each row includes raw
        count + completed/failed breakdown, derived `success_rate`, p95
        completion-duration (Mongo 7+ only; `None` otherwise), and the
        most-recent `created` timestamp as `last_seen`.
        """
        match_query = _build_filter_query(times=times)
        match_query["submitter"] = {"$exists": True}

        failure_list = sorted(TERMINAL_FAILURE)
        success_list = sorted(TERMINAL_SUCCESS)

        group_stage: dict[str, Any] = {
            "_id": "$submitter",
            "count": {"$sum": 1},
            "completed_count": _count_if_status_in(success_list),
            "failed_count": _count_if_status_in(failure_list),
            "last_seen": {"$max": "$created"},
        }
        if self._supports_percentile:
            group_stage["completed_durations"] = _push_success_durations(success_list)

        terminal_count_expr = {"$add": ["$completed_count", "$failed_count"]}
        project_stage: dict[str, Any] = {
            "_id": 0,
            "submitter": "$_id",
            "count": 1,
            "completed": "$completed_count",
            "failed": "$failed_count",
            "last_seen": 1,
            "success_rate": {
                "$cond": [
                    {"$gt": [terminal_count_expr, 0]},
                    {"$divide": ["$completed_count", terminal_count_expr]},
                    None,
                ]
            },
        }
        if self._supports_percentile:
            project_stage["p95_duration_seconds"] = _percentile_first_of(
                "completed_durations", 0.95
            )
        else:
            project_stage["p95_duration_seconds"] = {"$literal": None}

        pipeline: list[dict[str, Any]] = [
            {"$match": match_query},
            {"$group": group_stage},
            {"$project": project_stage},
            {"$sort": {"count": -1}},
            {"$limit": top},
        ]

        status_collection, _ = self.get_job_collection()
        return [
            SubmitterStats(
                submitter=doc["submitter"],
                count=int(doc["count"]),
                completed=int(doc["completed"]),
                failed=int(doc["failed"]),
                success_rate=(
                    float(doc["success_rate"])
                    if doc["success_rate"] is not None
                    else None
                ),
                p95_duration_seconds=(
                    float(doc["p95_duration_seconds"])
                    if doc["p95_duration_seconds"] is not None
                    else None
                ),
                last_seen=doc["last_seen"],
            )
            async for doc in status_collection.aggregate(pipeline)
        ]

    async def tier_stats(
        self,
        *,
        times: JobTimeFilter | None = None,
    ) -> list[TierStats]:
        """Per-tier activity rows.

        `active` counts in-flight jobs on a tier (no time filter applies).
        `completed` / `failed` / `durations` / `last_activity` reflect
        terminal jobs within the supplied `times` window (filters on
        `completed` if present; otherwise no time bound).

        Duration percentiles are populated only on Mongo 7+; otherwise
        their fields are `None`. Returned tiers are those that appear in
        the data; callers that want a fixed set of tier rows can pad with
        zeros.
        """
        window_match = _build_filter_query(times=times)
        # Window-pipeline only sees terminal jobs with both timestamps so
        # duration math is meaningful.
        _require_nonnull(window_match, "completed")
        _require_nonnull(window_match, "created")

        failure_list = sorted(TERMINAL_FAILURE)
        success_list = sorted(TERMINAL_SUCCESS)
        non_terminal_list = sorted(NON_TERMINAL)

        window_group: dict[str, Any] = {
            "_id": "$data_tiers",
            "completed": _count_if_status_in(success_list),
            "failed": _count_if_status_in(failure_list),
            "last_activity": {"$max": "$completed"},
            "avg_seconds": {"$avg": _DURATION_EXPR},
            "min_seconds": {"$min": _DURATION_EXPR},
            "max_seconds": {"$max": _DURATION_EXPR},
        }
        if self._supports_percentile:
            # Percentile input matches avg/min/max — all terminal jobs in
            # window, success and failure alike. window_match has already
            # filtered to docs with non-null timestamps.
            window_group["percentiles"] = {
                "$percentile": {
                    "input": _DURATION_EXPR,
                    "p": [0.5, 0.95, 0.99],
                    "method": "approximate",
                }
            }

        pipeline: list[dict[str, Any]] = [
            {"$match": {"data_tiers": {"$exists": True, "$ne": []}}},
            {"$unwind": "$data_tiers"},
            {
                "$facet": {
                    "active": [
                        {"$match": {"status": {"$in": non_terminal_list}}},
                        {"$group": {"_id": "$data_tiers", "count": {"$sum": 1}}},
                    ],
                    "in_window": [
                        {"$match": window_match},
                        {"$group": window_group},
                    ],
                }
            },
        ]

        status_collection, _ = self.get_job_collection()
        facet_docs = await status_collection.aggregate(pipeline).to_list(length=1)
        if not facet_docs:
            return []

        facet = facet_docs[0]
        active_by_tier: dict[int, int] = {
            int(row["_id"]): int(row["count"]) for row in facet.get("active", [])
        }
        window_by_tier: dict[int, dict[str, Any]] = {
            int(row["_id"]): row for row in facet.get("in_window", [])
        }

        tiers = sorted(set(active_by_tier) | set(window_by_tier))
        return [
            self._tier_stats_row(tier, active_by_tier, window_by_tier) for tier in tiers
        ]

    @staticmethod
    def _tier_stats_row(
        tier: int,
        active_by_tier: dict[int, int],
        window_by_tier: dict[int, dict[str, Any]],
    ) -> TierStats:
        """Merge a single tier's active + window rows into a TierStats."""
        window = window_by_tier.get(tier, {})
        percentiles = window.get("percentiles")
        durations: TierDurationStats = {
            "avg_seconds": (
                float(window["avg_seconds"])
                if window.get("avg_seconds") is not None
                else None
            ),
            "min_seconds": (
                float(window["min_seconds"])
                if window.get("min_seconds") is not None
                else None
            ),
            "max_seconds": (
                float(window["max_seconds"])
                if window.get("max_seconds") is not None
                else None
            ),
            "p50_seconds": float(percentiles[0]) if percentiles else None,
            "p95_seconds": float(percentiles[1]) if percentiles else None,
            "p99_seconds": float(percentiles[2]) if percentiles else None,
        }
        return TierStats(
            tier=tier,
            active=active_by_tier.get(tier, 0),
            completed=int(window.get("completed", 0)),
            failed=int(window.get("failed", 0)),
            durations=durations,
            last_activity=window.get("last_activity"),
        )

    async def submitter_tier_stats(
        self,
        *,
        times: JobTimeFilter | None = None,
    ) -> list[SubmitterTierStats]:
        """Per-(submitter, tier) cells for the heatmaps view.

        Returns one row per non-empty (submitter, tier) pair in the
        window. A job whose `data_tiers` list spans both tiers
        contributes to BOTH cells (mirrors `tier_stats` semantics).
        Duration percentiles populate on Mongo 7+; otherwise `None`.
        """
        match_query = _build_filter_query(times=times)
        match_query["submitter"] = {"$exists": True}
        match_query["data_tiers"] = {"$exists": True, "$ne": []}

        failure_list = sorted(TERMINAL_FAILURE)
        success_list = sorted(TERMINAL_SUCCESS)

        group_stage: dict[str, Any] = {
            "_id": {"submitter": "$submitter", "tier": "$data_tiers"},
            "count": {"$sum": 1},
            "completed_count": _count_if_status_in(success_list),
            "failed_count": _count_if_status_in(failure_list),
            # `last_seen` = latest submission (created), matching submitter_stats
            # semantics. Using `completed` would yield None for cells whose
            # most recent job is still running.
            "last_seen": {"$max": "$created"},
            "avg_seconds": {"$avg": _DURATION_EXPR},
        }
        if self._supports_percentile:
            group_stage["completed_durations"] = _push_success_durations(success_list)

        project_stage: dict[str, Any] = {
            "_id": 0,
            "submitter": "$_id.submitter",
            "tier": "$_id.tier",
            "count": 1,
            "completed": "$completed_count",
            "failed": "$failed_count",
            "last_seen": 1,
            "avg_seconds": 1,
        }
        if self._supports_percentile:
            project_stage["p95_seconds"] = _percentile_first_of(
                "completed_durations", 0.95
            )
        else:
            project_stage["p95_seconds"] = {"$literal": None}

        pipeline: list[dict[str, Any]] = [
            {"$match": match_query},
            {"$unwind": "$data_tiers"},
            {"$group": group_stage},
            {"$project": project_stage},
            {"$sort": {"submitter": 1, "tier": 1}},
        ]

        status_collection, _ = self.get_job_collection()
        return [
            SubmitterTierStats(
                submitter=doc["submitter"],
                tier=int(doc["tier"]),
                count=int(doc["count"]),
                completed=int(doc["completed"]),
                failed=int(doc["failed"]),
                avg_seconds=(
                    float(doc["avg_seconds"])
                    if doc.get("avg_seconds") is not None
                    else None
                ),
                p95_seconds=(
                    float(doc["p95_seconds"])
                    if doc.get("p95_seconds") is not None
                    else None
                ),
                last_seen=doc.get("last_seen"),
            )
            async for doc in status_collection.aggregate(pipeline)
        ]

    async def failure_breakdown_by_submitter(
        self,
        *,
        times: JobTimeFilter | None = None,
    ) -> list[FailureBreakdownRow]:
        """One row per (submitter, failure-status) pair with counts."""
        match_query = _build_filter_query(times=times)
        match_query["submitter"] = {"$exists": True}
        match_query["status"] = {"$in": sorted(TERMINAL_FAILURE)}

        pipeline: list[dict[str, Any]] = [
            {"$match": match_query},
            {
                "$group": {
                    "_id": {"submitter": "$submitter", "status": "$status"},
                    "count": {"$sum": 1},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "key": "$_id.submitter",
                    "status": "$_id.status",
                    "count": 1,
                }
            },
            {"$sort": {"key": 1, "status": 1}},
        ]
        status_collection, _ = self.get_job_collection()
        return [
            FailureBreakdownRow(
                key=doc["key"], status=doc["status"], count=int(doc["count"])
            )
            async for doc in status_collection.aggregate(pipeline)
        ]

    async def failure_breakdown_by_tier(
        self,
        *,
        times: JobTimeFilter | None = None,
    ) -> list[FailureBreakdownRow]:
        """One row per (tier, failure-status) pair with counts.

        `data_tiers` is `$unwind`ed so a multi-tier job contributes to
        each of its tiers (same semantics as `tier_stats`).
        """
        match_query = _build_filter_query(times=times)
        match_query["data_tiers"] = {"$exists": True, "$ne": []}
        match_query["status"] = {"$in": sorted(TERMINAL_FAILURE)}

        pipeline: list[dict[str, Any]] = [
            {"$match": match_query},
            {"$unwind": "$data_tiers"},
            {
                "$group": {
                    "_id": {"tier": "$data_tiers", "status": "$status"},
                    "count": {"$sum": 1},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "key": {"$toString": "$_id.tier"},
                    "status": "$_id.status",
                    "count": 1,
                }
            },
            {"$sort": {"key": 1, "status": 1}},
        ]
        status_collection, _ = self.get_job_collection()
        return [
            FailureBreakdownRow(
                key=doc["key"], status=doc["status"], count=int(doc["count"])
            )
            async for doc in status_collection.aggregate(pipeline)
        ]

    async def get_logs(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        level: LogLevel = "DEBUG",
        job_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Return a generator of a filtered set of logs."""
        query = _build_log_query(start, end, level)
        if job_id is not None:
            query["extra.job_id"] = {"$regex": job_id}

        async for document in self._yield_logs(query):
            yield document

    async def get_server_logs(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        level: LogLevel = "DEBUG",
    ) -> AsyncGenerator[dict[str, Any]]:
        """Return a generator of logs not associated with any job.

        Matches docs where `extra.job_id` is absent — the logs Retriever
        itself emits (lifecycle, background refreshes, tier drivers,
        cache activity). Sibling to `get_logs`.
        """
        query = _build_log_query(start, end, level)
        query["extra.job_id"] = {"$exists": False}

        async for document in self._yield_logs(query):
            yield document

    async def _yield_logs(
        self, query: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Run the log query and yield decoded log documents (oldest first)."""
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
        async for line in _format_flat(self.get_logs(start, end, level, job_id)):
            yield line

    async def get_flat_server_logs(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        level: LogLevel = "DEBUG",
    ) -> AsyncGenerator[str]:
        """Return flat-formatted logs that aren't associated with a job."""
        async for line in _format_flat(self.get_server_logs(start, end, level)):
            yield line

    async def get_non_terminal_with_worker_info(
        self,
    ) -> list[tuple[str, int | None, datetime | None]]:
        """Return (job_id, worker_pid, worker_started_at) for all non-terminal jobs.

        Jobs created before worker tracking was added will have None for both worker fields.
        """
        status_collection, _ = self.get_job_collection()
        projection = {"_id": 0, "job_id": 1, "worker_pid": 1, "worker_started_at": 1}
        cursor = status_collection.find(
            {"status": {"$in": sorted(NON_TERMINAL)}}, projection
        )
        return [
            (doc["job_id"], doc.get("worker_pid"), doc.get("worker_started_at"))
            async for doc in cursor
        ]

    async def mark_jobs_abandoned(self, job_ids: list[str]) -> int:
        """Mark non-terminal jobs whose worker died as Failed/abandoned.

        Updates both collections so they stay in sync for TTL purposes,
        and sets an `abandoned` flag so `get_job_response` can tell that
        `docs.doc` (if any) still holds the original query bytes rather
        than a response.

        Each side has its own guard against the race where a job
        completed between the orphan-detection snapshot and this write:
        the status side checks status is still non-terminal; the docs
        side checks `completed` hasn't been set by a ResponseState
        write. Returns the count of status rows updated.
        """
        if not job_ids:
            return 0
        status_collection, docs_collection = self.get_job_collection()
        now = datetime.now().astimezone()
        fields = {"completed": now, "touched": now, "abandoned": True}
        result = await status_collection.update_many(
            {"job_id": {"$in": job_ids}, "status": {"$in": sorted(NON_TERMINAL)}},
            {"$set": {"status": "Failed", **fields}},
        )
        await docs_collection.update_many(
            {"job_id": {"$in": job_ids}, "completed": None},
            {"$set": fields},
        )
        return result.modified_count

    async def close(self) -> None:
        """Clean up and close MongoDB client threads/connections."""
        self.client.close()


class MongoQueue(BatchedAction):
    """An asynchronous queue for sending items to MongoDB."""

    # Essentially should flush every interval
    batch_size: int = 1000
    flush_time: float = 0

    client: ClassVar[MongoClient] = MongoClient()

    @override
    async def wrapup(self) -> None:
        """Flush pending writes during graceful shutdown.

        The polling loop is cancelled and awaited *before* the flush, so it
        can't race the flush by pulling items off the queue concurrently.
        """
        log.info(f"{type(self).__name__} wrapping up.")
        for task in self.tasks:
            _ = task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            async with asyncio.timeout(CONFIG.mongo.shutdown_timeout):
                await self.flush()
        except TimeoutError:
            log.warning(
                f"MongoQueue flush timed out after {CONFIG.mongo.shutdown_timeout}s; some queued writes may be lost."
            )

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

    def qsize(self) -> int:
        """Return the total number of pending items across all action queues."""
        return sum(q.qsize() for q in self.action_queues.values())
