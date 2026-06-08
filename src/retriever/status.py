"""FastAPI route handlers for the `/status/*` dashboard API.

A thin glue layer over MongoClient / RedisClient aggregations. Mounted on
the main app by `server.py`'s `app.include_router(router)`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Annotated, Literal, cast

import psutil
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from retriever.config.general import CONFIG
from retriever.config.openapi import OPENAPI_CONFIG
from retriever.data_tiers import tier_manager
from retriever.types.general import LogLevel
from retriever.types.status import (
    ActiveJobRow,
    ActivePage,
    CompletedJobRow,
    CompletedPage,
    CountsLinks,
    CountsSnapshot,
    CountsWindow,
    DurationsResponse,
    FailedJobRow,
    FailedPage,
    FailureBreakdownCell,
    FailureBreakdownResponse,
    JobDetail,
    Links,
    StatusMetaKG,
    StatusMongo,
    StatusProcess,
    StatusProcesses,
    StatusRedis,
    StatusServedBy,
    StatusSnapshot,
    StatusSubclassMap,
    StatusTier,
    StatusVersion,
    SubmitterRow,
    SubmitterTierRow,
    TierDurations,
    TierRow,
    TimelineBucket,
)
from retriever.utils import service_health, worker
from retriever.utils.job_status import (
    NON_TERMINAL,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
    resolve_status_filter,
)
from retriever.utils.logs import objs_to_json
from retriever.utils.mongo import (
    CursorDecodeError,
    JobIdentityFilter,
    JobSortSpec,
    JobStatus,
    JobStatusPage,
    JobTimeFilter,
    MongoClient,
    MongoQueue,
    TimeRange,
)
from retriever.utils.redis import FreshnessRecord, RedisClient
from retriever.utils.version import get_version

MAX_LIMIT = 500
"""Server-side cap on per-page row count."""

DEFAULT_LIMIT = 100

STUCK_DEFAULT_HOURS = 1.0
"""How long a job must have been running to count as 'stuck' by default."""

HOURS_PER_WEEK = 168.0

_BYTES_PER_MB = 1024 * 1024
"""Mebibyte (binary). What `ps`, `top`, `free`, and friends report."""

STATUS_DESCRIPTIONS = OPENAPI_CONFIG.response_descriptions.status
"""Per-endpoint response_description strings keyed by short name."""


def _now_aware() -> datetime:
    """Tz-aware current time."""
    return datetime.now().astimezone()


def _resolve_time_window(
    *,
    since: datetime | None,
    until: datetime | None,
    lookback: float | None,
    field: Literal["created", "completed"],
) -> JobTimeFilter | None:
    """Build a JobTimeFilter from `since`/`until`/`lookback` query params.

    `lookback` (hours back from now) overrides `since` when supplied.
    """
    effective_since = since
    if lookback is not None:
        effective_since = _now_aware() - timedelta(hours=lookback)

    range_: TimeRange = {}
    if effective_since is not None:
        range_["after"] = effective_since
    if until is not None:
        range_["before"] = until
    if not range_:
        return None

    return (
        JobTimeFilter(created=range_)
        if field == "created"
        else JobTimeFilter(completed=range_)
    )


def _build_links(request: Request, job_id: str) -> Links:
    """Drill-down URLs for the existing per-job endpoints - absolute URLs.

    Uses `request.base_url` so links are clickable when the dashboard is
    served from a different host / port than what a relative path would
    resolve to.
    """
    base = str(request.base_url)
    return Links(
        logs=f"{base}logs?job_id={job_id}", response=f"{base}response/{job_id}"
    )


def _enrich_active(row: JobStatus, now: datetime, request: Request) -> ActiveJobRow:
    """Project a JobStatus into the /active row shape."""
    # `created` is always set on upsert; cast away the NotRequired narrowing.
    created = cast(datetime, row.get("created"))
    return ActiveJobRow(
        job_id=row["job_id"],
        created=created,
        submitter=row.get("submitter", ""),
        data_tier=row.get("data_tier"),
        age_seconds=int((now - created).total_seconds()),
        links=_build_links(request, row["job_id"]),
    )


def _enrich_completed(row: JobStatus, request: Request) -> CompletedJobRow:
    """Project a JobStatus into the /completed row shape."""
    created = cast(datetime, row.get("created"))
    completed = cast(datetime, row.get("completed"))
    return CompletedJobRow(
        job_id=row["job_id"],
        submitter=row.get("submitter", ""),
        created=created,
        completed=completed,
        duration_seconds=float((completed - created).total_seconds()),
        data_tier=row.get("data_tier"),
        results=int(row.get("results", 0)),
        links=_build_links(request, row["job_id"]),
    )


def _enrich_failed(row: JobStatus, request: Request) -> FailedJobRow:
    """Project a JobStatus into the /failed row shape."""
    created = cast(datetime, row.get("created"))
    completed = cast(datetime, row.get("completed"))
    return FailedJobRow(
        job_id=row["job_id"],
        submitter=row.get("submitter", ""),
        created=created,
        completed=completed,
        duration_seconds=float((completed - created).total_seconds()),
        data_tier=row.get("data_tier"),
        status=row["status"],
        links=_build_links(request, row["job_id"]),
    )


def _get_version() -> StatusVersion:
    """Return the /status `version` block from the cached version lookup."""
    sha, branch = get_version()
    # 12-char short form is enough for the dashboard's at-a-glance display.
    return StatusVersion(git_commit=sha[:12], git_branch=branch)


def _bytes_to_mb(value: int | None) -> float | None:
    """Convert a byte count (or None) to MB (mebibytes)."""
    return None if value is None else value / _BYTES_PER_MB


def _get_rss_mb(pid: int) -> float | None:
    """Return the resident set size for `pid` in MB, or None if not inspectable."""
    try:
        rss = psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
        return None
    return _bytes_to_mb(rss)


def _process_row(pid: int, started_at: datetime, now: datetime) -> StatusProcess:
    """Turn a registry entry into the StatusProcess response shape."""
    return StatusProcess(
        pid=pid,
        uptime_seconds=float((now - started_at).total_seconds()),
        rss_mb=_get_rss_mb(pid),
    )


def _tier_snapshot() -> list[StatusTier]:
    """Iterate configured tiers (0, 1) and report each driver's health."""
    backends = (CONFIG.tier0.backend, CONFIG.tier1.backend)
    rows: list[StatusTier] = []
    for tier_idx, backend in enumerate(backends):
        try:
            health = tier_manager.get_driver(tier_idx).status()
        except Exception as exc:
            rows.append(
                StatusTier(
                    tier=tier_idx,
                    backend=backend,
                    up=False,
                    last_outage=None,
                    last_recovery=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        rows.append(
            StatusTier(
                tier=tier_idx,
                backend=backend,
                up=health["up"],
                last_outage=health["last_outage"],
                last_recovery=health["last_recovery"],
                error=health["error"],
            )
        )
    return rows


def _metakg_row(
    record: FreshnessRecord | None,
    *,
    self_reported: bool,
    now: datetime,
) -> StatusMetaKG:
    """Build the metakg block; `stale` flips True past twice the refresh cadence."""
    if record is None:
        return StatusMetaKG(
            refreshed_at=None, count=None, self_reported=self_reported, stale=False
        )
    refreshed_at = record["refreshed_at"]
    stale_after = timedelta(seconds=CONFIG.job.metakg.build_time * 2)
    return StatusMetaKG(
        refreshed_at=refreshed_at,
        count=int(record["count"]),
        self_reported=self_reported,
        stale=(now - refreshed_at) > stale_after,
    )


def _subclass_map_row(record: FreshnessRecord | None) -> StatusSubclassMap:
    """Build the subclass_map block; `available` is False when no map can be served."""
    if record is None:
        return StatusSubclassMap(refreshed_at=None, count=None, available=False)
    return StatusSubclassMap(
        refreshed_at=record["refreshed_at"],
        count=int(record["count"]),
        available=True,
    )


def _unwrap(result: object) -> object:
    """Return the value from a `gather(return_exceptions=True)` slot, or None on raise."""
    return None if isinstance(result, BaseException) else result


async def _window_snapshot(
    request: Request,
    times: JobTimeFilter | None,
    lookback_hours: float | None,
) -> CountsWindow:
    """Total + status-breakdown + drill-down links for one /counts window."""
    try:
        async with asyncio.TaskGroup() as tg:
            total_task = tg.create_task(MongoClient().count_jobs(times=times))
            counts_task = tg.create_task(
                MongoClient().group_jobs(group_by="status", times=times)
            )
    except BaseExceptionGroup as eg:
        # TaskGroup always wraps; preserve the original exception shape
        # for callers and FastAPI exception handlers when only one
        # branch actually failed.
        if len(eg.exceptions) == 1:
            raise eg.exceptions[0] from None
        raise
    total = total_task.result()
    counts = counts_task.result()
    base = str(request.base_url)
    if lookback_hours is None:
        links = CountsLinks(
            completed=f"{base}status/completed", failed=f"{base}status/failed"
        )
    else:
        # `:g` strips trailing zeros on whole-number hours (24, 168).
        suffix = f"?lookback={lookback_hours:g}"
        links = CountsLinks(
            completed=f"{base}status/completed{suffix}",
            failed=f"{base}status/failed{suffix}",
        )
    return CountsWindow(
        total=total,
        counts={str(row["key"]): int(row["count"]) for row in counts},
        links=links,
    )


router = APIRouter(prefix="/status", tags=["status"])


@router.get(
    "",
    response_model=StatusSnapshot,
    response_description=STATUS_DESCRIPTIONS.get("root", ""),
)
async def status_root() -> StatusSnapshot:
    """Operator-glance health snapshot; per-dependency failures degrade fields, not the response."""
    redis_client = RedisClient()
    mongo_client = MongoClient()
    stuck_cutoff = _now_aware() - timedelta(hours=STUCK_DEFAULT_HOURS)

    # Fan out independent IO in parallel. Both Redis and Mongo calls
    # are gated on their health flags - otherwise redis-py's retry-with-
    # backoff and Motor's 30s server-selection timeout would each stall
    # the response during an outage.
    #
    # `gather(..., return_exceptions=True)` loses per-element type
    # inference; cast back to a tuple of `object` so call sites stay typed.
    main_registry_r: object = None
    background_registry_r: object = None
    worker_registry_r: object = None
    redis_mem_r: object = None
    metakg_r: object = None
    subclass_r: object = None
    if redis_client.up:
        redis_results = cast(
            tuple[object, object, object, object, object, object],
            await asyncio.gather(
                redis_client.list_main(),
                redis_client.list_background(),
                redis_client.list_workers(),
                redis_client.used_memory_bytes(),
                redis_client.metakg_freshness(),
                redis_client.subclass_freshness(),
                return_exceptions=True,
            ),
        )
        (
            main_registry_r,
            background_registry_r,
            worker_registry_r,
            redis_mem_r,
            metakg_r,
            subclass_r,
        ) = redis_results
        if any(isinstance(r, BaseException) for r in redis_results):
            redis_client.request_health_check()

    mongo_storage_r: object = None
    in_flight_r: object = None
    stuck_r: object = None
    if mongo_client.up:
        mongo_results = cast(
            tuple[object, object, object],
            await asyncio.gather(
                mongo_client.db_storage_bytes(),
                mongo_client.count_in_flight(),
                mongo_client.count_stuck(stuck_cutoff),
                return_exceptions=True,
            ),
        )
        mongo_storage_r, in_flight_r, stuck_r = mongo_results
        if any(isinstance(r, BaseException) for r in mongo_results):
            mongo_client.request_health_check()

    main_registry = _unwrap(main_registry_r)
    background_registry = _unwrap(background_registry_r)
    worker_registry = _unwrap(worker_registry_r)
    mongo_storage = _unwrap(mongo_storage_r)
    redis_memory = _unwrap(redis_mem_r)
    active_job_count = _unwrap(in_flight_r)
    stuck_job_count = _unwrap(stuck_r)
    metakg_record = cast(FreshnessRecord | None, _unwrap(metakg_r))
    subclass_record = cast(FreshnessRecord | None, _unwrap(subclass_r))

    # `registry_available` flips False when *any* of the three registry
    # reads failed (they all hit Redis; one failing means we can't trust
    # the others either).
    registry_available = (
        main_registry is not None
        and background_registry is not None
        and worker_registry is not None
    )

    now = _now_aware()
    main = (
        next(
            (
                _process_row(pid, started, now)
                for pid, started in sorted(
                    cast(dict[int, datetime], main_registry).items()
                )
            ),
            None,
        )
        if main_registry is not None
        else None
    )
    background = (
        next(
            (
                _process_row(pid, started, now)
                for pid, started in sorted(
                    cast(dict[int, datetime], background_registry).items()
                )
            ),
            None,
        )
        if background_registry is not None
        else None
    )
    if worker_registry is not None:
        workers = [
            _process_row(pid, started, now)
            for pid, started in sorted(
                cast(dict[int, datetime], worker_registry).items()
            )
        ]
    else:
        # Registry unavailable - fall back to self-reporting the worker
        # serving this request so the dashboard still shows *something*.
        self_pid = worker.get_pid()
        self_started = worker.get_started_at()
        workers = (
            [_process_row(self_pid, self_started, now)]
            if self_pid is not None and self_started is not None
            else []
        )

    mongo_health = mongo_client.status()
    redis_health = redis_client.status()

    return StatusSnapshot(
        version=_get_version(),
        served_by=_served_by(now),
        processes=StatusProcesses(
            main=main,
            background=background,
            workers=workers,
            registry_available=registry_available,
        ),
        active_job_count=cast(int, active_job_count)
        if active_job_count is not None
        else None,
        stuck_job_count=cast(int, stuck_job_count)
        if stuck_job_count is not None
        else None,
        mongo=StatusMongo(
            up=mongo_health["up"],
            last_outage=mongo_health["last_outage"],
            last_recovery=mongo_health["last_recovery"],
            error=mongo_health["error"],
            storage_mb=cast(int, mongo_storage) / _BYTES_PER_MB
            if mongo_storage is not None
            else None,
            queue_depth=MongoQueue().qsize(),
        ),
        redis=StatusRedis(
            up=redis_health["up"],
            last_outage=redis_health["last_outage"],
            last_recovery=redis_health["last_recovery"],
            error=redis_health["error"],
            used_memory_mb=cast(int, redis_memory) / _BYTES_PER_MB
            if redis_memory is not None
            else None,
        ),
        tiers=_tier_snapshot(),
        # `self_reported=False`: this path serves the Redis-published copy.
        # Workers serving a degraded local copy would need to flip this
        # True on their own snapshot - not currently wired.
        metakg=_metakg_row(metakg_record, self_reported=False, now=now),
        subclass_map=_subclass_map_row(subclass_record),
    )


def _served_by(now: datetime) -> StatusServedBy:
    """Identify the worker answering this `/status` request; `started_at` defaults to `now`."""
    pid = worker.get_pid() or 0
    started_at = worker.get_started_at() or now
    return StatusServedBy(pid=pid, started_at=started_at)


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetail,
    response_description=STATUS_DESCRIPTIONS.get("jobs", ""),
    responses={404: {"description": "No job_status doc with that job_id."}},
)
async def status_job(request: Request, job_id: str) -> JobDetail:
    """Return the full job_status doc for one job_id (metadata + geometry)."""
    service_health.require_mongo(
        "MongoDB is unavailable; job detail cannot be retrieved."
    )
    job_id = job_id.lower()
    row = await MongoClient().get_job_status(job_id)
    if row is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail=f"Job {job_id!r} not found."
        )
    created = row.get("created")
    completed = row.get("completed")
    duration: float | None = None
    if isinstance(created, datetime) and isinstance(completed, datetime):
        duration = float((completed - created).total_seconds())
    return JobDetail(
        job_id=row["job_id"],
        status=row["status"],
        description=row.get("description"),
        submitter=row.get("submitter"),
        data_tier=row.get("data_tier"),
        is_async=row.get("is_async"),
        job_timeout=row.get("job_timeout"),
        created=created,
        completed=completed,
        duration_seconds=duration,
        qnodes=row.get("qnodes"),
        qedges=row.get("qedges"),
        qpaths=row.get("qpaths"),
        knodes=row.get("knodes"),
        kedges=row.get("kedges"),
        aux_graphs=row.get("aux_graphs"),
        results=row.get("results"),
        links=_build_links(request, job_id),
    )


@router.get(
    "/running",
    response_model=ActivePage,
    response_description=STATUS_DESCRIPTIONS.get("active", ""),
)
async def status_running(
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> ActivePage:
    """Paged scan of currently-in-flight jobs, newest first."""
    service_health.require_mongo(
        "MongoDB is unavailable; active jobs cannot be listed."
    )
    page = await _paged_jobs(
        identity=JobIdentityFilter(status=sorted(NON_TERMINAL)),
        times=None,
        sort={"field": "created", "direction": "desc"},
        limit=limit,
        cursor=cursor,
    )
    now = _now_aware()
    return ActivePage(
        items=[_enrich_active(row, now, request) for row in page["items"]],
        next_cursor=page["next_cursor"],
    )


@router.get(
    "/stuck",
    response_model=ActivePage,
    response_description=STATUS_DESCRIPTIONS.get("stuck", ""),
)
async def status_stuck(
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    min_age: Annotated[
        float | None,
        Query(
            ge=0,
            description="Hours; jobs running longer than this are returned.",
        ),
    ] = None,
) -> ActivePage:
    """Paged scan of active jobs older than `min_age` (default 1h)."""
    service_health.require_mongo("MongoDB is unavailable; stuck jobs cannot be listed.")
    effective_min_age = min_age if min_age is not None else STUCK_DEFAULT_HOURS
    cutoff = _now_aware() - timedelta(hours=effective_min_age)
    page = await _paged_jobs(
        identity=JobIdentityFilter(status=sorted(NON_TERMINAL)),
        times=JobTimeFilter(created={"before": cutoff}),
        sort={"field": "created", "direction": "desc"},
        limit=limit,
        cursor=cursor,
    )
    now = _now_aware()
    return ActivePage(
        items=[_enrich_active(row, now, request) for row in page["items"]],
        next_cursor=page["next_cursor"],
    )


@router.get(
    "/counts",
    response_model=CountsSnapshot,
    response_description=STATUS_DESCRIPTIONS.get("counts", ""),
)
async def status_counts(request: Request) -> CountsSnapshot:
    """Status breakdown across 24h / week / all-time windows."""
    service_health.require_mongo("MongoDB is unavailable; job counts cannot be served.")
    now = _now_aware()
    last_24h_window = JobTimeFilter(created={"after": now - timedelta(hours=24)})
    last_week_window = JobTimeFilter(
        created={"after": now - timedelta(hours=HOURS_PER_WEEK)}
    )
    last_24h, last_week, all_time = await asyncio.gather(
        _window_snapshot(request, last_24h_window, 24.0),
        _window_snapshot(request, last_week_window, HOURS_PER_WEEK),
        _window_snapshot(request, None, None),
    )
    return CountsSnapshot(
        windows={
            "last_24h": last_24h,
            "last_week": last_week,
            "all_time": all_time,
        }
    )


@router.get(
    "/timeline",
    response_model=list[TimelineBucket],
    response_description=STATUS_DESCRIPTIONS.get("timeline", ""),
)
async def status_timeline(  # noqa: PLR0913 Each arg is a query knob
    field: Literal["created", "completed"] = "completed",
    granularity: Literal["hour", "day", "week", "month"] = "hour",
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = 24.0,
    status: Annotated[
        str | None,
        Query(
            description=(
                "AsyncQueryStatusResponse lifecycle code "
                "(Running, Completed, Failed); maps to stored outcome internally."
            ),
        ),
    ] = None,
    data_tier: Annotated[
        int | None,
        Query(ge=0, le=2, description="Restrict to jobs run on this tier."),
    ] = None,
) -> list[TimelineBucket]:
    """Time-bucketed throughput counts."""
    service_health.require_mongo(
        "MongoDB is unavailable; timeline data cannot be served."
    )
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field=field
    )
    identity: JobIdentityFilter | None = None
    resolved = resolve_status_filter(status)
    if resolved is not None or data_tier is not None:
        identity = JobIdentityFilter()
        if resolved is not None:
            identity["status"] = resolved
        if data_tier is not None:
            identity["data_tier"] = data_tier  # pyright: ignore[reportGeneralTypeIssues]

    buckets = await MongoClient().bucket_jobs_over_time(
        field=field, granularity=granularity, identity=identity, times=times
    )
    return [
        TimelineBucket(bucket_start=row["bucket_start"], count=int(row["count"]))
        for row in buckets
    ]


@router.get(
    "/durations",
    response_model=DurationsResponse,
    response_description=STATUS_DESCRIPTIONS.get("durations", ""),
)
async def status_durations(
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = 24.0,
    status: Annotated[
        str | None,
        Query(
            description=(
                "AsyncQueryStatusResponse lifecycle code "
                "(Running, Completed, Failed); maps to stored outcome internally."
            ),
        ),
    ] = "Completed",
) -> DurationsResponse:
    """Aggregate runtime stats over terminal jobs in the window."""
    service_health.require_mongo(
        "MongoDB is unavailable; duration stats cannot be served."
    )
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="completed"
    )
    identity: JobIdentityFilter | None = None
    resolved = resolve_status_filter(status)
    if resolved is not None:
        identity = JobIdentityFilter(status=resolved)

    stats = await MongoClient().compute_durations(identity=identity, times=times)
    return DurationsResponse(
        sample_size=stats["sample_size"],
        min_seconds=stats["min_seconds"],
        max_seconds=stats["max_seconds"],
        avg_seconds=stats["avg_seconds"],
        p50_seconds=stats.get("p50_seconds"),
        p95_seconds=stats.get("p95_seconds"),
        p99_seconds=stats.get("p99_seconds"),
    )


@router.get(
    "/submitters",
    response_model=list[SubmitterRow],
    response_description=STATUS_DESCRIPTIONS.get("submitters", ""),
)
async def status_submitters(
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = None,
    top: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SubmitterRow]:
    """Per-submitter activity leaderboard."""
    service_health.require_mongo(
        "MongoDB is unavailable; submitter stats cannot be served."
    )
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="created"
    )
    rows = await MongoClient().submitter_stats(times=times, top=top)
    return [SubmitterRow(**row) for row in rows]


@router.get(
    "/tiers",
    response_model=list[TierRow],
    response_description=STATUS_DESCRIPTIONS.get("tiers", ""),
)
async def status_tiers(
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = 24.0,
) -> list[TierRow]:
    """Per-tier activity + duration percentiles."""
    service_health.require_mongo("MongoDB is unavailable; tier stats cannot be served.")
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="completed"
    )
    rows = await MongoClient().tier_stats(times=times)
    return [
        TierRow(
            tier=row["tier"],
            active=row["active"],
            completed=row["completed"],
            failed=row["failed"],
            durations=TierDurations(**row["durations"]),
            last_activity=row["last_activity"],
        )
        for row in rows
    ]


@router.get(
    "/submitter_tier_stats",
    response_model=list[SubmitterTierRow],
    response_description=STATUS_DESCRIPTIONS.get("submitter_tier_stats", ""),
)
async def status_submitter_tier_stats(
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = None,
) -> list[SubmitterTierRow]:
    """Per-(submitter, tier) cells. Drives the heatmaps view."""
    service_health.require_mongo(
        "MongoDB is unavailable; submitter/tier stats cannot be served."
    )
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="created"
    )
    rows = await MongoClient().submitter_tier_stats(times=times)
    return [SubmitterTierRow(**row) for row in rows]


@router.get(
    "/failure_breakdown",
    response_model=FailureBreakdownResponse,
    response_description=STATUS_DESCRIPTIONS.get("failure_breakdown", ""),
)
async def status_failure_breakdown(
    by: Annotated[
        Literal["submitter", "tier"],
        Query(description="Group key for the breakdown."),
    ],
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = None,
) -> FailureBreakdownResponse:
    """Failure-status counts pivoted by submitter or by tier."""
    service_health.require_mongo(
        "MongoDB is unavailable; failure breakdown cannot be served."
    )
    # Time window filters on `completed` so "failures in the last 24h"
    # means jobs that finished failing in that window - matches /failed
    # semantics and what the dashboard window selector intuitively
    # implies.
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="completed"
    )
    if by == "submitter":
        raw = await MongoClient().failure_breakdown_by_submitter(times=times)
    else:
        raw = await MongoClient().failure_breakdown_by_tier(times=times)
    return FailureBreakdownResponse(
        by=by, rows=[FailureBreakdownCell(**row) for row in raw]
    )


def _identity_with(
    *,
    status: str | list[str],
    submitter: str | None,
    data_tier: int | None,
) -> JobIdentityFilter:
    """Build a JobIdentityFilter from the common history-scan query params."""
    identity: JobIdentityFilter = {"status": status}
    if submitter is not None:
        identity["submitter"] = submitter
    if data_tier is not None:
        identity["data_tier"] = data_tier  # pyright: ignore[reportGeneralTypeIssues]
    return identity


async def _paged_jobs(
    *,
    identity: JobIdentityFilter,
    times: JobTimeFilter | None,
    sort: JobSortSpec,
    limit: int,
    cursor: str | None,
) -> JobStatusPage:
    """Fetch a paged JobStatus result, translating cursor errors to HTTP 400."""
    try:
        return await MongoClient().get_job_statuses(
            identity=identity,
            times=times,
            sort=sort,
            limit=limit,
            cursor=cursor,
        )
    except CursorDecodeError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e


@router.get(
    "/completed",
    response_model=CompletedPage,
    response_description=STATUS_DESCRIPTIONS.get("completed", ""),
)
async def status_completed(  # noqa: PLR0913 Each arg is a query knob
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = None,
    data_tier: Annotated[int | None, Query(ge=0, le=2)] = None,
    submitter: str | None = None,
    sort: Annotated[
        Literal["created", "completed", "duration", "results", "submitter", "status"],
        Query(description="Sort field; default `completed`."),
    ] = "completed",
    direction: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort direction; default `desc`."),
    ] = "desc",
) -> CompletedPage:
    """Paged scan of successfully-terminated jobs."""
    service_health.require_mongo(
        "MongoDB is unavailable; completed jobs cannot be listed."
    )
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="completed"
    )
    identity = _identity_with(
        status=sorted(TERMINAL_SUCCESS), submitter=submitter, data_tier=data_tier
    )
    page = await _paged_jobs(
        identity=identity,
        times=times,
        sort={"field": sort, "direction": direction},
        limit=limit,
        cursor=cursor,
    )
    return CompletedPage(
        items=[_enrich_completed(row, request) for row in page["items"]],
        next_cursor=page["next_cursor"],
    )


@router.get(
    "/failed",
    response_model=FailedPage,
    response_description=STATUS_DESCRIPTIONS.get("failed", ""),
)
async def status_failed(  # noqa: PLR0913 Each arg is a query knob
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    since: datetime | None = None,
    until: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `since`."),
    ] = None,
    data_tier: Annotated[int | None, Query(ge=0, le=2)] = None,
    submitter: str | None = None,
    reason: Annotated[
        str | None,
        Query(description="Narrow to one specific failure status."),
    ] = None,
    sort: Annotated[
        Literal["created", "completed", "duration", "results", "submitter", "status"],
        Query(description="Sort field; default `completed`."),
    ] = "completed",
    direction: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort direction; default `desc`."),
    ] = "desc",
) -> FailedPage:
    """Paged scan of failed-or-errored terminal jobs."""
    service_health.require_mongo(
        "MongoDB is unavailable; failed jobs cannot be listed."
    )
    times = _resolve_time_window(
        since=since, until=until, lookback=lookback, field="completed"
    )
    status_filter: str | list[str] = (
        reason if reason is not None else sorted(TERMINAL_FAILURE)
    )
    identity = _identity_with(
        status=status_filter, submitter=submitter, data_tier=data_tier
    )
    page = await _paged_jobs(
        identity=identity,
        times=times,
        sort={"field": sort, "direction": direction},
        limit=limit,
        cursor=cursor,
    )
    return FailedPage(
        items=[_enrich_failed(row, request) for row in page["items"]],
        next_cursor=page["next_cursor"],
    )


@router.get(
    "/server_logs",
    response_description=STATUS_DESCRIPTIONS.get("server_logs", ""),
)
async def status_server_logs(
    start: datetime | None = None,
    end: datetime | None = None,
    lookback: Annotated[
        float | None,
        Query(description="Hours back from now; overrides `start`."),
    ] = None,
    level: LogLevel = "DEBUG",
    fmt: Literal["flat", "struct"] = "flat",
) -> StreamingResponse:
    """Stream logs not associated with any job."""
    service_health.require_mongo(
        "MongoDB is unavailable; server logs cannot be served."
    )
    effective_start = start
    if lookback is not None:
        effective_start = _now_aware() - timedelta(hours=lookback)

    body: AsyncGenerator[str]
    if fmt == "flat":
        body = MongoClient().get_flat_server_logs(effective_start, end, level)
        media = "text/plain"
    else:
        body = objs_to_json(
            MongoClient().get_server_logs(effective_start, end, level),
            jsonl=True,
        )
        media = "application/json"

    return StreamingResponse(body, media_type=media)
