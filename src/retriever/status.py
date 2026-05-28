"""FastAPI route handlers for the `/status/*` dashboard API.

A thin glue layer over MongoClient / RedisClient aggregations. Mounted on
the main app by `server.py`'s `app.include_router(router)`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal, cast

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
    StatusFreshness,
    StatusMongo,
    StatusProcess,
    StatusProcesses,
    StatusRedis,
    StatusSnapshot,
    StatusTier,
    StatusVersion,
    SubmitterRow,
    SubmitterTierRow,
    TierDurations,
    TierRow,
    TimelineBucket,
)
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    """Drill-down URLs for the existing per-job endpoints — absolute URLs.

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
        data_tiers=list(row.get("data_tiers", [])),
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
        data_tiers=list(row.get("data_tiers", [])),
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
        data_tiers=list(row.get("data_tiers", [])),
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
    """Iterate configured tiers (0, 1) and report connection state."""
    backends = (CONFIG.tier0.backend, CONFIG.tier1.backend)
    rows: list[StatusTier] = []
    for tier_idx, backend in enumerate(backends):
        try:
            connected = not tier_manager.get_driver(tier_idx).is_failed
        except Exception:
            connected = False
        rows.append(StatusTier(tier=tier_idx, backend=backend, connected=connected))
    return rows


def _freshness_row(record: FreshnessRecord | None) -> StatusFreshness | None:
    """Project a FreshnessRecord into the response shape; None passes through."""
    if record is None:
        return None
    return StatusFreshness(
        refreshed_at=record["refreshed_at"], count=int(record["count"])
    )


async def _window_snapshot(
    request: Request,
    times: JobTimeFilter | None,
    lookback_hours: float | None,
) -> CountsWindow:
    """Total + status-breakdown + drill-down links for one /counts window."""
    total, counts = await asyncio.gather(
        MongoClient().count_jobs(times=times),
        MongoClient().group_jobs(group_by="status", times=times),
    )
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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/status", tags=["status"])


# === Bare /status root ====================================================


@router.get(
    "",
    response_model=StatusSnapshot,
    response_description=STATUS_DESCRIPTIONS.get("root", ""),
)
async def status_root() -> StatusSnapshot:
    """Operator-glance health snapshot."""
    redis_client = RedisClient()
    mongo_client = MongoClient()
    stuck_cutoff = _now_aware() - timedelta(hours=STUCK_DEFAULT_HOURS)

    # Fan out all the independent IO calls in parallel — these are the
    # bulk of the endpoint's wall time.
    async with asyncio.TaskGroup() as tg:
        t_main = tg.create_task(redis_client.list_main())
        t_bg = tg.create_task(redis_client.list_background())
        t_workers = tg.create_task(redis_client.list_workers())
        t_mongo_ping = tg.create_task(mongo_client.ping())
        t_mongo_storage = tg.create_task(mongo_client.db_storage_bytes())
        t_redis_ping = tg.create_task(redis_client.ping())
        t_redis_mem = tg.create_task(redis_client.used_memory_bytes())
        t_in_flight = tg.create_task(mongo_client.count_in_flight())
        t_stuck = tg.create_task(mongo_client.count_stuck(stuck_cutoff))
        t_metakg = tg.create_task(redis_client.metakg_freshness())
        t_subclass = tg.create_task(redis_client.subclass_freshness())

    main_registry = t_main.result()
    background_registry = t_bg.result()
    worker_registry = t_workers.result()
    mongo_connected = t_mongo_ping.result()
    mongo_storage = t_mongo_storage.result()
    redis_connected = t_redis_ping.result()
    redis_memory = t_redis_mem.result()
    active_job_count = t_in_flight.result()
    stuck_job_count = t_stuck.result()
    metakg = t_metakg.result()
    subclass = t_subclass.result()

    now = _now_aware()
    main = next(
        (
            _process_row(pid, started, now)
            for pid, started in sorted(main_registry.items())
        ),
        None,
    )
    background = next(
        (
            _process_row(pid, started, now)
            for pid, started in sorted(background_registry.items())
        ),
        None,
    )
    workers = [
        _process_row(pid, started, now)
        for pid, started in sorted(worker_registry.items())
    ]

    return StatusSnapshot(
        version=_get_version(),
        processes=StatusProcesses(main=main, background=background, workers=workers),
        active_job_count=active_job_count,
        stuck_job_count=stuck_job_count,
        mongo=StatusMongo(
            connected=mongo_connected,
            storage_mb=mongo_storage / _BYTES_PER_MB,
            queue_depth=MongoQueue().qsize(),
        ),
        redis=StatusRedis(
            connected=redis_connected,
            used_memory_mb=redis_memory / _BYTES_PER_MB,
        ),
        tiers=_tier_snapshot(),
        metakg=_freshness_row(metakg),
        subclass_map=_freshness_row(subclass),
    )


# === Per-job lookup =======================================================


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetail,
    response_description=STATUS_DESCRIPTIONS.get("jobs", ""),
    responses={404: {"description": "No job_status doc with that job_id."}},
)
async def status_job(request: Request, job_id: str) -> JobDetail:
    """Return the full job_status doc for one job_id (metadata + geometry)."""
    job_id = job_id.lower()
    row = await MongoClient().get_job_status(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    created = row.get("created")
    completed = row.get("completed")
    duration: float | None = None
    if isinstance(created, datetime) and isinstance(completed, datetime):
        duration = float((completed - created).total_seconds())
    return JobDetail(
        job_id=row["job_id"],
        status=row["status"],
        submitter=row.get("submitter"),
        data_tiers=list(row.get("data_tiers", [])),
        is_async=row.get("is_async"),
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


# === Realtime =============================================================


@router.get(
    "/active",
    response_model=ActivePage,
    response_description=STATUS_DESCRIPTIONS.get("active", ""),
)
async def status_active(
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> ActivePage:
    """Paged scan of currently-in-flight jobs, newest first."""
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


# === Dashboard headlines ==================================================


@router.get(
    "/counts",
    response_model=CountsSnapshot,
    response_description=STATUS_DESCRIPTIONS.get("counts", ""),
)
async def status_counts(request: Request) -> CountsSnapshot:
    """Status breakdown across 24h / week / all-time windows."""
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
        Query(description="Specific status or alias 'failed'."),
    ] = None,
    data_tier: Annotated[
        int | None,
        Query(ge=0, le=2, description="Restrict to jobs run on this tier."),
    ] = None,
) -> list[TimelineBucket]:
    """Time-bucketed throughput counts."""
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
            identity["data_tiers"] = [data_tier]  # pyright: ignore[reportGeneralTypeIssues]

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
        Query(description="Specific status or alias 'failed'."),
    ] = "Complete",
) -> DurationsResponse:
    """Aggregate runtime stats over terminal jobs in the window."""
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
    # Time window filters on `completed` so "failures in the last 24h"
    # means jobs that finished failing in that window — matches /failed
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


# === History scans ========================================================


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
        identity["data_tiers"] = [data_tier]  # pyright: ignore[reportGeneralTypeIssues]
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
        raise HTTPException(status_code=400, detail=str(e)) from e


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


# === Logs =================================================================


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
    effective_start = start
    if lookback is not None:
        effective_start = _now_aware() - timedelta(hours=lookback)

    if fmt == "flat":
        body: AsyncGenerator[str | dict[str, Any]] = MongoClient().get_flat_server_logs(
            effective_start, end, level
        )
        media = "text/plain"
    else:
        body = objs_to_json(
            MongoClient().get_server_logs(effective_start, end, level),
            jsonl=True,
        )
        media = "application/json"

    return StreamingResponse(body, media_type=media)
