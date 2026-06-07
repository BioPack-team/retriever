"""Response shapes for the `/status/*` dashboard API.

These are plain TypedDicts - the dashboard endpoints build responses from
trusted internal data, so there's nothing to validate. FastAPI uses the
return-type annotations to generate the OpenAPI schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, NotRequired, TypedDict


class Links(TypedDict):
    """Drill-down URLs into the existing per-job endpoints."""

    logs: str
    response: str


class ActiveJobRow(TypedDict):
    """A single currently-in-flight job."""

    job_id: str
    created: datetime
    submitter: str
    data_tier: int | None
    age_seconds: int
    links: Links


class CompletedJobRow(TypedDict):
    """A single terminal-success job row."""

    job_id: str
    submitter: str
    created: datetime
    completed: datetime
    duration_seconds: float
    data_tier: int | None
    results: int
    links: Links


class FailedJobRow(TypedDict):
    """A single terminal-failure job row.

    Same as `CompletedJobRow` but with `status` carrying the specific
    failure flavor (Failed / QueryNotTraversable / UnsupportedConstraint).
    """

    job_id: str
    submitter: str
    created: datetime
    completed: datetime
    duration_seconds: float
    data_tier: int | None
    status: str
    links: Links


class ActivePage(TypedDict):
    """A page of in-flight job rows."""

    items: list[ActiveJobRow]
    next_cursor: str | None


class CompletedPage(TypedDict):
    """A page of completed job rows."""

    items: list[CompletedJobRow]
    next_cursor: str | None


class FailedPage(TypedDict):
    """A page of failed job rows."""

    items: list[FailedJobRow]
    next_cursor: str | None


class JobDetail(TypedDict):
    """Full per-job status for the dashboard's drill-down modal."""

    job_id: str
    status: str
    description: NotRequired[str | None]
    submitter: NotRequired[str | None]
    data_tier: int | None
    is_async: NotRequired[bool | None]
    job_timeout: NotRequired[float | None]
    created: NotRequired[datetime | None]
    completed: NotRequired[datetime | None]
    duration_seconds: NotRequired[float | None]
    # Query-graph geometry (set at enqueue)
    qnodes: NotRequired[int | None]
    qedges: NotRequired[int | None]
    qpaths: NotRequired[int | None]
    # Response-graph geometry (set on completion)
    knodes: NotRequired[int | None]
    kedges: NotRequired[int | None]
    aux_graphs: NotRequired[int | None]
    results: NotRequired[int | None]
    links: Links


class TimelineBucket(TypedDict):
    """A single time-bucketed throughput count."""

    bucket_start: datetime
    count: int


class DurationsResponse(TypedDict):
    """Aggregate runtime stats over terminal jobs in the window."""

    sample_size: int
    min_seconds: float
    max_seconds: float
    avg_seconds: float
    p50_seconds: NotRequired[float | None]
    p95_seconds: NotRequired[float | None]
    p99_seconds: NotRequired[float | None]


class SubmitterRow(TypedDict):
    """A single submitter leaderboard row."""

    submitter: str
    count: int
    completed: int
    failed: int
    success_rate: float | None
    p95_duration_seconds: float | None
    last_seen: datetime


class TierDurations(TypedDict):
    """Per-tier duration stats; all `None` when no completions."""

    avg_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None


class TierRow(TypedDict):
    """A single tier-activity row."""

    tier: int
    active: int
    completed: int
    failed: int
    durations: TierDurations
    last_activity: datetime | None


class SubmitterTierRow(TypedDict):
    """A single (submitter, tier) cell."""

    submitter: str
    tier: int
    count: int
    completed: int
    failed: int
    avg_seconds: float | None
    p95_seconds: float | None
    last_seen: datetime | None


class FailureBreakdownCell(TypedDict):
    """One {key, status} cell from `/status/failure_breakdown`.

    `key` is the submitter name or the tier index stringified, depending
    on the `by` query param.
    """

    key: str
    status: str
    count: int


class FailureBreakdownResponse(TypedDict):
    """`/status/failure_breakdown` response."""

    by: Literal["submitter", "tier"]
    rows: list[FailureBreakdownCell]


class CountsLinks(TypedDict):
    """Per-window drill-down links to /completed and /failed."""

    completed: str
    failed: str


class CountsWindow(TypedDict):
    """One window's status breakdown + drill-down links.

    `counts` maps status string -> count for shallow JSON access:
    `counts.Failed` rather than walking a list.
    """

    total: int
    counts: dict[str, int]
    links: CountsLinks


class CountsSnapshot(TypedDict):
    """`/counts` response - three fixed windows (last_24h, last_week, all_time)."""

    windows: dict[str, CountsWindow]


class StatusVersion(TypedDict):
    """Running-build provenance."""

    git_commit: str
    git_branch: str | None


class StatusProcess(TypedDict):
    """A registered Retriever process."""

    pid: int
    uptime_seconds: float
    rss_mb: float | None


class BackendHealth(TypedDict):
    """Uniform health projection emitted by every BackendClient."""

    up: bool
    last_outage: datetime | None
    last_recovery: datetime | None
    error: str | None


class StatusMongo(BackendHealth):
    """MongoDB liveness + capacity."""

    storage_mb: float | None
    queue_depth: int


class StatusRedis(BackendHealth):
    """Dragonfly liveness + capacity."""

    used_memory_mb: float | None


class StatusTier(BackendHealth):
    """One configured tier's health + identity."""

    tier: int
    backend: str


class StatusMetaKG(TypedDict):
    """Freshness of the published MetaKG plus per-worker caveats."""

    refreshed_at: datetime | None
    count: int | None
    self_reported: bool
    """True if the answering worker served its in-memory copy instead of the Redis sidecar."""
    stale: bool
    """True past roughly twice the refresh cadence; signals instance-0 not refreshing."""


class StatusSubclassMap(TypedDict):
    """Subclass mapping freshness; `available` flips false when no map can be served."""

    refreshed_at: datetime | None
    count: int | None
    available: bool


class StatusProcesses(TypedDict):
    """Process tree for the running Retriever instance."""

    main: StatusProcess | None
    background: StatusProcess | None
    workers: list[StatusProcess]
    """Just the answering worker if the registry isn't available."""
    registry_available: bool
    """False when the worker registry couldn't be read (typically a Redis outage)."""


class StatusServedBy(TypedDict):
    """Identity of the worker that answered this `/status` request."""

    pid: int
    started_at: datetime


class StatusSnapshot(TypedDict):
    """Bare `/status` health snapshot."""

    version: StatusVersion
    served_by: StatusServedBy
    processes: StatusProcesses
    active_job_count: int | None
    stuck_job_count: int | None
    mongo: StatusMongo
    redis: StatusRedis
    tiers: list[StatusTier]
    metakg: StatusMetaKG
    subclass_map: StatusSubclassMap
