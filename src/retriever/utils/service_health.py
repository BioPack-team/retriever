"""Service-health snapshot and degradation policy.

`Snapshot()` captures a frozen, process-local view of every backend
dependency's `up`/`error` state at the moment it's constructed; its
methods answer the per-request questions that follow from that view
- what HTTP status to return, which tier to route to, what warning
text to attach. No cross-process consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Literal

from fastapi import HTTPException

from retriever.data_tiers import tier_manager
from retriever.types.general import ErrorDetail
from retriever.types.trapi import LogEntryDict
from retriever.types.trapi_pydantic import (
    AsyncQuery as TRAPIAsyncQuery,
)
from retriever.types.trapi_pydantic import (
    Query as TRAPIQuery,
)
from retriever.types.trapi_pydantic import (
    TierNumber,
)
from retriever.utils.backend_client import BackendClient, StatusDict
from retriever.utils.logs import format_trapi_log
from retriever.utils.mongo import MongoClient
from retriever.utils.redis import RedisClient

Endpoint = Literal[
    "/query",
    "/asyncquery",
    "/asyncquery_status",
    "/response",
    "/logs",
    "/status/server_logs",
    "/status/counts",
    "/metadata",
    "/meta_knowledge_graph",
]
"""Routes whose behavior the policy module decides about."""

_T2: TierNumber = 2
"""Tier 2 - the one tier without a fallback peer."""


_MONGO_OUTAGE_TEXT = (
    "MongoDB connection failed, job state will not be persisted after this response"
)

_SUBCLASS_UNAVAILABLE_TEXT = (
    "Subclass mapping unavailable due to Redis outage; query results "
    "may omit subclass expansion."
)


def _tier_fallback_text(requested: TierNumber, fallback: TierNumber) -> str:
    return f"Tier {requested} is down; falling back to Tier {fallback}."


def mongo_outage_warning() -> LogEntryDict:
    """LogEntry attached when a Mongo write fails during request execution."""
    return format_trapi_log("WARNING", _MONGO_OUTAGE_TEXT)


def subclass_unavailable_warning() -> LogEntryDict:
    """LogEntry attached when subclass expansion fails for outage reasons."""
    return format_trapi_log("WARNING", _SUBCLASS_UNAVAILABLE_TEXT)


def tier_fallback_warning(requested: TierNumber, fallback: TierNumber) -> LogEntryDict:
    """LogEntry attached when a query was routed to a fallback tier."""
    return format_trapi_log("WARNING", _tier_fallback_text(requested, fallback))


def outage_detail(detail: str, dependency: BackendClient) -> ErrorDetail:
    """Build an `ErrorDetail` enriched with `dependency`'s outage info.

    `additional_info` carries `outage_time` (the dependency's
    `last_outage`) and `outage_error` (its `last_error`) so callers
    handling 424s can correlate the failure with `/status` data.
    """
    status = dependency.status()
    return ErrorDetail(
        detail=detail,
        additional_info={
            "outage_time": status["last_outage"],
            "outage_error": status["error"],
        },
    )


def require_mongo(detail: str) -> None:
    """Fail-fast 424 when MongoDB is unreachable.

    Convenience for read-only endpoints that touch Mongo and have no
    fallback path. Saves each call site from the Motor 30s server-
    selection timeout and ensures a uniform `ErrorDetail` body.
    """
    mongo = MongoClient()
    if mongo.up:
        return
    raise HTTPException(
        HTTPStatus.FAILED_DEPENDENCY,
        detail=outage_detail(detail, mongo),
    )


@dataclass(frozen=True, slots=True, init=False)
class Snapshot:
    """Frozen process-local view of backend health, built at construction time.

    Immutable so every decision within a request shares one view, even
    if a backend's flag flips mid-flight. `tier2` is `None` when that
    tier isn't implemented in this build.
    """

    mongo: StatusDict
    redis: StatusDict
    tier0: StatusDict
    tier1: StatusDict
    tier2: StatusDict | None

    def __init__(self) -> None:
        """Build a snapshot from the current `BackendClient` singletons."""
        object.__setattr__(self, "mongo", MongoClient().status())
        object.__setattr__(self, "redis", RedisClient().status())
        object.__setattr__(self, "tier0", tier_manager.get_driver(0).status())
        object.__setattr__(self, "tier1", tier_manager.get_driver(1).status())
        tier2 = (
            tier_manager.get_driver(_T2).status()
            if _T2 in tier_manager.IMPLEMENTED_TIERS
            else None
        )
        object.__setattr__(self, "tier2", tier2)

    def _tier_health(self, tier: TierNumber) -> StatusDict | None:
        """`StatusDict` for `tier`, or `None` if unimplemented."""
        if tier == 0:
            return self.tier0
        if tier == 1:
            return self.tier1
        return self.tier2

    def http_status_for(self, endpoint: Endpoint) -> HTTPStatus | None:
        """Short-circuit HTTP status for `endpoint`, or `None` to proceed.

        Per-tier endpoints (`/metadata?tier=N`, `/meta_knowledge_graph?tier=N`)
        can't be decided from the snapshot alone - callers should use
        `select_tier(..., allow_fallback=False)`. Aggregated
        `/meta_knowledge_graph` runs as long as at least one implemented
        tier is up; only 424s when all are down.
        """
        if endpoint in (
            "/asyncquery_status",
            "/response",
            "/logs",
            "/status/server_logs",
            "/status/counts",
        ):
            return None if self.mongo["up"] else HTTPStatus.FAILED_DEPENDENCY
        if endpoint == "/meta_knowledge_graph":
            return (
                None
                if (self.tier0["up"] or self.tier1["up"])
                else HTTPStatus.FAILED_DEPENDENCY
            )
        return None

    def select_tier(
        self,
        body: TRAPIQuery | TRAPIAsyncQuery | None,
        requested: TierNumber,
        *,
        allow_fallback: bool = True,
    ) -> tuple[TierNumber, list[LogEntryDict]] | tuple[HTTPStatus, ErrorDetail]:
        """Resolve `requested` into an effective tier, or a 424/501 body to return.

        Success: `(effective_tier, fallback_warnings)`. `fallback_warnings`
        carries a tier-fallback `LogEntry` when the request was redirected;
        empty list otherwise.

        Failure: `(status, ErrorDetail)`. Status is `501 Not Implemented`
        for tier 2, `424 Failed Dependency` otherwise.

        `allow_fallback=False` (per-tier endpoints with no body) disables
        fallback explicitly; a `tier_fallback=False` parameter on a query
        body has the same effect.
        """
        fallback_enabled = allow_fallback and (
            body.parameters.tier_fallback
            if body is not None and body.parameters is not None
            else True
        )
        requested_health = self._tier_health(requested)

        if requested_health is not None and requested_health["up"]:
            return requested, []

        if requested == _T2 and requested_health is None:
            return HTTPStatus.NOT_IMPLEMENTED, ErrorDetail(
                detail="Tier 2 not yet implemented.",
            )

        if requested == _T2:
            return HTTPStatus.FAILED_DEPENDENCY, outage_detail(
                "Tier 2 is unavailable with no fallback.",
                tier_manager.get_driver(_T2),
            )

        if not fallback_enabled:
            return HTTPStatus.FAILED_DEPENDENCY, outage_detail(
                f"Tier {requested} is unavailable; tier fallback was disabled by the request.",
                tier_manager.get_driver(requested),
            )

        other: TierNumber = 1 if requested == 0 else 0
        other_health = self._tier_health(other)
        if other_health is not None and other_health["up"]:
            return other, [tier_fallback_warning(requested, other)]

        return HTTPStatus.FAILED_DEPENDENCY, outage_detail(
            f"Tier {requested} is unavailable; fallback Tier {other} is also unavailable.",
            tier_manager.get_driver(requested),
        )
