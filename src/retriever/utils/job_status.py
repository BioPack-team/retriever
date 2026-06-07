"""Job-status vocabulary shared across the /status API surface.

Centralizes the canonical sets of status strings that Retriever's lookup
path actually writes, plus the "failed" alias used at the API boundary.
Imported by MongoClient query helpers and (eventually) FastAPI route
handlers so the vocabulary stays in one place.
"""

from __future__ import annotations

NON_TERMINAL: frozenset[str] = frozenset({"Running"})
"""Statuses for jobs that have not reached a terminal state.

Retriever doesn't queue async jobs so `Queued` never appears in our
storage.
"""

TERMINAL_SUCCESS: frozenset[str] = frozenset({"Success", "Complete"})
"""Statuses for successfully-terminated jobs."""

TERMINAL_FAILURE: frozenset[str] = frozenset(
    {"Failed", "QueryNotTraversable", "UnsupportedConstraint"}
)
"""Statuses for failed-or-errored terminal jobs."""


# TRAPI 1.6 splits status vocabularies: Response uses outcome shortcodes
# (Success, QueryNotTraversable, KPsNotAvailable, ...) while
# AsyncQueryStatusResponse uses lifecycle shortcodes (Queued, Running,
# Completed, Failed). We store outcomes (the more specific value) and
# collapse to lifecycle only when filling an AsyncQueryStatusResponse.
_ASYNC_LIFECYCLE_MAP: dict[str, str] = {
    "Success": "Completed",
    "Complete": "Completed",  # legacy stored value
    "QueryNotTraversable": "Failed",
    "UnsupportedConstraint": "Failed",
}


def to_async_lifecycle(status: str) -> str:
    """Map a stored outcome status to the TRAPI AsyncQueryStatusResponse vocabulary.

    Unknown values pass through so we never lose information for statuses
    we haven't classified (e.g. future Retriever-specific failure flavors).
    """
    return _ASYNC_LIFECYCLE_MAP.get(status, status)


def resolve_status_filter(status: str | None) -> str | list[str] | None:
    """Map a lifecycle status to the stored outcome code(s) MongoDB indexes against.

    `Completed` -> `Success`. `Failed` (and legacy lowercase `failed`)
    expand to the failure union so all failure flavors match. `Running`
    passes through unchanged. Unknown values pass through too so a
    caller can still filter by a Retriever-specific code directly.

    Output shape matches `JobIdentityFilter["status"]` so callers can
    drop the result straight into `_build_filter_query` without further
    massaging.
    """
    if status is None:
        return None
    if status in {"Failed", "failed"}:
        return sorted(TERMINAL_FAILURE)
    if status == "Completed":
        return "Success"
    return status
