"""Job-status vocabulary shared across the /status API surface.

Centralizes the canonical sets of status strings that Retriever's lookup
path actually writes, plus the "failed" alias used at the API boundary.
Imported by MongoClient query helpers and (eventually) FastAPI route
handlers so the vocabulary stays in one place.
"""

from __future__ import annotations

NON_TERMINAL: frozenset[str] = frozenset({"Running", "Accepted"})
"""Statuses for jobs that have not reached a terminal state."""

TERMINAL_SUCCESS: frozenset[str] = frozenset({"Complete"})
"""Statuses for successfully-terminated jobs."""

TERMINAL_FAILURE: frozenset[str] = frozenset(
    {"Failed", "QueryNotTraversable", "UnsupportedConstraint"}
)
"""Statuses for failed-or-errored terminal jobs."""


def resolve_status_filter(status: str | None) -> str | list[str] | None:
    """Expand the "failed" alias to the failure union; pass other values through.

    Output shape matches `JobIdentityFilter["status"]` so callers can drop
    the result straight into `_build_filter_query` without further
    massaging.
    """
    if status is None:
        return None
    if status == "failed":
        return sorted(TERMINAL_FAILURE)
    return status
