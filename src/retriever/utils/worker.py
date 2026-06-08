"""Per-worker process identity, set once at lifespan startup.

Populated by `server.py` lifespan via `init()`. Read by `query.py` at
job-creation time. A separate module avoids the server → query circular import.
"""

from __future__ import annotations

import os
from datetime import datetime

# Mutable container avoids module-level `global` statements.
_identity: dict[str, int | datetime | None] = {"pid": None, "started_at": None}


def init(started_at: datetime) -> None:
    """Capture this worker's PID and start time. Called once from lifespan."""
    _identity["pid"] = os.getpid()
    _identity["started_at"] = started_at


def get_pid() -> int | None:
    """Return the PID captured at init, or None if init has not been called."""
    pid = _identity["pid"]
    return pid if isinstance(pid, int) else None


def get_started_at() -> datetime | None:
    """Return the started_at captured at init, or None if not yet called."""
    started_at = _identity["started_at"]
    return started_at if isinstance(started_at, datetime) else None
