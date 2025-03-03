from typing import Any, Literal, TypedDict

LogLevel = Literal[
    "TRACE",
    "trace",
    "DEBUG",
    "debug",
    "INFO",
    "info",
    "SUCCESS",
    "success",
    "WARNING",
    "warning",
    "ERROR",
    "error",
    "CRITICAL",
    "critical",
]


class Query(TypedDict):
    """All information needed to understand what a client is asking for."""

    endpoint: str
    method: str
    body: dict[str, Any] | None
