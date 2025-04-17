from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Basic FastAPI error response body."""

    detail: str


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


class TRAPILog(TypedDict):
    """A TRAPI-formatted log object."""

    level: str
    message: str
    timestamp: str
    code: str | None
    trace: NotRequired[str]


class Query(TypedDict):
    """All information needed to understand what a client is asking for."""

    endpoint: str
    method: str
    body: dict[str, Any]
    job_id: str
