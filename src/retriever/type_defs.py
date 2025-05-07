from typing import Literal

from pydantic import BaseModel
from reasoner_pydantic import AsyncQuery, Query


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


class QueryInfo(BaseModel):
    """All information needed to understand what a client is asking for."""

    endpoint: str
    method: str
    body: Query | AsyncQuery | None
    job_id: str
