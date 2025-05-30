from typing import Literal

from pydantic import BaseModel
from reasoner_pydantic import CURIE, AsyncQuery, Edge, QEdge, Query


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


AdjacencyGraph = dict[str, dict[str, QEdge]]

KAdjacencyGraph = dict[str, dict[CURIE, dict[CURIE, list[Edge]]]]

QEdgeIDMap = dict[QEdge, str]

# A pair of Qnode and CURIE, used to uniquely identify partial results
QNodeCURIEPair = tuple[str, CURIE]

# A set of Qnode, CURIE, and QEdgeID, used to uniquely identify subquery results and partials
SuperpositionHop = tuple[CURIE, str]
