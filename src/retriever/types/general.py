from typing import Annotated, Literal, NamedTuple, TypedDict

from fastapi import BackgroundTasks, Request, Response
from pydantic import BaseModel, Field
from reasoner_pydantic import (
    CURIE,
    AsyncQuery,
    QEdge,
    Query,
)
from reasoner_pydantic.shared import EdgeIdentifier

from retriever.types.trapi import (
    AuxGraphDict,
    KnowledgeGraphDict,
    LogEntryDict,
    ResultDict,
)

TierNumber = Annotated[int, Field(ge=0, le=2)]


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


class APIInfo(NamedTuple):
    """Information relating to the FastAPI request."""

    request: Request
    response: Response
    background_tasks: BackgroundTasks | None = None


class QueryInfo(NamedTuple):
    """All information needed to understand what a client is asking for."""

    endpoint: str
    method: str
    body: Query | AsyncQuery | None
    job_id: str
    tiers: set[TierNumber]


class BackendResults(TypedDict):
    """Transformed results of a query to a given database backend."""

    results: list[ResultDict]
    knowledge_graph: KnowledgeGraphDict
    auxiliary_graphs: dict[str, AuxGraphDict]


class LookupArtifacts(NamedTuple):
    """The parts of a TRAPI response that a lookup will update.

    (results, kg, logs)
    """

    results: list[ResultDict]
    kgraph: KnowledgeGraphDict
    aux_graphs: dict[str, AuxGraphDict]
    logs: list[LogEntryDict]
    error: bool | None = None


AdjacencyGraph = dict[str, dict[str, list[QEdge]]]

KAdjacencyGraph = dict[str, dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]]

QEdgeIDMap = dict[QEdge, str]

# A pair of Qnode and CURIE, used to uniquely identify partial results
QNodeCURIEPair = tuple[str, CURIE]

# A set of Qnode, CURIE, and QEdgeID, used to uniquely identify subquery results and partials
SuperpositionHop = tuple[CURIE, str]
