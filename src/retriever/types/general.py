from typing import Annotated, Literal, NamedTuple, TypedDict

from fastapi import BackgroundTasks, Request, Response
from pydantic import BaseModel, BeforeValidator
from reasoner_pydantic import QEdge

from retriever.types.trapi import (
    CURIE,
    AuxGraphID,
    AuxiliaryGraphDict,
    EdgeIdentifier,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeID,
    QNodeID,
    ResultDict,
)
from retriever.types.trapi_pydantic import AsyncQuery, Query, TierNumber


class ErrorDetail(BaseModel):
    """Basic FastAPI error response body."""

    detail: str


LogLevel = Annotated[
    Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ],
    BeforeValidator(lambda a: str(a).upper()),
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
    timeout: float


class BackendResult(TypedDict):
    """Transformed results of a query to a given database backend."""

    results: list[ResultDict]
    knowledge_graph: KnowledgeGraphDict
    auxiliary_graphs: dict[AuxGraphID, AuxiliaryGraphDict]


class LookupArtifacts(NamedTuple):
    """The parts of a TRAPI response that a lookup will update.

    (results, kg, logs)
    """

    results: list[ResultDict]
    kgraph: KnowledgeGraphDict
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict]
    logs: list[LogEntryDict]
    error: bool | None = None


AdjacencyGraph = dict[QNodeID, dict[QNodeID, list[QEdge]]]

KAdjacencyGraph = dict[QEdgeID, dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]]

QEdgeIDMap = dict[QEdge, QEdgeID]

# A pair of Qnode and CURIE, used to uniquely identify partial results
QNodeCURIEPair = tuple[QNodeID, CURIE]
