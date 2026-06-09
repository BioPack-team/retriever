from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, NamedTuple, NotRequired, TypedDict

from fastapi import BackgroundTasks, Request, Response
from fastapi.datastructures import Headers
from pydantic import BeforeValidator

from retriever.types.trapi import (
    CURIE,
    AsyncQueryDict,
    AuxGraphID,
    AuxiliaryGraphDict,
    EdgeIdentifier,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeDict,
    QEdgeID,
    QNodeID,
    QueryDict,
    ResultDict,
)
from retriever.types.trapi_pydantic import TierNumber


class ErrorDetail(TypedDict):
    """Basic FastAPI error response body."""

    detail: str
    additional_info: NotRequired[dict[str, Any]]


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
    headers: Headers
    body: QueryDict | AsyncQueryDict | None
    job_id: str
    tier: TierNumber | None
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
    status: str = "Success"


AdjacencyGraph = dict[QNodeID, dict[QNodeID, list[QEdgeDict]]]

KAdjacencyGraph = dict[QEdgeID, dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]]

QEdgeIDMap = dict[int, QEdgeID]

# A pair of Qnode and CURIE, used to uniquely identify partial results
QNodeCURIEPair = tuple[QNodeID, CURIE]


EntityToEntityMapping = Mapping[CURIE, list[CURIE]]
