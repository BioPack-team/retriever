"""TRAPI as represented in TypedDict.

More performant than reasoner-pydantic, but doesn't offer validation or other utility.
To be used only for TRAPI constructed by Retriever, that doesn't need validation.
This way we get the performance of pure dict, with static type checking.
(Note that we're using TypedDict over Dataclass because they're less finicky
and more compatible with dependencies which return pure dicts)

A few notes:
    - Individual class docstrings solely refer to their name in the TRAPI Spec.
    - Because this is only for internal use, some items that are reducible to strings
      are simply annotated as `str`
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NewType, NotRequired, TypedDict


class SetInterpretationEnum(str, Enum):
    """Enumeration for set interpretation."""

    BATCH = "BATCH"
    ALL = "ALL"
    MANY = "MANY"


class OperatorEnum(str, Enum):
    """Enumeration of possible operators for attribute constraints."""

    EQUAL = "=="
    STRICT_EQUAL = "==="
    GT = ">"
    LT = "<"
    MATCH = "matches"


class ParametersDict(TypedDict):
    """Query Parameters."""

    tiers: NotRequired[list[int]]
    timeout: NotRequired[float]


class QueryDict(TypedDict):
    """Query."""

    message: MessageDict
    log_level: NotRequired[LogLevel | None]
    workflow: NotRequired[list[dict[str, Any]] | None]
    submitter: NotRequired[str | None]
    bypass_cache: NotRequired[bool]
    parameters: NotRequired[ParametersDict | None]


class AsyncQueryDict(QueryDict):
    """AsyncQuery."""

    callback: URL


class AsyncQueryResponseDict(TypedDict):
    """AsyncQueryResponse."""

    status: NotRequired[str | None]
    description: NotRequired[str | None]
    job_id: str


class AsyncQueryStatusResponseDict(TypedDict):
    """AsyncQueryStatus."""

    status: str
    description: str
    logs: list[LogEntryDict]
    response_url: NotRequired[URL | None]


class ResponseDict(TypedDict):
    """Response."""

    message: MessageDict
    status: NotRequired[str | None]
    description: NotRequired[str | None]
    logs: NotRequired[list[LogEntryDict]]
    workflow: NotRequired[list[dict[str, Any]] | None]
    parameters: NotRequired[ParametersDict | None]
    schema_version: NotRequired[str | None]
    biolink_version: NotRequired[str | None]


class MessageDict(TypedDict):
    """Message."""

    results: NotRequired[list[ResultDict] | None]
    query_graph: NotRequired[QueryGraphDict | PathfinderQueryGraphDict | None]
    knowledge_graph: NotRequired[KnowledgeGraphDict | None]
    auxiliary_graphs: NotRequired[dict[AuxGraphID, AuxiliaryGraphDict] | None]


class LogEntryDict(TypedDict):
    """LogEntry."""

    timestamp: str
    level: NotRequired[LogLevel | None]
    code: NotRequired[str | None]
    message: str


class LogLevel(str, Enum):
    """Log level."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class ResultDict(TypedDict):
    """Result."""

    node_bindings: dict[QNodeID, list[NodeBindingDict]]
    analyses: list[AnalysisDict | PathfinderAnalysisDict]


class NodeBindingDict(TypedDict):
    """NodeBinding."""

    id: CURIE
    query_id: NotRequired[QNodeID | None]
    attributes: list[AttributeDict]


class BaseAnalysisDict(TypedDict):
    """Base Analysis, shared by Analysis and PathfinderAnalysis."""

    resource_id: Infores
    score: NotRequired[float | None]
    support_graphs: NotRequired[list[AuxGraphID] | None]
    scoring_method: NotRequired[str | None]
    attributes: NotRequired[list[AttributeDict] | None]


class AnalysisDict(BaseAnalysisDict):
    """Analysis."""

    edge_bindings: dict[QEdgeID, list[EdgeBindingDict]]


class PathfinderAnalysisDict(BaseAnalysisDict):
    """PathfinderAnalysis."""

    path_bindings: dict[QPathID, list[PathBindingDict]]


class EdgeBindingDict(TypedDict):
    """EdgeBinding."""

    id: EdgeIdentifier
    attributes: list[AttributeDict]


class PathBindingDict(TypedDict):
    """PathBinding."""

    id: AuxGraphID


class AuxiliaryGraphDict(TypedDict):
    """AuxiliaryGraph."""

    edges: list[EdgeIdentifier]
    attributes: list[AttributeDict]


class KnowledgeGraphDict(TypedDict):
    """KnowledgeGraph."""

    nodes: dict[CURIE, NodeDict]
    edges: dict[EdgeIdentifier, EdgeDict]


class BaseQueryGraphDict(TypedDict):
    """Base QueryGraph."""

    nodes: dict[QNodeID, QNodeDict]


class QueryGraphDict(BaseQueryGraphDict):
    """Normal QueryGraph."""

    edges: dict[QEdgeID, QEdgeDict]


class PathfinderQueryGraphDict(BaseQueryGraphDict):
    """Pathfinder QueryGraph."""

    paths: dict[QPathID, QPathDict]


class QNodeDict(TypedDict):
    """QNode."""

    ids: NotRequired[list[CURIE] | None]
    categories: NotRequired[list[BiolinkEntity] | None]
    set_interpretation: NotRequired[SetInterpretationEnum | None]
    member_ids: NotRequired[list[CURIE] | None]
    constraints: NotRequired[list[AttributeConstraintDict] | None]


class QEdgeDict(TypedDict):
    """QEdge."""

    knowledge_type: NotRequired[str | None]
    predicates: NotRequired[list[BiolinkPredicate] | None]
    subject: QNodeID
    object: QNodeID
    attribute_constraints: NotRequired[list[AttributeConstraintDict]]
    qualifier_constraints: NotRequired[list[QualifierConstraintDict]]


class QPathDict(TypedDict):
    """QPath."""

    subject: QNodeID
    object: QNodeID
    predicates: NotRequired[list[BiolinkPredicate] | None]
    constraints: NotRequired[list[PathConstraintDict] | None]


class PathConstraintDict(TypedDict):
    """PathContraint."""

    intermediate_categories: NotRequired[list[BiolinkEntity] | None]


class NodeDict(TypedDict):
    """Node."""

    name: NotRequired[str | None]
    categories: list[BiolinkEntity]
    attributes: list[AttributeDict]
    is_set: NotRequired[bool | None]


class AttributeDict(TypedDict):
    """Attribute."""

    attribute_type_id: str
    original_attribute_name: NotRequired[str | None]
    value: Any
    value_type_id: NotRequired[str | None]
    attribute_source: NotRequired[str | None]
    value_url: NotRequired[URL | None]
    attributes: NotRequired[list[AttributeDict] | None]


class EdgeDict(TypedDict):
    """Edge."""

    predicate: BiolinkPredicate
    subject: CURIE
    object: CURIE
    attributes: NotRequired[list[AttributeDict] | None]
    qualifiers: NotRequired[list[QualifierDict] | None]
    sources: list[RetrievalSourceDict]


class QualifierDict(TypedDict):
    """Qualifier."""

    qualifier_type_id: QualifierTypeID
    qualifier_value: str


class QualifierConstraintDict(TypedDict):
    """QualifierConstraint."""

    qualifier_set: list[QualifierDict]


BiolinkEntity = NewType("BiolinkEntity", str)
BiolinkPredicate = NewType("BiolinkPredicate", str)
CURIE = NewType("CURIE", str)


class MetaKnowledgeGraphDict(TypedDict):
    """MetaKnowledgeGraph."""

    nodes: dict[BiolinkEntity, MetaNodeDict]
    edges: list[MetaEdgeDict]


class MetaNodeDict(TypedDict):
    """MetaNode."""

    id_prefixes: list[str]
    attributes: NotRequired[list[MetaAttributeDict] | None]


class MetaEdgeDict(TypedDict):
    """MetaEdge."""

    subject: BiolinkEntity
    predicate: BiolinkPredicate
    object: BiolinkEntity
    knowledge_types: NotRequired[list[str] | None]
    attributes: NotRequired[list[MetaAttributeDict] | None]
    qualifiers: NotRequired[list[MetaQualifierDict] | None]


class MetaQualifierDict(TypedDict):
    """MetaQualifier."""

    qualifier_type_id: QualifierTypeID
    applicable_values: NotRequired[list[str]]


class MetaAttributeDict(TypedDict):
    """MetaAttribute."""

    attribute_type_id: str
    attribute_source: NotRequired[str | None]
    original_attribute_names: NotRequired[list[str] | None]
    constraint_use: NotRequired[bool]
    constraint_name: NotRequired[str | None]


# AttributeConstraint
AttributeConstraintDict = TypedDict(
    "AttributeConstraintDict",
    {
        "id": str,
        "name": str,
        "not": NotRequired[bool],
        "operator": OperatorEnum,
        "value": Any,
        "unit_id": NotRequired[str | None],
        "unit_name": NotRequired[str | None],
    },
)


class RetrievalSourceDict(TypedDict):
    """RetrievalSource."""

    resource_id: Infores
    resource_role: str
    upstream_resource_ids: NotRequired[list[Infores] | None]
    source_record_urls: NotRequired[list[str] | None]


# These don't offer any special behavior, but make type annotation less confusable
EdgeIdentifier = NewType("EdgeIdentifier", str)
AuxGraphID = NewType("AuxGraphID", str)
QNodeID = NewType("QNodeID", str)
QEdgeID = NewType("QEdgeID", str)
QPathID = NewType("QPathID", str)
Infores = NewType("Infores", str)
QualifierTypeID = NewType("QualifierTypeID", str)
URL = NewType("URL", str)
