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

from collections.abc import Hashable
from typing import Any, NotRequired, TypedDict

from reasoner_pydantic import LogLevel
from reasoner_pydantic.qgraph import SetInterpretationEnum


class QueryDict(TypedDict):
    """Query."""

    message: MessageDict
    log_level: NotRequired[LogLevel]
    workflow: NotRequired[list[Hashable]]
    submitter: NotRequired[str]
    bypass_cache: NotRequired[bool]


class AsyncQueryDict(QueryDict):
    """AsyncQuery."""

    callback: str


class AsyncQueryResponseDict(TypedDict):
    """AsyncQueryResponse."""

    status: NotRequired[str]
    description: NotRequired[str]
    job_id: str


class AsyncQueryStatusResponseDict(TypedDict):
    """AsyncQueryStatus."""

    status: str
    description: str
    logs: list[LogEntryDict]
    response_url: NotRequired[str]


class ResponseDict(TypedDict):
    """Response."""

    message: MessageDict
    status: NotRequired[str]
    description: NotRequired[str]
    logs: list[LogEntryDict]
    workflow: NotRequired[list[Hashable]]
    schema_version: NotRequired[str]
    biolink_version: NotRequired[str]


class MessageDict(TypedDict):
    """Message."""

    results: NotRequired[list[ResultDict]]
    query_graph: NotRequired[QueryGraphDict]
    knowledge_graph: NotRequired[KnowledgeGraphDict]
    auxiliary_graphs: NotRequired[dict[AuxGraphID, AuxGraphDict]]


class LogEntryDict(TypedDict):
    """LogEntry."""

    timestamp: str
    level: NotRequired[LogLevel]
    code: NotRequired[str]
    message: str


class ResultDict(TypedDict):
    """Result."""

    node_bindings: dict[QNodeID, list[NodeBindingDict]]
    analyses: list[AnalysisDict]


class NodeBindingDict(TypedDict):
    """NodeBinding."""

    id: CURIE
    query_id: NotRequired[QNodeID]
    attributes: list[AttributeDict]


class AnalysisDict(TypedDict):
    """Analysis."""

    resource_id: Infores
    score: NotRequired[float]
    edge_bindings: dict[QEdgeID, list[EdgeBindingDict]]
    support_graphs: NotRequired[list[AuxGraphID]]
    scoring_method: NotRequired[str]
    attributes: NotRequired[list[AttributeDict]]


class EdgeBindingDict(TypedDict):
    """EdgeBinding."""

    id: EdgeIdentifier
    attributes: list[AttributeDict]


class AuxGraphDict(TypedDict):
    """AuxiliaryGraph."""

    edges: list[EdgeIdentifier]
    attributes: list[AttributeDict]


class KnowledgeGraphDict(TypedDict):
    """KnowledgeGraph."""

    nodes: dict[CURIE, NodeDict]
    edges: dict[EdgeIdentifier, EdgeDict]


class QueryGraphDict(TypedDict):
    """QueryGraph."""

    nodes: dict[QNodeID, QNodeDict]
    edges: dict[QEdgeID, QEdgeDict]


class QNodeDict(TypedDict):
    """QNode."""

    ids: NotRequired[list[CURIE]]
    categories: NotRequired[list[BiolinkCategory]]
    set_interpretation: NotRequired[SetInterpretationEnum]
    member_ids: NotRequired[list[CURIE]]
    constraints: list[AttributeConstraintDict]


class QEdgeDict(TypedDict):
    """QEdge."""

    knowledge_type: NotRequired[str]
    predicates: NotRequired[BiolinkPredicate]
    subject: CURIE
    object: CURIE
    attribute_constraints: list[AttributeConstraintDict]
    qualifier_constraints: list[QualifierConstraintDict]


class NodeDict(TypedDict):
    """Node."""

    name: NotRequired[str]
    categories: list[BiolinkCategory]
    attributes: list[AttributeDict]
    is_set: NotRequired[bool]


class AttributeDict(TypedDict):
    """Attribute."""

    attribute_type_id: str
    original_attribute_name: NotRequired[str]
    value: Any
    value_type_id: NotRequired[str]
    attribute_source: NotRequired[str]
    value_url: NotRequired[str]
    attributes: NotRequired[list[AttributeDict]]


class EdgeDict(TypedDict):
    """Edge."""

    predicate: BiolinkPredicate
    subject: CURIE
    object: CURIE
    attributes: NotRequired[list[AttributeDict]]
    qualifiers: NotRequired[list[QualifierDict]]
    sources: list[RetrievalSourceDict]


class QualifierDict(TypedDict):
    """Qualifier."""

    qualifier_type_id: str
    qualifier_value: str


class QualifierConstraintDict(TypedDict):
    """QualifierConstraint."""

    qualifier_set: list[QualifierDict]


class MetaKnowledgeGraphDict(TypedDict):
    """MetaKnowledgeGraph."""

    nodes: dict[BiolinkCategory, MetaNodeDict]
    edges: dict[BiolinkPredicate, MetaEdgeDict]


class MetaNodeDict(TypedDict):
    """MetaNode."""

    id_prefixes: list[str]
    attributes: list[MetaAttributeDict]


class MetaEdgeDict(TypedDict):
    """MetaEdge."""

    subject: BiolinkCategory
    predicate: BiolinkPredicate
    object: BiolinkCategory
    knowledge_types: NotRequired[list[str]]
    attributes: NotRequired[list[MetaAttributeDict]]
    qualifiers: NotRequired[list[MetaQualifierDict]]


class MetaQualifierDict(TypedDict):
    """MetaQualifier."""

    qualifier_type_id: str
    applicable_values: list[str]


class MetaAttributeDict(TypedDict):
    """MetaAttribute."""

    attribute_type_id: str
    attribute_source: NotRequired[str]
    original_attribute_name: NotRequired[str]
    constraint_use: bool
    constraint_name: NotRequired[str]


# AttributeConstraint
AttributeConstraintDict = TypedDict(
    "AttributeConstraintDict",
    {
        "id": str,
        "name": str,
        "not": bool,
        "operator": str,
        "value": Hashable,
        "unit_id": NotRequired[str],
        "unit_name": NotRequired[str],
    },
)


class RetrievalSourceDict(TypedDict):
    """RetrievalSource."""

    resource_id: Infores
    resource_role: str
    upstream_resource_ids: NotRequired[list[Infores]]
    source_record_urls: NotRequired[list[str]]


# These don't offer any special behavior, but make type annotation less confusable
CURIE = str
EdgeIdentifier = str
BiolinkCategory = str
BiolinkPredicate = str
AuxGraphID = str
QNodeID = str
QEdgeID = str
Infores = str
