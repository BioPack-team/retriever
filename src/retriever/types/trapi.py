"""TRAPI as represented in TypedDict.

More performant than reasoner-pydantic, but doesn't offer validation or other utility.
To be used only for TRAPI constructed by Retriever, that doesn't need validation.
This way we get the performance of pure dict, with static type checking.

A few notes:
    - Individual class docstrings solely refer to their name in the TRAPI Spec.
    - Because this is only for internal use, some items that are reducible to strings
      (e.g. CURIES, etc.) are simply annotated as `str`
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import NotRequired, TypedDict

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
    auxiliary_graphs: NotRequired[AuxGraphDict]


class LogEntryDict(TypedDict):
    """LogEntry."""

    timestamp: str
    level: NotRequired[LogLevel]
    code: NotRequired[str]
    message: str


class ResultDict(TypedDict):
    """Result."""

    node_bindings: dict[str, list[NodeBindingDict]]
    analyses: list[AnalysisDict]


class NodeBindingDict(TypedDict):
    """NodeBinding."""

    id: str
    query_id: NotRequired[str]
    attributes: list[AttributeDict]


class AnalysisDict(TypedDict):
    """Analysis."""

    resource_id: str
    score: NotRequired[float]
    edge_bindings: dict[str, list[EdgeBindingDict]]
    support_graphs: NotRequired[list[str]]
    scoring_method: NotRequired[str]
    attributes: NotRequired[list[AttributeDict]]


class EdgeBindingDict(TypedDict):
    """EdgeBinding."""

    id: str
    attributes: list[AttributeDict]


class AuxGraphDict(TypedDict):
    """AuxiliaryGraph."""

    edges: list[str]
    attributes: list[AttributeDict]


class KnowledgeGraphDict(TypedDict):
    """KnowledgeGraph."""

    nodes: dict[str, NodeDict]
    edges: dict[str, EdgeDict]


class QueryGraphDict(TypedDict):
    """QueryGraph."""

    nodes: dict[str, QNodeDict]
    edges: dict[str, QEdgeDict]


class QNodeDict(TypedDict):
    """QNode."""

    ids: NotRequired[list[str]]
    categories: NotRequired[list[str]]
    set_interpretation: NotRequired[SetInterpretationEnum]
    member_ids: NotRequired[list[str]]
    constraints: list[AttributeConstraintDict]


class QEdgeDict(TypedDict):
    """QEdge."""

    knowledge_type: NotRequired[str]
    predicates: NotRequired[str]
    subject: str
    object: str
    attribute_constraints: list[AttributeConstraintDict]
    qualifier_constraints: list[QualifierConstraintDict]


class NodeDict(TypedDict):
    """Node."""

    name: NotRequired[str]
    categories: list[str]
    attributes: list[AttributeDict]
    is_set: NotRequired[bool]


class AttributeDict(TypedDict):
    """Attribute."""

    attribute_type_id: str
    original_attribute_name: NotRequired[str]
    value: Hashable
    value_type_id: NotRequired[str]
    attribute_source: NotRequired[str]
    value_url: NotRequired[str]
    attributes: NotRequired[list[AttributeDict]]


class EdgeDict(TypedDict):
    """Edge."""

    predicate: str
    subject: str
    object: str
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

    nodes: dict[str, MetaNodeDict]
    edges: dict[str, MetaEdgeDict]


class MetaNodeDict(TypedDict):
    """MetaNode."""

    id_prefixes: list[str]
    attributes: list[MetaAttributeDict]


class MetaEdgeDict(TypedDict):
    """MetaEdge."""

    subject: str
    predicate: str
    object: str
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

    resource_id: str
    resource_role: str
    upstream_resource_ids: NotRequired[list[str]]
    source_record_urls: NotRequired[list[str]]
