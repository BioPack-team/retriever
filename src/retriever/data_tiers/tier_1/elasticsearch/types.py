from dataclasses import dataclass
from typing import Any, NotRequired, Self, TypedDict

import orjson

from retriever.data_tiers.tier_1.elasticsearch.attribute_types import (
    AttributeFilterQuery,
)
from retriever.data_tiers.tier_1.elasticsearch.qualifier_types import (
    ESQueryForSingleQualifierConstraint,
    ESTermClause,
)
from retriever.data_tiers.utils import (
    DINGO_KG_EDGE_TOPLEVEL_VALUES,
    DINGO_KG_NODE_TOPLEVEL_VALUES,
)
from retriever.types.trapi import (
    RetrievalSourceDict,
)
from retriever.utils import biolink


class ESFilterClause(TypedDict):
    """An Elasticsearch filter clause."""

    terms: dict[str, list[str]]


class ESBooleanQuery(TypedDict):
    """An Elasticsearch boolean query."""

    filter: list[ESFilterClause | ESTermClause]
    should: NotRequired[list[ESQueryForSingleQualifierConstraint | ESTermClause]]
    minimum_should_match: NotRequired[int]
    must: NotRequired[list[AttributeFilterQuery]]
    must_not: NotRequired[list[AttributeFilterQuery]]


class ESQueryContext(TypedDict):
    """An Elasticsearch query context."""

    bool: ESBooleanQuery


class ESPayload(TypedDict):
    """An Elasticsearch query payload."""

    query: ESQueryContext


class ESDocument(TypedDict):
    """A source document returned from Elasticsearch."""

    _source: dict[str, Any]
    _index: NotRequired[str]
    sort: list[Any]


class ESHits(TypedDict):
    """A collection of Elasticsearch documents returned as hits."""

    hits: list[ESDocument]


class ESResponse(TypedDict):
    """An Elasticsearch query response."""

    hits: ESHits


@dataclass(frozen=True, kw_only=True, slots=True)
class ESNode:
    """A knowledge node as represented by ES, with some convenience features."""

    id: str
    name: str
    category: list[str]
    attributes: dict[str, Any]

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> Self:
        """Parse part of an ES document as an Edge."""
        attributes = dict[str, Any]()
        for key, value in doc.items():
            if key in DINGO_KG_NODE_TOPLEVEL_VALUES:
                continue
            else:
                attributes[key] = value

        return cls(
            id=doc.get("id", "NOT_PROVIDED"),
            name=doc.get("name", "NOT_PROVIDED"),
            category=doc.get("category", "NOT_PROVIDED"),
            attributes=attributes,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class ESEdge:
    """Edge information as represented by ES, with some convenience features."""

    _index: str | None
    id: str
    subject: ESNode
    object: ESNode
    predicate: str
    predicate_ancestors: list[str]
    sources: list[RetrievalSourceDict]
    source_inforeses: list[str]
    qualifiers: dict[str, str]
    attributes: dict[str, Any]

    @classmethod
    def from_dict(cls, doc: ESDocument) -> Self:
        """Parse an ES document as an Edge."""
        qualifiers = dict[str, str]()
        attributes = dict[str, Any]()
        for key, value in doc["_source"].items():
            if key in DINGO_KG_EDGE_TOPLEVEL_VALUES:
                continue
            if biolink.is_qualifier(key):
                if not isinstance(value, str):
                    qualifiers[key] = orjson.dumps(value).decode()
                else:
                    qualifiers[key] = str(value)
            else:
                attributes[key] = value

        sbj_node = ESNode.from_dict(doc["_source"]["subject"])
        obj_node = ESNode.from_dict(doc["_source"]["object"])

        return cls(
            _index=doc.get("_index"),
            id=doc["_source"].get("id", "NOT_PROVIDED"),
            subject=sbj_node,
            object=obj_node,
            predicate=doc["_source"].get("predicate", "related_to"),
            predicate_ancestors=doc["_source"].get(
                "predicate_ancestors", ["related_to"]
            ),
            sources=doc["_source"].get("sources", []),
            source_inforeses=doc["_source"].get("source_inforeses", []),
            qualifiers=qualifiers,
            attributes=attributes,
        )
