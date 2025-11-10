from typing import Any, NotRequired, TypedDict

from retriever.types.trapi import CURIE, Infores


class ESFilterClause(TypedDict):
    """An Elasticsearch filter clause."""

    terms: dict[str, list[str]]


class ESBooleanQuery(TypedDict):
    """An Elasticsearch boolean query."""

    filter: list[ESFilterClause]


class ESQueryContext(TypedDict):
    """An Elasticsearch query context."""

    bool: ESBooleanQuery


class ESPayload(TypedDict):
    """An Elasticsearch query payload."""

    query: ESQueryContext


class ESPublicationsInfo(TypedDict):
    """Information regarding publications."""

    pmid: str
    publication_date: NotRequired[str]
    sentence: NotRequired[str]
    subject_score: NotRequired[str]
    object_score: NotRequired[str]


class ESNode(TypedDict):
    """A knowledge node as represented in Elasticsearch."""

    id: CURIE
    name: str
    category: str
    all_names: NotRequired[list[str]]
    all_categories: list[str]
    iri: NotRequired[str]
    description: str
    equivalent_curies: list[str]
    publications: NotRequired[list[str]]


class ESHit(TypedDict):
    """The main data of an Elasticsearch hit."""

    subject: ESNode
    object: ESNode
    predicate: str
    primary_knowledge_source: Infores
    publications: NotRequired[list[str]]
    publications_info: NotRequired[list[ESPublicationsInfo | None]]
    kg2_ids: list[str]
    domain_range_exclusion: bool
    knowledge_level: str | None
    agent_type: str | None
    id: int
    qualified_object_aspect: NotRequired[str]
    qualified_object_direction: NotRequired[str]
    qualified_predicate: NotRequired[str]


class ESDocument(TypedDict):
    """A source document returned from Elasticsearch."""

    _source: ESHit
    sort: list[Any]


class ESHits(TypedDict):
    """A collection of Elasticsearch documents returned as hits."""

    hits: list[ESDocument]


class ESResponse(TypedDict):
    """An Elasticsearch query response."""

    hits: ESHits
