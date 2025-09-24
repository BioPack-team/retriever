from typing import NotRequired, TypedDict

from retriever.types.trapi import CURIE, BiolinkEntity, BiolinkPredicate, Infores


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
    publication_date: str
    sentence: str
    subject_score: str


class ESNode(TypedDict):
    """A knowledge node as represented in Elasticsearch."""

    id: CURIE
    name: str
    category: BiolinkEntity
    all_names: list[str]
    all_categories: list[BiolinkEntity]
    iri: NotRequired[str]
    description: str
    equivalent_curies: str
    publications: NotRequired[list[str]]


class ESHit(TypedDict):
    """The main data of an Elasticsearch hit."""

    subject: ESNode
    object: ESNode
    predicate: BiolinkPredicate
    primary_knowledge_source: Infores
    publications: NotRequired[list[str]]
    publications_info: NotRequired[list[ESPublicationsInfo]]
    kg2_ids: list[str]
    domain_range_exclusion: bool
    knowledge_level: str
    agent_type: str
    id: int
    qualified_object_aspect: NotRequired[str]
    qualified_object_direction: NotRequired[str]
    qualified_predicate: NotRequired[str]
