from typing import Any, NotRequired, TypedDict

from retriever.data_tiers.tier_1.elasticsearch.attribute_types import (
    AttributeFilterQuery,
)
from retriever.data_tiers.tier_1.elasticsearch.qualifier_types import (
    ESQueryForSingleQualifierConstraint,
    ESTermClause,
)
from retriever.types.trapi import CURIE


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


class ESPublicationsInfo(TypedDict):
    """Information regarding publications."""

    pmid: str
    publication_date: NotRequired[str]
    sentence: NotRequired[str]
    subject_score: NotRequired[str]
    object_score: NotRequired[str]


class ESSourceInfo(TypedDict):
    """Information regarding sources."""

    resource_id: str
    resource_role: str
    upstream_resource_ids: NotRequired[list[str]]
    source_record_urls: NotRequired[list[str]]


class ESNode(TypedDict):
    """A knowledge node as represented in Elasticsearch."""

    id: CURIE
    name: str
    category: str
    description: str
    equivalent_identifiers: list[str]
    in_taxon: NotRequired[list[str]]
    information_content: NotRequired[float]
    inheritance: NotRequired[str]
    provided_by: NotRequired[list[str]]


class ESHit(TypedDict):
    """The main data of an Elasticsearch hit."""

    _index: NotRequired[str]
    subject: ESNode
    object: ESNode
    predicate: str
    sources: list[ESSourceInfo]
    id: NotRequired[str]
    agent_type: NotRequired[str]
    knowledge_level: NotRequired[str]
    publications: list[str]
    qualified_predicate: NotRequired[str]
    predicate_ancestors: list[str]
    source_inforeses: list[str]
    subject_form_or_variant_qualifier: NotRequired[str]
    disease_context_qualifier: NotRequired[str]
    frequency_qualifier: NotRequired[str]
    onset_qualifier: NotRequired[str]
    sex_qualifier: NotRequired[str]
    original_subject: NotRequired[str]
    original_predicate: NotRequired[str]
    original_object: NotRequired[str]
    allelic_requirement: NotRequired[str]
    update_date: NotRequired[str]
    z_score: NotRequired[float]
    has_evidence: list[str]
    has_confidence_score: NotRequired[float]
    has_count: NotRequired[float]
    has_total: NotRequired[float]
    has_percentage: NotRequired[float]
    has_quotient: NotRequired[float]
    category: list[str]
    seq_: NotRequired[int]
    negated: NotRequired[bool]


class ESDocument(TypedDict):
    """A source document returned from Elasticsearch."""

    _source: ESHit
    _index: NotRequired[str]
    sort: list[Any]


class ESHits(TypedDict):
    """A collection of Elasticsearch documents returned as hits."""

    hits: list[ESDocument]


class ESResponse(TypedDict):
    """An Elasticsearch query response."""

    hits: ESHits
