from typing import TypedDict


class ESTermClause(TypedDict):
    """An Elasticsearch term clause."""

    term: dict[str, str]


class ESQualifierBooleanQuery(TypedDict):
    """Must container for matching both fields within one qualifier (type_id, value)"""

    must: list[ESTermClause]


class ESQualifierQuery(TypedDict):
    """Bool container for `and` relationships of both fields"""

    bool: ESQualifierBooleanQuery


class ESNestedQuery(TypedDict):
    """Full nested field query generated for one qualifier"""

    path: str
    query: ESQualifierQuery


class ESQueryForOneQualifierEntry(TypedDict):
    """Nested query container for one qualifier"""

    nested: ESNestedQuery


class ESBoolQueryForSingleQualifierConstraint(TypedDict):
    """Bool query combining nested queries for qualifiers for one constraint."""

    must: list[ESQueryForOneQualifierEntry]


class ESQueryForSingleQualifierConstraint(TypedDict):
    """Bool container for `and` relationship between qualifiers within one constraint."""

    bool: ESBoolQueryForSingleQualifierConstraint


class ESConstraintsChainedQuery(TypedDict):
    """Query entry specifying an OR relationship between constraints"""

    should: list[ESQueryForSingleQualifierConstraint | ESQueryForOneQualifierEntry]


class ESConstraintsQueryContext(TypedDict):
    """Boolean wrapper for chained queries generated for a list of constraints"""

    bool: ESConstraintsChainedQuery | ESQueryForSingleQualifierConstraint | ESQueryForOneQualifierEntry


class ESConstraintsQuery(TypedDict):
    """Full query for a list of constraints."""

    query: ESConstraintsQueryContext
