from typing import TypedDict


class ESTermClause(TypedDict):
    """An Elasticsearch term clause."""

    term: dict[str, str]


class ESBoolQueryForSingleQualifierConstraint(TypedDict):
    """Bool query combining nested queries for qualifiers for one constraint."""

    must: list[ESTermClause]


class ESQueryForSingleQualifierConstraint(TypedDict):
    """Bool container for `and` relationship between qualifiers within one constraint."""

    bool: ESBoolQueryForSingleQualifierConstraint


ESQualifierConstraintQuery = ESQueryForSingleQualifierConstraint | ESTermClause


class ESConstraintsChainedQuery(TypedDict):
    """Query entry specifying an OR relationship between constraints."""

    should: list[ESQualifierConstraintQuery]
