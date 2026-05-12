from typing import NotRequired, TypedDict


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


class ESEquivalentQualifierPair(TypedDict):
    """Wrapper for equivalent qualifier constraint pairs."""

    should: list[ESTermClause]
    minimum_should_match: NotRequired[int]


class ESEquivalentQualifierPairCollection(TypedDict):
    """Wrapper for equivalent qualifier constraint pairs."""

    bool: ESEquivalentQualifierPair


class ESMustQueryForExpandedQualifiers(TypedDict):
    """Must query combining nested queries for qualifiers for one constraint with proper expansion."""

    must: list[ESEquivalentQualifierPairCollection]


class ESBoolQueryForExpandedQualifiers(TypedDict):
    """Bool wrapper for ESMustQueryForExpandedQualifiers."""

    bool: ESMustQueryForExpandedQualifiers


class ESConstraintsChainedQuery(TypedDict):
    """Query entry specifying an OR relationship between constraints."""

    should: list[ESEquivalentQualifierPairCollection | ESBoolQueryForExpandedQualifiers]
