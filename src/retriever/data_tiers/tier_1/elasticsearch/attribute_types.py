from typing import Literal, NotRequired, TypedDict

AttrValType = Literal["text", "num", "date", "keyword"]
AttrContainerType = Literal["scalar", "array"]


class AttrFieldMeta(TypedDict):
    """Stored attribute field metadata."""

    container: AttrContainerType
    value_type: AttrValType
    curie: bool


ComparisonOperator = Literal["lt", "gt", "lte", "gte"]
ComparisonValue = str | int | float

ComparisonTerm = dict[ComparisonOperator, ComparisonValue]


class ESValueComparisonQuery(TypedDict):
    """ES comparison term wrapper."""

    range: dict[str, ComparisonTerm]


class ESTermComparisonClause(TypedDict):
    """ES comparison clause."""

    term: dict[str, str | int | float]


class RegexTerm(TypedDict):
    """ES Regex clause.."""

    value: str
    case_sensitive: NotRequired[bool]


class ESRegexQuery(TypedDict):
    """ES Query content for `match` filter."""

    regexp: dict[str, RegexTerm]


AttributeFilterQuery = ESRegexQuery | ESTermComparisonClause | ESValueComparisonQuery


class SingleAttributeFilterQueryPayload(TypedDict):
    """ES query generated based on one single attribute constraint."""

    query: AttributeFilterQuery
    negate: bool
