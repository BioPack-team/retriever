from typing import TypedDict, Literal, NotRequired, Union

AttrValType = Literal["text", "num", "date", "keyword"]



class AttrFieldMeta(TypedDict):
    container: Literal["scalar", "array"]
    value_type: AttrValType
    curie: bool


ComparisonOperator = Literal["lt", "gt", "lte", "gte"]
ComparisonValue = Union[str, int, float]

ComparisonTerm = dict[ComparisonOperator, ComparisonValue]

class ESValueComparisonQuery(TypedDict):
    range: dict[str, ComparisonTerm]


class ESTermComparisonClause(TypedDict):
    term: dict[str, str| int | float]


class RegexTerm(TypedDict):
    value: str
    case_sensitive: NotRequired[bool]

class ESRegexQuery(TypedDict):
    regexp: dict[str, RegexTerm]