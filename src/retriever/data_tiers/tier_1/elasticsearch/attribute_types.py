from typing import TypedDict, Literal, NotRequired


class AttrFieldMeta(TypedDict):
    container: Literal["scalar", "array"]
    value_type: Literal["bool", "text", "num", "date", "keyword"]
    curie: bool


class RegexTerm(TypedDict):
    value: str
    case_sensitive: NotRequired[bool]

class ESRegexQuery(TypedDict):
    regexp: dict[str, RegexTerm]