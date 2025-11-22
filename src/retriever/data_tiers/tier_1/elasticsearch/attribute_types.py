from typing import TypedDict, Literal, NotRequired


AttrValType = Literal["text", "num", "date", "keyword"]

class AttrFieldMeta(TypedDict):
    container: Literal["scalar", "array"]
    value_type: AttrValType
    curie: bool


class RegexTerm(TypedDict):
    value: str
    case_sensitive: NotRequired[bool]

class ESRegexQuery(TypedDict):
    regexp: dict[str, RegexTerm]