import re
from typing import Any

from retriever.data_tiers.tier_1.elasticsearch.attribute_types import (
    AttrFieldMeta,
    ESRegexQuery,
    RegexTerm,
)


def validate_regex(pattern: Any) -> str:
    """Validate regex for `match` query."""
    # filter empty regex
    if not isinstance(pattern, str) or pattern.strip() == "":
        raise ValueError("Regex term cannot be empty")

    # filter long regex
    max_allowed_regex_length = 50
    if len(pattern) > max_allowed_regex_length:
        raise ValueError(
            f"Regex term cannot exceed {max_allowed_regex_length} characters"
        )

    pattern = pattern.strip()

    # filter leading wildcard
    if re.match(r"^\(*(?:(?<!\\)\.|(?<!\\)\[[^]]+]|\([^)]+\))(?<!\\)[*+?{]", pattern):
        raise ValueError("Regex with leading wildcard (.* or .+) is not supported")

    # filter nested quantifier
    if re.search(r"\([^)]+(?<!\\)[*+?{]\)(?<!\\)[*+?{]", pattern):
        raise ValueError("Regex with nested quantifiers is not supported")

    # filter lazy quantifier
    if re.search(r"(?<!\\)([*+?])\?|(?<!\\)}\?", pattern):
        raise ValueError("Regex with lazy quantifier is not supported")

    # unsupported types in ES
    # 0. anchor
    if re.search(r"(?<!\\)[\^$]", pattern):
        raise ValueError("ES does not support anchor operators")
    # 1. lookaround
    if re.search(r"\(\?(?<!\\)[=!<]", pattern):
        raise ValueError("ES does not support lookaround")
    # 2. escapes
    if re.search(r"(?<!\\)\\([dDsSwWbB])", pattern):
        raise ValueError(r"ES regex does not support escapes like \d, \w, \s.")

    try:
        re.compile(pattern)
    except re.error as err:
        raise ValueError("Not a valid regular expression") from err

    return pattern


def handle_match(
    field_meta_info: AttrFieldMeta, raw_value: Any, target_field_name: str
) -> ESRegexQuery:
    """Generate ES query term for match-regex constraint."""
    # `matches`

    # 0. rule out meaningless queries
    if field_meta_info["value_type"] not in ["keyword", "text"]:
        """
        !!! be very careful with regex against text field.
        While it's allowed, A regex term `GO:` will not match the text `We have GO:123` because of standard tokenizer used by ES
        """
        raise TypeError(
            f"{field_meta_info['value_type']} fields does not support RegEx query"
        )

    # 1. check if valid / efficient regex
    regex_term = validate_regex(raw_value)

    # 2. generate ES query

    """
    example
    {
        "regexp": {
            agent_type: {
                "value": "manual_*",
                "case_insensitive": True
            }
        }
    }
    """

    query_term = ESRegexQuery(regexp={target_field_name: RegexTerm(value=regex_term)})

    return query_term
