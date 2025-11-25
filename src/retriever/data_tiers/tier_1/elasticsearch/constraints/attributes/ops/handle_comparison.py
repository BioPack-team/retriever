from typing import cast

from retriever.data_tiers.tier_1.elasticsearch.attribute_types import ComparisonOperator, ESValueComparisonQuery, \
     ESTermComparisonClause


def get_operator(raw_operator: str) -> ComparisonOperator | str:
    op_mapping: dict[str, ComparisonOperator] = {
        ">": "gt",
        "<": "lt",
        ">=": "gte",
        "<=": "lte",
    }

    if raw_operator in op_mapping:
        return op_mapping[raw_operator]

    # otherwise unchanged
    return raw_operator

def handle_simple_comparison(
        value: int | float | str,
        raw_operator: str,
        target_field_name: str
) -> ESValueComparisonQuery | ESTermComparisonClause:
    if raw_operator in ["==", "==="]:
        return ESTermComparisonClause(
            term={
                target_field_name: value
            }
        )

    operator = cast(ComparisonOperator, get_operator(raw_operator))

    return ESValueComparisonQuery(
        range={
            target_field_name: {
                operator: value
            }
        }
    )


def handle_negation(raw_operator: str, should_negate: bool):
    if should_negate and raw_operator not in ["==", "==="]:
        if raw_operator[0] == ">":
            negated_operator = "<"
        else:
            negated_operator = '>'

        if len(raw_operator) == 1:
            negated_operator += '='

        raw_operator = negated_operator
        should_negate = False

    return raw_operator, should_negate
