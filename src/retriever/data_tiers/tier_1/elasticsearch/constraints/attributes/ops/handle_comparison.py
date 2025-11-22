def get_operator(raw_operator: str):
    op_mapping = {
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
        value: any,
        raw_operator: str,
        target_field_name: str
):
    if raw_operator in ["==", "==="]:
        return {
            "term": {
                target_field_name: value
            }
        }

    operator = get_operator(raw_operator)

    return {
        "range": {
            target_field_name: {
                operator: value,
            }
        }
    }


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
