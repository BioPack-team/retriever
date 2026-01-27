# general guidance https://github.com/BioPack-team/retriever/issues/30#issuecomment-3549549249
# scheme reference
# - edge:
#   - https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#qedge-
#   - field name: attribute_constraints
# - node:
#   - https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#qnode-
#   - field name: constraints

# decision tree:
#
#        - check operator
#            - matches : evaluate value( a regex) against `target` field or each element thereof, true if any (OR)
#            - ===     : evaluate type and strict positional equality (AND) must:
#                            publications.keyword == a [a,b,c]
#
#            - ==, >, <, >=, <= :
#                    - `target` field is an array (can only be an array of strings)
#                            |- yes:
#                                - the `value` is
#                                    - an array of strings:
#                                            - operator is ==
#                                                    |- yes:
#                                                        cross check (e.g. "publications", "==",  ["PMID:123", "PMID:234"]), true if any (OR)
#                                                    |- no:
#                                                        throw error, not meaningful comparison (ES does not support check like "publications", ">",  ["PMID:123", "PMID:234"])
#                                    - an array of numbers:
#                                            cross check against array length (e.g. "publications", "==" / ">",  [1,2,3]), true if any (OR)
#                                    - a number:
#                                            evaluate against array length (e.g. "publications", ">=", 2.5)
#                                    - OTHER:
#                                            throw error, not a meaningful comparison (e.g. "publications", ">", [1, "PMID:123", 5.5])"
#
#                            | - no:
#                                - cross check meaningful comparison (i.e. of same type), true if any (OR)
#                                    examples:
#                                        - meaningful:
#                                            - "original_predicate", "==", "RO:001231
#                                            - "original_predicate", ">", ["RO:001231"] (extraneous, but lexically meaningful)
#                                        - not meaningful:
#                                            - "original_predicate", ">=", 5


from datetime import datetime
from typing import Any

from retriever.data_tiers.tier_1.elasticsearch.constraints.attributes.meta_info import (
    EDGE_ATTR_META,
    NODE_ATTR_META,
)
from retriever.data_tiers.tier_1.elasticsearch.constraints.attributes.ops.handle_comparison import (
    handle_negation,
    handle_simple_comparison,
)
from retriever.data_tiers.tier_1.elasticsearch.constraints.attributes.ops.handle_match import (
    handle_match,
)

# from functools import partial
from retriever.data_tiers.tier_1.elasticsearch.constraints.types.attribute_types import (
    AttrFieldMeta,
    AttributeFilterQuery,
    AttributeOrigin,
    AttrValType,
    SingleAttributeFilterQueryPayload,
)
from retriever.types.trapi import AttributeConstraintDict
from retriever.utils import biolink

# sample
# {
#   "id": "biolink:publications",
#   "name": "Must have 3+ publications",
#   "operator": OperatorEnum.GT,
#   "value": [2,3],
# }


def validate_constraint(constraint: AttributeConstraintDict) -> None:
    """Validate attribute constraint for fields."""
    required_fields = ["id", "operator", "value"]

    for field in required_fields:
        if field not in constraint:
            raise AttributeError(f"Attribute constraint must have the field {field}")


def validate_operator(operator: Any) -> None:
    """Validate allowed operator in attribute constraints."""
    allowed_ops = {"matches", "===", "==", ">", "<", ">=", "<="}

    if not isinstance(operator, str) or operator not in allowed_ops:
        raise AttributeError(f"Operator must be one of {allowed_ops}")


def validate_date(candidate: Any) -> str | int | None:
    """Validate date payload.

    only supports:
    - str: must be in or can be formatted into YYYY-MM-DD, or
    - int: treated as an Epoch number
    """
    if isinstance(candidate, str):
        try:
            parsed = datetime.strptime(candidate, "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            # only allow YYYY-MM-DD
            return None
    if isinstance(candidate, int):
        return candidate

    return None


def ensure_type_consistency(field_type: AttrValType, raw_value: Any) -> Any:
    """Check type between compared values. Transform date string where possible."""
    if field_type == "date":
        return validate_date(raw_value)

    if field_type == "num" and type(raw_value) not in [int, float]:
        return None

    if field_type in ["keyword", "text"] and not isinstance(raw_value, str):
        return None

    return raw_value


def process_single_constraint(
    constraint: AttributeConstraintDict,
    origin: AttributeOrigin,
) -> SingleAttributeFilterQueryPayload | None:
    """Process a single attribute constraint."""
    validate_constraint(constraint)

    # todo unit?
    raw_operator = constraint.get("operator")
    raw_value = constraint.get("value")
    should_negate = constraint.get("not", False)
    target_field_name = biolink.rmprefix(constraint.get("id"))

    validate_operator(raw_operator)

    attr_meta = EDGE_ATTR_META if origin == "edge" else NODE_ATTR_META

    field_meta_info: AttrFieldMeta | None = attr_meta.get(target_field_name, None)

    if field_meta_info is None:
        # todo consider not met?
        return None

    if raw_operator == "matches":
        # regex is not easily reversed

        return SingleAttributeFilterQueryPayload(
            query=handle_match(field_meta_info, raw_value, target_field_name),
            negate=should_negate,
        )

    # scalar vs scalar
    if not isinstance(raw_value, list) and field_meta_info["container"] != "array":
        # ensure type consistency
        value = ensure_type_consistency(field_meta_info["value_type"], raw_value)

        if value is None:
            if field_meta_info["value_type"] == "date":
                raise AttributeError(
                    f"{raw_value} is not a supported date. Must be in YYYY-MM-DD format"
                )

            raise AttributeError(
                f"{field_meta_info['value_type']} field {target_field_name} is being compared with f{type(raw_value)}"
            )

        # ensure meaningful comparison
        if field_meta_info["value_type"] == "text" and raw_operator != "==":
            raise AttributeError(
                f"text field{target_field_name} only supports loose match operator '==' "
            )

        # translate negation where possible
        raw_operator, should_negate = handle_negation(raw_operator, should_negate)

        # pass to translator
        return SingleAttributeFilterQueryPayload(
            query=handle_simple_comparison(value, raw_operator, target_field_name),
            negate=should_negate,
        )
    # field: scalar, attr: array
    # elif isinstance(raw_value, list):
    #     # rule out meaningless op:
    #     if raw_operator == "===":
    #         raise AttributeError(f"field {target_field_name} is a scalar, cannot be compared with a list using ===")
    #
    #     value_processor = partial(ensure_type_consistency, field_type=field_meta_info['value_type'])
    #     values = list(filter(None, map(value_processor, raw_value)))
    #
    #     # zero valid value to compared to
    #     if not values:
    #         return None
    #
    #     # translate negation where possible
    #     raw_operator, should_negate = handle_negation(raw_operator, should_negate)
    #
    #     if raw_operator == "==":
    #         return {
    #             "terms": {
    #                 target_field_name: values
    #             },
    #         }

    # we will handle more complex filtering at a later time
    # post filtering needed for some cases
    else:
        return None


def process_attribute_constraints(
    constraints: list[AttributeConstraintDict],
    origin: AttributeOrigin = "edge",
) -> tuple[list[AttributeFilterQuery], list[AttributeFilterQuery]]:
    """Generate ES query for attribute constraint field."""
    must: list[AttributeFilterQuery] = []
    must_not: list[AttributeFilterQuery] = []

    # fail fast. exception => not met => fails everything
    for constraint in constraints:
        # attribute error will ba raised if illegal
        payload = process_single_constraint(constraint, origin)

        # None will be returned if not a supported filtering
        if not payload:
            raise AttributeError(f"Constraint not currently supported: {constraint}.")

        if payload["negate"]:
            must_not.append(payload["query"])
        else:
            must.append(payload["query"])

    return must, must_not
