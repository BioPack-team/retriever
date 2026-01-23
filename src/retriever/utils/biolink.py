from functools import lru_cache

import bmt
from reasoner_pydantic import BiolinkQualifier

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.types.trapi import (
    BiolinkPredicate,
    QualifierConstraintDict,
    QualifierDict,
    QualifierTypeID,
)

biolink = bmt.Toolkit(
    schema=f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{OPENAPI_CONFIG.x_translator.biolink_version}/biolink-model.yaml",
    predicate_map=f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{OPENAPI_CONFIG.x_translator.biolink_version}/predicate_mapping.yaml",
)


def ensure_prefix(item: str) -> str:
    """Add a `biolink:` prefix to the given string.

    Replaces the prefix if it's already present.
    """
    return f"biolink:{rmprefix(item)}"


def rmprefix(item: str) -> str:
    """Remove the `biolink:` prefix from the given string.

    Returns the string if it has no prefix.
    """
    return item.removeprefix("biolink:")


def expand(items: str | set[str]) -> set[str]:
    """Safely expand a set of biolink categories or predicates to their descendants.

    Accepts either with or without biolink prefix, but always outputs with biolink prefix.
    """
    initial = {items} if isinstance(items, str) else items
    expanded = set(initial)
    for item in initial:
        expanded.update(biolink.get_descendants(item, formatted=True))
    return expanded


@lru_cache
def get_all_qualifiers() -> set[str]:
    """Return all qualifiers in the biolink model."""
    slots = biolink.get_all_edge_properties()
    return {
        slot.replace(" ", "_")
        for slot in slots
        if biolink.is_qualifier(slot) and slot != "qualifier"
    }


def get_inverse(predicate: BiolinkPredicate) -> BiolinkPredicate | None:
    """Return the inverse of a given predicate."""
    inverse = biolink.get_inverse_predicate(predicate, formatted=True)
    return BiolinkPredicate(inverse) if inverse else None


def get_descendant_values(qualifier_type: BiolinkQualifier, value: str) -> set[str]:
    """Given a biolink qualifier and an associated value, return applicable descendant values."""
    ranges = biolink.get_slot_range(qualifier_type)  # pyright:ignore[reportUnknownMemberType]

    # Handle qualified_predicate
    if "predicate" in qualifier_type:
        return {rmprefix(predicate) for predicate in expand(value)}

    permissible_values: set[str] = {value}
    for value_type in ranges:
        if biolink.is_enum(value_type):
            permissible_values.update(
                biolink.get_permissible_value_descendants(value, value_type)
            )

    return permissible_values


def reverse_qualifier_constraints(
    qualifier_constraints: list[QualifierConstraintDict],
) -> list[QualifierConstraintDict]:
    """Reverse a given list of qualifier constraints."""
    new = list[QualifierConstraintDict]()
    for constraint in qualifier_constraints:
        new_qualifier_set = list[QualifierDict]()
        for qualifier in constraint["qualifier_set"]:
            new_qualifier = QualifierDict(**qualifier)
            if "object" in qualifier["qualifier_type_id"]:
                new_qualifier["qualifier_type_id"] = QualifierTypeID(
                    qualifier["qualifier_type_id"].replace("object", "subject")
                )
            elif "subject" in qualifier["qualifier_type_id"]:
                new_qualifier["qualifier_type_id"] = QualifierTypeID(
                    qualifier["qualifier_type_id"].replace("subject", "object")
                )
            elif inverse := qualifier[
                "qualifier_type_id"
            ] == "biolink:qualified_predicate" and get_inverse(
                BiolinkPredicate(qualifier["qualifier_value"])
            ):
                # BUG: Technically invalid if we can't reverse the predicate
                # but this is vanishingly rare and not worth addressing right now
                new_qualifier["qualifier_value"] = inverse
            new_qualifier_set.append(new_qualifier)
        new.append(constraint)
    return new


is_qualifier = biolink.is_qualifier
is_symmetric = biolink.is_symmetric
is_predicate = biolink.is_predicate
is_category = biolink.is_category
