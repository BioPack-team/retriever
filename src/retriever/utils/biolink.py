import re
from functools import lru_cache
from typing import cast

import bmt
from reasoner_pydantic import BiolinkQualifier

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.types.trapi import (
    AttributeConstraintDict,
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


def get_descendants[T: str](value: T) -> list[T]:
    """Get the descendants for a given biolink concept."""
    return cast(list[T], biolink.get_descendants(value, formatted=True))


def get_descendant_values(qualifier_type: BiolinkQualifier, value: str) -> set[str]:
    """Given a biolink qualifier and an associated value, return applicable descendant values."""
    if "predicate" in qualifier_type:
        return {rmprefix(predicate) for predicate in expand(value)}

    permissible_values: set[str] = {value}
    for enum_name, enum_def in biolink.view.all_enums().items():
        if value in cast(dict[str, object], enum_def.permissible_values or {}):
            permissible_values.update(
                biolink.get_permissible_value_descendants(value, enum_name)
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
            elif qualifier["qualifier_type_id"] == "biolink:qualified_predicate":
                inverse = get_inverse(BiolinkPredicate(qualifier["qualifier_value"]))
                if inverse is None:
                    # Can't correctly reverse the constraint; signal the caller so
                    # it can skip the reverse subquery instead of emitting an invalid one.
                    raise ValueError(
                        f"qualified_predicate '{qualifier['qualifier_value']}' has no "
                        + "inverse; qualifier constraint cannot be reversed."
                    )
                new_qualifier["qualifier_value"] = inverse
            new_qualifier_set.append(new_qualifier)
        new.append(QualifierConstraintDict(qualifier_set=new_qualifier_set))
    return new


_SUBJECT_RE = re.compile(r"(?<![a-zA-Z])subject(?![a-zA-Z])")
_OBJECT_RE = re.compile(r"(?<![a-zA-Z])object(?![a-zA-Z])")


def _swap_subject_object(value: str) -> str:
    """Swap 'subject'<->'object' in a directional attribute type token."""
    if _OBJECT_RE.search(value):
        value = _OBJECT_RE.sub("subject", value)
    return value


def reverse_attribute_constraints(
    attribute_constraints: list[AttributeConstraintDict],
) -> list[AttributeConstraintDict]:
    """Reverse a given list of edge attribute constraints.

    Mostly just consists of swapping subject/object.
    """
    new = list[AttributeConstraintDict]()
    for constraint in attribute_constraints:
        new_constraint: AttributeConstraintDict = {**constraint}
        new_constraint["id"] = _swap_subject_object(constraint["id"])
        new_constraint["name"] = _swap_subject_object(constraint["name"])
        new.append(new_constraint)
    return new


is_qualifier = biolink.is_qualifier
is_symmetric = biolink.is_symmetric
is_predicate = biolink.is_predicate
is_category = biolink.is_category
