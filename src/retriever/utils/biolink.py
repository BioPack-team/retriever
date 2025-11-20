from functools import lru_cache

import bmt

from retriever.config.openapi import OPENAPI_CONFIG

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


is_qualifier = biolink.is_qualifier
