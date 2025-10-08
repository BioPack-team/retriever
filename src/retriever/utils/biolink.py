from functools import lru_cache

import bmt

biolink = bmt.Toolkit()


def expand(items: set[str]) -> set[str]:
    """Expand a set of biolink categories or predicates."""
    expanded = set(items)
    for item in items:
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
