from functools import lru_cache

import bmt

biolink = bmt.Toolkit()


def expand(items: str | set[str]) -> set[str]:
    """Safely expand a set of biolink categories or predicates to their descendants.

    Accepts either with or without biolink prefix, but always outputs with biolink prefix.
    """
    initial = {items} if isinstance(items, str) else items
    expanded = set(initial)
    for item in initial:
        # Have to strip biolink prefix due to bug in bmt, see https://github.com/biolink/biolink-model-toolkit/issues/154
        lookup = item.replace("biolink:", "")
        expanded.update(biolink.get_descendants(lookup, formatted=True))
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
