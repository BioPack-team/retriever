import bmt

biolink = bmt.Toolkit()


def expand(items: set[str]) -> set[str]:
    """Expand a set of biolink categories or predicates."""
    expanded = set(items)
    for item in items:
        expanded.update(biolink.get_descendants(item))
    return expanded
