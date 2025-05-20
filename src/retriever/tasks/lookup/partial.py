import json
from typing import override

from reasoner_pydantic import CURIE


class Partial:
    """A partial result that can be back-propagated for branch reconciliation."""

    def __init__(
        self,
        node_bindings: list[tuple[str, CURIE]],
        edge_bindings: list[tuple[str, CURIE, CURIE]],
    ) -> None:
        """Initialize a partial result."""
        self.node_bindings: list[tuple[str, CURIE]] = node_bindings
        self.edge_bindings: set[tuple[str, CURIE, CURIE]] = set(edge_bindings)

    @override
    def __str__(self) -> str:
        return json.dumps(
            {
                "node_bindings": {
                    qnode: str(node) for qnode, node in sorted(self.node_bindings)
                },
                "edge_bindings": list[self.edge_bindings],
            }
        )

    @override
    def __hash__(self) -> int:
        """Hash a Partial by its str representation."""
        return hash(self.__str__())

    def clone(self) -> "Partial":
        """Return a clone of the Partial with no mutable overlap."""
        return Partial(self.node_bindings.copy(), list(self.edge_bindings.copy()))

    def combine(self, other: "Partial") -> "Partial":
        """Return a new Partial that combines self and other."""
        return Partial(
            [*self.node_bindings, *other.node_bindings],
            [*self.edge_bindings, *other.edge_bindings],
        )

    def reconcile(self, other: "Partial") -> "Partial | None":
        """Reconcile a Partial with another Partial, returning their combination.

        Returns:
            Partial if the two are reconcilable; None if they weren't.
        """
        combined_nodes = dict(self.node_bindings)
        for qnode, node in other.node_bindings:
            if combined_nodes.get(qnode, node) != node:
                return None
            else:
                combined_nodes[qnode] = node

        return Partial(
            list(combined_nodes.items()), [*self.edge_bindings, *other.edge_bindings]
        )
