import json
from typing import override

from reasoner_pydantic import (
    CURIE,
    Edge,
)


class Partial:
    """A partial result that can be back-propagated for branch reconciliation."""

    def __init__(
        self,
        node_bindings: list[tuple[str, CURIE]],
        edge_bindings: list[tuple[str, Edge]],
    ) -> None:
        """Initialize a partial result."""
        self.node_bindings: list[tuple[str, CURIE]] = node_bindings
        self.edge_bindings: list[tuple[str, Edge]] = edge_bindings

    @override
    def __str__(self) -> str:
        return json.dumps(
            {
                "node_bindings": {
                    qnode: str(node) for qnode, node in self.node_bindings
                },
                "edge_bindings": {
                    qedge: hash(edge) for qedge, edge in self.edge_bindings
                },
            }
        )

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
