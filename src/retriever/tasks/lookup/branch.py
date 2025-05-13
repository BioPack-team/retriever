from itertools import zip_longest
from typing import cast, override

from reasoner_pydantic import (
    CURIE,
    QEdge,
    QueryGraph,
)

from retriever.type_defs import AdjacencyGraph, EdgeIDMap


class Branch:
    """A tracking class for identifying a branch/superposition.

    A branch is uniquely identified by its nodes and edges.
    Superpositions are different curies on the same branch.
    A superposition is thus uniquely identified by its curies.
    """

    def __init__(
        self,
        qgraph: QueryGraph,
        agraph: AdjacencyGraph,
        edge_id_map: EdgeIDMap,
        starting_node: str,
        starting_curie: CURIE,
    ) -> None:
        """Initialize a Branch instance.

        Expects that nodes, curies, and edges are all the same length.
        """
        self.qgraph: QueryGraph = qgraph
        self.agraph: AdjacencyGraph = agraph
        self.edge_id_map: EdgeIDMap = edge_id_map
        self.nodes: list[str] = [starting_node]
        self.curies: list[CURIE] = [starting_curie]
        self.edges: list[str] = []

    @override
    def __eq__(self, other: object) -> bool:
        """Check if self and other branch are equivalent."""
        if not isinstance(other, Branch):
            return False
        return hash(self) == hash(other)

    @override
    def __hash__(self) -> int:
        """Hash the branch.

        Does not take into account superposition.
        """
        return hash(self.branch_name)

    @property
    def branch_name(self) -> str:
        """Return the branch name.

        A Branch is named solely by its QNodes and QEdges.
        """
        return "-".join(
            f"{node}-{edge}"
            for node, edge in zip_longest(self.nodes, self.edges, fillvalue="")
        )

    @property
    def superposition_name(self) -> str:
        """Return the superposition name.

        A superposition is named taking into account curies.
        """
        return "-".join(
            f"{node}[{curie}]-{edge}"
            for node, curie, edge in zip_longest(
                self.nodes, self.curies, self.edges, fillvalue=""
            )
        )

    @property
    def current_edge(self) -> str:
        """Get the currently-focused edge of the branch."""
        return self.edges[-1]

    @property
    def start_node(self) -> str:
        """Get the starting node of the branch."""
        return self.nodes[0]

    @property
    def end_node(self) -> str:
        """Get the current ending node of the branch."""
        return self.nodes[-1]

    def advance(
        self, edge: str, node: str, last_curie: CURIE | None = None
    ) -> "Branch":
        """Return a new branch that is advanced by the given step."""
        branch = Branch(
            self.qgraph, self.agraph, self.edge_id_map, self.start_node, self.curies[0]
        )
        branch.nodes = [*self.nodes, node]
        branch.curies = [*self.curies]
        if last_curie:
            branch.curies.append(last_curie)
        branch.edges = [*self.edges, edge]
        return branch

    def get_next_steps(self, curies: list[CURIE]) -> list["Branch"]:
        """Find next possible edges to execute.

        Returns:
            A 2d dict of next-step Branches by curie and QEdge ID.
        """
        next_steps = list[Branch]()

        current_edge = cast(QEdge, self.edge_id_map[self.current_edge])
        for curie in curies:
            for next_node, next_edge in self.agraph[self.end_node].items():
                if next_edge is current_edge:
                    continue

                next_edge_id = cast(str, self.edge_id_map[next_edge])

                next_steps.append(self.advance(next_edge_id, next_node, curie))

        return next_steps

    @staticmethod
    def get_start_branches(
        qg: QueryGraph, ag: AdjacencyGraph, em: EdgeIDMap
    ) -> list["Branch"]:
        """Get starting edges from a query graph.

        Starting edges are all edges connected to nodes which have ids.
        """
        start = list[Branch]()
        for node_id, node in qg.nodes.items():
            if node.ids is None:  # Only start on nodes with curies
                continue
            for curie in node.ids:
                for edge in ag[node_id].values():
                    edge_id = cast(str, em[edge])
                    next_node = edge.object if edge.subject == node_id else edge.subject
                    start.append(
                        Branch(qg, ag, em, node_id, curie).advance(edge_id, next_node)
                    )
        return start
