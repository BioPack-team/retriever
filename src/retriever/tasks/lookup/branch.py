import asyncio
from collections.abc import Iterable
from functools import cached_property
from itertools import zip_longest
from typing import override

from reasoner_pydantic import (
    CURIE,
    QEdge,
    QueryGraph,
)

from retriever.type_defs import AdjacencyGraph, QEdgeIDMap


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
        edge_id_map: QEdgeIDMap,
        starting_node: str,
        starting_curie: CURIE,
    ) -> None:
        """Initialize a Branch instance.

        Expects that nodes, curies, and edges are all the same length.
        """
        self.qgraph: QueryGraph = qgraph
        self.agraph: AdjacencyGraph = agraph
        self.edge_id_map: QEdgeIDMap = edge_id_map
        self.nodes: list[str] = [starting_node]
        self.curies: list[CURIE] = [starting_curie]
        self.edges: list[str] = []
        self.reversed: bool = False  # executing reversed relative to current_edge
        self.next_steps: set[str] = set()
        self.skipped_steps: set[str] = set()

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

    @cached_property
    def branch_name(self) -> str:
        """Return the branch name.

        A Branch is named solely by its QNodes and QEdges.
        """
        return "-".join(
            f"{node}-{edge}"
            for node, edge in zip_longest(self.nodes, self.edges, fillvalue="")
        )

    @cached_property
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

    @cached_property
    def hop_name(self) -> tuple[CURIE, str]:
        """Return a name representing the branch's input superposition.

        Used to identify the specific curie-qedge hop, which might be executed by
        other superpositions.
        """
        return (
            self.input_curie,
            self.current_edge,
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
    def input_node(self) -> str:
        """Get the input node of the current edge.

        Relative to Branch direction, not QEdge direction.
        """
        return self.nodes[-2]

    @property
    def input_curie(self) -> CURIE:
        """Get the input curie of the current edge.

        Relative to Branch direction, not QEdge direction.
        """
        return self.curies[-1]

    @property
    def output_node(self) -> str:
        """Get the output node of the current edge.

        Relative to Branch direction, not QEdge direction.
        """
        return self.nodes[-1]

    @cached_property
    def next_edges(self) -> dict[str, QEdge]:
        """Return the next potential edges adjacent to the current edge."""
        return {
            qnode_id: qedge
            for qnode_id, qedge in self.agraph[self.output_node].items()
            if qnode_id != self.input_node
        }

    def advance(
        self,
        edge: str,
        node: str,
        last_curie: CURIE | None = None,
        reverse: bool = False,
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
        if reverse:
            branch.reversed = True
        return branch

    def get_next_steps(self, curies: Iterable[CURIE]) -> list["Branch"]:
        """Find next possible edges to execute.

        Returns:
            A 2d dict of next-step Branches by curie and QEdge ID.
        """
        next_steps = list[Branch]()

        current_edge = self.qgraph.edges[self.current_edge]
        for next_node_id, next_edge in self.next_edges.items():
            if self.edge_id_map[next_edge] == self.edge_id_map[current_edge]:
                continue
            for curie in curies:

                next_edge_id = self.edge_id_map[next_edge]

                next_steps.append(
                    self.advance(
                        next_edge_id,
                        next_node_id,
                        curie,
                        reverse=next_edge.subject == next_node_id,
                    )
                )

        return next_steps

    async def has_claim(
        self, qedge_claims: dict[str, "Branch | None"], lock: asyncio.Lock
    ) -> bool:
        """Check if the branch's current edge is claimable, and claim it if so.

        Returns:
            True if branch already has claim or is able to make claim.
            False if another branch has already claimed the edge.
        """
        async with lock:
            if (
                qedge_claims[self.current_edge] is not None
                and qedge_claims[self.current_edge] != self
            ):
                return False
            qedge_claims[self.current_edge] = self
            return True

    @staticmethod
    def get_start_branches(
        qg: QueryGraph, ag: AdjacencyGraph, em: QEdgeIDMap
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
                    edge_id = em[edge]
                    next_node = edge.object if edge.subject == node_id else edge.subject
                    reverse = edge.object == node_id
                    start.append(
                        Branch(qg, ag, em, node_id, curie).advance(
                            edge_id, next_node, reverse=reverse
                        )
                    )
        return start
