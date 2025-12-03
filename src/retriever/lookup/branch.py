import asyncio
import itertools
from collections.abc import Iterable
from functools import cached_property
from typing import override

from retriever.metakg.metakg import OperationPlan
from retriever.types.general import AdjacencyGraph, QEdgeIDMap
from retriever.types.metakg import Operation
from retriever.types.trapi import CURIE, QEdgeDict, QEdgeID, QNodeID, QueryGraphDict

BranchHop = tuple[QNodeID, QEdgeID | None]
SuperpositionHop = tuple[QNodeID, CURIE | None, QEdgeID | None]

BranchID = tuple[BranchHop, ...]
SuperpositionID = tuple[SuperpositionHop, ...]

QGraphInfo = tuple[QueryGraphDict, AdjacencyGraph, QEdgeIDMap, OperationPlan]


class Branch:
    """A tracking class for identifying a branch/superposition.

    A branch is uniquely identified by its nodes and edges.
    Superpositions are different curies on the same branch.
    A superposition is thus uniquely identified by its curies.
    """

    def __init__(
        self, starting_node: QNodeID, starting_curie: CURIE, qgraph_info: QGraphInfo
    ) -> None:
        """Initialize a Branch instance.

        Expects that nodes, curies, and edges are all the same length.
        """
        qgraph, q_agraph, edge_id_map, op_plan = qgraph_info
        self.qgraph: QueryGraphDict = qgraph
        self.q_agraph: AdjacencyGraph = q_agraph
        self.edge_id_map: QEdgeIDMap = edge_id_map
        self.op_plan: OperationPlan = op_plan
        self.nodes: list[QNodeID] = [starting_node]
        self.curies: list[CURIE] = [starting_curie]
        self.edges: list[QEdgeID] = []
        self.reversed: bool = False  # executing reversed relative to current_edge
        self.next_steps: set[BranchID] = set()
        self.skipped_steps: set[BranchID] = set()

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
    def branch_id(self) -> BranchID:
        """Return the branch identifier.

        A Branch is named solely by its QNodes and QEdges.
        """
        return (
            *[(node, edge) for node, edge in zip(self.nodes, self.edges, strict=False)],
            (self.output_node, None),
        )

    @staticmethod
    def branch_id_to_name(branch_id: BranchID) -> str:
        """Return a string representation of a given branch ID.

        Cannot be reliably split back.
        """
        return (
            "-".join(
                "-".join((qnode_id, str(qedge_id)))
                for qnode_id, qedge_id in branch_id[:-1]
            )
            + f"-{branch_id[-1][0]}"
        )

    @cached_property
    def branch_name(self) -> str:
        """Return a string representation of the Branch.

        Cannot be reliably split back.
        """
        return Branch.branch_id_to_name(self.branch_id)

    @cached_property
    def superposition_id(self) -> SuperpositionID:
        """Return the superposition name.

        A superposition is named taking into account curies.
        """
        return (
            *[
                (node, curie, edge)
                for node, curie, edge in zip(
                    self.nodes, self.curies, self.edges, strict=False
                )
            ],
            (self.output_node, None, None),
        )

    @staticmethod
    def superposition_id_to_name(superposition_id: SuperpositionID) -> str:
        """Return a string representation of a given Superposition.

        Cannot be reliably split back.
        """
        return (
            "-".join(
                f"{qnode_id}({curie})-{qedge_id}"
                for qnode_id, curie, qedge_id in superposition_id[:-1]
            )
            + f"-{superposition_id[-1][0]}"
        )

    @cached_property
    def superposition_name(self) -> str:
        """Return a string representation of the Superposition.

        Cannot be reliably split back.
        """
        return Branch.superposition_id_to_name(self.superposition_id)

    @cached_property
    def hop_id(self) -> SuperpositionHop:
        """Return a name representing the branch's input superposition.

        Used to identify the specific curie-qedge hop, which might be executed by
        other superpositions.
        """
        return (
            self.input_node,
            self.input_curie,
            self.current_edge,
        )

    @property
    def current_edge(self) -> QEdgeID:
        """Get the currently-focused edge of the branch."""
        return self.edges[-1]

    @property
    def start_node(self) -> QNodeID:
        """Get the starting node of the branch."""
        return self.nodes[0]

    @property
    def input_node(self) -> QNodeID:
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
    def output_node(self) -> QNodeID:
        """Get the output node of the current edge.

        Relative to Branch direction, not QEdge direction.
        """
        return self.nodes[-1]

    @property
    def operations(self) -> list[Operation]:
        """Get the planned operations for the current edge."""
        return self.op_plan[self.current_edge]

    @cached_property
    def next_edges(self) -> dict[QNodeID, list[QEdgeDict]]:
        """Return the next potential edges adjacent to the current edge."""
        edges = dict[QNodeID, list[QEdgeDict]]()
        for qnode_id, qedge in self.q_agraph[self.output_node].items():
            if qnode_id not in edges:
                edges[qnode_id] = list[QEdgeDict]()
            edges[qnode_id].extend(qedge)
        return edges

    def advance(
        self,
        edge: QEdgeID,
        node: QNodeID,
        last_curie: CURIE | None = None,
        reverse: bool = False,
    ) -> "Branch":
        """Return a new branch that is advanced by the given step."""
        branch = Branch(
            self.start_node,
            self.curies[0],
            (self.qgraph, self.q_agraph, self.edge_id_map, self.op_plan),
        )
        branch.nodes = [*self.nodes, node]
        branch.curies = [*self.curies]
        if last_curie:
            branch.curies.append(last_curie)
        branch.edges = [*self.edges, edge]
        if reverse:
            branch.reversed = True
        return branch

    async def get_next_steps(
        self,
        curies: Iterable[CURIE],
        qedge_claims: dict[QEdgeID, "Branch | None"],
        lock: asyncio.Lock,
    ) -> list["Branch"]:
        """Find next possible edges to execute.

        Returns:
            A 2d dict of next-step Branches by curie and QEdge ID.
        """
        next_steps = list[Branch]()

        current_edge = self.qgraph["edges"][self.current_edge]
        for next_qnode_id, edges in self.next_edges.items():
            for next_edge in edges:
                if (
                    self.edge_id_map[id(next_edge)]
                    == self.edge_id_map[id(current_edge)]
                ):
                    continue

                claim_checked = False
                for curie in curies:
                    next_qedge_id = self.edge_id_map[id(next_edge)]
                    branch = self.advance(
                        next_qedge_id,
                        next_qnode_id,
                        curie,
                        reverse=next_edge["subject"] == next_qnode_id,
                    )
                    if not claim_checked and not await branch.has_claim(
                        qedge_claims, lock
                    ):
                        break
                    claim_checked = True
                    next_steps.append(branch)

        return next_steps

    async def has_claim(
        self, qedge_claims: dict[QEdgeID, "Branch | None"], lock: asyncio.Lock
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
    async def get_start_branches(
        qedge_claims: dict[QEdgeID, "Branch | None"],
        lock: asyncio.Lock,
        qgraph_info: QGraphInfo,
    ) -> list["Branch"]:
        """Get starting edges from a query graph.

        Starting edges are all edges connected to nodes which have ids.
        """
        qg, ag, em, tiers = qgraph_info
        start = list[Branch]()
        for node_id, node in qg["nodes"].items():
            node_id = QNodeID(node_id)  # noqa:PLW2901
            if (
                "ids" not in node or node["ids"] is None
            ):  # Only start on nodes with curies
                continue
            for curie in node["ids"]:
                for edge in itertools.chain(*ag[node_id].values()):
                    edge_id = em[id(edge)]
                    next_node = QNodeID(
                        edge["object"]
                        if edge["subject"] == node_id
                        else edge["subject"]
                    )
                    reverse = edge["object"] == node_id
                    branch = Branch(node_id, CURIE(curie), (qg, ag, em, tiers)).advance(
                        edge_id, next_node, reverse=reverse
                    )
                    if await branch.has_claim(qedge_claims, lock):
                        start.append(branch)
        return start
