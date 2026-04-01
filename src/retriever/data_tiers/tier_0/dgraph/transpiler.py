import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, override

import orjson
from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers.base_transpiler import Tier0Transpiler
from retriever.data_tiers.tier_0.dgraph import result_models as dg
from retriever.lookup.partial import Partial
from retriever.lookup.subclass_format import solve_subclass_edges
from retriever.lookup.utils import QueryDumper
from retriever.types.general import BackendResult, KAdjacencyGraph
from retriever.types.trapi import (
    CURIE,
    AttributeConstraintDict,
    AttributeDict,
    AuxGraphID,
    AuxiliaryGraphDict,
    BiolinkEntity,
    BiolinkPredicate,
    EdgeDict,
    EdgeIdentifier,
    Infores,
    KnowledgeGraphDict,
    NodeDict,
    QEdgeDict,
    QEdgeID,
    QNodeDict,
    QNodeID,
    QualifierConstraintDict,
    QualifierDict,
    QualifierTypeID,
    QueryGraphDict,
    RetrievalSourceDict,
)
from retriever.utils import biolink
from retriever.utils.trapi import (
    append_aggregator_source,
    attributes_meet_contraints,
    hash_edge,
    hash_hex,
)


@dataclass(frozen=True)
class EdgeTraversalContext:
    """Context for building an edge traversal query."""

    edge_id: QEdgeID
    edge: QEdgeDict
    target_id: QNodeID
    target_node: QNodeDict
    edge_direction: str  # "out" or "in"
    visited: set[QNodeID]
    nodes: Mapping[QNodeID, QNodeDict]
    edges: Mapping[QEdgeID, QEdgeDict]


@dataclass(frozen=True)
class DirectionTraversalContext:
    """Context for building a single direction traversal."""

    edge_name: str
    edge_reverse_field: str
    predicate_field: str
    filter_clause: str
    target_id: QNodeID
    target_filter: str
    child_visited: set[QNodeID]
    nodes: Mapping[QNodeID, QNodeDict]
    edges: Mapping[QEdgeID, QEdgeDict]


@dataclass(frozen=True)
class BranchStep:
    """A single edge-and-node alias pair that represents one hop in a planned branch."""

    edge_alias: str
    node_alias: str


@dataclass(frozen=True)
class EdgeGroupPlan:
    """The set of alternative branches that may fulfill one qedge in the query graph."""

    qedge_id: QEdgeID
    branches: tuple["BranchPlan", ...]


@dataclass(frozen=True)
class NodePlan:
    """A recursive plan describing which edge groups must match beneath a node alias."""

    node_alias: str
    edge_groups: tuple[EdgeGroupPlan, ...]


@dataclass(frozen=True)
class BranchPlan:
    """A specific multi-step path that must end in a target node plan."""

    steps: tuple[BranchStep, ...]
    target: NodePlan


@dataclass(frozen=True)
class QueryBuildResult:
    """Container for the emitted DQL and the node plan used to validate its results."""

    dql: str
    plan: NodePlan


class FilterValueProtocol(Protocol):
    """Protocol for values that can be used in filters."""

    @override
    def __str__(self) -> str: ...


@dataclass(slots=True)
class CascadePlanMatcher:
    """Match a result tree against a NodePlan and prune unmatched edges in place."""

    def match_node_plan(self, node: dg.Node, node_plan: NodePlan) -> bool:
        """Return whether a node satisfies a plan and keep only matching child edges."""
        if node.raw_alias != node_plan.node_alias:
            return False

        kept_edges: list[dg.Edge] = []

        for group in node_plan.edge_groups:
            group_matches = self.match_edge_group(node, group)
            if not group_matches:
                return False
            kept_edges.extend(group_matches)

        node.edges[:] = self._dedupe_edges(kept_edges)
        return True

    def match_edge_group(
        self,
        node: dg.Node,
        group: EdgeGroupPlan,
    ) -> list[dg.Edge]:
        """Return all edges that satisfy any branch in an edge group."""
        kept: list[dg.Edge] = []

        for branch in group.branches:
            matched = self.match_branch(node, branch)
            if matched:
                kept.extend(matched)

        return kept

    def match_branch(
        self,
        node: dg.Node,
        branch: BranchPlan,
    ) -> list[dg.Edge] | None:
        """Return the matched edges for a branch, or None when the branch fails."""
        return self.match_branch_steps(node, branch.steps, 0, branch.target)

    def match_branch_steps(
        self,
        node: dg.Node,
        steps: tuple[BranchStep, ...],
        index: int,
        target_plan: NodePlan,
    ) -> list[dg.Edge] | None:
        """Recursively match a branch's steps starting at the given step index."""
        step = steps[index]
        matched_edges: list[dg.Edge] = []

        for edge in node.edges:
            if edge.raw_alias != step.edge_alias:
                continue
            if edge.node.raw_alias != step.node_alias:
                continue

            if index == len(steps) - 1:
                if self.match_node_plan(edge.node, target_plan):
                    matched_edges.append(edge)
                continue

            tail = self.match_branch_steps(
                edge.node,
                steps,
                index + 1,
                target_plan,
            )
            if tail is not None:
                matched_edges.append(edge)

        return matched_edges or None

    def _dedupe_edges(self, edges: list[dg.Edge]) -> list[dg.Edge]:
        seen: set[tuple[str, int]] = set()
        out: list[dg.Edge] = []

        for edge in edges:
            key = (edge.raw_alias, id(edge))
            if key in seen:
                continue
            seen.add(key)
            out.append(edge)

        return out


class DgraphTranspiler(Tier0Transpiler):
    """Transpiler for converting TRAPI queries into Dgraph GraphQL queries."""

    # --- Constants for Pinnedness Algorithm ---
    PINNEDNESS_DEFAULT_TOTAL_NODES: int = (
        1_000_000  # Default assumption for total nodes in the graph
    )
    PINNEDNESS_DEFAULT_EDGES_PER_NODE: int = (
        25  # Default assumption for average edges per node
    )
    PINNEDNESS_RECURSION_DEPTH: int = (
        2  # Max recursion depth for pinnedness calculation
    )
    PINNEDNESS_ADJ_WEIGHT: float = (
        0.1  # dampen adjacency contribution relative to the base ID selectivity
    )

    FilterScalar: TypeAlias = str | int | float | bool  # noqa: UP040
    FilterValue: TypeAlias = FilterScalar | list[FilterScalar]  # noqa: UP040
    version: str | None
    prefix: str

    # Feature flags
    _symmetric_edges_enabled: bool
    _subclass_edges_enabled: bool

    # Normalization mappings for injection prevention
    _node_id_map: dict[QNodeID, str]
    _edge_id_map: dict[QEdgeID, str]
    _reverse_node_map: dict[str, QNodeID]
    _reverse_edge_map: dict[str, QEdgeID]

    # Mapping for TRAPI conversion of implicit subclass handling
    subclass_backmap: dict[CURIE, CURIE]

    # =========================================================================
    # Construction and Shared State
    # =========================================================================

    def __init__(
        self,
        version: str | None = None,
        enable_symmetric_edges: bool | None = None,
        enable_subclass_edges: bool | None = None,
    ) -> None:
        """Initialize a Transpiler instance.

        Args:
            version: An optional version string to prefix to all schema fields.
            enable_symmetric_edges: Enable symmetric edge expansion. If None, uses config value.
            enable_subclass_edges: Enable subclass edge expansion. If None, uses config value.
        """
        super().__init__()
        self.kgraph: KnowledgeGraphDict = KnowledgeGraphDict(nodes={}, edges={})
        self.k_agraph: KAdjacencyGraph
        self.version = version
        self.prefix = f"{version}_" if version else ""

        # Load feature flags from config, but allow override via parameters
        self._symmetric_edges_enabled = (
            enable_symmetric_edges
            if enable_symmetric_edges is not None
            else CONFIG.tier0.dgraph.enable_symmetric_edges
        )
        self._subclass_edges_enabled = (
            enable_subclass_edges
            if enable_subclass_edges is not None
            else CONFIG.tier0.dgraph.enable_subclass_edges
        )

        # Initialize normalization mappings
        self._node_id_map = {}
        self._edge_id_map = {}
        self._reverse_node_map = {}
        self._reverse_edge_map = {}

        self._symmetric_edge_map: dict[QEdgeID, tuple[str, str]] = {}
        self._subclass_edge_map: dict[QEdgeID, list[str]] = {}
        self._query_plan: NodePlan | None = None

        self.subclass_backmap = {}

    # =========================================================================
    # Identifier and Alias Helpers
    # =========================================================================

    def _v(self, field: str) -> str:
        """Return the versioned field name."""
        return f"{self.prefix}{field}"

    def _aliased_fields(self, fields: list[str]) -> str:
        """Return a string of aliased fields if a version is set, otherwise return just the field names."""
        if self.version:
            return " ".join(f"{field}: {self._v(field)}" for field in fields) + " "
        return " ".join(fields) + " "

    def _node_has_ids(self, node: QNodeDict) -> bool:
        """Check if node has IDs specified."""
        ids = node.get("ids")
        return bool(ids and len(ids) > 0)

    def _node_has_categories(self, node: QNodeDict) -> bool:
        """Check if node has categories specified."""
        cats = node.get("categories")
        return bool(cats and len(cats) > 0)

    def _is_subclass_predicate(self, predicates: Sequence[str] | None) -> bool:
        """Return True if predicates contain biolink:subclass_of."""
        if not predicates:
            predicates = ["biolink:related_to"]
        return any(
            biolink.ensure_prefix(p) in biolink.SUBCLASS_SKIP_PREDICATES
            for p in predicates
        )

    def _subclass_edge_filter(self) -> str:
        """Filter clause for subclass_of edges only."""
        return f'eq({self._v("predicate_ancestors")}, "subclass_of")'

    # =========================================================================
    # Query Graph Analysis and Special-Edge Detection
    # =========================================================================

    def _detect_symmetric_and_subclass_edges(
        self,
        edges: Mapping[QEdgeID, QEdgeDict],
        nodes: Mapping[QNodeID, QNodeDict],
    ) -> None:
        """Pre-detect all symmetric and subclass edges in the query graph.

        This must be called before building the query to ensure cascade
        clauses can correctly identify nodes with special edges.

        Args:
            edges: Dictionary of all edges in the query graph
            nodes: Dictionary of all nodes in the query graph
        """
        self._symmetric_edge_map.clear()
        self._subclass_edge_map.clear()

        for edge_id, edge in edges.items():
            predicates = edge.get("predicates") or []
            is_symmetric = any(biolink.is_symmetric(str(pred)) for pred in predicates)
            is_subclass = self._is_subclass_predicate(predicates)

            if is_symmetric and self._symmetric_edges_enabled:
                normalized_edge_id = self._get_normalized_edge_id(edge_id)
                primary = f"in_edges_{normalized_edge_id}"
                symmetric = f"out_edges-symmetric_{normalized_edge_id}"
                self._symmetric_edge_map[edge_id] = (primary, symmetric)

            # Detect subclass expansion cases (skip if edge itself is subclass_of)
            if not is_subclass and self._subclass_edges_enabled:
                source_id = edge["subject"]
                target_id = edge["object"]
                source_node = nodes[source_id]
                target_node = nodes[target_id]

                normalized_edge_id = self._get_normalized_edge_id(edge_id)
                subclass_forms: list[str] = []

                # Case 1: ID -> predicate -> ID
                if self._node_has_ids(source_node) and self._node_has_ids(target_node):
                    subclass_forms.extend(
                        [
                            f"in_edges-subclassB_{normalized_edge_id}",
                            f"out_edges-subclassC_{normalized_edge_id}",
                            f"in_edges-subclassD_{normalized_edge_id}",
                        ]
                    )
                # Case 2: ID -> predicate -> CAT
                elif (
                    self._node_has_ids(source_node)
                    and self._node_has_categories(target_node)
                    and not self._node_has_ids(target_node)
                ):
                    subclass_forms.append(f"in_edges-subclassB_{normalized_edge_id}")
                # Mirrored Case 2: CAT -> predicate -> ID
                elif (
                    self._node_has_ids(target_node)
                    and self._node_has_categories(source_node)
                    and not self._node_has_ids(source_node)
                ):
                    subclass_forms.append(
                        f"out_edges-subclassObjB_{normalized_edge_id}"
                    )

                if subclass_forms:
                    self._subclass_edge_map[edge_id] = subclass_forms

    # =========================================================================
    # Query Graph Normalization
    # =========================================================================

    def _normalize_qgraph_ids(self, qgraph: QueryGraphDict) -> None:
        """Create normalized mappings for node and edge IDs to prevent injection attacks.

        This method creates safe, predictable identifiers (n0, n1, e0, e1, etc.) that are
        used in the generated Dgraph query. The mappings allow us to convert back to the
        original user-provided IDs when parsing results.

        Args:
            qgraph: The query graph to normalize
        """
        # Clear any existing mappings
        self._node_id_map.clear()
        self._edge_id_map.clear()
        self._reverse_node_map.clear()
        self._reverse_edge_map.clear()

        # Create normalized node IDs (n0, n1, n2, ...)
        # Sort for deterministic ordering
        for i, node_id in enumerate(sorted(qgraph["nodes"].keys())):
            normalized = f"n{i}"
            self._node_id_map[node_id] = normalized
            self._reverse_node_map[normalized] = node_id
            logger.debug(f"Normalized node ID: {node_id} -> {normalized}")

        # Create normalized edge IDs (e0, e1, e2, ...)
        # Sort for deterministic ordering
        for i, edge_id in enumerate(sorted(qgraph["edges"].keys())):
            normalized = f"e{i}"
            self._edge_id_map[edge_id] = normalized
            self._reverse_edge_map[normalized] = edge_id
            logger.debug(f"Normalized edge ID: {edge_id} -> {normalized}")

    def _get_normalized_node_id(self, node_id: QNodeID) -> str:
        """Get the normalized ID for a node.

        Args:
            node_id: The original node ID from the query graph

        Returns:
            The normalized node ID (e.g., 'n0', 'n1')
        """
        return self._node_id_map.get(node_id, str(node_id))

    def _get_normalized_edge_id(self, edge_id: QEdgeID) -> str:
        """Get the normalized ID for an edge.

        Args:
            edge_id: The original edge ID from the query graph

        Returns:
            The normalized edge ID (e.g., 'e0', 'e1')
        """
        return self._edge_id_map.get(edge_id, str(edge_id))

    def _get_original_node_id(self, normalized_id: str) -> QNodeID:
        """Get the original node ID from a normalized ID.

        Args:
            normalized_id: The normalized ID (e.g., 'n0')

        Returns:
            The original node ID from the query graph
        """
        return self._reverse_node_map.get(normalized_id, QNodeID(normalized_id))

    def _get_original_edge_id(self, normalized_id: str) -> QEdgeID:
        """Get the original edge ID from a normalized ID.

        Args:
            normalized_id: The normalized ID (e.g., 'e0')

        Returns:
            The original edge ID from the query graph
        """
        return self._reverse_edge_map.get(normalized_id, QEdgeID(normalized_id))

    # =========================================================================
    # Result-Path Validation
    # =========================================================================

    def _filter_cascaded_with_or(
        self,
        nodes: list[dg.Node],
        plan: NodePlan | QueryGraphDict,
    ) -> list[dg.Node]:
        """Filter results to enforce cascade with OR logic between symmetric edge directions.

        Ensure all required path hops are present. Since Dgraph doesn't natively support
        this logic, we implement it as a post-processing step on the raw results.

        Args:
            nodes: Parsed Dgraph response nodes
            plan: The traversal plan used to validate path completeness

        Returns:
            Filtered list of nodes that satisfy cascade with OR logic and full path requirements
        """
        resolved_plan = self._resolve_result_filter_plan(nodes, plan)
        matcher = CascadePlanMatcher()
        return [node for node in nodes if matcher.match_node_plan(node, resolved_plan)]

    def _resolve_result_filter_plan(
        self,
        nodes: list[dg.Node],
        plan: NodePlan | QueryGraphDict,
    ) -> NodePlan:
        if not isinstance(plan, dict):
            return plan

        qgraph = plan
        if not self._reverse_node_map or not self._reverse_edge_map:
            self._normalize_qgraph_ids(qgraph)

        qnodes = qgraph["nodes"]
        qedges = qgraph["edges"]

        start_node_id = self._infer_start_node_id_from_results(nodes)
        if start_node_id is None:
            start_node_id = self._find_start_node(qnodes, qedges)

        self._detect_symmetric_and_subclass_edges(qedges, qnodes)
        return self._build_node_plan(
            start_node_id,
            qnodes,
            qedges,
            query_index=0,
        )

    def _infer_start_node_id_from_results(
        self,
        nodes: list[dg.Node],
    ) -> QNodeID | None:
        if not nodes:
            return None

        normalized_root_id = self._extract_normalized_root_id(nodes[0].raw_alias)
        if normalized_root_id is None:
            return None

        return self._get_original_node_id(normalized_root_id)

    def _extract_normalized_root_id(self, raw_alias: str) -> str | None:
        if raw_alias.startswith("q") and "_node_" in raw_alias:
            return raw_alias.split("_node_", 1)[1]
        if raw_alias.startswith("node_"):
            return raw_alias.removeprefix("node_")
        return None

    # =========================================================================
    # Public Transpiler Entrypoints
    # =========================================================================

    @override
    def process_qgraph(
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> str:
        payload = super().process_qgraph(qgraph, *additional_qgraphs)

        if CONFIG.tier0.dump_queries:
            QueryDumper().put(
                "write_tier0",
                orjson.dumps(
                    {"trapi": qgraph, "dgraph": payload},
                    option=orjson.OPT_APPEND_NEWLINE,
                ),
            )

        return payload

    @override
    def convert_multihop(self, qgraph: QueryGraphDict) -> str:
        """Convert a TRAPI multi-hop graph to a proper Dgraph multihop query.

        Args:
            qgraph: The TRAPI query graph

        Returns:
            A Dgraph query string with normalized, injection-safe identifiers
        """
        # Normalize IDs first to prevent injection attacks
        self._normalize_qgraph_ids(qgraph)

        nodes = qgraph["nodes"]
        edges = qgraph["edges"]

        # Identify the starting node
        start_node_id = self._find_start_node(nodes, edges)

        self._detect_symmetric_and_subclass_edges(edges, nodes)
        self._query_plan = self._build_node_plan(
            start_node_id,
            nodes,
            edges,
            query_index=0,
        )

        # Build query from the starting node, providing a default query_index of 0
        query = self._build_node_query(start_node_id, nodes, edges, query_index=0)

        return "{ " + query + " }"

    # =========================================================================
    # Start-Node Selection
    # =========================================================================

    def _find_start_node(
        self,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
    ) -> QNodeID:
        """Find the best node to start the traversal from.

        First, check if any node has IDs specified - if so, use that.
        Otherwise, use pinnedness algorithm to find the most constrained node.
        """
        if not nodes:
            raise ValueError("Query graph must have at least one node.")

        # Use pinnedness algorithm to find the most constrained node
        qgraph = QueryGraphDict(nodes=dict(nodes), edges=dict(edges))
        pinnedness_scores = {
            node_id: self._get_pinnedness(qgraph, node_id) for node_id in nodes
        }

        # Return the node_id with the maximum pinnedness score
        # Use node_id as tiebreaker for deterministic ordering
        return max(
            pinnedness_scores, key=lambda nid: (pinnedness_scores[nid], -ord(nid[-1]))
        )

    # =========================================================================
    # Start-Node Scoring (Pinnedness Algorithm)
    # =========================================================================

    def _get_adjacency_matrix(
        self, qgraph: QueryGraphDict
    ) -> defaultdict[str, defaultdict[str, int]]:
        """Get adjacency matrix."""
        adjacency_matrix: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for qedge in qgraph["edges"].values():
            adjacency_matrix[qedge["subject"]][qedge["object"]] += 1
            adjacency_matrix[qedge["object"]][qedge["subject"]] += 1
        return adjacency_matrix

    def _get_num_ids(self, qgraph: QueryGraphDict) -> dict[str, int]:
        """Get the number of ids for each node, defaulting to N."""
        num_ids_map: dict[str, int] = {}
        for qnode_id, qnode in qgraph["nodes"].items():
            ids = qnode.get("ids")
            if ids:
                num_ids_map[qnode_id] = len(ids)
            else:
                num_ids_map[qnode_id] = self.PINNEDNESS_DEFAULT_TOTAL_NODES
        return num_ids_map

    def _compute_log_expected_n(
        self,
        adjacency_mat: defaultdict[str, defaultdict[str, int]],
        num_ids: dict[str, int],
        qnode_id: str,
        last: str | None = None,
        level: int = 0,
    ) -> float:
        """Compute the log of the expected number of unique knodes bound to the specified qnode.

        The base term is log(ids_count). Neighbor contributions are heavily dampened by
        PINNEDNESS_ADJ_WEIGHT and limited by PINNEDNESS_RECURSION_DEPTH to ensure
        'fewer IDs always preferred' dominates adjacency effects.
        """
        log_expected_n = math.log(num_ids[qnode_id])

        if level < self.PINNEDNESS_RECURSION_DEPTH:
            for neighbor, num_edges in adjacency_mat[qnode_id].items():
                if neighbor == last:
                    continue

                # Neighbor expectation (non-negative)
                neighbor_log = max(
                    self._compute_log_expected_n(
                        adjacency_mat, num_ids, neighbor, qnode_id, level + 1
                    ),
                    0,
                )

                # Baseline per-edge connectivity factor
                baseline = math.log(
                    self.PINNEDNESS_DEFAULT_EDGES_PER_NODE
                    / self.PINNEDNESS_DEFAULT_TOTAL_NODES
                )

                # Dampen adjacency term so ID selectivity dominates
                contribution = (
                    self.PINNEDNESS_ADJ_WEIGHT
                    * num_edges
                    * max(neighbor_log + baseline, 0)
                )

                log_expected_n += contribution

        return log_expected_n

    def _get_pinnedness(self, qgraph: QueryGraphDict, qnode_id: str) -> float:
        """Get pinnedness of a single node.

        Higher pinnedness is better. With dampened adjacency, fewer IDs (more selective)
        produce higher pinnedness (since -log(ids) is closer to 0 than a larger -log).
        """
        adjacency_mat = self._get_adjacency_matrix(qgraph)
        num_ids = self._get_num_ids(qgraph)
        # Pinnedness is negative expected log-N so smaller expected set -> larger pinnedness
        return -self._compute_log_expected_n(adjacency_mat, num_ids, qnode_id)

    # =========================================================================
    # Filter Compilation
    # =========================================================================

    def _build_node_filter(self, node: QNodeDict, *, primary: bool = False) -> str:
        """Build a filter expression for a node based on its properties.

        If `primary` is True, it returns the most selective filter for a `func:` block.
        Otherwise, it returns all applicable filters for an `@filter` block.

        Filter hierarchy (most to least selective):
        1. IDs (uniquely identify nodes - sufficient alone)
        2. Categories (narrow down by type)
        3. Constraints (additional filtering)
        """
        ids = node.get("ids")
        categories = node.get("categories")
        constraints = node.get("constraints")

        # For primary filter, return the most selective single option
        if primary:
            if ids:
                return self._create_id_filter(ids)
            if categories:
                return self._create_category_filter(categories)
            if constraints:
                return self._convert_constraints_to_filters(constraints)[0]
            return f"has({self._v('id')})"

        # For secondary filters (@filter)
        # If we have IDs, they uniquely identify the node(s) - don't add redundant filters
        if ids:
            return self._create_id_filter(ids)

        # If no IDs, combine categories and constraints
        filters: list[str] = []
        if categories:
            filters.append(self._create_category_filter(categories))
        if constraints:
            filters.extend(self._convert_constraints_to_filters(constraints))

        return " AND ".join(filters) if filters else ""

    def _build_edge_filter(self, edge: QEdgeDict) -> str:
        """Build a filter expression for an edge based on its properties."""
        filters: list[str] = []

        # Handle predicates (multiple) filtering
        predicates = edge.get("predicates")
        if predicates:
            if len(predicates) == 1:
                filters.append(
                    f'eq({self._v("predicate_ancestors")}, "{predicates[0].replace("biolink:", "")}")'
                )
            elif len(predicates) > 1:
                predicates_str = ", ".join(
                    f'"{pred.replace("biolink:", "")}"' for pred in predicates
                )
                filters.append(
                    f"eq({self._v('predicate_ancestors')}, [{predicates_str}])"
                )

        # NOTE: Attribute constraints are intentionally not applied at the Dgraph query level.
        # They are enforced via Python-level post-filtering in `_build_results` using
        # `attributes_meet_contraints`. Full transpiler-level support is pending.
        # TODO: Handle attribute constraints
        # attribute_constraints = edge.get("attribute_constraints")
        # if attribute_constraints:
        #     filters.extend(self._convert_constraints_to_filters(attribute_constraints))

        # Qualifier constraints
        qualifier_constraints: Sequence[QualifierConstraintDict] | None = edge.get(
            "qualifier_constraints"
        )
        if qualifier_constraints:
            qc_filter = self._convert_qualifier_constraints_to_filter(
                qualifier_constraints
            )
            if qc_filter:
                filters.append(qc_filter)

        # If no filters, return empty string
        if not filters:
            return ""

        # Combine all filters with AND
        if len(filters) == 1:
            return filters[0]
        else:
            return " AND ".join(filters)

    def _convert_constraints_to_filters(
        self, constraints: list[AttributeConstraintDict]
    ) -> list[str]:
        """Convert TRAPI attribute constraints to Dgraph filter expressions."""
        filters: list[str] = []

        for constraint in constraints:
            filter_expr = self._create_filter_expression(constraint)
            filters.append(filter_expr)

        return filters

    def _convert_qualifier_constraints_to_filter(
        self,
        qualifier_constraints: Sequence[QualifierConstraintDict],
    ) -> str:
        """Convert TRAPI qualifier_constraints into a Dgraph filter string.

        Within a single `qualifier_set`, items are ANDed. Multiple sets are ORed.
        """
        if not qualifier_constraints:
            return ""

        set_filters: list[str] = []

        for qc in qualifier_constraints:
            qset: Sequence[QualifierDict] = qc["qualifier_set"]
            and_filters: list[str] = []
            for q in qset:
                qtype = biolink.rmprefix(q["qualifier_type_id"])
                qval = (
                    biolink.rmprefix(q["qualifier_value"])
                    if "qualified_predicate" in qtype
                    else q["qualifier_value"]
                )
                if not qtype or qval == "":
                    continue
                field = self._v(qtype)
                and_filters.append(self._get_operator_filter(field, "==", qval))
            if and_filters:
                # AND items within the set; wrap in parentheses if more than one
                set_filters.append(
                    " AND ".join(and_filters)
                    if len(and_filters) == 1
                    else f"({' AND '.join(and_filters)})"
                )

        if not set_filters:
            return ""
        # OR the sets together; wrap in parentheses if more than one
        return (
            " OR ".join(set_filters)
            if len(set_filters) == 1
            else f"({' OR '.join(set_filters)})"
        )

    def _create_filter_expression(self, constraint: AttributeConstraintDict) -> str:
        """Create a filter expression for a single constraint."""
        field_name = self._v(constraint["id"].replace("biolink:", ""))
        value = constraint["value"]
        operator = constraint["operator"]
        is_negated = constraint.get("not", False)

        # Generate the appropriate filter expression based on the operator
        filter_expr = self._get_operator_filter(field_name, operator, value)

        # Handle negation
        if is_negated:
            filter_expr = f"NOT({filter_expr})"

        return filter_expr

    def _get_operator_filter(self, field_name: str, operator: str, value: Any) -> str:
        """Generate filter expression based on operator type."""
        # Group operators by their filter type
        if operator == "in":
            # List membership requires special handling
            return self._create_in_filter(field_name, value)

        if operator == "matches":
            # Text matching
            return f"regexp({field_name}, {value})"

        # All other operators map to Dgraph functions
        func_map = {
            "===": "eq",
            "==": "eq",
            "=": "eq",
            ">": "gt",
            ">=": "ge",
            "<": "lt",
            "<=": "le",
        }

        # Get the function name or default to 'eq'
        func = func_map.get(operator, "eq")

        # Return the constructed filter
        return f'{func}({field_name}, "{value}")'

    def _create_in_filter(self, field_name: str, value: FilterValue) -> str:
        """Create a filter expression for 'in' operator."""
        if isinstance(value, list):
            quoted_items = [f'"{item!s}"' for item in value] if value else []
            values_str = ", ".join(quoted_items)
            return f"eq({field_name}, [{values_str}])"
        else:
            return f'eq({field_name}, "{value!s}")'

    def _create_id_filter(self, ids: Sequence[str] | Sequence[CURIE]) -> str:
        """Create a filter for ID fields."""
        id_list = [str(i) for i in ids]
        if len(id_list) == 1:
            id_value = id_list[0]
            if "," in id_value:
                # Multiple IDs in a single string - split them
                split_ids = [id_val.strip() for id_val in id_value.split(",")]
                ids_str = ", ".join(f'"{id_val}"' for id_val in split_ids)
                return f"eq({self._v('id')}, [{ids_str}])"
            # Single ID - most selective query
            return f'eq({self._v("id")}, "{id_value}")'

        # Multiple IDs in array
        ids_str = ", ".join(f'"{id_val}"' for id_val in id_list)
        return f"eq({self._v('id')}, [{ids_str}])"

    def _create_category_filter(
        self,
        categories: Sequence[str] | Sequence[BiolinkEntity],
    ) -> str:
        """Create a filter for category fields."""
        cat_vals = [str(c).replace("biolink:", "") for c in categories]
        if len(cat_vals) == 1:
            return f'eq({self._v("category")}, "{cat_vals[0].replace("biolink:", "")}")'
        categories_str = ", ".join(f'"{cat}"' for cat in cat_vals)
        return f"eq({self._v('category')}, [{categories_str}])"

    def _get_primary_and_secondary_filters(
        self, node: QNodeDict
    ) -> tuple[str, list[str]]:
        """Extract primary and secondary filters from a node."""
        primary_filter = f"has({self._v('id')})"  # Default
        secondary_filters: list[str] = []

        # Choose the most selective filter for primary (usually ID)
        ids = node.get("ids")
        if ids:
            primary_filter = self._create_id_filter(ids)
        else:
            # Use category as primary if no IDs
            categories = node.get("categories")
            if categories:
                primary_filter = self._create_category_filter(categories)

        # Build secondary filters
        # If we used IDs as primary, add categories as secondary
        categories = node.get("categories")
        if ids and categories:
            category_filter = self._create_category_filter(categories)
            secondary_filters.append(category_filter)

        # Add constraints as secondary filters
        constraints = node.get("constraints")
        if constraints:
            secondary_filters.extend(self._convert_constraints_to_filters(constraints))

        return primary_filter, secondary_filters

    def _add_standard_node_fields(self) -> str:
        """Generate standard node fields with versioned aliases."""
        return f"expand({self.prefix}Node) "

    def _add_standard_edge_fields(self) -> str:
        """Generate standard edge fields with versioned aliases."""
        return f"expand({self.prefix}Edge) {{ {self.prefix}sources expand({self.prefix}Source) }} "

    def _build_filter_clause(self, filters: list[str]) -> str:
        """Build filter clause from a list of filters."""
        if not filters:
            return ""
        elif len(filters) == 1:
            return f" @filter({filters[0]})"
        else:
            return f" @filter({' AND '.join(filters)})"

    # Create a context class to bundle related arguments
    class EdgeConnectionContext:
        """Context object for edge connection data."""

        def __init__(
            self,
            nodes: Mapping[QNodeID, QNodeDict],
            edges: Mapping[QEdgeID, QEdgeDict],
            visited: set[QNodeID],
        ) -> None:
            """Initialize edge connection context.

            Args:
                nodes: Dictionary of query nodes
                edges: Dictionary of query edges
                visited: Set of already visited node IDs
            """
            self.nodes: Mapping[QNodeID, QNodeDict] = nodes
            self.edges: Mapping[QEdgeID, QEdgeDict] = edges
            self.visited: set[QNodeID] = visited

    def _process_edge_connection(
        self,
        edge: QEdgeDict,
        source: QNodeDict,
        source_id: QNodeID,
        context: EdgeConnectionContext,
    ) -> str:
        """Process an edge connection and build the query for it."""
        # Build edge filter
        edge_filter = self._build_edge_filter(edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""

        query = f"in_edges: ~{self._v('source')}{filter_clause} {{ "

        # Include all standard edge fields
        query += self._add_standard_edge_fields()

        # Get primary and secondary filters for source node
        primary_filter, secondary_filters = self._get_primary_and_secondary_filters(
            source
        )

        # Create the source filter clause
        source_filter_clause = ""
        if primary_filter != f"has({self._v('id')})":
            if not secondary_filters:
                source_filter_clause = f" @filter({primary_filter})"
            else:
                source_filter_clause = (
                    f" @filter({primary_filter} AND {' AND '.join(secondary_filters)})"
                )
        elif secondary_filters:
            source_filter_clause = self._build_filter_clause(secondary_filters)

        query += f"node: {self._v('target')}{source_filter_clause} {{ "

        # Include all standard node fields for the target
        query += self._add_standard_node_fields()

        # Recursively add further hops
        query += self._build_further_hops(
            source_id, context.nodes, context.edges, context.visited.copy()
        )

        # Close the blocks
        query += "} } "

        return query

    # =========================================================================
    # Traversal Constraints
    # =========================================================================

    def _build_node_cascade_clause(
        self,
        node_id: QNodeID,
        edges: Mapping[QEdgeID, QEdgeDict],
        visited: set[QNodeID],
    ) -> str:
        """Build a @cascade(...) clause for a node block.

        Always require id. Only require reverse predicates (~subject, ~object) for edges
        that are NOT symmetric or subclass-expanded (those use OR logic in post-processing).
        """
        cascade_fields: list[str] = [self._v("id")]

        # Check outgoing edges (node as subject) to unvisited objects
        has_non_special_out = False
        for e_id, e in edges.items():
            if (
                e["subject"] == node_id
                and e["object"] not in visited
                and e_id not in self._symmetric_edge_map
                and e_id not in self._subclass_edge_map
            ):
                has_non_special_out = True
                break

        # Only require ~subject if there are non-special outgoing edges
        if has_non_special_out:
            cascade_fields.append(f"~{self._v('subject')}")

        # Check incoming edges (node as object) to unvisited subjects
        has_non_special_in = False
        for e_id, e in edges.items():
            if (
                e["object"] == node_id
                and e["subject"] not in visited
                and e_id not in self._symmetric_edge_map
                and e_id not in self._subclass_edge_map
            ):
                has_non_special_in = True
                break

        # Only require ~object if there are non-special incoming edges
        if has_non_special_in:
            cascade_fields.append(f"~{self._v('object')}")

        return f" @cascade({', '.join(cascade_fields)})"

    def _detect_symmetric_edges(
        self,
        edges: Mapping[QEdgeID, QEdgeDict],
    ) -> None:
        """Pre-detect all symmetric edges in the query graph.

        This must be called before building the query to ensure cascade
        clauses can correctly identify nodes with symmetric edges.

        Args:
            edges: Dictionary of all edges in the query graph
        """
        self._symmetric_edge_map.clear()

        for edge_id, edge in edges.items():
            predicates = edge.get("predicates") or []
            is_symmetric = any(biolink.is_symmetric(str(pred)) for pred in predicates)

            if is_symmetric:
                normalized_edge_id = self._get_normalized_edge_id(edge_id)
                # Store with placeholder names - actual direction doesn't matter for detection
                primary = f"in_edges_{normalized_edge_id}"
                symmetric = f"out_edges-symmetric_{normalized_edge_id}"
                self._symmetric_edge_map[edge_id] = (primary, symmetric)

    # =========================================================================
    # Query Planning
    # =========================================================================

    def _build_node_plan(  # noqa: PLR0913
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        visited: set[QNodeID] | None = None,
        *,
        query_index: int | None = None,
        node_alias: str | None = None,
    ) -> NodePlan:
        """Build the traversal plan for a node and its descendants."""
        current_visited: set[QNodeID] = set() if visited is None else set(visited)
        normalized_node_id = self._get_normalized_node_id(node_id)

        if node_alias is None:
            node_alias = f"node_{normalized_node_id}"
            if query_index is not None:
                node_alias = f"q{query_index}_{node_alias}"

        edge_groups = self._build_edge_group_plans(
            node_id,
            nodes,
            edges,
            current_visited | {node_id},
        )

        return NodePlan(
            node_alias=node_alias,
            edge_groups=tuple(edge_groups),
        )

    def _build_edge_group_plans(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        visited: set[QNodeID],
    ) -> list[EdgeGroupPlan]:
        """Build all edge-group plans for a node."""
        plans: list[EdgeGroupPlan] = []

        for edge_id, edge in edges.items():
            if edge["subject"] == node_id:
                outbound_target_id: QNodeID = edge["object"]
                if outbound_target_id not in visited:
                    plans.append(
                        self._build_edge_group_plan(
                            edge_id=edge_id,
                            edge=edge,
                            target_id=outbound_target_id,
                            edge_direction="out",
                            visited=visited,
                            nodes=nodes,
                            edges=edges,
                        )
                    )

            if edge["object"] == node_id:
                inbound_target_id: QNodeID = edge["subject"]
                if inbound_target_id not in visited:
                    plans.append(
                        self._build_edge_group_plan(
                            edge_id=edge_id,
                            edge=edge,
                            target_id=inbound_target_id,
                            edge_direction="in",
                            visited=visited,
                            nodes=nodes,
                            edges=edges,
                        )
                    )

        return plans

    def _build_edge_group_plan(  # noqa: PLR0913
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        target_id: QNodeID,
        edge_direction: str,
        visited: set[QNodeID],
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
    ) -> EdgeGroupPlan:
        """Build the plan for one qedge and all of its valid branches."""
        normalized_edge_id = self._get_normalized_edge_id(edge_id)
        normalized_target_id = self._get_normalized_node_id(target_id)
        target_alias = f"node_{normalized_target_id}"

        if edge_direction == "out":
            primary_edge_alias = f"out_edges_{normalized_edge_id}"
            symmetric_edge_alias = f"in_edges-symmetric_{normalized_edge_id}"
        else:
            primary_edge_alias = f"in_edges_{normalized_edge_id}"
            symmetric_edge_alias = f"out_edges-symmetric_{normalized_edge_id}"

        target_plan = self._build_node_plan(
            target_id,
            nodes,
            edges,
            visited,
            node_alias=target_alias,
        )

        branches: list[BranchPlan] = [
            BranchPlan(
                steps=(
                    BranchStep(edge_alias=primary_edge_alias, node_alias=target_alias),
                ),
                target=target_plan,
            )
        ]

        if edge_id in self._symmetric_edge_map:
            branches.append(
                BranchPlan(
                    steps=(
                        BranchStep(
                            edge_alias=symmetric_edge_alias, node_alias=target_alias
                        ),
                    ),
                    target=target_plan,
                )
            )

        if edge_id in self._subclass_edge_map and not self._is_subclass_predicate(
            edge.get("predicates")
        ):
            branches.extend(
                self._build_subclass_branch_plans(
                    edge_id=edge_id,
                    edge=edge,
                    target_id=target_id,
                    edge_direction=edge_direction,
                    target_plan=target_plan,
                    nodes=nodes,
                )
            )

        return EdgeGroupPlan(
            qedge_id=edge_id,
            branches=tuple(branches),
        )

    def _build_subclass_branch_plans(  # noqa: PLR0913
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        target_id: QNodeID,
        edge_direction: str,
        target_plan: NodePlan,
        nodes: Mapping[QNodeID, QNodeDict],
    ) -> list[BranchPlan]:
        """Build all subclass expansion branches for a qedge."""
        normalized_edge_id = self._get_normalized_edge_id(edge_id)
        normalized_target_id = self._get_normalized_node_id(target_id)
        normalized_source_id = self._get_normalized_node_id(edge["subject"])

        source_node = nodes[edge["subject"]]
        target_node = nodes[edge["object"]]
        branches: list[BranchPlan] = []

        source_has_ids = self._node_has_ids(source_node)
        target_has_ids = self._node_has_ids(target_node)
        source_has_categories = self._node_has_categories(source_node)
        target_has_categories = self._node_has_categories(target_node)

        follows_requested_direction = (
            edge_direction == "out" and target_id == edge["object"]
        ) or (edge_direction == "in" and target_id == edge["subject"])

        def add_branch(*steps: tuple[str, str]) -> None:
            branches.append(
                BranchPlan(
                    steps=tuple(
                        BranchStep(edge_alias=edge_alias, node_alias=node_alias)
                        for edge_alias, node_alias in steps
                    ),
                    target=target_plan,
                )
            )

        if source_has_ids and target_has_ids:
            add_branch(
                (
                    f"in_edges-subclassB_{normalized_edge_id}",
                    f"node_intermediate_{normalized_source_id}",
                ),
                (
                    f"out_edges-subclassB-mid_{normalized_edge_id}",
                    f"node_{normalized_target_id}",
                ),
            )
            add_branch(
                (
                    f"out_edges-subclassC_{normalized_edge_id}",
                    f"node_intermediate_{normalized_target_id}",
                ),
                (
                    f"out_edges-subclassC-tail_{normalized_edge_id}",
                    f"node_{normalized_target_id}",
                ),
            )
            add_branch(
                (
                    f"in_edges-subclassD_{normalized_edge_id}",
                    f"node_intermediateA_{normalized_source_id}",
                ),
                (
                    f"out_edges-subclassD-mid_{normalized_edge_id}",
                    f"node_intermediateB_{normalized_target_id}",
                ),
                (
                    f"out_edges-subclassD-tail_{normalized_edge_id}",
                    f"node_{normalized_target_id}",
                ),
            )
        elif source_has_ids and target_has_categories and not target_has_ids:
            if follows_requested_direction:
                add_branch(
                    (
                        f"in_edges-subclassB_{normalized_edge_id}",
                        f"node_intermediate_{normalized_source_id}",
                    ),
                    (
                        f"out_edges-subclassB-mid_{normalized_edge_id}",
                        f"node_{normalized_target_id}",
                    ),
                )
        elif source_has_categories and not source_has_ids and target_has_ids:
            if follows_requested_direction:
                add_branch(
                    (
                        f"out_edges-subclassObjB_{normalized_edge_id}",
                        f"node_intermediate_{normalized_target_id}",
                    ),
                    (
                        f"out_edges-subclassObjB-tail_{normalized_edge_id}",
                        f"node_{normalized_target_id}",
                    ),
                )

        return branches

    # =========================================================================
    # DQL Emission
    # =========================================================================

    def _build_node_query(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int | None = None,
    ) -> str:
        """Recursively build a query for a node and its connected nodes.

        Args:
            node_id: The original node ID from the query graph
            nodes: Dictionary of all nodes in the query graph
            edges: Dictionary of all edges in the query graph
            query_index: Optional index for batch queries

        Returns:
            Query fragment string with normalized identifiers
        """
        # Pre-detect symmetric AND subclass edges before building any queries
        self._detect_symmetric_and_subclass_edges(edges, nodes)

        node = nodes[node_id]

        # Use normalized node ID for query generation
        normalized_node_id = self._get_normalized_node_id(node_id)

        # Get the primary filter for the `func:` block of the starting node.
        primary_filter = self._build_node_filter(node, primary=True)

        # Create the root node key, with an optional query index prefix
        root_node_key = f"node_{normalized_node_id}"
        if query_index is not None:
            root_node_key = f"q{query_index}_{root_node_key}"

        # Start the query with the primary filter, using the generated key
        query = f"{root_node_key}(func: {primary_filter})"

        # Prepare visited set and add node-level cascade
        visited = {node_id}
        query += self._build_node_cascade_clause(node_id, edges, visited)

        # Open node block
        query += " { "

        # Include all standard node fields
        query += self._add_standard_node_fields()

        # Start the recursive traversal
        query += self._build_further_hops(node_id, nodes, edges, visited)

        # Close the node block
        query += "} "
        return query

    def _build_edge_traversal(self, ctx: EdgeTraversalContext) -> str:
        """Build query fragment for traversing an edge in a specific direction.

        Args:
            ctx: Edge traversal context containing all necessary information

        Returns:
            Query fragment string with normalized identifiers
        """
        query = ""

        # Use normalized edge ID for query generation
        normalized_edge_id = self._get_normalized_edge_id(ctx.edge_id)

        # Check if this edge was already detected as symmetric
        is_symmetric = ctx.edge_id in self._symmetric_edge_map

        # Check if this edge was already detected as subclass expansion
        is_subclass = self._is_subclass_predicate(ctx.edge.get("predicates"))

        edge_filter = self._build_edge_filter(ctx.edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""

        # Build the filter for the target node
        target_filter = self._build_node_filter(ctx.target_node)
        child_visited = ctx.visited | {ctx.target_id}

        # Determine field names based on direction
        if ctx.edge_direction == "out":
            edge_name = f"out_edges_{normalized_edge_id}"
            symmetric_edge_name = f"in_edges-symmetric_{normalized_edge_id}"
            edge_reverse_field = self._v("subject")
            predicate_field = self._v("object")
            symmetric_edge_reverse_field = self._v("object")
            symmetric_predicate_field = self._v("subject")
        else:  # "in"
            edge_name = f"in_edges_{normalized_edge_id}"
            symmetric_edge_name = f"out_edges-symmetric_{normalized_edge_id}"
            edge_reverse_field = self._v("object")
            predicate_field = self._v("subject")
            symmetric_edge_reverse_field = self._v("subject")
            symmetric_predicate_field = self._v("object")

        # Build primary direction
        primary_ctx = DirectionTraversalContext(
            edge_name=edge_name,
            edge_reverse_field=edge_reverse_field,
            predicate_field=predicate_field,
            filter_clause=filter_clause,
            target_id=ctx.target_id,
            target_filter=target_filter,
            child_visited=child_visited,
            nodes=ctx.nodes,
            edges=ctx.edges,
        )
        query += self._build_single_direction_traversal(primary_ctx)

        # If symmetric, also build reverse direction
        if is_symmetric:
            reverse_ctx = DirectionTraversalContext(
                edge_name=symmetric_edge_name,
                edge_reverse_field=symmetric_edge_reverse_field,
                predicate_field=symmetric_predicate_field,
                filter_clause=filter_clause,
                target_id=ctx.target_id,
                target_filter=target_filter,
                child_visited=child_visited,
                nodes=ctx.nodes,
                edges=ctx.edges,
            )
            query += self._build_single_direction_traversal(reverse_ctx)

        if ctx.edge_id in self._subclass_edge_map and not is_subclass:
            source_id = ctx.edge["subject"]
            target_id = ctx.edge["object"]
            source_node = ctx.nodes[source_id]
            target_node = ctx.nodes[target_id]

            # Case 1: ID -> predicate -> ID (build all three forms regardless of direction)
            if self._node_has_ids(source_node) and self._node_has_ids(target_node):
                query += self._build_subclass_form_b(ctx, normalized_edge_id)
                query += self._build_subclass_form_c(ctx, normalized_edge_id)
                query += self._build_subclass_form_d(ctx, normalized_edge_id)

            # Case 2: ID -> predicate -> CAT (only Form B, source must have IDs)
            elif (
                self._node_has_ids(source_node)
                and self._node_has_categories(target_node)
                and not self._node_has_ids(target_node)
            ):
                # Build Form B when we're at the ID node (source) traversing toward the CAT node
                # This works regardless of which direction the pinnedness algorithm chose
                if ctx.edge_direction == "out" and ctx.target_id == target_id:
                    # Forward: at source (ID), going to object (CAT)
                    query += self._build_subclass_form_b(ctx, normalized_edge_id)
                elif ctx.edge_direction == "in" and ctx.target_id == source_id:
                    # Backward: at object (CAT), going to subject (ID) - still need Form B
                    query += self._build_subclass_form_b(ctx, normalized_edge_id)

            # Mirrored Case 2: CAT -> predicate -> ID (only mirrored Form B, target must have IDs)
            elif (
                self._node_has_categories(source_node)
                and not self._node_has_ids(source_node)
                and self._node_has_ids(target_node)
            ):
                # Build mirrored Form B when we're at the ID node (target) looking back to CAT
                if ctx.edge_direction == "out" and ctx.target_id == target_id:
                    # Forward: at CAT (source), going to ID (object)
                    query += self._build_subclass_object_case3_form_b(
                        ctx, normalized_edge_id
                    )
                elif ctx.edge_direction == "in" and ctx.target_id == source_id:
                    # Backward: at ID (object), going back to CAT (subject)
                    query += self._build_subclass_object_case3_form_b(
                        ctx, normalized_edge_id
                    )

        return query

    def _build_single_direction_traversal(self, ctx: DirectionTraversalContext) -> str:
        """Build a single directional edge traversal query fragment.

        Args:
            ctx: Direction traversal context containing all necessary information

        Returns:
            Query fragment string with normalized identifiers
        """
        # Use normalized target node ID for query generation
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)

        query = f"{ctx.edge_name}: ~{ctx.edge_reverse_field}{ctx.filter_clause}"
        query += f" @cascade({self._v('predicate')}, {ctx.predicate_field}) {{ "
        query += self._add_standard_edge_fields()

        query += f"node_{normalized_target_id}: {ctx.predicate_field}"
        if ctx.target_filter:
            query += f" @filter({ctx.target_filter})"

        query += self._build_node_cascade_clause(
            ctx.target_id, ctx.edges, ctx.child_visited
        )

        query += " { "
        query += self._add_standard_node_fields()
        query += self._build_further_hops(
            ctx.target_id, ctx.nodes, ctx.edges, ctx.child_visited
        )
        query += "} } "

        return query

    def _build_further_hops(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        visited: set[QNodeID],
    ) -> str:
        """Recursively build query for all hops connected to the current node_id."""
        query = ""

        # Find outgoing edges from this node (where this node is the SUBJECT)
        for edge_id, edge in edges.items():
            if edge["subject"] == node_id:
                object_id: QNodeID = edge["object"]
                if object_id in visited:
                    continue  # Skip cycles

                object_node = nodes[object_id]
                ctx = EdgeTraversalContext(
                    edge_id=edge_id,
                    edge=edge,
                    target_id=object_id,
                    target_node=object_node,
                    edge_direction="out",
                    visited=visited,
                    nodes=nodes,
                    edges=edges,
                )
                query += self._build_edge_traversal(ctx)

        # Find incoming edges to this node (where this node is the OBJECT)
        for edge_id, edge in edges.items():
            if edge["object"] == node_id:
                source_id: QNodeID = edge["subject"]
                if source_id in visited:
                    continue  # Skip cycles

                source_node = nodes[source_id]
                ctx = EdgeTraversalContext(
                    edge_id=edge_id,
                    edge=edge,
                    target_id=source_id,
                    target_node=source_node,
                    edge_direction="in",
                    visited=visited,
                    nodes=nodes,
                    edges=edges,
                )
                query += self._build_edge_traversal(ctx)

        return query

    # =========================================================================
    # Subclass Expansion Emitters
    # =========================================================================

    def _build_subclass_form_b(self, ctx: EdgeTraversalContext, norm_eid: str) -> str:
        """Form B: A' subclass_of→ A; A' → predicate1 → B."""
        # The intermediate node is on the subject side (in_edges), so it shares
        # the source node alias (node before the predicate edge).
        # ctx.edge_direction == "out": current node is source (n0), target is object (n1)
        #   -> intermediate is between source and predicate -> use source node alias
        # ctx.edge_direction == "in": current node is object (n1), target is source (n0)
        #   -> intermediate is between target (n0) and predicate -> use target node alias
        normalized_source_id = self._get_normalized_node_id(ctx.edge["subject"])
        intermediate_alias = f"node_intermediate_{normalized_source_id}"

        alias = f"in_edges-subclassB_{norm_eid}"
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        query = f"{alias}: ~{self._v('object')}{subclass_filter_clause} @cascade({self._v('predicate')}, {self._v('subject')}) {{ "
        query += self._add_standard_edge_fields()
        mid_edge_alias = f"out_edges-subclassB-mid_{norm_eid}"
        query += f"{intermediate_alias}: {self._v('subject')} @filter(has({self._v('id')})) @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        query += self._add_standard_node_fields()
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        query += f"{mid_edge_alias}: ~{self._v('subject')}{pred_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()
        query += f"node_{normalized_target_id}: {self._v('object')}"
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(
            ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id}
        )
        query += (
            " { "
            + self._add_standard_node_fields()
            + self._build_further_hops(
                ctx.target_id, ctx.nodes, ctx.edges, ctx.visited | {ctx.target_id}
            )
            + " } } } } "
        )
        return query

    def _build_subclass_form_c(self, ctx: EdgeTraversalContext, norm_eid: str) -> str:
        """Form C: A → predicate1 → B'; B' subclass_of→ B."""
        # The intermediate node is on the object side (out_edges), so it shares
        # the target node alias (node after the predicate edge).
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        intermediate_alias = f"node_intermediate_{normalized_target_id}"

        alias = f"out_edges-subclassC_{norm_eid}"
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        query = f"{alias}: ~{self._v('subject')}{pred_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()
        tail_edge_alias = f"out_edges-subclassC-tail_{norm_eid}"
        query += f"{intermediate_alias}: {self._v('object')} @filter(has({self._v('id')})) @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        query += self._add_standard_node_fields()
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        query += f"{tail_edge_alias}: ~{self._v('subject')}{subclass_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()
        query += f"node_{normalized_target_id}: {self._v('object')} "
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(
            ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id}
        )
        query += (
            " { "
            + self._add_standard_node_fields()
            + self._build_further_hops(
                ctx.target_id, ctx.nodes, ctx.edges, ctx.visited | {ctx.target_id}
            )
            + " } } } } "
        )
        return query

    def _build_subclass_form_d(self, ctx: EdgeTraversalContext, norm_eid: str) -> str:
        """Form D: A' subclass_of→ A; A' → predicate1 → B'; B' subclass_of→ B."""
        # intermediateA is on the subject side -> use source node alias
        # intermediateB is on the object side -> use target node alias
        normalized_source_id = self._get_normalized_node_id(ctx.edge["subject"])
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        intermediate_a_alias = f"node_intermediateA_{normalized_source_id}"
        intermediate_b_alias = f"node_intermediateB_{normalized_target_id}"

        alias = f"in_edges-subclassD_{norm_eid}"
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        query = f"{alias}: ~{self._v('object')}{subclass_filter_clause} @cascade({self._v('predicate')}, {self._v('subject')}) {{ "
        query += self._add_standard_edge_fields()
        mid_edge_alias = f"out_edges-subclassD-mid_{norm_eid}"
        query += f"{intermediate_a_alias}: {self._v('subject')} @filter(has({self._v('id')})) @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        query += self._add_standard_node_fields()
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        query += f"{mid_edge_alias}: ~{self._v('subject')}{pred_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()
        tail_edge_alias = f"out_edges-subclassD-tail_{norm_eid}"
        query += f"{intermediate_b_alias}: {self._v('object')} @filter(has({self._v('id')})) @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        query += self._add_standard_node_fields()
        query += f"{tail_edge_alias}: ~{self._v('subject')}{subclass_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()
        query += f"node_{normalized_target_id}: {self._v('object')} "
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(
            ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id}
        )
        query += (
            " { "
            + self._add_standard_node_fields()
            + self._build_further_hops(
                ctx.target_id, ctx.nodes, ctx.edges, ctx.visited | {ctx.target_id}
            )
            + " } } } } } } "
        )
        return query

    def _build_subclass_object_case3_form_b(
        self, ctx: EdgeTraversalContext, norm_eid: str
    ) -> str:
        """Mirrored Case 2 (CAT->P->ID)."""
        # The intermediate node is the subclassed object reached by the
        # predicate edge, so it shares the target node alias.
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        intermediate_alias = f"node_intermediate_{normalized_target_id}"

        alias = f"out_edges-subclassObjB_{norm_eid}"
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        query = (
            f"{alias}: ~{self._v('subject')}{pred_filter_clause} "
            f"@cascade({self._v('predicate')}, {self._v('object')}) {{ "
        )
        query += self._add_standard_edge_fields()
        query += (
            f"{intermediate_alias}: {self._v('object')} "
            f"@filter(has({self._v('id')})) "
            f"@cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        )
        query += self._add_standard_node_fields()
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        tail_edge_alias = f"out_edges-subclassObjB-tail_{norm_eid}"
        query += (
            f"{tail_edge_alias}: ~{self._v('subject')}{subclass_filter_clause} "
            f"@cascade({self._v('predicate')}, {self._v('object')}) {{ "
        )
        query += self._add_standard_edge_fields()
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        query += f"node_{normalized_target_id}: {self._v('object')}"
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(
            ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id}
        )
        query += (
            " { "
            + self._add_standard_node_fields()
            + self._build_further_hops(
                ctx.target_id, ctx.nodes, ctx.edges, ctx.visited | {ctx.target_id}
            )
            + " } } } } "
        )
        return query

    @override
    def convert_batch_multihop(self, qgraphs: list[QueryGraphDict]) -> str:
        """Convert a TRAPI multi-hop batch graph to a batch of Dgraph queries.

        Args:
            qgraphs: List of query graphs to batch

        Returns:
            A combined Dgraph query containing all sub-queries with normalized identifiers
        """
        # Process each query graph in the list
        blocks: list[str] = []
        for i, sub_qgraph in enumerate(qgraphs):
            # Normalize IDs for each qgraph independently
            self._normalize_qgraph_ids(sub_qgraph)

            sub_qgraph_typed: QueryGraphDict = sub_qgraph
            nodes = sub_qgraph_typed["nodes"]
            edges = sub_qgraph_typed["edges"]
            start_node_id = self._find_start_node(nodes, edges)
            # Build each query block with its corresponding batch index
            blocks.append(
                self._build_node_query(start_node_id, nodes, edges, query_index=i)
            )

        # Combine all queries into one batch query
        return "{ " + " ".join(blocks) + " }"

    # =========================================================================
    # TRAPI Translation and Result Assembly
    # =========================================================================

    def _build_trapi_node(self, node: dg.Node) -> NodeDict:
        """Convert a Dgraph Node to a TRAPI NodeDict."""
        attributes: list[AttributeDict] = []

        # Cases that require additional formatting to be TRAPI-compliant
        special_cases: dict[str, tuple[str, Any]] = {}

        for attr_id, value in node.attributes.items():
            if attr_id in special_cases:
                continue
            if value is not None and value not in ([], ""):
                attributes.append(
                    AttributeDict(
                        attribute_type_id=biolink.ensure_prefix(attr_id), value=value
                    )
                )

        for name, value in special_cases.values():
            if value is not None and value not in ([], ""):
                attributes.append(AttributeDict(attribute_type_id=name, value=value))

        trapi_node = NodeDict(
            name=node.name,
            categories=[
                BiolinkEntity(biolink.ensure_prefix(cat)) for cat in node.category
            ],
            attributes=attributes,
        )

        return trapi_node

    def _build_trapi_edge(self, edge: dg.Edge, initial_curie: str) -> EdgeDict:
        """Convert a Dgraph Edge to a TRAPI EdgeDict."""
        attributes: list[AttributeDict] = []
        qualifiers: list[QualifierDict] = []
        sources: list[RetrievalSourceDict] = []

        # Cases that require additional formatting to be TRAPI-compliant
        special_cases: dict[str, tuple[str, Any]] = {
            "category": (
                "biolink:category",
                [
                    BiolinkEntity(biolink.ensure_prefix(cat))
                    for cat in edge.attributes.get("category", [])
                ],
            ),
        }
        for attr_id, value in edge.attributes.items():
            if attr_id in special_cases:
                continue
            if value is not None and value not in ([], ""):
                attributes.append(
                    AttributeDict(
                        attribute_type_id=biolink.ensure_prefix(attr_id), value=value
                    )
                )

        for name, value in special_cases.values():
            if value is not None and value not in ([], ""):
                attributes.append(AttributeDict(attribute_type_id=name, value=value))

        # Build qualifiers
        for qualifier_id, value in edge.qualifiers.items():
            qualifiers.append(
                QualifierDict(
                    qualifier_type_id=QualifierTypeID(
                        biolink.ensure_prefix(qualifier_id)
                    ),
                    qualifier_value=biolink.ensure_prefix(value)
                    if "qualified_predicate" in qualifier_id
                    else value,
                )
            )

        # Build Sources
        for source in edge.sources:
            retrieval_source = RetrievalSourceDict(
                resource_id=Infores(source.resource_id),
                resource_role=source.resource_role,
            )
            if len(source.upstream_resource_ids):
                retrieval_source["upstream_resource_ids"] = [
                    Infores(upstream) for upstream in source.upstream_resource_ids
                ]
            if len(source.source_record_urls):
                retrieval_source["source_record_urls"] = source.source_record_urls
            sources.append(retrieval_source)

        # Build Edge
        trapi_edge = EdgeDict(
            predicate=BiolinkPredicate(biolink.ensure_prefix(edge.predicate)),
            subject=CURIE(edge.node.id if edge.direction == "in" else initial_curie),
            object=CURIE(initial_curie if edge.direction == "in" else edge.node.id),
            sources=sources,
        )
        if len(attributes) > 0:
            trapi_edge["attributes"] = attributes
        if len(qualifiers) > 0:
            trapi_edge["qualifiers"] = qualifiers

        append_aggregator_source(trapi_edge, Infores(CONFIG.tier0.backend_infores))

        return trapi_edge

    def _update_graphs(
        self,
        qedge_id: QEdgeID,
        trapi_edge: EdgeDict,
    ) -> None:
        """Update the knowledge graph and adjacency graph with the given edge.

        Args:
            qedge_id: The original query edge ID that the edge fulfills.
            trapi_edge: The TRAPI representation of the edge fulfilling the QEdge.
        """
        subject_id = trapi_edge["subject"]
        object_id = trapi_edge["object"]
        edge_hash = EdgeIdentifier(hash_hex(hash_edge(trapi_edge)))
        # Update kgraph
        if edge_hash not in self.kgraph["edges"]:
            self.kgraph["edges"][edge_hash] = trapi_edge

        # Update k_agraph
        if subject_id not in self.k_agraph[qedge_id]:
            self.k_agraph[qedge_id][subject_id] = dict[CURIE, list[EdgeIdentifier]]()
        if object_id not in self.k_agraph[qedge_id][subject_id]:
            self.k_agraph[qedge_id][subject_id][object_id] = list[EdgeIdentifier]()
        self.k_agraph[qedge_id][subject_id][object_id].append(edge_hash)

    def _add_node_to_kgraph(self, node: dg.Node, qg: QueryGraphDict) -> bool:
        """Add the node to the kgraph if it isn't already added.

        Returns True if the node meets node constraints.
        """
        if node.id not in self.kgraph["nodes"]:
            trapi_node = self._build_trapi_node(node)
            constraints = qg["nodes"][node.binding].get("constraints", []) or []
            attributes = trapi_node.get("attributes", []) or []

            if not attributes_meet_contraints(constraints, attributes):
                return False

            self.kgraph["nodes"][node.id] = trapi_node

        return True

    def _reconcile_partials(
        self, partials: dict[QEdgeID, list[Partial]]
    ) -> list[Partial]:
        """Reconcile partials for each branch/qedge."""
        reconciled = list[Partial]()
        for combo in itertools.product(*partials.values()):
            if len(combo) == 1:
                reconciled.append(combo[0])
                continue
            reconcile_attempt = combo[0]
            for part in combo[1:]:
                reconcile_attempt = reconcile_attempt.reconcile(part)
                if reconcile_attempt is None:
                    break
            if reconcile_attempt is not None:
                reconciled.append(reconcile_attempt)
        return reconciled

    def _build_results(
        self,
        node: dg.Node,
        qg: QueryGraphDict,
    ) -> list[Partial]:
        """Recursively build results from dgraph response.

        Args:
            node: Parsed node from Dgraph response
            qg: The TRAPI query graph

        Returns:
            List of partial results with original node/edge IDs
        """
        original_node_id = QNodeID(node.binding)

        # The node CURIE (whether it binds to a real QNode or is a subclass)
        node_curie = CURIE(node.id)

        # All nodes (normal *and* subclass) are added to the KGraph
        met_constraints = self._add_node_to_kgraph(node, qg)
        if not met_constraints:
            return []

        # There are two slightly different end conditions
        normal_end_node = len(node.edges) == 0
        subclass_end_node = node.is_subclass_of_expansion and all(
            edge.is_subclass_of_expansion
            and edge.predicate == "subclass_of"
            and len(edge.node.edges) == 0
            for edge in node.edges
        )

        # One case for obtaining subclass backmap
        if subclass_end_node:
            self.subclass_backmap[node_curie] = node.edges[0].node.id
            self._add_node_to_kgraph(node.edges[0].node, qg)

        # When we reach an end condition, return a Partial to kick off result formation
        if normal_end_node or subclass_end_node:
            return [Partial([(original_node_id, node_curie)], [])]

        partials: dict[str, list[Partial]] = {}

        # Loop through the existing edges...subclassing can cause new edges to check
        next_edges = list(node.edges)
        while len(next_edges) > 0:
            edge = next_edges.pop()
            qedge_id = QEdgeID(edge.binding)

            # Skip implicit subclassing subclass_of edges
            if edge.predicate == "subclass_of" and edge.is_subclass_of_expansion:
                # Build subclass backmap
                subclass = node_curie if node.is_subclass_of_expansion else edge.node.id
                ancestor = edge.node.id if node.is_subclass_of_expansion else node_curie
                self.subclass_backmap[subclass] = ancestor

                # Continue on with edges after subclass edge
                next_edges.extend(edge.node.edges)
                continue

            # Build the KGraph edge, doesn't matter if it uses a subclass node or not
            trapi_edge = self._build_trapi_edge(edge, node_curie)

            constraints = qg["edges"][qedge_id].get("constraints", []) or []
            attributes = trapi_edge.get("attributes", []) or []
            if not attributes_meet_contraints(constraints, attributes):
                continue  # We don't continue down the path because this edge breaks it

            self._update_graphs(qedge_id, trapi_edge)

            # Add the current hop to partials from continuing down each path
            for partial in self._build_results(edge.node, qg):
                if qedge_id not in partials:
                    partials[qedge_id] = []
                partials[qedge_id].append(
                    partial.combine(
                        Partial(
                            [(original_node_id, node_curie)],
                            [(qedge_id, trapi_edge["subject"], trapi_edge["object"])],
                        )
                    )
                )

        # Reconciling partials builds up a result for each path
        return self._reconcile_partials(partials)

    # =========================================================================
    # Result Conversion Entrypoint
    # =========================================================================

    @override
    def convert_results(
        self, qgraph: QueryGraphDict, results: list[dg.Node]
    ) -> BackendResult:
        """Convert Dgraph JSON results back to TRAPI BackendResults.

        Args:
            qgraph: The original query graph (for reverse mapping)
            results: Parsed Dgraph response nodes

        Returns:
            TRAPI-formatted results with original node/edge IDs
        """
        logger.info("Begin transforming records")

        # Ensure normalization mappings exist
        # (In case convert_results is called without convert_multihop first)
        if not self._reverse_node_map or not self._reverse_edge_map:
            self._normalize_qgraph_ids(qgraph)

        self.k_agraph = {
            QEdgeID(qedge_id): dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]()
            for qedge_id in qgraph["edges"]
        }

        partial_count = 0
        reconciled = dict[int, Partial]()

        # Apply cascade filtering using the generated query plan when available.
        if self._query_plan is not None:
            logger.debug("Applying OR cascade filter using generated query plan")
            results = self._filter_cascaded_with_or(results, self._query_plan)
            logger.debug(f"Filtered to {len(results)} valid result paths")

        for node in results:
            partials = self._build_results(node, qgraph)
            partial_count += len(partials)
            reconciled.update({hash(part): part for part in partials})

        logger.debug(
            f"Reconciled {partial_count} partials into {len(reconciled)} results."
        )
        trapi_results = [part.as_result(self.k_agraph) for part in reconciled.values()]

        aux_graphs = dict[AuxGraphID, AuxiliaryGraphDict]()
        if len(trapi_results) > 0:
            solve_subclass_edges(
                self.subclass_backmap, self.kgraph, trapi_results, aux_graphs
            )

        logger.info("Finished transforming records")
        return BackendResult(
            results=trapi_results,
            knowledge_graph=self.kgraph,
            auxiliary_graphs=aux_graphs,
        )
