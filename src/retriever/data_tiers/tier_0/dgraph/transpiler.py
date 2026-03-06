import itertools
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, override

import orjson
from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers.base_transpiler import Tier0Transpiler
from retriever.data_tiers.tier_0.dgraph import result_models as dg
from retriever.lookup.partial import Partial
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


class FilterValueProtocol(Protocol):
    """Protocol for values that can be used in filters."""

    @override
    def __str__(self) -> str: ...


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

    # Normalization mappings for injection prevention
    _node_id_map: dict[QNodeID, str]
    _edge_id_map: dict[QEdgeID, str]
    _reverse_node_map: dict[str, QNodeID]
    _reverse_edge_map: dict[str, QEdgeID]

    def __init__(
        self,
        version: str | None = None,
        subclassing_enabled: bool = True,
        symmetric_root_enabled: bool = True,
    ) -> None:
        """Initialize a Transpiler instance.

        Args:
            version: An optional version string to prefix to all schema fields.
            subclassing_enabled: Whether to emit subclass-based subqueries per hop.
            symmetric_root_enabled: When True, emit symmetric-direction traversals
                as separate root query blocks instead of inline siblings.
        """
        super().__init__()
        self.kgraph: KnowledgeGraphDict = KnowledgeGraphDict(nodes={}, edges={})
        self.k_agraph: KAdjacencyGraph
        self.version = version
        self.prefix = f"{version}_" if version else ""

        # Initialize normalization mappings
        self._node_id_map = {}
        self._edge_id_map = {}
        self._reverse_node_map = {}
        self._reverse_edge_map = {}

        self._symmetric_edge_map: dict[QEdgeID, tuple[str, str]] = {}
        self.subclassing_enabled = subclassing_enabled
        self.symmetric_root_enabled = symmetric_root_enabled
        # Set to the edge being rendered in symmetric-root mode; None otherwise.
        self._symmetric_root_edge_id: QEdgeID | None = None
        # Set to the edge being rendered in embedded CAT→ID subclassing mode; None otherwise.
        self._subclassing_cat_to_id_edge_id: QEdgeID | None = None

    def _v(self, field: str) -> str:
        """Return the versioned field name."""
        return f"{self.prefix}{field}"

    def _aliased_fields(self, fields: list[str]) -> str:
        """Return a string of aliased fields if a version is set, otherwise return just the field names."""
        if self.version:
            return " ".join(f"{field}: {self._v(field)}" for field in fields) + " "
        return " ".join(fields) + " "

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

    def _filter_cascaded_with_or(
        self, nodes: list[dg.Node], symmetric_edges: dict[QEdgeID, tuple[str, str]]
    ) -> list[dg.Node]:
        """Filter results to enforce cascade with OR logic between symmetric edge directions.

        Args:
            nodes: Parsed Dgraph response nodes
            symmetric_edges: Map of edge IDs to their (primary, symmetric) field name pairs
                        e.g., {QEdgeID('e0'): ('in_edges_e0', 'out_edges-symmetric_e0')}

        Returns:
            Filtered list of nodes that satisfy cascade with OR logic
        """

        def validate_edge_path(
            node: dg.Node, symmetric_map: Mapping[QEdgeID, tuple[str, str]]
        ) -> bool:
            """Recursively validate all edges in the path."""
            if not node.edges:
                return True

            # Group edges by their query edge ID
            edges_by_qid: dict[QEdgeID, list[dg.Edge]] = {}
            for edge in node.edges:
                qid = QEdgeID(edge.binding)
                if qid not in edges_by_qid:
                    edges_by_qid[qid] = []
                edges_by_qid[qid].append(edge)

            # Check each edge group - must have at least one valid path
            for qid, edge_group in edges_by_qid.items():
                if qid in symmetric_map:
                    # For symmetric edges, check if at least one direction has valid nested paths
                    has_valid = False
                    for edge in edge_group:
                        if validate_edge_path(edge.node, symmetric_map):
                            has_valid = True
                            break
                    if not has_valid:
                        return False
                else:
                    # For non-symmetric edges, all must be valid
                    for edge in edge_group:
                        if not validate_edge_path(edge.node, symmetric_map):
                            return False

            return True

        # Filter top-level nodes
        filtered: list[dg.Node] = [
            node for node in nodes if validate_edge_path(node, symmetric_edges)
        ]

        return filtered

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

        # Build query from the starting node, providing a default query_index of 0
        main_block = self._build_node_query(start_node_id, nodes, edges, query_index=0)

        subclassing_blocks = ""
        if self.subclassing_enabled:
            subclassing_blocks = self._build_subclassing_root_queries(
                nodes, edges, start_node_id=start_node_id, query_index=0
            )

        symmetric_blocks = ""
        if self.symmetric_root_enabled:
            symmetric_blocks = self._build_symmetric_root_queries(
                nodes, edges, start_node_id, query_index=0
            )

        return "{ " + main_block + subclassing_blocks + symmetric_blocks + "}"

    # --- Split Query API ---

    def build_split_queries(
        self, qgraph: QueryGraphDict
    ) -> tuple[str, list[str], list[str]]:
        """Build separate, standalone DQL queries rather than one combined query.

        Each symmetric edge and each subclassing expansion form is emitted as its
        own ``{ ... }`` DQL query string.  Running them independently (in parallel)
        allows partial failures to be isolated and keeps individual queries smaller.

        Args:
            qgraph: The TRAPI query graph to transpile.

        Returns:
            A 3-tuple of:
            - ``main_query``: The primary traversal DQL query.
            - ``symmetric_queries``: One DQL query per symmetric edge (may be empty).
            - ``subclassing_queries``: One DQL query per subclassing expansion form
              per eligible edge (may be empty).

        Note:
            Results from all three groups must be merged before calling
            :meth:`convert_split_results`.  Main + symmetric results can be passed
            directly; subclassing results will have their intermediate nodes
            collapsed automatically inside ``convert_split_results``.
        """
        self._normalize_qgraph_ids(qgraph)
        nodes = qgraph["nodes"]
        edges = qgraph["edges"]
        start_node_id = self._find_start_node(nodes, edges)

        # Main query (also triggers _detect_symmetric_edges via _build_node_query)
        main_block = self._build_node_query(start_node_id, nodes, edges, query_index=0)
        main_query = "{ " + main_block + " }"

        # One standalone query per symmetric edge
        symmetric_queries: list[str] = []
        if self.symmetric_root_enabled:
            for block_str, n_id in self._iter_symmetric_query_strings(
                nodes, edges, start_node_id, query_index=0
            ):
                symmetric_queries.append(
                    self._make_standalone_query(block_str, query_index=0, normalized_node_id=n_id)
                )

        # One standalone query per subclassing expansion form
        subclassing_queries: list[str] = []
        if self.subclassing_enabled:
            for block_str, n_id in self._iter_subclassing_query_strings(
                nodes, edges, start_node_id, query_index=0
            ):
                subclassing_queries.append(
                    self._make_standalone_query(block_str, query_index=0, normalized_node_id=n_id)
                )

        return main_query, symmetric_queries, subclassing_queries

    def _make_standalone_query(
        self, block_str: str, *, query_index: int, normalized_node_id: str
    ) -> str:
        """Wrap a root block string as a self-contained DQL query.

        The block's custom name (e.g. ``q0subclassing_formB_e0_node_n0``) is
        replaced with a standard ``q<N>_node_<id>`` alias so that
        :class:`~retriever.data_tiers.tier_0.dgraph.result_models.DgraphResponse`
        can parse the result into ``data["q<N>"]`` without any changes to the
        existing parser.
        """
        paren_idx = block_str.index("(")
        block_body = block_str[paren_idx:]
        standard_alias = f"q{query_index}_node_{normalized_node_id}"
        return "{ " + standard_alias + block_body + " }"

    def _iter_subclassing_query_strings(
        self,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        start_node_id: QNodeID,
        query_index: int,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(block_str, normalized_root_node_id)`` for every subclassing form.

        Mirrors the logic of :meth:`_build_subclassing_root_queries` but yields
        each expansion form block individually rather than concatenating them.
        """
        for edge_id, edge in edges.items():
            if self._is_subclass_of_predicate(edge):
                continue

            subject_node = nodes[edge["subject"]]
            object_node = nodes[edge["object"]]
            subject_has_ids = self._node_has_ids(subject_node)
            object_has_ids = self._node_has_ids(object_node)
            subject_cat_only = self._node_has_categories_only(subject_node)
            object_cat_only = self._node_has_categories_only(object_node)

            n_edge_id = self._get_normalized_edge_id(edge_id)
            n_subject_id = self._get_normalized_node_id(edge["subject"])
            n_object_id = self._get_normalized_node_id(edge["object"])

            kwargs: dict = dict(
                edge_id=edge_id,
                edge=edge,
                subject_node=subject_node,
                object_node=object_node,
                nodes=nodes,
                edges=edges,
                query_index=query_index,
                normalized_edge_id=n_edge_id,
                normalized_subject_id=n_subject_id,
                normalized_object_id=n_object_id,
            )

            # Case 1: ID → P → ID  →  Forms B, C, D
            if subject_has_ids and object_has_ids:
                yield self._build_subclassing_form_b_id_to_id(**kwargs), n_subject_id
                yield self._build_subclassing_form_c(**kwargs), n_subject_id
                yield self._build_subclassing_form_d(**kwargs), n_subject_id

            # Case 2: ID → P → CAT-only  →  Form B only
            elif subject_has_ids and object_cat_only:
                yield self._build_subclassing_form_b_id_to_cat(**kwargs), n_subject_id

            # Case 3: CAT-only → P → ID  →  Embedded Form B from the main start node
            elif subject_cat_only and object_has_ids:
                n_start_id = self._get_normalized_node_id(start_node_id)
                yield (
                    self._build_subclassing_form_b_cat_to_id_embedded(
                        start_node_id=start_node_id, **kwargs
                    ),
                    n_start_id,
                )

    def _iter_symmetric_query_strings(
        self,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        start_node_id: QNodeID,
        query_index: int,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(block_str, normalized_root_node_id)`` for every symmetric edge.

        Mirrors the logic of :meth:`_build_symmetric_root_queries` but yields
        each per-edge block individually rather than concatenating them.

        Returns nothing if no symmetric edges were detected.
        """
        if not self._symmetric_edge_map:
            return

        n_start_id = self._get_normalized_node_id(start_node_id)
        start_node = nodes[start_node_id]
        start_filter = self._build_node_filter(start_node, primary=True)

        for edge_id in self._symmetric_edge_map:
            n_edge_id = self._get_normalized_edge_id(edge_id)
            block_name = f"q{query_index}symmetric_{n_edge_id}_node_{n_start_id}"

            self._symmetric_root_edge_id = edge_id
            try:
                q = f"{block_name}(func: {start_filter})"
                q += self._build_node_cascade_clause(start_node_id, edges, {start_node_id})
                q += " { "
                q += self._add_standard_node_fields()
                q += self._build_further_hops(start_node_id, nodes, edges, {start_node_id})
                q += "} "
            finally:
                self._symmetric_root_edge_id = None

            yield q, n_start_id

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

    # --- Pinnedness Algorithm Methods ---

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

    # --- Nodes and Edges Methods ---

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
                qtype = str(q["qualifier_type_id"]).replace("biolink:", "")
                qval = q["qualifier_value"]
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

    def _build_node_cascade_clause(
        self,
        node_id: QNodeID,
        edges: Mapping[QEdgeID, QEdgeDict],
        visited: set[QNodeID],
    ) -> str:
        """Build a @cascade(...) clause for a node block.

        Always require id. When symmetric_root_enabled, symmetric edges are traversed
        in their own root block so they are treated the same as regular edges here.
        When symmetric_root_enabled is False (legacy), symmetric edges are omitted from
        cascade because the inline OR logic handles them in post-processing.
        """
        cascade_fields: list[str] = [self._v("id")]

        # When symmetric_root_enabled, treat symmetric edges like ordinary edges;
        # otherwise exclude them (handled by post-processing OR logic).
        def _include_edge(e_id: QEdgeID) -> bool:
            return self.symmetric_root_enabled or e_id not in self._symmetric_edge_map

        # Check outgoing edges (node as subject) to unvisited objects.
        # - Normal edges add ~vJ_subject to cascade.
        # - CAT→ID embedded subclassing edges add ~vJ_object (traversal is reversed).
        # - Symmetric root edges traverse the *reverse* direction (~vJ_object) in their
        #   standalone query, so intermediate nodes must include ~vJ_object in @cascade
        #   to enforce that the node actually participates in that reversed traversal.
        has_relevant_out = False
        has_relevant_out_subclassing = False
        has_symmetric_root_out = False
        for e_id, e in edges.items():
            if (
                e["subject"] == node_id
                and e["object"] not in visited
                and _include_edge(e_id)
            ):
                if e_id == self._subclassing_cat_to_id_edge_id:
                    has_relevant_out_subclassing = True
                elif e_id == self._symmetric_root_edge_id:
                    # Symmetric expansion for this edge traverses ~vJ_object, not
                    # ~vJ_subject, so the node cascade must reflect that.
                    has_symmetric_root_out = True
                else:
                    has_relevant_out = True

        if has_relevant_out:
            cascade_fields.append(f"~{self._v('subject')}")

        # All cases that require ~vJ_object in cascade (deduplicated):
        #   - CAT→ID embedded subclassing edges (traversal is reversed)
        #   - Symmetric root outgoing edges (reverse direction uses ~vJ_object)
        #   - Incoming edges (node as object, subject is unvisited)
        needs_object_cascade = has_relevant_out_subclassing or has_symmetric_root_out

        # Check incoming edges (node as object) to unvisited subjects
        if not needs_object_cascade:
            for e_id, e in edges.items():
                if (
                    e["object"] == node_id
                    and e["subject"] not in visited
                    and _include_edge(e_id)
                ):
                    needs_object_cascade = True
                    break

        if needs_object_cascade:
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

    # --- Subclassing Expansion Methods ---

    def _node_has_ids(self, node: QNodeDict) -> bool:
        """Return True if the node has explicit ID constraints."""
        return bool(node.get("ids"))

    def _node_has_categories_only(self, node: QNodeDict) -> bool:
        """Return True if the node has categories but no IDs."""
        return not self._node_has_ids(node) and bool(node.get("categories"))

    def _is_subclass_of_predicate(self, edge: QEdgeDict) -> bool:
        """Return True if ALL predicates on this edge are subclass_of."""
        predicates = edge.get("predicates") or []
        if not predicates:
            return False
        return all(
            str(p).replace("biolink:", "") == "subclass_of" for p in predicates
        )

    def _build_subclassing_root_queries(
        self,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        start_node_id: QNodeID,
        query_index: int,
    ) -> str:
        """Build all subclassing expansion root query blocks for every eligible hop.

        For each edge, determine which case applies and emit the corresponding
        extra root query blocks alongside the main query.

        Cases:
            0  – predicate is subclass_of → skip entirely.
            1  – ID→P→ID   → Forms B, C, D.
            2  – ID→P→CAT  → Form B only.
            3  – CAT→P→ID  → Mirrored Form B only.

        Args:
            nodes: All query nodes.
            edges: All query edges.
            query_index: Batch index used in root block naming.

        Returns:
            Zero or more additional root query block strings (no outer braces).
        """
        result = ""

        for edge_id, edge in edges.items():
            # Case 0: skip subclass_of predicates entirely
            if self._is_subclass_of_predicate(edge):
                logger.debug(
                    f"Subclassing: skipping edge {edge_id} (predicate is subclass_of)"
                )
                continue

            subject_node = nodes[edge["subject"]]
            object_node = nodes[edge["object"]]
            subject_has_ids = self._node_has_ids(subject_node)
            object_has_ids = self._node_has_ids(object_node)
            subject_cat_only = self._node_has_categories_only(subject_node)
            object_cat_only = self._node_has_categories_only(object_node)

            n_edge_id = self._get_normalized_edge_id(edge_id)
            n_subject_id = self._get_normalized_node_id(edge["subject"])
            n_object_id = self._get_normalized_node_id(edge["object"])

            kwargs: dict = dict(
                edge_id=edge_id,
                edge=edge,
                subject_node=subject_node,
                object_node=object_node,
                nodes=nodes,
                edges=edges,
                query_index=query_index,
                normalized_edge_id=n_edge_id,
                normalized_subject_id=n_subject_id,
                normalized_object_id=n_object_id,
            )

            # Case 1: ID → P → ID  →  Forms B, C, D
            if subject_has_ids and object_has_ids:
                logger.debug(
                    f"Subclassing: edge {edge_id} matches Case 1 (ID→P→ID), "
                    "emitting Forms B, C, D"
                )
                result += self._build_subclassing_form_b_id_to_id(**kwargs)
                result += self._build_subclassing_form_c(**kwargs)
                result += self._build_subclassing_form_d(**kwargs)

            # Case 2: ID → P → CAT-only  →  Form B only
            elif subject_has_ids and object_cat_only:
                logger.debug(
                    f"Subclassing: edge {edge_id} matches Case 2 (ID→P→CAT), "
                    "emitting Form B"
                )
                result += self._build_subclassing_form_b_id_to_cat(**kwargs)

            # Case 3: CAT-only → P → ID  →  Embedded Form B from the main start node.
            # We traverse from the start node, follow all preceding hops normally,
            # and at this edge emit: CAT←P←X; X→subclass_of→ID (inline subclassing).
            elif subject_cat_only and object_has_ids:
                logger.debug(
                    f"Subclassing: edge {edge_id} matches Case 3 (CAT→P→ID), "
                    "emitting embedded Form B from start node"
                )
                result += self._build_subclassing_form_b_cat_to_id_embedded(
                    start_node_id=start_node_id,
                    **kwargs,
                )

        return result

    def _build_subclass_of_edge_segment(self, *, incoming: bool) -> str:
        """Build a subclass_of traversal segment for an intermediate node.

        Each hop includes expand(vJ_Edge) for edge fields and the intermediate node
        receives expand(vJ_Node), @filter(has(vJ_id)), and a @cascade that includes
        the outbound traversal direction used by the caller's next hop.

        Args:
            incoming: If True, traverses ~object so that the returned intermediate
                      node A' satisfies (A' subclass_of A).
                      Dgraph path: A → ~object → subclass_of_edge → subject → A'
                      If False, traverses ~subject so that the returned intermediate
                      node B' satisfies (B' subclass_of B).
                      Dgraph path: B → ~subject → subclass_of_edge → object → B'

        Returns:
            Open Dgraph query fragment (caller must close the two open braces —
            one for the intermediate node block and one for the edge block).
        """
        subclass_filter = f'eq({self._v("predicate_ancestors")}, "subclass_of")'
        if incoming:
            # A → ~vJ_object → subclass_of_edge → vJ_subject → A'
            # Next hop from A' will traverse ~vJ_subject, so cascade on that.
            return (
                f"~{self._v('object')} @filter({subclass_filter})"
                f" @cascade({self._v('predicate')}, {self._v('subject')}) {{ "
                f"{self._add_standard_edge_fields()}"
                f"node_intermediate: {self._v('subject')}"
                f" @filter(has({self._v('id')}))"
                f" @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
                f"{self._add_standard_node_fields()}"
            )
        else:
            # B → ~vJ_subject → subclass_of_edge → vJ_object → B'
            # Next hop from B' will traverse ~vJ_object, so cascade on that.
            return (
                f"~{self._v('subject')} @filter({subclass_filter})"
                f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
                f"{self._add_standard_edge_fields()}"
                f"node_intermediate: {self._v('object')}"
                f" @filter(has({self._v('id')}))"
                f" @cascade({self._v('id')}, ~{self._v('object')}) {{ "
                f"{self._add_standard_node_fields()}"
            )

    def _build_predicate_edge_segment(
        self,
        edge: QEdgeDict,
        *,
        outgoing: bool,
        target_node: QNodeDict,
        normalized_target_id: str,
    ) -> str:
        """Build the main predicate edge segment with constraints applied.

        Args:
            edge: The query edge carrying predicates/qualifier constraints.
            outgoing: If True, forward traversal (subject→object).
                      If False, reverse traversal (object→subject).
            target_node: QNodeDict for the far-end node.
            normalized_target_id: Normalized ID string for the target node alias.

        Returns:
            Open Dgraph query fragment (caller must close the two open braces).
        """
        edge_filter = self._build_edge_filter(edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""
        target_filter = self._build_node_filter(target_node)

        if outgoing:
            reverse_field = self._v("subject")
            predicate_field = self._v("object")
        else:
            reverse_field = self._v("object")
            predicate_field = self._v("subject")

        segment = (
            f"~{reverse_field}{filter_clause}"
            f" @cascade({self._v('predicate')}, {predicate_field}) {{ "
            f"{self._add_standard_edge_fields()}"
            f"node_{normalized_target_id}: {predicate_field}"
        )
        if target_filter:
            segment += f" @filter({target_filter})"
        segment += f" @cascade({self._v('id')}) {{ "
        segment += self._add_standard_node_fields()
        return segment

    # ── Form B (Case 1, ID→ID): A'←subclass_of←A; A'→P→B ─────────────────────

    def _build_subclassing_form_b_id_to_id(
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        subject_node: QNodeDict,
        object_node: QNodeDict,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int,
        normalized_edge_id: str,
        normalized_subject_id: str,
        normalized_object_id: str,
    ) -> str:
        """Case 1 Form B: root at A (ID), expand subclasses A', then A'→P→B.

        Traversal:
            ID:A →  in_edges-subclassB  → ~vJ_object → subclass_of → vJ_subject → A'
            A'   →  out_edges-subclassB-mid → ~vJ_subject → P → vJ_object → ID:B
        """
        subject_filter = self._build_node_filter(subject_node, primary=True)
        block_name = (
            f"q{query_index}subclassing_formB_{normalized_edge_id}"
            f"_node_{normalized_subject_id}"
        )
        # Root cascade requires the first edge hop (~vJ_object) to be present.
        q = f"{block_name}(func: {subject_filter}) @cascade({self._v('id')}, ~{self._v('object')}) {{ "
        q += self._add_standard_node_fields()

        # A → ~vJ_object → subclass_of → vJ_subject → A'  (intermediate node with expand)
        q += f"in_edges-subclassB_{normalized_edge_id}: "
        q += self._build_subclass_of_edge_segment(incoming=True)

        # A' → ~vJ_subject → P → vJ_object → ID:B
        q += f"out_edges-subclassB-mid_{normalized_edge_id}: "
        q += self._build_predicate_edge_segment(
            edge,
            outgoing=True,
            target_node=object_node,
            normalized_target_id=normalized_object_id,
        )
        # Close: B-node { }, pred-edge { }, intermediate-node { }, subclass-edge { }, root { }
        q += "} } } } } "
        return q

    # ── Form C (Case 1, ID→ID): A→P→B'; B'←subclass_of←B ─────────────────────

    def _build_subclassing_form_c(
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        subject_node: QNodeDict,
        object_node: QNodeDict,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int,
        normalized_edge_id: str,
        normalized_subject_id: str,
        normalized_object_id: str,
    ) -> str:
        """Case 1 Form C: root at A (ID), A→P→B', where B' is a subclass of B.

        Traversal:
            ID:A →  out_edges-subclassC  → ~vJ_subject → P → vJ_object → B'
            B'   →  out_edges-subclassC-tail → ~vJ_subject → subclass_of → vJ_object → ID:B
        """
        subject_filter = self._build_node_filter(subject_node, primary=True)
        block_name = (
            f"q{query_index}subclassing_formC_{normalized_edge_id}"
            f"_node_{normalized_subject_id}"
        )
        edge_filter = self._build_edge_filter(edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""
        subclass_filter = f'eq({self._v("predicate_ancestors")}, "subclass_of")'

        # Root cascade requires the first edge hop (~vJ_subject) to be present.
        q = f"{block_name}(func: {subject_filter}) @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        q += self._add_standard_node_fields()

        # A → ~vJ_subject → P → vJ_object → B'  (intermediate node with expand)
        q += f"out_edges-subclassC_{normalized_edge_id}: "
        q += (
            f"~{self._v('subject')}{filter_clause}"
            f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
            f"{self._add_standard_edge_fields()}"
            f"node_intermediate: {self._v('object')}"
            f" @filter(has({self._v('id')}))"
            f" @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        )
        q += self._add_standard_node_fields()

        # B' → ~vJ_subject → subclass_of → vJ_object → B  (final target with full expand)
        q += f"out_edges-subclassC-tail_{normalized_edge_id}: "
        q += (
            f"~{self._v('subject')} @filter({subclass_filter})"
            f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
            f"{self._add_standard_edge_fields()}"
            f"node_{normalized_object_id}: {self._v('object')}"
            f" @filter({self._build_node_filter(object_node)})"
            f" @cascade({self._v('id')}) {{ "
            f"{self._add_standard_node_fields()}"
        )

        # Close: B-node { }, subclass-tail-edge { }, intermediate-node { }, pred-edge { }, root { }
        q += "} } } } } "
        return q

    # ── Form D (Case 1, ID→ID): A'←sc←A; A'→P→B'; B'←sc←B ───────────────────

    def _build_subclassing_form_d(
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        subject_node: QNodeDict,
        object_node: QNodeDict,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int,
        normalized_edge_id: str,
        normalized_subject_id: str,
        normalized_object_id: str,
    ) -> str:
        """Case 1 Form D: both endpoints expanded via subclassing.

        Traversal:
            ID:A →  in_edges-subclassD  → ~vJ_object → subclass_of → vJ_subject → A'
            A'   →  out_edges-subclassD-mid → ~vJ_subject → P → vJ_object → B'
            B'   →  out_edges-subclassD-tail → ~vJ_subject → subclass_of → vJ_object → ID:B
        """
        subject_filter = self._build_node_filter(subject_node, primary=True)
        block_name = (
            f"q{query_index}subclassing_formD_{normalized_edge_id}"
            f"_node_{normalized_subject_id}"
        )
        edge_filter = self._build_edge_filter(edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""
        subclass_filter = f'eq({self._v("predicate_ancestors")}, "subclass_of")'

        # Root cascade requires the first edge hop (~vJ_object) to be present.
        q = f"{block_name}(func: {subject_filter}) @cascade({self._v('id')}, ~{self._v('object')}) {{ "
        q += self._add_standard_node_fields()

        # A → ~vJ_object → subclass_of → vJ_subject → A'  (intermediate node with expand)
        q += f"in_edges-subclassD_{normalized_edge_id}: "
        q += self._build_subclass_of_edge_segment(incoming=True)

        # A' → ~vJ_subject → P → vJ_object → B'  (intermediate node with expand)
        q += f"out_edges-subclassD-mid_{normalized_edge_id}: "
        q += (
            f"~{self._v('subject')}{filter_clause}"
            f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
            f"{self._add_standard_edge_fields()}"
            f"node_intermediate_B: {self._v('object')}"
            f" @filter(has({self._v('id')}))"
            f" @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        )
        q += self._add_standard_node_fields()

        # B' → ~vJ_subject → subclass_of → vJ_object → B  (final target with full expand)
        q += f"out_edges-subclassD-tail_{normalized_edge_id}: "
        q += (
            f"~{self._v('subject')} @filter({subclass_filter})"
            f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
            f"{self._add_standard_edge_fields()}"
            f"node_{normalized_object_id}: {self._v('object')}"
            f" @filter({self._build_node_filter(object_node)})"
            f" @cascade({self._v('id')}) {{ "
            f"{self._add_standard_node_fields()}"
        )

        # Close: B-node { }, subclass-tail-edge { }, intermediate_B { }, pred-mid-edge { },
        #        intermediate_A { } (from _build_subclass_of_edge_segment), subclass-in-edge { }, root { }
        q += "} } } } } } } "
        return q

    # ── Form B (Case 2, ID→CAT): A'←subclass_of←A; A'→P→CAT ─────────────────

    def _build_subclassing_form_b_id_to_cat(
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        subject_node: QNodeDict,
        object_node: QNodeDict,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int,
        normalized_edge_id: str,
        normalized_subject_id: str,
        normalized_object_id: str,
    ) -> str:
        """Case 2 Form B: root at A (ID), expand subclasses A', then A'→P→CAT.

        Traversal:
            ID:A →  in_edges-subclassB  → ~vJ_object → subclass_of → vJ_subject → A'
            A'   →  out_edges-subclassB-mid → ~vJ_subject → P → vJ_object → CAT:B
        """
        subject_filter = self._build_node_filter(subject_node, primary=True)
        block_name = (
            f"q{query_index}subclassing_formB_{normalized_edge_id}"
            f"_node_{normalized_subject_id}"
        )
        # Root cascade requires the first edge hop (~vJ_object) to be present.
        q = f"{block_name}(func: {subject_filter}) @cascade({self._v('id')}, ~{self._v('object')}) {{ "
        q += self._add_standard_node_fields()

        # A → ~vJ_object → subclass_of → vJ_subject → A'  (intermediate node with expand)
        q += f"in_edges-subclassB_{normalized_edge_id}: "
        q += self._build_subclass_of_edge_segment(incoming=True)

        # A' → ~vJ_subject → P → vJ_object → CAT:B
        q += f"out_edges-subclassB-mid_{normalized_edge_id}: "
        q += self._build_predicate_edge_segment(
            edge,
            outgoing=True,
            target_node=object_node,
            normalized_target_id=normalized_object_id,
        )
        # Close: CAT-node { }, pred-edge { }, intermediate-node { }, subclass-edge { }, root { }
        q += "} } } } } "
        return q

    # ── Mirrored Form B (Case 3, CAT→ID): CAT→P→B'; B'←subclass_of←B ─────────

    def _build_subclassing_form_b_cat_to_id(
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        subject_node: QNodeDict,
        object_node: QNodeDict,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int,
        normalized_edge_id: str,
        normalized_subject_id: str,
        normalized_object_id: str,
    ) -> str:
        """Case 3 Mirrored Form B: root at B (ID), expand subclasses B', confirm CAT→P→B'.

        B has IDs. We root from B, walk backward through subclass_of to find B'
        (things that are subclasses of B), then confirm CAT→P→B' exists.

        Traversal:
            ID:B →  out_edges-subclassB-mir  → ~vJ_subject → subclass_of → vJ_object → B'
            B'   →  in_edges-subclassB-mir-mid → ~vJ_object → P → vJ_subject → CAT:A
        """
        object_filter = self._build_node_filter(object_node, primary=True)
        block_name = (
            f"q{query_index}subclassing_formB_mirrored_{normalized_edge_id}"
            f"_node_{normalized_object_id}"
        )
        # Root cascade requires the first edge hop (~vJ_subject) to be present.
        q = f"{block_name}(func: {object_filter}) @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        q += self._add_standard_node_fields()

        # B → ~vJ_subject → subclass_of → vJ_object → B'  (intermediate node with expand)
        # _build_subclass_of_edge_segment(incoming=False) traverses ~vJ_subject → vJ_object.
        # The intermediate node gets @cascade(vJ_id, ~vJ_object) so next hop (~vJ_object) is enforced.
        q += f"out_edges-subclassB-mir_{normalized_edge_id}: "
        q += self._build_subclass_of_edge_segment(incoming=False)

        # B' → ~vJ_object → P → vJ_subject → CAT:A
        q += f"in_edges-subclassB-mir-mid_{normalized_edge_id}: "
        q += self._build_predicate_edge_segment(
            edge,
            outgoing=False,
            target_node=subject_node,
            normalized_target_id=normalized_subject_id,
        )
        # Close: CAT-node { }, pred-edge { }, intermediate-node { }, subclass-edge { }, root { }
        q += "} } } } } "
        return q

    # ── Embedded Form B (Case 3, CAT→ID): start from main start node ─────────

    def _build_subclassing_form_b_cat_to_id_embedded(
        self,
        *,
        edge_id: QEdgeID,
        edge: QEdgeDict,
        subject_node: QNodeDict,
        object_node: QNodeDict,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int,
        normalized_edge_id: str,
        normalized_subject_id: str,
        normalized_object_id: str,
        start_node_id: QNodeID,
    ) -> str:
        """Case 3 Embedded Form B: root at the main start node; embed subclassing inline.

        Instead of starting from the ID-end node (B), we start from start_node,
        traverse all preceding hops normally, and at this CAT→ID edge emit:

            CAT ← P ← X  (X is some intermediate node that relates to CAT)
            X → subclass_of → ID:B

        This keeps the full traversal path anchored at the query's start node,
        avoiding the disconnected root problem of the mirrored approach.
        """
        n_start_id = self._get_normalized_node_id(start_node_id)
        start_node = nodes[start_node_id]
        start_filter = self._build_node_filter(start_node, primary=True)
        block_name = (
            f"q{query_index}subclassing_formB_{normalized_edge_id}"
            f"_node_{n_start_id}"
        )

        self._subclassing_cat_to_id_edge_id = edge_id
        try:
            q = f"{block_name}(func: {start_filter})"
            q += self._build_node_cascade_clause(start_node_id, edges, {start_node_id})
            q += " { "
            q += self._add_standard_node_fields()
            q += self._build_further_hops(start_node_id, nodes, edges, {start_node_id})
            q += "} "
        finally:
            self._subclassing_cat_to_id_edge_id = None

        return q

    def _build_subclassing_cat_to_id_inline(
        self,
        ctx: EdgeTraversalContext,
    ) -> str:
        """Build the inline subclassing traversal for a CAT→ID edge (Case 3 embedded).

        From the CAT node (edge subject), traverse in the *reverse* (symmetric) direction
        to find intermediate nodes X, then confirm X→subclass_of→ID.

        Traversal from CAT (ctx's current positional node):
            CAT ← P ← X          (in_edges-subclassObjB: ~vJ_object @filter(P))
            X → subclass_of → ID  (out_edges-subclassObjB-tail: ~vJ_subject @filter(subclass_of))
        """
        normalized_edge_id = self._get_normalized_edge_id(ctx.edge_id)
        edge_filter = self._build_edge_filter(ctx.edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""
        object_filter = self._build_node_filter(ctx.target_node)
        subclass_filter = f'eq({self._v("predicate_ancestors")}, "subclass_of")'
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)

        # CAT ← P ← X  (CAT is the object of P, X is the subject)
        q = f"in_edges-subclassObjB_{normalized_edge_id}: "
        q += f"~{self._v('object')}{filter_clause}"
        q += f" @cascade({self._v('predicate')}, {self._v('subject')}) {{ "
        q += self._add_standard_edge_fields()

        # Intermediate node X
        q += f"node_intermediate: {self._v('subject')}"
        q += f" @filter(has({self._v('id')}))"
        q += f" @cascade({self._v('id')}, ~{self._v('subject')}) {{ "
        q += self._add_standard_node_fields()

        # X → subclass_of → ID:B
        q += f"out_edges-subclassObjB-tail_{normalized_edge_id}: "
        q += f"~{self._v('subject')} @filter({subclass_filter})"
        q += f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        q += self._add_standard_edge_fields()

        q += f"node_{normalized_target_id}: {self._v('object')}"
        if object_filter:
            q += f" @filter({object_filter})"
        q += f" @cascade({self._v('id')}) {{ "
        q += self._add_standard_node_fields()

        # Close: node_ID { }, subclass-tail-edge { }, intermediate { }, main-edge { }
        q += "} } } } "
        return q

    # --- Subclassing Result Flattening Methods ---

    def _find_terminal_real_node(
        self,
        node: dg.Node,
        qgraph: QueryGraphDict,
        _depth: int = 0,
    ) -> dg.Node | None:
        """Return the first real (qgraph-bound) node reachable via subclass_of hops.

        Starting from *node* (which may be an intermediate produced by a
        subclassing query), the method follows any edges whose predicate contains
        ``"subclass_of"`` until it finds a node whose :attr:`~dg.Node.binding`
        matches a key in ``qgraph["nodes"]``.

        Args:
            node: Starting node (may be intermediate).
            qgraph: The original TRAPI query graph used to identify real nodes.
            _depth: Recursion guard (max 8 levels).

        Returns:
            The first real node found, or ``None`` if none is reachable.
        """
        if _depth > 8:
            return None
        if node.binding in qgraph["nodes"]:
            return node
        for edge in node.edges:
            if "subclass_of" in edge.predicate:
                found = self._find_terminal_real_node(edge.node, qgraph, _depth + 1)
                if found is not None:
                    return found
        return None

    def _flatten_subclassing_edges(
        self,
        edges: list[dg.Edge],
        qgraph: QueryGraphDict,
        _depth: int = 0,
    ) -> list[dg.Edge]:
        """Collapse intermediate subclassing nodes in an edge list.

        Subclassing expansion queries introduce intermediate nodes (e.g. A' in
        ``A→subclass_of→A'→P→B``).  These nodes are not in the query-graph and
        would cause :meth:`_build_results` to crash.  This method rewrites the
        edge list so that every edge points directly to a real (qgraph-bound)
        target node:

        * **Subclass_of hop → intermediate**: skip the hop; absorb its children
          into the current level.
        * **Predicate hop → intermediate**: keep the edge's predicate/sources/
          qualifiers/attributes but replace the target with the real terminal
          node found via :meth:`_find_terminal_real_node`.
        * **Any hop → real node**: recurse into the target normally.

        Args:
            edges: Raw edge list from a parsed subclassing result node.
            qgraph: The original TRAPI query graph.
            _depth: Recursion guard (max 8 levels).

        Returns:
            Rewritten edge list whose every target is a real qgraph node.
        """
        if _depth > 8:
            return []

        result: list[dg.Edge] = []
        for edge in edges:
            target = edge.node
            if target.binding in qgraph["nodes"]:
                # Direct edge to a real node — recurse into the node's own edges
                flattened_target = dataclasses.replace(
                    target,
                    edges=self._flatten_subclassing_edges(
                        target.edges, qgraph, _depth + 1
                    ),
                )
                result.append(dataclasses.replace(edge, node=flattened_target))
            else:
                # Intermediate node
                if "subclass_of" in edge.predicate:
                    # Subclass_of hop — skip it and absorb real edges from intermediate
                    result.extend(
                        self._flatten_subclassing_edges(
                            target.edges, qgraph, _depth + 1
                        )
                    )
                else:
                    # Real predicate hop into an intermediate — find the terminal real node
                    terminal = self._find_terminal_real_node(target, qgraph)
                    if terminal is not None:
                        result.append(dataclasses.replace(edge, node=terminal))

        return result

    def _flatten_subclassing_node(
        self, node: dg.Node, qgraph: QueryGraphDict
    ) -> dg.Node:
        """Return *node* with all intermediate subclassing hops collapsed.

        Applies :meth:`_flatten_subclassing_edges` to the node's top-level edge
        list so that the result can be processed by the standard
        :meth:`_build_results` pipeline without modifications.

        Args:
            node: A root-level node from a subclassing expansion query result.
            qgraph: The original TRAPI query graph.

        Returns:
            A copy of *node* whose edges point only to real qgraph nodes.
        """
        return dataclasses.replace(
            node,
            edges=self._flatten_subclassing_edges(node.edges, qgraph),
        )

    # --- Symmetric Root Query Methods ---

    def _build_symmetric_root_queries(
        self,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        start_node_id: QNodeID,
        query_index: int,
    ) -> str:
        """Emit one separate root query block per symmetric edge.

        For each symmetric edge in the query graph, a new root block is emitted
        that traverses that edge in the reverse (symmetric) direction while
        continuing all other hops normally.  The main query block only contains
        primary-direction traversals.

        Args:
            nodes: All query nodes.
            edges: All query edges.
            start_node_id: The anchor node used for the main query.
            query_index: Batch index used in root block naming.

        Returns:
            Zero or more additional root query block strings (no outer braces).
        """
        if not self._symmetric_edge_map:
            return ""

        n_start_id = self._get_normalized_node_id(start_node_id)
        start_node = nodes[start_node_id]
        start_filter = self._build_node_filter(start_node, primary=True)
        result = ""

        for edge_id in self._symmetric_edge_map:
            n_edge_id = self._get_normalized_edge_id(edge_id)
            block_name = (
                f"q{query_index}symmetric_{n_edge_id}_node_{n_start_id}"
            )
            logger.debug(
                f"Symmetric root: emitting block {block_name} for edge {edge_id}"
            )

            self._symmetric_root_edge_id = edge_id
            try:
                q = f"{block_name}(func: {start_filter})"
                q += self._build_node_cascade_clause(start_node_id, edges, {start_node_id})
                q += " { "
                q += self._add_standard_node_fields()
                q += self._build_further_hops(start_node_id, nodes, edges, {start_node_id})
                q += "} "
            finally:
                self._symmetric_root_edge_id = None

            result += q

        return result

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
        # Pre-detect symmetric edges before building any queries
        self._detect_symmetric_edges(edges)

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
        # Build the reverse-direction context (used in symmetric root mode or legacy inline)
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
        else:
            reverse_ctx = None

        if (
            self._subclassing_cat_to_id_edge_id is not None
            and self._subclassing_cat_to_id_edge_id == ctx.edge_id
            and ctx.edge_direction == "out"
        ):
            # Embedded CAT→ID subclassing: traverse via ~vJ_object + subclass_of tail.
            query += self._build_subclassing_cat_to_id_inline(ctx)
        elif (
            self.symmetric_root_enabled
            and is_symmetric
            and self._symmetric_root_edge_id == ctx.edge_id
        ):
            # Symmetric-root mode for this specific edge: emit reverse direction only.
            assert reverse_ctx is not None
            query += self._build_single_direction_traversal(reverse_ctx)
        else:
            # Primary direction.
            query += self._build_single_direction_traversal(primary_ctx)
            # Legacy inline symmetric (only when symmetric_root is disabled).
            if not self.symmetric_root_enabled and is_symmetric:
                assert reverse_ctx is not None
                query += self._build_single_direction_traversal(reverse_ctx)

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

            if self.subclassing_enabled:
                sc = self._build_subclassing_root_queries(nodes, edges, start_node_id=start_node_id, query_index=i)
                if sc:
                    blocks.append(sc)

            if self.symmetric_root_enabled:
                sym = self._build_symmetric_root_queries(
                    nodes, edges, start_node_id, query_index=i
                )
                if sym:
                    blocks.append(sym)

        # Combine all queries into one batch query
        return "{ " + " ".join(blocks) + " }"

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
                    qualifier_value=value,
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

    def _build_results(self, node: dg.Node, qg: QueryGraphDict) -> list[Partial]:
        """Recursively build results from dgraph response.

        Args:
            node: Parsed node from Dgraph response (with bindings already restored to original IDs)
            qg: The TRAPI query graph used in getting the response.

        Returns:
            List of partial results with original node/edge IDs
        """
        original_node_id = QNodeID(node.binding)

        if node.id not in self.kgraph["nodes"]:
            trapi_node = self._build_trapi_node(node)
            constraints = qg["nodes"][original_node_id].get("constraints", []) or []
            attributes = trapi_node.get("attributes", []) or []

            if not attributes_meet_contraints(constraints, attributes):
                return []

            self.kgraph["nodes"][CURIE(node.id)] = trapi_node

        # If we hit a stop condition, return partial for the node
        if not len(node.edges):
            return [Partial([(original_node_id, CURIE(node.id))], [])]

        partials = {QEdgeID(edge.binding): list[Partial]() for edge in node.edges}

        for edge in node.edges:
            qedge_id = QEdgeID(edge.binding)
            trapi_edge = self._build_trapi_edge(edge, node.id)

            constraints = qg["edges"][qedge_id].get("constraints", []) or []
            attributes = trapi_edge.get("attributes", []) or []

            if not attributes_meet_contraints(constraints, attributes):
                continue

            self._update_graphs(qedge_id, trapi_edge)

            for partial in self._build_results(edge.node, qg):
                partials[qedge_id].append(
                    partial.combine(
                        Partial(
                            [(original_node_id, CURIE(node.id))],
                            [(qedge_id, trapi_edge["subject"], trapi_edge["object"])],
                        )
                    )
                )

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

        # Apply symmetric cascade filtering BEFORE building results
        if self._symmetric_edge_map:
            logger.debug(
                f"Applying symmetric cascade filter for {len(self._symmetric_edge_map)} edges"
            )
            results = self._filter_cascaded_with_or(results, self._symmetric_edge_map)
            logger.debug(f"Filtered to {len(results)} valid result paths")

        for node in results:
            partials = self._build_results(node, qgraph)
            partial_count += len(partials)
            reconciled.update({hash(part): part for part in partials})

        logger.debug(
            f"Reconciled {partial_count} partials into {len(reconciled)} results."
        )
        trapi_results = [part.as_result(self.k_agraph) for part in reconciled.values()]

        logger.info("Finished transforming records")
        return BackendResult(
            results=trapi_results,
            knowledge_graph=self.kgraph,
            auxiliary_graphs=dict[AuxGraphID, AuxiliaryGraphDict](),
        )

    def convert_split_results(
        self,
        qgraph: QueryGraphDict,
        main_and_sym_nodes: list[dg.Node],
        subclassing_nodes: list[dg.Node],
    ) -> BackendResult:
        """Convert results from split queries back to TRAPI.

        This is the counterpart to :meth:`build_split_queries`.  It handles the
        two conceptually different result sets that come back from parallel
        query execution:

        * **main_and_sym_nodes**: Root nodes from the main traversal query and
          any per-symmetric-edge queries.  These have the same nested structure
          as a regular ``convert_multihop`` response and are processed as-is
          (after the symmetric OR-cascade filter is applied).

        * **subclassing_nodes**: Root nodes from subclassing expansion queries
          (Forms B, C, D and embedded Form B).  Their trees contain intermediate
          nodes (A', B') that are not present in *qgraph*.  These are first
          passed through :meth:`_flatten_subclassing_node` to collapse the
          intermediate hops into direct ``A → predicate → B`` edges before
          being fed into :meth:`_build_results`.

        Args:
            qgraph: The original TRAPI query graph.
            main_and_sym_nodes: Merged node list from the main + symmetric queries.
            subclassing_nodes: Merged node list from all subclassing expansion
                queries.

        Returns:
            ``BackendResult`` with the combined TRAPI knowledge graph and results.
        """
        logger.info("Begin transforming records (split query mode)")

        # Ensure normalization mappings exist
        if not self._reverse_node_map or not self._reverse_edge_map:
            self._normalize_qgraph_ids(qgraph)

        self.k_agraph = {
            QEdgeID(qedge_id): dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]()
            for qedge_id in qgraph["edges"]
        }

        # Apply symmetric cascade filter to main + symmetric nodes
        if self._symmetric_edge_map:
            logger.debug(
                f"Applying symmetric cascade filter for {len(self._symmetric_edge_map)} edges"
            )
            main_and_sym_nodes = self._filter_cascaded_with_or(
                main_and_sym_nodes, self._symmetric_edge_map
            )
            logger.debug(
                f"Filtered to {len(main_and_sym_nodes)} valid main/symmetric paths"
            )

        # Flatten intermediate nodes from subclassing expansion results
        flattened_subclassing: list[dg.Node] = [
            self._flatten_subclassing_node(node, qgraph) for node in subclassing_nodes
        ]
        logger.debug(
            f"Flattened {len(subclassing_nodes)} subclassing root nodes"
        )

        all_nodes = main_and_sym_nodes + flattened_subclassing

        partial_count = 0
        reconciled = dict[int, Partial]()

        for node in all_nodes:
            partials = self._build_results(node, qgraph)
            partial_count += len(partials)
            reconciled.update({hash(part): part for part in partials})

        logger.debug(
            f"Reconciled {partial_count} partials into {len(reconciled)} results."
        )
        trapi_results = [part.as_result(self.k_agraph) for part in reconciled.values()]

        logger.info("Finished transforming records")
        return BackendResult(
            results=trapi_results,
            knowledge_graph=self.kgraph,
            auxiliary_graphs=dict[AuxGraphID, AuxiliaryGraphDict](),
        )
