import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, override

from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers.base_transpiler import Tier0Transpiler
from retriever.data_tiers.tier_0.dgraph import result_models as dg
from retriever.lookup.partial import Partial
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
    subclassing_enabled: bool

    # Normalization mappings for injection prevention
    _node_id_map: dict[QNodeID, str]
    _edge_id_map: dict[QEdgeID, str]
    _reverse_node_map: dict[str, QNodeID]
    _reverse_edge_map: dict[str, QEdgeID]

    def __init__(self, version: str | None = None, subclassing_enabled: bool = True) -> None:
        """Initialize a Transpiler instance.

        Args:
            version: An optional version string to prefix to all schema fields.
            subclassing_enabled: Enable subclass expansion (default True).
        """
        super().__init__()
        self.kgraph: KnowledgeGraphDict = KnowledgeGraphDict(nodes={}, edges={})
        self.k_agraph: KAdjacencyGraph
        self.version = version
        self.prefix = f"{version}_" if version else ""
        self.subclassing_enabled = subclassing_enabled

        # Initialize normalization mappings
        self._node_id_map = {}
        self._edge_id_map = {}
        self._reverse_node_map = {}
        self._reverse_edge_map = {}

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

    @override
    def process_qgraph(
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> str:
        return super().process_qgraph(qgraph, *additional_qgraphs)

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
        return (
            "{ "
            + self._build_node_query(start_node_id, nodes, edges, query_index=0)
            + "}"
        )

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
        return max(pinnedness_scores, key=lambda nid: (pinnedness_scores[nid], nid))

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

        # Handle attribute constraints
        attribute_constraints = edge.get("attribute_constraints")
        if attribute_constraints:
            filters.extend(self._convert_constraints_to_filters(attribute_constraints))

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

        Always require id. Require edge aliases (out_edges_*, in_edges_*)
        if there are corresponding traversals from this node to not-yet-visited nodes.
        """
        cascade_fields: list[str] = [self._v("id")]

        # If this node has any outgoing edges (node as subject) to unvisited objects,
        # include their alias names so the block cascades only when such edges exist.
        for eid, e in edges.items():
            if e["subject"] == node_id and e["object"] not in visited:
                normalized_eid = self._get_normalized_edge_id(eid)
                cascade_fields.append(f"out_edges_{normalized_eid}")

        # If this node has any incoming edges (node as object) to unvisited subjects,
        # include their alias names similarly.
        for eid, e in edges.items():
            if e["object"] == node_id and e["subject"] not in visited:
                normalized_eid = self._get_normalized_edge_id(eid)
                cascade_fields.append(f"in_edges_{normalized_eid}")

        # Always emit a cascade; at minimum it will include the id
        return f" @cascade({', '.join(cascade_fields)})"

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

    def _is_subclass_predicate(self, predicates: Sequence[str] | None) -> bool:
        """Return True if predicates contain biolink:subclass_of."""
        if not predicates:
            return False
        return any(str(p).endswith("subclass_of") for p in predicates)

    def _node_has_ids(self, node: QNodeDict) -> bool:
        ids = node.get("ids")
        return bool(ids and len(ids) > 0)

    def _node_has_categories(self, node: QNodeDict) -> bool:
        cats = node.get("categories")
        return bool(cats and len(cats) > 0)

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

        # Check if predicate is symmetric
        predicates = ctx.edge.get("predicates") or []
        is_symmetric = any(biolink.is_symmetric(str(pred)) for pred in predicates)
        is_subclass = self._is_subclass_predicate(predicates)

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

        # --- Subclass expansions (Cases 1/2), skip if subclass_of itself (Cases 0a/0b) ---
        if self.subclassing_enabled and not is_subclass:
            # Original source/target nodes
            source_id = ctx.edge["subject"]
            target_id = ctx.edge["object"]
            source_node = ctx.nodes[source_id]
            target_node = ctx.nodes[target_id]

            # Case 1: ID -> predicate -> ID
            if self._node_has_ids(source_node) and self._node_has_ids(target_node):
                query += self._build_subclass_form_b(ctx, normalized_edge_id)  # A' -> predicate -> B
                query += self._build_subclass_form_c(ctx, normalized_edge_id)  # A -> predicate -> B'
                query += self._build_subclass_form_d(ctx, normalized_edge_id)  # A' -> predicate -> B'
            # Case 2: ID -> predicate -> CAT
            elif self._node_has_ids(source_node) and self._node_has_categories(target_node):
                query += self._build_subclass_form_b(ctx, normalized_edge_id)

        return query

    def _subclass_edge_filter(self) -> str:
        """Filter clause for subclass_of edges only."""
        return f'eq({self._v("predicate_ancestors")}, "subclass_of")'

    def _build_subclass_form_b(self, ctx: EdgeTraversalContext, norm_eid: str) -> str:
        """Form B: A' subclass_of→ A; A' → predicate1 → B. Alias: in_edges-subclassB_eX."""
        # Traverse reverse from subclass A' into current node A
        alias = f"in_edges-subclassB_{norm_eid}"
        # No constraints on subclass edge; use only subclass_of filter
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        query = f"{alias}: ~{self._v('object')}{subclass_filter_clause} {{ "  # A' -> subclass_of -> A (reverse)
        query += self._add_standard_edge_fields()

        # Now from A', traverse the original predicate1 to B with original edge filters
        # Emit child node with original constraints preserved for target B
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        query += f"out_edges_{norm_eid}: ~{self._v('subject')}{pred_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()
        query += f"node_{normalized_target_id}: {self._v('object')}"
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id})
        query += " { " + self._add_standard_node_fields() + " } } } "
        return query

    def _build_subclass_form_c(self, ctx: EdgeTraversalContext, norm_eid: str) -> str:
        """Form C: A → predicate1 → B'; B' subclass_of→ B. Alias: in_edges-subclassC_eX."""
        alias = f"in_edges-subclassC_{norm_eid}"
        # First traverse original predicate1 to B'
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        query = f"{alias}: ~{ctx.edge_direction == 'out' and self._v('subject') or self._v('object')}{pred_filter_clause} @cascade({self._v('predicate')}, {ctx.edge_direction == 'out' and self._v('object') or self._v('subject')}) {{ "
        query += self._add_standard_edge_fields()

        # Then from B', traverse subclass_of to B (reverse on object side)
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        query += f"in_edges-subclassC-tail_{norm_eid}: ~{self._v('object')}{subclass_filter_clause} {{ "
        query += self._add_standard_edge_fields()
        query += f"node_{normalized_target_id}: {self._v('object')} "
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id})
        query += " { " + self._add_standard_node_fields() + " } } } "
        return query

    def _build_subclass_form_d(self, ctx: EdgeTraversalContext, norm_eid: str) -> str:
        """Form D: A' subclass_of→ A; A' → predicate1 → B'; B' subclass_of→ B. Alias: in_edges-subclassD_eX."""
        alias = f"in_edges-subclassD_{norm_eid}"
        # A' -> subclass_of -> A
        subclass_filter_clause = f" @filter({self._subclass_edge_filter()})"
        query = f"{alias}: ~{self._v('object')}{subclass_filter_clause} {{ "
        query += self._add_standard_edge_fields()

        # A' -> predicate1 -> B'
        pred_edge_filter = self._build_edge_filter(ctx.edge)
        pred_filter_clause = f" @filter({pred_edge_filter})" if pred_edge_filter else ""
        query += f"out_edges-subclassD-mid_{norm_eid}: ~{self._v('subject')}{pred_filter_clause} @cascade({self._v('predicate')}, {self._v('object')}) {{ "
        query += self._add_standard_edge_fields()

        # B' -> subclass_of -> B
        normalized_target_id = self._get_normalized_node_id(ctx.target_id)
        query += f"in_edges-subclassD-tail_{norm_eid}: ~{self._v('object')}{subclass_filter_clause} {{ "
        query += self._add_standard_edge_fields()
        query += f"node_{normalized_target_id}: {self._v('object')} "
        target_filter = self._build_node_filter(ctx.target_node)
        if target_filter:
            query += f" @filter({target_filter})"
        query += self._build_node_cascade_clause(ctx.target_id, ctx.edges, ctx.visited | {ctx.target_id})
        query += " { " + self._add_standard_node_fields() + " } } } } "
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

        # Combine all queries into one batch query
        return "{ " + " ".join(blocks) + " }"

    def _build_trapi_node(self, node: dg.Node) -> NodeDict:
        """Convert a Dgraph Node to a TRAPI NodeDict."""
        attributes: list[AttributeDict] = []

        # Cases that require additional formatting to be TRAPI-compliant
        special_cases: dict[str, tuple[str, Any]] = {
            "equivalent_identifiers": (
                "biolink:xref",
                [CURIE(i) for i in node.attributes.get("equivalent_identifiers", [])],
            )
        }

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
