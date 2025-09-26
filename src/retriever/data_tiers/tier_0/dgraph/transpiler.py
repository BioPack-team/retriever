from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, override

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    CURIE,
    AttributeConstraintDict,
    BiolinkEntity,
    QEdgeDict,
    QEdgeID,
    QNodeDict,
    QNodeID,
    QueryGraphDict,
)


class FilterValueProtocol(Protocol):
    """Protocol for values that can be used in filters."""

    @override
    def __str__(self) -> str: ...


class DgraphTranspiler(Transpiler):
    """Transpiler for converting TRAPI queries into Dgraph GraphQL queries."""

    FilterScalar: TypeAlias = str | int | float | bool  # noqa: UP040
    FilterValue: TypeAlias = FilterScalar | list[FilterScalar]  # noqa: UP040

    @override
    def convert_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> str:
        """Convert a single triple to Dgraph query format.

        Not implemented for Dgraph as it's Tier 0 only.
        """
        raise NotImplementedError("Dgraph is Tier 0 only. Use multi-hop methods.")

    @override
    def convert_batch_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> str:
        """Convert a batch of triples to Dgraph query format.

        Not implemented for Dgraph as it's Tier 0 only.
        """
        raise NotImplementedError("Dgraph is Tier 0 only. Use batch multi-hop methods.")

    @override
    def _convert_multihop(self, qgraph: QueryGraphDict) -> str:
        """Convert a TRAPI multi-hop graph to a proper Dgraph multihop query."""
        nodes = qgraph["nodes"]
        edges = qgraph["edges"]

        # Identify the starting node
        start_node_id = self._find_start_node(nodes, edges)

        # Build query from the starting node
        return "{ " + self._build_node_query(start_node_id, nodes, edges) + "}"

    def _find_start_node(
        self,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
    ) -> QNodeID:
        """Find the best node to start the traversal from."""
        # Start with a node that has IDs and is the object of at least one edge
        for node_id, node in nodes.items():
            if node.get("ids") and any(e["object"] == node_id for e in edges.values()):
                return node_id

        # If no ideal start node, just take the first one
        return next(iter(nodes))

    def _build_node_filter(self, node: QNodeDict) -> str:
        """Build a filter expression for a node based on its properties."""
        filters: list[str] = []

        # Handle ID filtering
        ids = node.get("ids")
        if ids:
            if len(ids) == 1:
                # Single ID case - check if it contains a comma (multiple IDs in one string)
                id_value = ids[0]
                if "," in id_value:
                    # Multiple IDs in a single string - split them
                    id_list = [id_val.strip() for id_val in id_value.split(",")]
                    ids_str = ', '.join(f'"{id_val}"' for id_val in id_list)
                    filters.append(f'eq(id, [{ids_str}])')
                else:
                    # Single ID
                    filters.append(f'eq(id, "{id_value}")')
            else:
                # Multiple IDs in array
                ids_str = ', '.join(f'"{id_val}"' for id_val in ids)
                filters.append(f'eq(id, [{ids_str}])')

        # Handle category filtering
        categories = node.get("categories")
        if categories:
            if len(categories) == 1:
                filters.append(f'eq(category, "{categories[0]}")')
            elif len(categories) > 1:
                categories_str = ', '.join(f'"{cat}"' for cat in categories)
                filters.append(f'eq(category, [{categories_str}])')

        # Handle attribute constraints
        constraints = node.get("constraints")
        if constraints:
            filters.extend(self._convert_constraints_to_filters(constraints))

        # If no filters, use a generic filter
        if not filters:
            return "has(id)"

        # Combine all filters with AND
        if len(filters) == 1:
            return filters[0]
        else:
            return " AND ".join(filters)

    def _build_edge_filter(self, edge: QEdgeDict) -> str:
        """Build a filter expression for an edge based on its properties."""
        filters: list[str] = []

        # Handle predicates (multiple) filtering
        predicates = edge.get("predicates")
        if predicates:
            if len(predicates) == 1:
                filters.append(f'eq(predicate, "{predicates[0]}")')
            elif len(predicates) > 1:
                predicates_str = ', '.join(f'"{pred}"' for pred in predicates)
                filters.append(f'eq(predicate, [{predicates_str}])')

        # Handle attribute constraints
        attribute_constraints = edge.get("attribute_constraints")
        if attribute_constraints:
            filters.extend(self._convert_constraints_to_filters(attribute_constraints))

        # If no filters, return empty string
        if not filters:
            return ""

        # Combine all filters with AND
        if len(filters) == 1:
            return filters[0]
        else:
            return " AND ".join(filters)

    def _convert_constraints_to_filters(self, constraints: list[AttributeConstraintDict]) -> list[str]:
        """Convert TRAPI attribute constraints to Dgraph filter expressions."""
        filters: list[str] = []

        for constraint in constraints:
            filter_expr = self._create_filter_expression(constraint)
            filters.append(filter_expr)

        return filters

    def _create_filter_expression(self, constraint: AttributeConstraintDict) -> str:
        """Create a filter expression for a single constraint."""
        field_name = constraint["id"]
        value = constraint["value"]
        operator = constraint["operator"]
        is_negated = constraint.get("not", False)

        # Generate the appropriate filter expression based on the operator
        filter_expr = self._get_operator_filter(field_name, operator, value)

        # Handle negation
        if is_negated:
            filter_expr = f'NOT({filter_expr})'

        return filter_expr

    def _get_operator_filter(self, field_name: str, operator: str, value: Any) -> str:
        """Generate filter expression based on operator type."""
        # Group operators by their filter type
        if operator == "in":
            # List membership requires special handling
            return self._create_in_filter(field_name, value)

        if operator == "matches":
            # Text matching
            return f'anyoftext({field_name}, "{value}")'

        # All other operators map to Dgraph functions
        func_map = {
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
            # value is list[FilterScalar] here; ensure string formatting
            quoted_items = [f'"{item!s}"' for item in value] if value else []
            values_str = ", ".join(quoted_items)
            return f'eq({field_name}, [{values_str}])'
        else:
            # Single scalar value
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
                return f'eq(id, [{ids_str}])'
            # Single ID - most selective query
            return f'eq(id, "{id_value}")'

        # Multiple IDs in array
        ids_str = ", ".join(f'"{id_val}"' for id_val in id_list)
        return f'eq(id, [{ids_str}])'

    def _create_category_filter(
        self,
        categories: Sequence[str] | Sequence[BiolinkEntity],
    ) -> str:
        """Create a filter for category fields."""
        cat_vals = [str(c) for c in categories]
        if len(cat_vals) == 1:
            return f'eq(category, "{cat_vals[0]}")'
        categories_str = ", ".join(f'"{cat}"' for cat in cat_vals)
        return f'eq(category, [{categories_str}])'

    def _get_primary_and_secondary_filters(self, node: QNodeDict) -> tuple[str, list[str]]:
        """Extract primary and secondary filters from a node."""
        primary_filter = "has(id)"  # Default
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
        """Generate standard node fields."""
        return ("id name category "
                "all_names all_categories "
                "iri equivalent_curies "
                "description publications ")

    def _add_standard_edge_fields(self) -> str:
        """Generate standard edge fields."""
        return ("predicate primary_knowledge_source "
                "knowledge_level agent_type "
                "kg2_ids domain_range_exclusion "
                "edge_id ")

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

        query = f"in_edges: ~source{filter_clause} {{ "

        # Include all standard edge fields
        query += self._add_standard_edge_fields()

        # Get primary and secondary filters for source node
        primary_filter, secondary_filters = self._get_primary_and_secondary_filters(source)

        # Create the source filter clause
        source_filter_clause = ""
        if primary_filter != "has(id)":
            if not secondary_filters:
                source_filter_clause = f" @filter({primary_filter})"
            else:
                source_filter_clause = f" @filter({primary_filter} AND {' AND '.join(secondary_filters)})"
        elif secondary_filters:
            source_filter_clause = self._build_filter_clause(secondary_filters)

        query += f"node: target{source_filter_clause} {{ "

        # Include all standard node fields for the target
        query += self._add_standard_node_fields()

        # Recursively add further hops
        query += self._build_further_hops(
            source_id, context.nodes, context.edges, context.visited.copy()
        )

        # Close the blocks
        query += "} } "

        return query

    def _build_node_query(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
    ) -> str:
        """Recursively build a query for a node and its connected nodes."""
        node = nodes[node_id]

        # Get primary and secondary filters
        primary_filter, secondary_filters = self._get_primary_and_secondary_filters(node)

        # Start the query with the primary filter
        query = f"node(func: {primary_filter})"

        # Add secondary filters if present
        query += self._build_filter_clause(secondary_filters)

        # Add cascade and open block
        query += " @cascade { "

        # Include all standard node fields
        query += self._add_standard_node_fields()

        # Find incoming edges to this node (where this node is the OBJECT)
        connected_edges: dict[QNodeID, list[QEdgeDict]] = {}
        for edge in edges.values():
            if edge["object"] == node_id:
                source_id: QNodeID = edge["subject"]
                if source_id not in connected_edges:
                    connected_edges[source_id] = []
                connected_edges[source_id].append(edge)

        # Create a context object for edge processing
        context = self.EdgeConnectionContext(nodes=nodes, edges=edges, visited={node_id})

        # For each source node, build the edge traversal
        for source_id, source_edges in connected_edges.items():
            edge = source_edges[0]
            source = nodes[source_id]
            query += self._process_edge_connection(
                edge=edge,
                source=source,
                source_id=source_id,
                context=context,
            )

        # Close the node block
        query += "} "
        return query

    def _build_further_hops(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        visited: set[QNodeID],
    ) -> str:
        """Build query for further hops beyond the first one."""
        # Prevent cycles
        if node_id in visited:
            return ""

        visited.add(node_id)

        query = ""

        # Find incoming edges to this node
        for edge in edges.values():
            if edge["object"] == node_id:
                # Get the source node and predicate
                source_id = edge["subject"]
                if source_id in visited:
                    continue  # Skip already visited nodes

                source = nodes[source_id]

                # Build edge filter
                edge_filter = self._build_edge_filter(edge)
                filter_clause = f" @filter({edge_filter})" if edge_filter else ""

                query += f"in_edges: ~source{filter_clause} {{ "

                # Include all standard edge fields
                query += self._add_standard_edge_fields()

                # Build source node filter using the node_filter helper
                source_filter = self._build_node_filter(source)
                source_filter_clause = f" @filter({source_filter})" if source_filter != "has(id)" else ""

                query += f"node: target{source_filter_clause} {{ "

                # Include all standard node fields for the target
                query += self._add_standard_node_fields()

                # Recursively add further hops with updated context
                new_visited = visited.copy()
                query += self._build_further_hops(source_id, nodes, edges, new_visited)

                # Close the blocks
                query += "} } "

        return query

    @override
    def _convert_batch_multihop(self, qgraphs: list[QueryGraphDict]) -> str:
        """Convert a TRAPI multi-hop batch graph to a batch of Dgraph queries.

        Returns:
            A combined Dgraph query containing all sub-queries
        """
        # Process each query graph in the list
        blocks: list[str] = []
        for i, sub_qgraph in enumerate(qgraphs):
            sub_qgraph_typed: QueryGraphDict = sub_qgraph
            query = self._convert_multihop(sub_qgraph_typed)
            query = query.replace("node(func:", f"node{i}(func:", 1)
            blocks.append(query.strip()[1:-1])

        # Combine all queries into one batch query
        return "{" + "".join(q.strip("{}") for q in blocks) + "}"

    @override
    def convert_results(
        self, qgraph: QueryGraphDict, results: Any
    ) -> BackendResult:
        """Convert Dgraph JSON results back to TRAPI BackendResults."""
        # Create a properly structured BackendResult with all required fields
        return BackendResult(
            results=results,
            knowledge_graph={"nodes": {}, "edges": {}},  # KnowledgeGraphDict requires nodes and edges
            auxiliary_graphs={}  # Required field in BackendResult
        )
