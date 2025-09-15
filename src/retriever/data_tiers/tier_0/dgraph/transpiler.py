from typing import Any, cast, override

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    AttributeConstraintDict,
    QEdgeDict,
    QNodeDict,
    QueryGraphDict,
)


class DgraphTranspiler(Transpiler):
    """Transpiler for converting TRAPI queries into Dgraph GraphQL queries."""

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
        return "{\n" + self._build_node_query(start_node_id, nodes, edges) + "}"

    def _find_start_node(self, nodes: dict[str, QNodeDict], edges: dict[str, QEdgeDict]) -> str:
        """Find the best node to start the traversal from."""
        # Start with a node that has IDs and is the object of at least one edge
        for node_id, node in nodes.items():
            if node.get("ids") and any(e["object"] == node_id for e in edges.values()):
                return node_id

        # If no ideal start node, just take the first one
        return next(iter(nodes))

    def _build_node_filter(self, node: QNodeDict) -> str:
        """Build a filter expression for a node based on its properties."""
        filters = []

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
        filters = []

        # Handle predicate filtering
        predicate = edge.get("predicate")
        if predicate:
            filters.append(f'eq(predicate, "{predicate}")')

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
        filters = []

        for constraint in constraints:
            field_name = constraint["id"]
            value = constraint["value"]
            operator = constraint["operator"]
            is_negated = constraint.get("not", False)

            # Handle different operators
            if operator in ("==", "="):
                filter_expr = f'eq({field_name}, "{value}")'
            elif operator == ">":
                filter_expr = f'gt({field_name}, "{value}")'
            elif operator == ">=":
                filter_expr = f'ge({field_name}, "{value}")'
            elif operator == "<":
                filter_expr = f'lt({field_name}, "{value}")'
            elif operator == "<=":
                filter_expr = f'le({field_name}, "{value}")'
            elif operator == "in":
                # Value should be a list
                if isinstance(value, list):
                    values_str = ', '.join(f'"{val}"' for val in value)
                    filter_expr = f'eq({field_name}, [{values_str}])'
                else:
                    # Single value
                    filter_expr = f'eq({field_name}, "{value}")'
            elif operator == "matches":
                # For string fields that should contain the value
                filter_expr = f'anyoftext({field_name}, "{value}")'
            else:
                # Default to equality
                filter_expr = f'eq({field_name}, "{value}")'

            # Handle negation
            if is_negated:
                filter_expr = f'NOT({filter_expr})'

            filters.append(filter_expr)

        return filters

    # Breaking down _get_primary_and_secondary_filters into smaller methods
    # to reduce branch complexity (PLR0912)
    def _create_id_filter(self, ids: list[str]) -> str:
        """Create a filter for ID fields."""
        if len(ids) == 1:
            id_value = ids[0]
            if "," in id_value:
                # Multiple IDs in a single string - split them
                id_list = [id_val.strip() for id_val in id_value.split(",")]
                ids_str = ', '.join(f'"{id_val}"' for id_val in id_list)
                return f'eq(id, [{ids_str}])'
            else:
                # Single ID - most selective query
                return f'eq(id, "{id_value}")'
        else:
            # Multiple IDs in array
            ids_str = ', '.join(f'"{id_val}"' for id_val in ids)
            return f'eq(id, [{ids_str}])'

    def _create_category_filter(self, categories: list[str]) -> str:
        """Create a filter for category fields."""
        if len(categories) == 1:
            return f'eq(category, "{categories[0]}")'
        else:
            categories_str = ', '.join(f'"{cat}"' for cat in categories)
            return f'eq(category, [{categories_str}])'

    def _get_primary_and_secondary_filters(self, node: QNodeDict) -> tuple[str, list[str]]:
        """Extract primary and secondary filters from a node."""
        primary_filter = "has(id)"  # Default
        secondary_filters = []

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
        if ids and node.get("categories"):
            categories = node["categories"]
            category_filter = self._create_category_filter(categories)
            secondary_filters.append(category_filter)

        # Add constraints as secondary filters
        constraints = node.get("constraints")
        if constraints:
            secondary_filters.extend(self._convert_constraints_to_filters(constraints))

        return primary_filter, secondary_filters

    def _add_standard_node_fields(self, indent: str) -> str:
        """Generate standard node fields."""
        return (f"{indent}id\n{indent}name\n{indent}category\n"
                f"{indent}all_names\n{indent}all_categories\n"
                f"{indent}iri\n{indent}equivalent_curies\n"
                f"{indent}description\n{indent}publications\n")

    def _add_standard_edge_fields(self, indent: str) -> str:
        """Generate standard edge fields."""
        return (f"{indent}predicate\n{indent}primary_knowledge_source\n"
                f"{indent}knowledge_level\n{indent}agent_type\n"
                f"{indent}kg2_ids\n{indent}domain_range_exclusion\n"
                f"{indent}edge_id\n")

    def _build_filter_clause(self, filters: list[str]) -> str:
        """Build filter clause from a list of filters."""
        if not filters:
            return ""
        elif len(filters) == 1:
            return f" @filter({filters[0]})"
        else:
            return f" @filter({' AND '.join(filters)})"

    # Fix for PLR0913 (too many arguments)
    # Create a context class to bundle related arguments
    class EdgeConnectionContext:
        """Context object for edge connection data."""

        def __init__(
            self,
            nodes: dict[str, QNodeDict],
            edges: dict[str, QEdgeDict],
            visited: set[str]
        ) -> None:
            """Initialize edge connection context.

            Args:
                nodes: Dictionary of query nodes
                edges: Dictionary of query edges
                visited: Set of already visited node IDs
            """
            self.nodes = nodes
            self.edges = edges
            self.visited = visited

    def _process_edge_connection(
        self,
        edge: QEdgeDict,
        source: QNodeDict,
        source_id: str,
        context: EdgeConnectionContext,
        indent_level: int,
    ) -> str:
        """Process an edge connection and build the query for it."""
        indent = "  " * indent_level

        # Build edge filter
        edge_filter = self._build_edge_filter(edge)
        filter_clause = f" @filter({edge_filter})" if edge_filter else ""

        query = f"{indent}in_edges: ~source{filter_clause} {{\n"

        # Include all standard edge fields
        query += self._add_standard_edge_fields(indent + "  ")

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

        query += f"{indent}  node: target{source_filter_clause} {{\n"

        # Include all standard node fields for the target
        query += self._add_standard_node_fields(indent + "    ")

        # Recursively add further hops
        query += self._build_further_hops(source_id, context.nodes, context.edges, indent_level + 2, context.visited.copy())

        # Close the blocks
        query += f"{indent}  }}\n{indent}}}\n"

        return query

    def _build_node_query(
        self,
        node_id: str,
        nodes: dict[str, QNodeDict],
        edges: dict[str, QEdgeDict],
        indent_level: int = 1
    ) -> str:
        """Recursively build a query for a node and its connected nodes."""
        node = nodes[node_id]
        indent = "  " * indent_level
        next_indent = "  " * (indent_level + 1)

        # Get primary and secondary filters
        primary_filter, secondary_filters = self._get_primary_and_secondary_filters(node)

        # Start the query with the primary filter
        query = f"{indent}node(func: {primary_filter})"

        # Add secondary filters if present
        query += self._build_filter_clause(secondary_filters)

        # Add cascade and open block
        query += " @cascade {\n"

        # Include all standard node fields
        query += self._add_standard_node_fields(next_indent)

        # Find incoming edges to this node (where this node is the OBJECT)
        connected_edges: dict[str, list[QEdgeDict]] = {}
        for edge in edges.values():
            if edge["object"] == node_id:
                # Get the source node and predicate
                source_id = edge["subject"]
                if source_id not in connected_edges:
                    connected_edges[source_id] = []
                connected_edges[source_id].append(edge)

        # Create a context object for edge processing
        context = self.EdgeConnectionContext(nodes=nodes, edges=edges, visited={node_id})

        # For each source node, build the edge traversal
        for source_id, source_edges in connected_edges.items():
            # Use the first edge
            edge = source_edges[0]
            source = nodes[source_id]
            query += self._process_edge_connection(
                edge=edge,
                source=source,
                source_id=source_id,
                context=context,
                indent_level=indent_level + 1,
            )

        # Close the node block
        query += f"{indent}}}\n"
        return query

    def _build_further_hops(
        self,
        node_id: str,
        nodes: dict[str, QNodeDict],
        edges: dict[str, QEdgeDict],
        indent_level: int,
        visited: set[str]
    ) -> str:
        """Build query for further hops beyond the first one."""
        # Prevent cycles
        if node_id in visited:
            return ""

        visited.add(node_id)

        query = ""
        indent = "  " * indent_level

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

                query += f"{indent}in_edges: ~source{filter_clause} {{\n"

                # Include all standard edge fields
                query += self._add_standard_edge_fields(indent + "  ")

                # Build source node filter using the node_filter helper
                source_filter = self._build_node_filter(source)
                source_filter_clause = f" @filter({source_filter})" if source_filter != "has(id)" else ""

                query += f"{indent}  node: target{source_filter_clause} {{\n"

                # Include all standard node fields for the target
                query += self._add_standard_node_fields(indent + "    ")

                # Recursively add further hops with updated context
                new_visited = visited.copy()
                query += self._build_further_hops(source_id, nodes, edges, indent_level + 2, new_visited)

                # Close the blocks
                query += f"{indent}  }}\n{indent}}}\n"

        return query

    @override
    def _convert_batch_multihop(self, qgraph: QueryGraphDict) -> str:
        """Convert a TRAPI multi-hop graph to a batch of Dgraph queries.

        This function handles a single QueryGraphDict that contains a special format
        where multiple query graphs are stored within it.

        Args:
            qgraph: A query graph that may contain multiple sub-graphs

        Returns:
            A combined Dgraph query containing all sub-queries
        """
        # Check if this is a "batch container" with sub-queries
        if "query_graphs" in qgraph and isinstance(qgraph["query_graphs"], list):
            # Process each query graph in the list
            queries = []
            for i, sub_qgraph in enumerate(qgraph["query_graphs"]):
                # Get the query for this graph and rename the node to make it unique
                query = self._convert_multihop(sub_qgraph)
                query = query.replace("node(func:", f"node{i}(func:")
                queries.append(query)

            # Combine all queries into one batch query
            return "{\n" + "\n".join(q.strip("{}") for q in queries) + "\n}"

        # Otherwise, process a standard query graph with possibly multiple IDs per node
        nodes = qgraph["nodes"]
        start_node_id = self._find_start_node(nodes, {})
        start_node = nodes[start_node_id]

        # If there are no IDs, just return a single query
        if not start_node.get("ids"):
            return self._convert_multihop(qgraph)

        # Create a query for each starting ID and collect
        queries = []
        for i, node_id in enumerate(start_node["ids"]):
            # Create a modified query graph with just one ID
            modified_qgraph = qgraph.copy()
            modified_qgraph["nodes"] = qgraph["nodes"].copy()
            modified_qgraph["nodes"][start_node_id] = start_node.copy()
            modified_qgraph["nodes"][start_node_id]["ids"] = [node_id]

            # Get the query for this ID and rename the node
            query = self._convert_multihop(modified_qgraph)
            query = query.replace("node(func:", f"node{i}(func:")
            queries.append(query)

        # Combine all queries into one
        return "{\n" + "\n".join(q.strip("{}") for q in queries) + "\n}"

    @override
    def convert_results(
        self, qgraph: QueryGraphDict, results: Any
    ) -> BackendResult:
        """Convert Dgraph JSON results back to TRAPI BackendResults."""
        # Create a properly structured BackendResult with required fields
        return cast(BackendResult, {
            "results": results,
            "knowledge_graph": {},  # Required field in BackendResult
            "auxiliary_graphs": {}  # Required field in BackendResult
        })
