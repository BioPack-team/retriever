from typing import Any, override

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResult
from retriever.types.trapi import AttributeConstraintDict, QEdgeDict, QNodeDict, QueryGraphDict


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
        if node.get("ids"):
            if len(node["ids"]) == 1:
                # Single ID case - check if it contains a comma (multiple IDs in one string)
                id_value = node["ids"][0]
                if "," in id_value:
                    # Multiple IDs in a single string - split them
                    ids = [id_val.strip() for id_val in id_value.split(",")]
                    ids_str = ', '.join(f'"{id_val}"' for id_val in ids)
                    filters.append(f'eq(id, [{ids_str}])')
                else:
                    # Single ID
                    filters.append(f'eq(id, "{id_value}")')
            else:
                # Multiple IDs in array
                ids_str = ', '.join(f'"{id_val}"' for id_val in node["ids"])
                filters.append(f'eq(id, [{ids_str}])')

        # Handle category filtering
        if node.get("categories"):
            categories = node["categories"]
            if len(categories) == 1:
                filters.append(f'eq(category, "{categories[0]}")')
            elif len(categories) > 1:
                categories_str = ', '.join(f'"{cat}"' for cat in categories)
                filters.append(f'eq(category, [{categories_str}])')

        # Handle attribute constraints
        if node.get("constraints"):
            filters.extend(self._convert_constraints_to_filters(node["constraints"]))

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
        if edge.get("predicate"):
            filters.append(f'eq(predicate, "{edge["predicate"]}")')

        # Handle predicates (multiple) filtering
        if edge.get("predicates"):
            predicates = edge["predicates"]
            if len(predicates) == 1:
                filters.append(f'eq(predicate, "{predicates[0]}")')
            elif len(predicates) > 1:
                predicates_str = ', '.join(f'"{pred}"' for pred in predicates)
                filters.append(f'eq(predicate, [{predicates_str}])')

        # Handle attribute constraints
        if edge.get("attribute_constraints"):
            filters.extend(self._convert_constraints_to_filters(edge["attribute_constraints"]))

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

        # Build the node filter - use a primary filter for func: and secondary filters for @filter
        primary_filter = "has(id)"  # Default
        secondary_filters = []

        # Choose the most selective filter for primary (usually ID)
        if node.get("ids"):
            if len(node["ids"]) == 1:
                id_value = node["ids"][0]
                if "," in id_value:
                    # Multiple IDs in a single string - split them
                    ids = [id_val.strip() for id_val in id_value.split(",")]
                    ids_str = ', '.join(f'"{id_val}"' for id_val in ids)
                    primary_filter = f'eq(id, [{ids_str}])'
                else:
                    # Single ID - most selective query
                    primary_filter = f'eq(id, "{id_value}")'
            else:
                # Multiple IDs in array
                ids_str = ', '.join(f'"{id_val}"' for id_val in node["ids"])
                primary_filter = f'eq(id, [{ids_str}])'
        elif node.get("categories"):
            # Use category as primary if no IDs
            categories = node["categories"]
            if len(categories) == 1:
                primary_filter = f'eq(category, "{categories[0]}")'
            else:
                categories_str = ', '.join(f'"{cat}"' for cat in categories)
                primary_filter = f'eq(category, [{categories_str}])'

        # Build secondary filters
        # If we used IDs as primary, add categories as secondary
        if node.get("ids") and node.get("categories"):
            categories = node["categories"]
            if len(categories) == 1:
                secondary_filters.append(f'eq(category, "{categories[0]}")')
            elif len(categories) > 1:
                categories_str = ', '.join(f'"{cat}"' for cat in categories)
                secondary_filters.append(f'eq(category, [{categories_str}])')

        # Add constraints as secondary filters
        if node.get("constraints"):
            secondary_filters.extend(self._convert_constraints_to_filters(node["constraints"]))

        # Start the query with the primary filter
        query = f"{indent}node(func: {primary_filter})"

        # Add secondary filters if present - use AND between filters
        if secondary_filters:
            if len(secondary_filters) == 1:
                query += f" @filter({secondary_filters[0]})"
            else:
                query += f" @filter({' AND '.join(secondary_filters)})"

        # Add cascade and open block
        query += f" @cascade {{\n"

        # Include all standard node fields
        query += f"{next_indent}id\n{next_indent}name\n{next_indent}category\n"
        query += f"{next_indent}all_names\n{next_indent}all_categories\n"
        query += f"{next_indent}iri\n{next_indent}equivalent_curies\n"
        query += f"{next_indent}description\n{next_indent}publications\n"

        # Find incoming edges to this node (where this node is the OBJECT)
        connected_edges: dict[str, list[QEdgeDict]] = {}
        for edge in edges.values():
            if edge["object"] == node_id:
                # Get the source node and predicate
                source_id = edge["subject"]
                if source_id not in connected_edges:
                    connected_edges[source_id] = []
                connected_edges[source_id].append(edge)

        # For each source node, build the edge traversal
        for source_id, source_edges in connected_edges.items():
            # Use the first edge
            edge = source_edges[0]
            source = nodes[source_id]

            # Build edge filter
            edge_filter = self._build_edge_filter(edge)
            filter_clause = f" @filter({edge_filter})" if edge_filter else ""

            query += f"{next_indent}in_edges: ~source{filter_clause} {{\n"

            # Include all standard edge fields
            query += f"{next_indent}  predicate\n{next_indent}  primary_knowledge_source\n"
            query += f"{next_indent}  knowledge_level\n{next_indent}  agent_type\n"
            query += f"{next_indent}  kg2_ids\n{next_indent}  domain_range_exclusion\n"
            query += f"{next_indent}  edge_id\n"

            # Build source node for target filter - need to split into primary/secondary
            primary_source_filter = "has(id)"  # Default
            secondary_source_filters = []

            # Choose the most selective filter for primary (usually ID)
            if source.get("ids"):
                if len(source["ids"]) == 1:
                    id_value = source["ids"][0]
                    if "," in id_value:
                        ids = [id_val.strip() for id_val in id_value.split(",")]
                        ids_str = ', '.join(f'"{id_val}"' for id_val in ids)
                        primary_source_filter = f'eq(id, [{ids_str}])'
                    else:
                        primary_source_filter = f'eq(id, "{id_value}")'
                else:
                    ids_str = ', '.join(f'"{id_val}"' for id_val in source["ids"])
                    primary_source_filter = f'eq(id, [{ids_str}])'
            elif source.get("categories"):
                categories = source["categories"]
                if len(categories) == 1:
                    primary_source_filter = f'eq(category, "{categories[0]}")'
                else:
                    categories_str = ', '.join(f'"{cat}"' for cat in categories)
                    primary_source_filter = f'eq(category, [{categories_str}])'

            # Build additional source filters
            if source.get("ids") and source.get("categories"):
                categories = source["categories"]
                if len(categories) == 1:
                    secondary_source_filters.append(f'eq(category, "{categories[0]}")')
                elif len(categories) > 1:
                    categories_str = ', '.join(f'"{cat}"' for cat in categories)
                    secondary_source_filters.append(f'eq(category, [{categories_str}])')

            # Add constraints as secondary filters
            if source.get("constraints"):
                secondary_source_filters.extend(self._convert_constraints_to_filters(source["constraints"]))

            # Create the source filter clause using AND for multiple filters
            source_filter_clause = ""
            if primary_source_filter != "has(id)":
                if not secondary_source_filters:
                    source_filter_clause = f" @filter({primary_source_filter})"
                else:
                    source_filter_clause = f" @filter({primary_source_filter} AND {' AND '.join(secondary_source_filters)})"
            elif secondary_source_filters:
                if len(secondary_source_filters) == 1:
                    source_filter_clause = f" @filter({secondary_source_filters[0]})"
                else:
                    source_filter_clause = f" @filter({' AND '.join(secondary_source_filters)})"

            query += f"{next_indent}  node: target{source_filter_clause} {{\n"

            # Include all standard node fields for the target
            query += f"{next_indent}    id\n{next_indent}    name\n{next_indent}    category\n"
            query += f"{next_indent}    all_names\n{next_indent}    all_categories\n"
            query += f"{next_indent}    iri\n{next_indent}    equivalent_curies\n"
            query += f"{next_indent}    description\n{next_indent}    publications\n"

            # Recursively add further hops
            visited = {node_id}
            query += self._build_further_hops(source_id, nodes, edges, indent_level + 2, visited)

            # Close the blocks
            query += f"{next_indent}  }}\n{next_indent}}}\n"

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
                query += f"{indent}  predicate\n{indent}  primary_knowledge_source\n"
                query += f"{indent}  knowledge_level\n{indent}  agent_type\n"
                query += f"{indent}  kg2_ids\n{indent}  domain_range_exclusion\n"
                query += f"{indent}  edge_id\n"

                # Build source node filter using the node_filter helper
                source_filter = self._build_node_filter(source)
                source_filter_clause = f" @filter({source_filter})" if source_filter != "has(id)" else ""

                query += f"{indent}  node: target{source_filter_clause} {{\n"

                # Include all standard node fields for the target
                query += f"{indent}    id\n{indent}    name\n{indent}    category\n"
                query += f"{indent}    all_names\n{indent}    all_categories\n"
                query += f"{indent}    iri\n{indent}    equivalent_curies\n"
                query += f"{indent}    description\n{indent}    publications\n"

                # Recursively add further hops
                query += self._build_further_hops(source_id, nodes, edges, indent_level + 2, visited.copy())

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
            return "{\n" + "\n".join([q.strip("{}") for q in queries]) + "\n}"

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
        return "{\n" + "\n".join([q.strip("{}") for q in queries]) + "\n}"

    @override
    def convert_results(
        self, qgraph: QueryGraphDict, results: Any
    ) -> BackendResult:
        """Convert Dgraph JSON results back to TRAPI BackendResults."""
        return {"results": results}
