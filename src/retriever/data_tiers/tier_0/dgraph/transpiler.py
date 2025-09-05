from typing import Any

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResults
from retriever.types.trapi import QEdgeDict, QNodeDict, QueryGraphDict


class DgraphTranspiler(Transpiler):
    """Transpiler for converting TRAPI queries into Dgraph GraphQL queries."""

    def convert_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> Any:
        """Convert a single triple to Dgraph query format.

        Not implemented for Dgraph as it's Tier 0 only.
        """
        raise NotImplementedError("Dgraph is Tier 0 only. Use multi-hop methods.")

    def convert_batch_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> Any:
        """Convert a batch of triples to Dgraph query format.

        Not implemented for Dgraph as it's Tier 0 only.
        """
        raise NotImplementedError("Dgraph is Tier 0 only. Use batch multi-hop methods.")

    def _convert_multihop(self, qgraph: QueryGraphDict) -> Any:
        """Convert a TRAPI multi-hop graph to a proper Dgraph multihop query."""
        nodes = qgraph["nodes"]
        edges = qgraph["edges"]

        # Identify the starting node
        start_node_id = self._find_start_node(nodes, edges)

        # Build query from the starting node
        return "{\n" + self._build_node_query(start_node_id, nodes, edges) + "}"

    def _find_start_node(self, nodes: dict, edges: dict) -> str:
        """Find the best node to start the traversal from."""
        # Start with a node that has IDs and is the object of at least one edge
        for node_id, node in nodes.items():
            if node.get("ids") and any(e["object"] == node_id for e in edges.values()):
                return node_id

        # If no ideal start node, just take the first one
        return next(iter(nodes))

    def _build_node_query(self, node_id: str, nodes: dict, edges: dict, indent_level: int = 1) -> str:
        """Recursively build a query for a node and its connected nodes."""
        node = nodes[node_id]
        indent = "  " * indent_level
        next_indent = "  " * (indent_level + 1)

        # Start the query with the node filter
        node_filter = "func: has(id)"
        if node.get("ids"):
            if len(node["ids"]) == 1:
                # Single ID case - check if it contains a comma (multiple IDs in one string)
                id_value = node["ids"][0]
                if "," in id_value:
                    # Multiple IDs in a single string - split them
                    ids = [id.strip() for id in id_value.split(",")]
                    ids_str = ', '.join(f'"{id_}"' for id_ in ids)
                    node_filter = f'func: eq(id, [{ids_str}])'
                else:
                    # Single ID
                    node_filter = f'func: eq(id, "{id_value}")'
            else:
                # Multiple IDs in array
                ids_str = ', '.join(f'"{id_}"' for id_ in node["ids"])
                node_filter = f'func: eq(id, [{ids_str}])'

        query = f"{indent}node({node_filter}) @cascade {{\n"
        query += f"{next_indent}id\n{next_indent}name\n{next_indent}category\n"

        # Find incoming edges to this node (where this node is the OBJECT)
        connected_edges = {}
        for edge in edges.values():
            if edge["object"] == node_id:
                # Get the source node and predicate
                source_id = edge["subject"]
                if source_id not in connected_edges:
                    connected_edges[source_id] = []
                connected_edges[source_id].append(edge)

        # For each source node, build the edge traversal
        for source_id, source_edges in connected_edges.items():
            # Use the first edge's predicate (if multiple exist between same nodes)
            predicate = source_edges[0].get("predicate", "")
            source = nodes[source_id]

            # Add the edge traversal
            predicate_filter = f'eq(predicate,"{predicate}")' if predicate else ""
            filter_clause = f" @filter({predicate_filter})" if predicate_filter else ""
            query += f"{next_indent}in_edges: ~source{filter_clause} {{\n"
            query += f"{next_indent}  predicate\n{next_indent}  primary_knowledge_source\n"

            # Source node
            source_filter = ""
            if source.get("ids"):
                if len(source["ids"]) == 1:
                    # Single ID case - check if it contains a comma (multiple IDs in one string)
                    id_value = source["ids"][0]
                    if "," in id_value:
                        # Multiple IDs in a single string - split them
                        ids = [id.strip() for id in id_value.split(",")]
                        ids_str = ', '.join(f'"{id_}"' for id_ in ids)
                        source_filter = f'eq(id, [{ids_str}])'
                    else:
                        # Single ID
                        source_filter = f'eq(id, "{id_value}")'
                else:
                    # Multiple IDs in array
                    ids_str = ', '.join(f'"{id_}"' for id_ in source["ids"])
                    source_filter = f'eq(id, [{ids_str}])'

            source_filter_clause = f" @filter({source_filter})" if source_filter else ""
            query += f"{next_indent}  node: target{source_filter_clause} {{\n"
            query += f"{next_indent}    id\n{next_indent}    name\n{next_indent}    category\n"

            # Recursively add further hops
            visited = {node_id}
            query += self._build_further_hops(source_id, nodes, edges, indent_level + 2, visited)

            # Close the blocks
            query += f"{next_indent}  }}\n{next_indent}}}\n"

        # Close the node block
        query += f"{indent}}}\n"
        return query

    def _build_further_hops(self, node_id: str, nodes: dict, edges: dict, indent_level: int, visited: set[str]) -> str:
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

                predicate = edge.get("predicate", "")
                source = nodes[source_id]

                # Add the edge traversal
                predicate_filter = f'eq(predicate,"{predicate}")' if predicate else ""
                filter_clause = f" @filter({predicate_filter})" if predicate_filter else ""
                query += f"{indent}in_edges: ~source{filter_clause} {{\n"
                query += f"{indent}  predicate\n{indent}  primary_knowledge_source\n"

                # Source node
                source_filter = ""
                if source.get("ids"):
                    if len(source["ids"]) == 1:
                        # Single ID case - check if it contains a comma (multiple IDs in one string)
                        id_value = source["ids"][0]
                        if "," in id_value:
                            # Multiple IDs in a single string - split them
                            ids = [id.strip() for id in id_value.split(",")]
                            ids_str = ', '.join(f'"{id_}"' for id_ in ids)
                            source_filter = f'eq(id, [{ids_str}])'
                        else:
                            # Single ID
                            source_filter = f'eq(id, "{id_value}")'
                    else:
                        # Multiple IDs in array
                        ids_str = ', '.join(f'"{id_}"' for id_ in source["ids"])
                        source_filter = f'eq(id, [{ids_str}])'

                source_filter_clause = f" @filter({source_filter})" if source_filter else ""
                query += f"{indent}  node: target{source_filter_clause} {{\n"
                query += f"{indent}    id\n{indent}    name\n{indent}    category\n"

                # Recursively add further hops
                query += self._build_further_hops(source_id, nodes, edges, indent_level + 2, visited.copy())

                # Close the blocks
                query += f"{indent}  }}\n{indent}}}\n"

        return query


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
        if "query_graphs" in qgraph:
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

    def convert_results(
        self, qgraph: QueryGraphDict, results: Any
    ) -> BackendResults:
        """Convert Dgraph JSON results back to TRAPI BackendResults."""
        return {"results": results}
