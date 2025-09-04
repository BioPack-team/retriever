from typing import Any, Dict, List, Set
import textwrap

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResults
from retriever.types.trapi import QEdgeDict, QNodeDict, QueryGraphDict


class DgraphTranspiler(Transpiler):
    """Transpiler for converting TRAPI queries into Dgraph GraphQL queries."""

    def convert_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> Any:
        raise NotImplementedError("Dgraph is Tier 0 only. Use multi-hop methods.")

    def convert_batch_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> Any:
        raise NotImplementedError("Dgraph is Tier 0 only. Use batch multi-hop methods.")

    def _convert_multihop(self, qgraph: QueryGraphDict) -> Any:
        """Convert a TRAPI multi-hop graph to a proper Dgraph multihop query."""
        nodes = qgraph["nodes"]
        edges = qgraph["edges"]

        # Identify the starting node
        start_node_id = self._find_start_node(nodes, edges)

        # Build query from the starting node
        return "{\n" + self._build_node_query(start_node_id, nodes, edges) + "}"

    def _find_start_node(self, nodes: Dict, edges: Dict) -> str:
        """Find the best node to start the traversal from."""
        # Start with a node that has IDs and is the subject of at least one edge
        for node_id, node in nodes.items():
            if "ids" in node and node["ids"] and any(e["subject"] == node_id for e in edges.values()):
                return node_id

        # If no ideal start node, just take the first one
        return next(iter(nodes))

    def _build_node_query(self, node_id: str, nodes: Dict, edges: Dict, indent_level: int = 1) -> str:
        """Recursively build a query for a node and its connected nodes."""
        node = nodes[node_id]
        indent = "  " * indent_level
        next_indent = "  " * (indent_level + 1)

        # Start the query with the node
        node_filter = f'eq(id, "{node["ids"][0]}")' if "ids" in node and node["ids"] else "has(id)"

        # Basic node template
        query_template = f"""{indent}node(func: {node_filter}) @cascade {{
        {next_indent}id
        {next_indent}name
        {next_indent}category
        """

        query_parts = [query_template]

        # Find outgoing edges from this node
        connected_edges = {}
        for edge_id, edge in edges.items():
            if edge["subject"] == node_id:
                # Get the target node and predicate
                target_id = edge["object"]
                if target_id not in connected_edges:
                    connected_edges[target_id] = []
                connected_edges[target_id].append(edge)

        # For each target node, build the edge traversal
        for target_id, target_edges in connected_edges.items():
            # Use the first edge's predicate (if multiple exist between same nodes)
            predicate = target_edges[0].get("predicate", "")
            target = nodes[target_id]

            # Create the edge traversal template
            predicate_filter = f'@filter(eq(predicate,"{predicate}"))' if predicate else ""
            target_filter = f'@filter(eq(id, "{target["ids"][0]}"))' if "ids" in target and target["ids"] else ""

            edge_template = f"""{next_indent}in_edges: ~source {predicate_filter} {{
            {next_indent}  predicate
            {next_indent}  primary_knowledge_source
            {next_indent}  node: target {target_filter} {{
            {next_indent}    id
            {next_indent}    name
            {next_indent}    category
            """
            query_parts.append(edge_template)

            # Recursively add further hops
            visited = set([node_id])
            further_hops = self._build_further_hops(target_id, nodes, edges, indent_level + 2, visited)
            if further_hops:
                query_parts.append(further_hops)

            # Close the target node and edge blocks
            query_parts.append(f"{next_indent}  }}\n{next_indent}}}\n")

        # Close the node block
        query_parts.append(f"{indent}}}\n")

        return "".join(query_parts)

    def _build_further_hops(self, node_id: str, nodes: Dict, edges: Dict, indent_level: int, visited: Set[str]) -> str:
        """Build query for further hops beyond the first one."""
        # Prevent cycles
        if node_id in visited:
            return ""

        visited.add(node_id)
        indent = "  " * indent_level
        next_indent = "  " * (indent_level + 1)

        query_parts = []

        # Find outgoing edges from this node
        for edge_id, edge in edges.items():
            if edge["subject"] == node_id:
                # Get the target node and predicate
                target_id = edge["object"]
                if target_id in visited:
                    continue  # Skip already visited nodes to prevent cycles

                predicate = edge.get("predicate", "")
                target = nodes[target_id]

                # Create edge and target templates
                predicate_filter = f'@filter(eq(predicate,"{predicate}"))' if predicate else ""
                target_filter = f'@filter(eq(id, "{target["ids"][0]}"))' if "ids" in target and target["ids"] else ""

                edge_template = f"""{indent}in_edges: ~source {predicate_filter} {{
                {indent}  predicate
                {indent}  primary_knowledge_source
                {indent}  node: target {target_filter} {{
                {indent}    id
                {indent}    name
                {indent}    category
                """
                query_parts.append(edge_template)

                # Recursively add further hops
                further_hops = self._build_further_hops(target_id, nodes, edges, indent_level + 2, visited.copy())
                if further_hops:
                    query_parts.append(further_hops)

                # Close the target node and edge blocks
                query_parts.append(f"{indent}  }}\n{indent}}}\n")

        return "".join(query_parts)

    def _convert_batch_multihop(self, qgraph: QueryGraphDict) -> Any:
        """Convert a TRAPI multi-hop graph to a batch of Dgraph queries."""
        nodes = qgraph["nodes"]
        start_node_id = self._find_start_node(nodes, {})
        start_node = nodes[start_node_id]

        # If there are no IDs, just return a single query
        if "ids" not in start_node or not start_node["ids"]:
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
