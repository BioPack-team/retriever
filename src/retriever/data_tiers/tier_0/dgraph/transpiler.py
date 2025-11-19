import itertools
import math  # <-- Add this import
from collections import defaultdict  # <-- Add this import
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, override

from loguru import logger

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
    QueryGraphDict,
    RetrievalSourceDict,
)
from retriever.utils import biolink
from retriever.utils.trapi import hash_edge, hash_hex


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
        10  # Max recursion depth for pinnedness calculation
    )

    FilterScalar: TypeAlias = str | int | float | bool  # noqa: UP040
    FilterValue: TypeAlias = FilterScalar | list[FilterScalar]  # noqa: UP040
    version: str | None
    prefix: str

    def __init__(self, version: str | None = None) -> None:
        """Initialize a Transpiler instance.

        Args:
            version: An optional version string to prefix to all schema fields.
        """
        super().__init__()
        self.kgraph: KnowledgeGraphDict = KnowledgeGraphDict(nodes={}, edges={})
        self.k_agraph: KAdjacencyGraph
        self.version = version
        self.prefix = f"{version}_" if version else ""

    def _v(self, field: str) -> str:
        """Return the versioned field name."""
        return f"{self.prefix}{field}"

    def _aliased_fields(self, fields: list[str]) -> str:
        """Return a string of aliased fields if a version is set, otherwise return just the field names."""
        if self.version:
            return " ".join(f"{field}: {self._v(field)}" for field in fields) + " "
        return " ".join(fields) + " "

    @override
    def process_qgraph(
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> str:
        return super().process_qgraph(qgraph, *additional_qgraphs)

    @override
    def convert_multihop(self, qgraph: QueryGraphDict) -> str:
        """Convert a TRAPI multi-hop graph to a proper Dgraph multihop query."""
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
        """Compute the log of the expected number of unique knodes bound to the specified qnode."""
        log_expected_n = math.log(num_ids[qnode_id])
        if level < self.PINNEDNESS_RECURSION_DEPTH:
            for neighbor, num_edges in adjacency_mat[qnode_id].items():
                if neighbor == last:
                    continue
                log_expected_n += num_edges * min(
                    max(
                        self._compute_log_expected_n(
                            adjacency_mat,
                            num_ids,
                            neighbor,
                            qnode_id,
                            level + 1,
                        ),
                        0,
                    )
                    + math.log(
                        self.PINNEDNESS_DEFAULT_EDGES_PER_NODE
                        / self.PINNEDNESS_DEFAULT_TOTAL_NODES
                    ),
                    0,
                )
        return log_expected_n

    def _get_pinnedness(self, qgraph: QueryGraphDict, qnode_id: str) -> float:
        """Get pinnedness of a single node."""
        adjacency_mat = self._get_adjacency_matrix(qgraph)
        num_ids = self._get_num_ids(qgraph)
        return -self._compute_log_expected_n(
            adjacency_mat,
            num_ids,
            qnode_id,
        )

    # --- Nodes and Edges Methods ---

    def _build_node_filter(self, node: QNodeDict, *, primary: bool = False) -> str:
        """Build a filter expression for a node based on its properties.

        If `primary` is True, it returns the most selective filter for a `func:` block.
        Otherwise, it returns all applicable filters for an `@filter` block.
        """
        filters: list[str] = []
        ids = node.get("ids")
        categories = node.get("categories")
        constraints = node.get("constraints")

        # For a primary filter, we want the single most selective option.
        if primary:
            if ids:
                return self._create_id_filter(ids)
            if categories:
                return self._create_category_filter(categories)
            if constraints:
                # Fallback to first constraint if no IDs or categories
                return self._convert_constraints_to_filters(constraints)[0]
            return f"has({self._v('id')})"

        # For secondary filters (@filter), we build a list of all constraints.
        if ids:
            filters.append(self._create_id_filter(ids))
        if categories:
            filters.append(self._create_category_filter(categories))
        if constraints:
            filters.extend(self._convert_constraints_to_filters(constraints))

        if not filters:
            return ""  # Return empty string if no filters apply

        return " AND ".join(filters)

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

    def _create_filter_expression(self, constraint: AttributeConstraintDict) -> str:
        """Create a filter expression for a single constraint."""
        field_name = self._v(constraint["id"])
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

        Always require id, and require reverse predicates (~subject, ~object)
        only if there are corresponding traversals from this node to not-yet-visited nodes.
        """
        cascade_fields: list[str] = [self._v("id")]

        # If this node has any outgoing edges (node as subject) to unvisited objects,
        # require ~subject in cascade to ensure at least one such edge exists.
        if any(
            e["subject"] == node_id and e["object"] not in visited
            for e in edges.values()
        ):
            cascade_fields.append(f"~{self._v('subject')}")

        # If this node has any incoming edges (node as object) to unvisited subjects,
        # require ~object in cascade to ensure at least one such edge exists.
        if any(
            e["object"] == node_id and e["subject"] not in visited
            for e in edges.values()
        ):
            cascade_fields.append(f"~{self._v('object')}")

        # Always emit a cascade; at minimum it will include the id
        return f" @cascade({', '.join(cascade_fields)})"

    def _build_node_query(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int | None = None,
    ) -> str:
        """Recursively build a query for a node and its connected nodes."""
        node = nodes[node_id]

        # Get the primary filter for the `func:` block of the starting node.
        primary_filter = self._build_node_filter(node, primary=True)

        # Create the root node key, with an optional query index prefix
        root_node_key = f"node_{node_id}"
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
                edge_filter = self._build_edge_filter(edge)
                filter_clause = f" @filter({edge_filter})" if edge_filter else ""

                # Edge block with cascade requiring predicate and the 'object' endpoint
                query += f"out_edges_{edge_id}: ~{self._v('subject')}{filter_clause}"
                query += f" @cascade({self._v('predicate')}, {self._v('object')}) {{ "
                query += self._add_standard_edge_fields()

                # Build the filter for the connected object node.
                object_filter = self._build_node_filter(object_node)
                query += f"node_{object_id}: {self._v('object')}"
                if object_filter:
                    query += f" @filter({object_filter})"

                # For the child node, include cascade(id, ~subject?, ~object?) based on remaining unvisited hops
                child_visited = visited | {object_id}
                query += self._build_node_cascade_clause(
                    object_id, edges, child_visited
                )

                # Open child node block
                query += " { "
                query += self._add_standard_node_fields()

                # Recurse from the newly connected object node
                query += self._build_further_hops(
                    object_id, nodes, edges, child_visited
                )

                # Close blocks
                query += "} } "

        # Find incoming edges to this node (where this node is the OBJECT)
        for edge_id, edge in edges.items():
            if edge["object"] == node_id:
                source_id: QNodeID = edge["subject"]
                if source_id in visited:
                    continue  # Skip cycles

                source_node = nodes[source_id]
                edge_filter = self._build_edge_filter(edge)
                filter_clause = f" @filter({edge_filter})" if edge_filter else ""

                # Edge block with cascade requiring predicate and the 'subject' endpoint
                query += f"in_edges_{edge_id}: ~{self._v('object')}{filter_clause}"
                query += f" @cascade({self._v('predicate')}, {self._v('subject')}) {{ "
                query += self._add_standard_edge_fields()

                # Build the filter for the connected source node.
                source_filter = self._build_node_filter(source_node)
                query += f"node_{source_id}: {self._v('subject')}"
                if source_filter:
                    query += f" @filter({source_filter})"

                # For the child node, include cascade(id, ~subject?, ~object?) based on remaining unvisited hops
                child_visited = visited | {source_id}
                query += self._build_node_cascade_clause(
                    source_id, edges, child_visited
                )

                # Open child node block
                query += " { "
                query += self._add_standard_node_fields()

                # Recurse from the newly connected source node
                query += self._build_further_hops(
                    source_id, nodes, edges, child_visited
                )

                # Close blocks
                query += "} } "

        return query

    @override
    def convert_batch_multihop(self, qgraphs: list[QueryGraphDict]) -> str:
        """Convert a TRAPI multi-hop batch graph to a batch of Dgraph queries.

        Returns:
            A combined Dgraph query containing all sub-queries
        """
        # Process each query graph in the list
        blocks: list[str] = []
        for i, sub_qgraph in enumerate(qgraphs):
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

        if not (
            len(node.equivalent_identifiers) == 1
            and node.equivalent_identifiers[0] == node.id
        ):
            attributes.append(
                AttributeDict(
                    attribute_type_id="biolink:xref",
                    value=[CURIE(i) for i in node.equivalent_identifiers],
                )
            )

        if node.description:
            attributes.append(
                AttributeDict(
                    attribute_type_id="biolink:description", value=node.description
                )
            )
        if node.in_taxon:
            attributes.append(
                AttributeDict(
                    attribute_type_id="biolink:in_taxon",
                    value=[CURIE(i) for i in node.in_taxon],
                )
            )
        if node.information_content is not None:
            attributes.append(
                AttributeDict(
                    attribute_type_id="biolink:information_content",
                    value=node.information_content,
                )
            )
        if node.inheritance:
            attributes.append(
                AttributeDict(
                    attribute_type_id="biolink:inheritance", value=node.inheritance
                )
            )
        if node.provided_by:
            attributes.append(
                AttributeDict(
                    attribute_type_id="biolink:provided_by",
                    value=[Infores(i) for i in node.provided_by],
                )
            )

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
        attribute_map = {
            "biolink:source_infores": edge.source_inforeses,
            "biolink:id": edge.id,
            "biolink:category": [
                BiolinkEntity(biolink.ensure_prefix(cat)) for cat in edge.category
            ],
            "biolink:knowledge_level": edge.knowledge_level,
            "biolink:agent_type": edge.agent_type,
            "biolink:publications": edge.publications,
            "biolink:qualified_predicate": edge.qualified_predicate,
            "biolink:subject_form_or_variant_qualifier": edge.subject_form_or_variant_qualifier,
            "biolink:disease_context_qualifier": edge.disease_context_qualifier,
            "biolink:frequency_qualifier": edge.frequency_qualifier,
            "biolink:onset_qualifier": edge.onset_qualifier,
            "biolink:sex_qualifier": edge.sex_qualifier,
            "biolink:original_subject": edge.original_subject,
            "biolink:original_predicate": edge.original_predicate,
            "biolink:original_object": edge.original_object,
            "biolink:allelic_requirement": edge.allelic_requirement,
            "biolink:update_date": edge.update_date,
            "biolink:z_score": edge.z_score,
            "biolink:has_evidence": edge.has_evidence,
            "biolink:has_confidence_score": edge.has_confidence_score,
            "biolink:has_count": edge.has_count,
            "biolink:has_total": edge.has_total,
            "biolink:has_percentage": edge.has_percentage,
            "biolink:has_quotient": edge.has_quotient,
        }
        for attr_id, value in attribute_map.items():
            if value is not None and value not in ([], ""):
                attributes.append(AttributeDict(attribute_type_id=attr_id, value=value))

        # --- Build Sources ---
        sources = [
            RetrievalSourceDict(
                resource_id=Infores(s.resource_id),
                resource_role=s.resource_role,
                upstream_resource_ids=[Infores(uid) for uid in s.upstream_resource_ids],
                source_record_urls=s.source_record_urls,
            )
            for s in edge.sources
        ]

        # --- Build Edge ---
        trapi_edge = EdgeDict(
            predicate=BiolinkPredicate(biolink.ensure_prefix(edge.predicate)),
            subject=CURIE(edge.node.id if edge.direction == "in" else initial_curie),
            object=CURIE(initial_curie if edge.direction == "in" else edge.node.id),
            sources=sources,
            attributes=attributes,
        )

        return trapi_edge

    def _build_results(self, node: dg.Node) -> list[Partial]:
        """Recursively build results from dgraph response."""
        if node.id not in self.kgraph["nodes"]:
            self.kgraph["nodes"][CURIE(node.id)] = self._build_trapi_node(node)

        # If we hit a stop condition, return partial for the node
        if not len(node.edges):
            return [Partial([(QNodeID(node.binding), CURIE(node.id))], [])]

        partials = {QEdgeID(edge.binding): list[Partial]() for edge in node.edges}

        for edge in node.edges:
            subject_id = CURIE(edge.node.id if edge.direction == "in" else node.id)
            object_id = CURIE(node.id if edge.direction == "in" else edge.node.id)
            qedge_id = QEdgeID(edge.binding)

            trapi_edge = self._build_trapi_edge(edge, node.id)
            edge_hash = EdgeIdentifier(hash_hex(hash_edge(trapi_edge)))

            # Update kgraph
            if edge_hash not in self.kgraph["edges"]:
                self.kgraph["edges"][edge_hash] = trapi_edge

            # Update k_agraph
            if subject_id not in self.k_agraph[qedge_id]:
                self.k_agraph[qedge_id][subject_id] = dict[
                    CURIE, list[EdgeIdentifier]
                ]()
            if object_id not in self.k_agraph[qedge_id][subject_id]:
                self.k_agraph[qedge_id][subject_id][object_id] = list[EdgeIdentifier]()
            self.k_agraph[qedge_id][subject_id][object_id].append(edge_hash)

            for partial in self._build_results(edge.node):
                partials[qedge_id].append(
                    partial.combine(
                        Partial(
                            [(QNodeID(node.binding), CURIE(node.id))],
                            [(qedge_id, subject_id, object_id)],
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
        """Convert Dgraph JSON results back to TRAPI BackendResults."""
        logger.info("Begin transforming records")
        self.k_agraph = {
            QEdgeID(qedge_id): dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]()
            for qedge_id in qgraph["edges"]
        }

        reconciled = list[Partial]()

        for node in results:
            reconciled.extend(self._build_results(node))

        trapi_results = [part.as_result(self.k_agraph) for part in reconciled]

        logger.info("Finished transforming records")
        return BackendResult(
            results=trapi_results,
            knowledge_graph=self.kgraph,
            auxiliary_graphs=dict[AuxGraphID, AuxiliaryGraphDict](),
        )
