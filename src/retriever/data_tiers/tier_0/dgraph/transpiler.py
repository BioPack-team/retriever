import contextlib
import itertools
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
                    ids_str = ", ".join(f'"{id_val}"' for id_val in id_list)
                    filters.append(f"eq({self._v('id')}, [{ids_str}])")
                else:
                    # Single ID
                    filters.append(f"eq({self._v('id')}, \"{id_value}\")")
            else:
                # Multiple IDs in array
                ids_str = ", ".join(f'"{id_val}"' for id_val in ids)
                filters.append(f"eq({self._v('id')}, [{ids_str}])")

        # Handle category filtering
        categories = node.get("categories")
        if categories:
            if len(categories) == 1:
                filters.append(
                    f'eq({self._v("all_categories")}, "{categories[0].replace("biolink:", "")}")'
                )
            elif len(categories) > 1:
                categories_str = ", ".join(
                    f'"{cat.replace("biolink:", "")}"' for cat in categories
                )
                filters.append(f"eq({self._v('all_categories')}, [{categories_str}])")

        # Handle attribute constraints
        constraints = node.get("constraints")
        if constraints:
            filters.extend(self._convert_constraints_to_filters(constraints))

        # If no filters, use a generic filter
        if not filters:
            return f"has({self._v('id')})"

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
                filters.append(
                    f'eq({self._v("all_predicates")}, "{predicates[0].replace("biolink:", "")}")'
                )
            elif len(predicates) > 1:
                predicates_str = ", ".join(
                    f'"{pred.replace("biolink:", "")}"' for pred in predicates
                )
                filters.append(f"eq({self._v('all_predicates')}, [{predicates_str}])")

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
            return f"eq({self._v('id')}, \"{id_value}\")"

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
            return f'eq({self._v("all_categories")}, "{cat_vals[0].replace("biolink:", "")}")'
        categories_str = ", ".join(f'"{cat}"' for cat in cat_vals)
        return f"eq({self._v('all_categories')}, [{categories_str}])"

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
        fields = [
            "id",
            "name",
            "category",
            "all_names",
            "all_categories",
            "iri",
            "equivalent_curies",
            "description",
            "publications",
        ]
        return self._aliased_fields(fields)

    def _add_standard_edge_fields(self) -> str:
        """Generate standard edge fields with versioned aliases."""
        fields = [
            "predicate",
            "primary_knowledge_source",
            "knowledge_level",
            "agent_type",
            "kg2_ids",
            "domain_range_exclusion",
            "qualified_object_aspect",
            "qualified_object_direction",
            "qualified_predicate",
            "publications",
            "publications_info",
        ]
        return self._aliased_fields(fields)

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

    def _build_node_query(
        self,
        node_id: QNodeID,
        nodes: Mapping[QNodeID, QNodeDict],
        edges: Mapping[QEdgeID, QEdgeDict],
        query_index: int | None = None,
    ) -> str:
        """Recursively build a query for a node and its connected nodes."""
        node = nodes[node_id]

        # Get primary and secondary filters
        primary_filter, secondary_filters = self._get_primary_and_secondary_filters(
            node
        )

        # Create the root node key, with an optional query index prefix
        root_node_key = f"node_{node_id}"
        if query_index is not None:
            root_node_key = f"q{query_index}_{root_node_key}"

        # Start the query with the primary filter, using the generated key
        query = f"{root_node_key}(func: {primary_filter})"

        # Add secondary filters if present
        query += self._build_filter_clause(secondary_filters)

        # Add cascade and open block
        query += " @cascade { "

        # Include all standard node fields
        query += self._add_standard_node_fields()

        # Start the recursive traversal
        visited = {node_id}
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

                # Use `~target` for outgoing edges
                query += f"out_edges_{edge_id}: ~{self._v('target')}{filter_clause} {{ "
                query += self._add_standard_edge_fields()

                object_filter = self._build_node_filter(object_node)
                object_filter_clause = (
                    f" @filter({object_filter})" if object_filter != f"has({self._v('id')})" else ""
                )

                # Use `node_{object_id}` for the connected node
                query += f"node_{object_id}: {self._v('source')}{object_filter_clause} {{ "
                query += self._add_standard_node_fields()

                # Recurse from the newly connected object node
                new_visited = visited | {object_id}
                query += self._build_further_hops(object_id, nodes, edges, new_visited)

                query += "} } "

        # Find incoming edges to this node (where this node is the OBJECT)
        for edge_id, edge in edges.items():
            if edge["object"] == node_id:
                # Get the target node and predicate
                source_id: QNodeID = edge["subject"]
                if source_id in visited:
                    continue  # Skip cycles

                source_node = nodes[source_id]
                edge_filter = self._build_edge_filter(edge)
                filter_clause = f" @filter({edge_filter})" if edge_filter else ""

                # Use `in_edges_{edge_id}` for incoming edges
                query += f"in_edges_{edge_id}: ~{self._v('source')}{filter_clause} {{ "
                query += self._add_standard_edge_fields()

                source_filter = self._build_node_filter(source_node)
                source_filter_clause = (
                    f" @filter({source_filter})" if source_filter != f"has({self._v('id')})" else ""
                )

                # Use `node_{source_id}` for the connected node
                query += f"node_{source_id}: {self._v('target')}{source_filter_clause} {{ "
                query += self._add_standard_node_fields()

                # Recurse from the newly connected source node
                new_visited = visited | {source_id}
                query += self._build_further_hops(source_id, nodes, edges, new_visited)

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
        trapi_node = NodeDict(
            name=node.name,
            categories=[
                BiolinkEntity(biolink.ensure_prefix(cat)) for cat in node.all_categories
            ],
            attributes=[
                AttributeDict(
                    attribute_type_id="biolink:xref",
                    value=node.equivalent_curies,
                )
            ],
        )
        if len(node.all_names):
            trapi_node["attributes"].append(
                AttributeDict(attribute_type_id="biolink:synonym", value=node.all_names)
            )

        # For some reason some publications have empty string
        with contextlib.suppress(ValueError):
            node.publications.remove("")

        if len(node.publications):
            trapi_node["attributes"].append(
                AttributeDict(
                    attribute_type_id="biolink:publications",
                    value=node.publications,
                )
            )
        return trapi_node

    def _build_trapi_edge(self, edge: dg.Edge, initial_curie: str) -> EdgeDict:
        trapi_edge = EdgeDict(
            predicate=BiolinkPredicate(biolink.ensure_prefix(edge.predicate)),
            subject=CURIE(edge.node.id if edge.direction == "in" else initial_curie),
            object=CURIE(initial_curie if edge.direction == "in" else edge.node.id),
            sources=[
                RetrievalSourceDict(
                    resource_id=Infores(
                        edge.primary_knowledge_source or "err_not_provided"
                    ),
                    resource_role="primary_knowledge_source",
                ),
                RetrievalSourceDict(
                    resource_id=Infores("infores:rtx-kg2"),
                    resource_role="aggregator_knowledge_source",
                    upstream_resource_ids=[
                        Infores(edge.primary_knowledge_source or "err_not_provided")
                    ],
                ),
            ],
            attributes=[
                AttributeDict(
                    attribute_type_id="biolink:knowledge_level",
                    value=edge.knowledge_level or "not_provided",
                ),
                AttributeDict(
                    attribute_type_id="biolink:agent_type",
                    value=edge.agent_type or "not_provided",
                ),
            ],
        )
        # TODO: publications

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
