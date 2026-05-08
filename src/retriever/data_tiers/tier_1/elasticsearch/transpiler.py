from typing import Any, Literal, cast, override

import orjson
from translator_tom import (
    CURIE,
    Attribute,
    AttributeConstraint,
    Biolink,
    Edge,
    EdgeID,
    KnowledgeGraph,
    Node,
    QEdge,
    QNode,
    Qualifier,
    QueryGraph,
    RetrievalSource,
    infores,
)

from retriever.config.general import CONFIG
from retriever.data_tiers.base_transpiler import Tier1Transpiler
from retriever.data_tiers.tier_1.elasticsearch.constraints.attributes.attribute import (
    process_attribute_constraints,
)
from retriever.data_tiers.tier_1.elasticsearch.constraints.qualifiers.qualifier import (
    process_qualifier_constraints,
)
from retriever.data_tiers.tier_1.elasticsearch.constraints.types.attribute_types import (
    AttributeFilterQuery,
    AttributeOrigin,
)
from retriever.data_tiers.tier_1.elasticsearch.constraints.types.qualifier_types import (
    ESEquivalentQualifierPairCollection,
)
from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESBooleanQuery,
    ESEdge,
    ESFilterClause,
    ESNode,
    ESPayload,
    ESQueryContext,
)
from retriever.lookup.utils import QueryDumper
from retriever.types.general import BackendResult

# TODO: Eventually we can roll this into the Tier2 x-bte transpiler
# And use x-bte annotations either on the SmartAPI for each Tier1 resource
# Or just a built-in annotation

SpecialCase = dict[str, tuple[str, Any]]


NODE_FIELDS_MAPPING = {
    "ids": "id",
    "categories": "category",
}


EDGE_FIELDS_MAPPING = {
    "predicates": "predicate_ancestors",
}


class ElasticsearchTranspiler(Tier1Transpiler):
    """Transpiler for TRAPI to/from Elasticsearch queries."""

    @override
    def process_qgraph(
        self, qgraph: QueryGraph, *additional_qgraphs: QueryGraph
    ) -> ESPayload | list[ESPayload]:
        payload = super().process_qgraph(qgraph, *additional_qgraphs)

        if CONFIG.tier1.dump_queries:
            QueryDumper().put(
                "write_tier1",
                orjson.dumps(
                    {"trapi": qgraph, "es": payload},
                    option=orjson.OPT_APPEND_NEWLINE,
                ),
            )

        return payload

    def generate_query_term(self, target: str, value: list[str]) -> ESFilterClause:
        """Common utility function to generate a termed query based on key-value pairs."""
        if type(value) is not list:
            raise TypeError("value must be a list")

        adjusted_value = value

        # to match both "category" and "categories"
        if "categor" in target or "predicate" in target:
            adjusted_value = [Biolink.rmprefix(cat) for cat in value]
        return {"terms": {f"{target}": adjusted_value}}

    def process_qnode(
        self, qnode: QNode, side: Literal["subject", "object"]
    ) -> list[ESFilterClause]:
        """Provide query terms based on given side and fields of a QNode.

        Example return value: { "terms": { "subject.id": ["NCBIGene:22828"] }},
        """
        query_fields = NODE_FIELDS_MAPPING.copy()

        # bypass categories if id is provided
        if qnode.ids:
            query_fields.pop("categories")

        return [
            self.generate_query_term(f"{side}.{es_field}", values)
            for qfield, es_field in query_fields.items()
            if (values := getattr(qnode, qfield))
        ]

    def process_qedge(self, qedge: QEdge) -> list[ESFilterClause]:
        """Provide query terms based on a given QEdge.

        Example return value: { "terms": { "predicates": ["Gene"] }},
        """
        # Check required field
        predicates = qedge.predicates

        if type(predicates) is not list or len(predicates) == 0:
            raise Exception("Invalid predicates values")

        # Scalable to more fields

        return [
            self.generate_query_term(f"{es_field}", values)
            for qfield, es_field in EDGE_FIELDS_MAPPING.items()
            if (values := getattr(qedge, qfield))
        ]

    def generate_attribute_constraints(
        self,
        in_node: QNode,
        edge: QEdge,
        out_node: QNode,
        query_kwargs: ESBooleanQuery,
    ) -> ESBooleanQuery:
        """Generate attribute constraints based on QNode/QEdge payload."""
        constraint_origins: list[AttributeOrigin] = ["edge", "subject", "object"]

        all_must: list[AttributeFilterQuery] = []
        all_must_not: list[AttributeFilterQuery] = []

        for origin in constraint_origins:
            entity = (
                edge
                if origin == "edge"
                else in_node
                if origin == "subject"
                else out_node
            )

            if origin == "edge":
                constraints = cast(QEdge, entity).attribute_constraints
            else:
                constraints = cast(QNode, entity).constraints

            if constraints:
                must, must_not = process_attribute_constraints(constraints, origin)
                if must:
                    all_must.extend(must)
                if must_not:
                    all_must_not.extend(must_not)

        if all_must:
            query_kwargs["must"] = all_must
        if all_must_not:
            query_kwargs["must_not"] = all_must_not

        return query_kwargs

    def generate_queries(
        self,
        in_node: QNode,
        edge: QEdge,
        out_node: QNode,
        gen_attribute_constraints: bool = False,  # disable attribute constraints for now
    ) -> ESPayload:
        """Generate query based on merged edges schema on Elasticsearch.

        Example payload:

        {
          "query": {
            "bool": {
              "filter": [
                { "terms": { "subject.id": ["NCBIGene:22828"] }},
                { "terms": { "object.id": ["NCBIGene:2801"] }}
              ]
            }
          }
        }
        """
        subject_terms = self.process_qnode(in_node, "subject")
        object_terms = self.process_qnode(out_node, "object")
        edge_terms = self.process_qedge(edge)

        query_kwargs: ESBooleanQuery = {
            "filter": [*subject_terms, *object_terms, *edge_terms]
        }

        qualifier_constraints = edge.qualifier_constraints
        qualifier_terms = process_qualifier_constraints(qualifier_constraints)

        if qualifier_terms:
            # if we have `should` in results, this is a multi-constraint
            if "should" in qualifier_terms:
                query_kwargs["should"] = qualifier_terms["should"]
                query_kwargs["minimum_should_match"] = (
                    1  # ensure `should` array is honored
                )

            # otherwise we have either
            # 0) `ESEquivalentQualifierPairCollection`, a single constraint of a should array, or
            # 1) `ESBoolQueryForExpandedQualifiers`, a single constraint of a must array
            # in both cases, there's a bool field that can be parsed/added to existing filter query
            elif "must" in qualifier_terms["bool"]:
                query_kwargs["filter"].extend(qualifier_terms["bool"]["must"])
            elif "should" in qualifier_terms["bool"]:
                query_kwargs["filter"].append(
                    cast(ESEquivalentQualifierPairCollection, qualifier_terms)
                )

        # generate constraint terms for edges and associated nodes
        # currently, this is DISABLED by default to favor post-processing
        if gen_attribute_constraints:
            query_kwargs = self.generate_attribute_constraints(
                in_node, edge, out_node, query_kwargs
            )

        return ESPayload(query=ESQueryContext(bool=ESBooleanQuery(**query_kwargs)))

    @override
    def convert_triple(self, qgraph: QueryGraph) -> ESPayload:
        """Provide an ES query body for given trio of Q-dicts."""
        edge = next(iter(qgraph.edges.values()), None)
        if edge is None:
            raise ValueError("Query graph must contain exactly one edge.")
        in_node = qgraph.nodes[edge.subject]
        out_node = qgraph.nodes[edge.object]
        return self.generate_queries(in_node, edge, out_node)

    @override
    def convert_batch_triple(self, qgraphs: list[QueryGraph]) -> list[ESPayload]:
        return [self.convert_triple(qgraph) for qgraph in qgraphs]

    def build_attributes(
        self, knowledge: ESEdge | ESNode, special_cases: SpecialCase
    ) -> list[Attribute]:
        """Build attributes from the given knowledge."""
        attributes: list[Attribute] = []

        for field, value in knowledge.attributes.items():
            if field in special_cases:
                continue
            if value is not None and value not in ([], ""):
                attributes.append(
                    Attribute.model_construct(
                        attribute_type_id=Biolink(field),
                        value=value,
                    )
                )

        for name, value in special_cases.values():
            if value is not None and value not in ([], ""):
                attributes.append(
                    Attribute.model_construct(attribute_type_id=name, value=value)
                )

        return attributes

    def build_single_node(
        self, node: ESNode, attributes: list[Attribute] | None = None
    ) -> Node:
        """Build a single TRAPI node from the given knowledge."""
        _attributes = [] if attributes is None else attributes

        if attributes is None:
            # Cases that require additional formatting to be TRAPI-compliant
            special_cases: SpecialCase = {}
            _attributes = self.build_attributes(node, special_cases)

        trapi_node = Node.model_construct(
            name=node.name,
            categories=[Biolink(cat) for cat in node.category],
            attributes=_attributes,
        )

        return trapi_node

    def build_nodes(
        self, edges: list[ESEdge], query_subject: QNode, query_object: QNode
    ) -> dict[CURIE, Node]:
        """Build TRAPI nodes from backend representation."""
        nodes = dict[CURIE, Node]()
        for edge in edges:
            node_ids = dict[str, CURIE]()
            for node_pos in ("subject", "object"):
                node: ESNode = getattr(edge, node_pos)
                node_id = CURIE(node.id)
                node_ids[node_pos] = node_id
                if node_id in nodes:
                    continue
                # Cases that require additional formatting to be TRAPI-compliant
                special_cases: SpecialCase = {}

                attributes = self.build_attributes(node, special_cases)

                constraints = (
                    query_subject if node_pos == "subject" else query_object
                ).constraints_list

                if not AttributeConstraint.set_met_by(constraints, attributes):
                    continue

                trapi_node = self.build_single_node(node, attributes)

                nodes[node_id] = trapi_node

        return nodes

    def build_edges(self, edges: list[ESEdge], qedge: QEdge) -> dict[EdgeID, Edge]:
        """Build TRAPI edges from backend representation."""
        trapi_edges = dict[EdgeID, Edge]()
        for edge in edges:
            attributes: list[Attribute] = []
            qualifiers: list[Qualifier] = []
            sources: list[RetrievalSource] = []

            # Cases that require additional formatting to be TRAPI-compliant
            special_cases: SpecialCase = {
                "category": (
                    Biolink("category"),
                    [Biolink(cat) for cat in edge.attributes.get("category", [])],
                ),
            }

            attributes = self.build_attributes(edge, special_cases)

            constraints = qedge.attribute_constraints_list
            if not AttributeConstraint.set_met_by(constraints, attributes):
                continue

            # Build Qualifiers
            for qtype, qval in edge.qualifiers.items():
                qualifiers.append(
                    Qualifier.model_construct(
                        qualifier_type_id=Biolink(qtype),
                        qualifier_value=qval
                        if "qualified_predicate" not in qtype
                        else Biolink(qval),
                    )
                )

            # Build Sources
            for source in edge.sources:
                retrieval_source = RetrievalSource.model_construct(
                    resource_id=infores(source.resource_id),
                    resource_role=source.resource_role,
                )
                if upstream_resource_ids := source.upstream_resource_ids:
                    retrieval_source.upstream_resource_ids = [
                        infores(upstream) for upstream in upstream_resource_ids
                    ]
                if source_record_urls := source.source_record_urls:
                    retrieval_source.source_record_urls = source_record_urls
                sources.append(retrieval_source)

            # Build Edge
            trapi_edge = Edge.model_construct(
                predicate=Biolink(edge.predicate),
                subject=edge.subject.id,
                object=edge.object.id,
                sources=sources,
            )
            if len(attributes) > 0:
                trapi_edge.attributes = attributes
            if len(qualifiers) > 0:
                trapi_edge.qualifiers = qualifiers

            trapi_edge.append_aggregator(infores(CONFIG.tier1.backend_infores))

            edge_hash = trapi_edge.hash()
            trapi_edges[edge_hash] = trapi_edge

        return trapi_edges

    @override
    def convert_results(
        self, qgraph: QueryGraph, results: list[ESEdge]
    ) -> BackendResult:
        edge = next(iter(qgraph.edges.values()))
        sbj = qgraph.nodes[edge.subject]
        obj = qgraph.nodes[edge.object]
        nodes = self.build_nodes(results, sbj, obj)
        edges = self.build_edges(results, edge)

        return BackendResult(
            results=[],
            knowledge_graph=KnowledgeGraph.model_construct(nodes=nodes, edges=edges),
            auxiliary_graphs={},
        )

    def convert_batch_results(
        self, qgraph_list: list[QueryGraph], results: list[list[ESEdge]]
    ) -> list[BackendResult]:
        """Wrapper for converting results for a batch query."""
        return [
            self.convert_results(qgraph, result)
            for qgraph, result in zip(qgraph_list, results, strict=False)
        ]
