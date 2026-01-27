from typing import Any, Literal, override

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
from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESBooleanQuery,
    ESEdge,
    ESFilterClause,
    ESNode,
    ESPayload,
    ESQueryContext,
)
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    CURIE,
    AttributeDict,
    BiolinkEntity,
    BiolinkPredicate,
    EdgeDict,
    EdgeIdentifier,
    Infores,
    KnowledgeGraphDict,
    NodeDict,
    QEdgeDict,
    QNodeDict,
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

# TODO: Eventually we can roll this into the Tier2 x-bte transpiler
# And use x-bte annotations either on the SmartAPI for each Tier1 resource
# Or just a built-in annotation

SpecialCaseDict = dict[str, tuple[str, Any]]


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
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> ESPayload | list[ESPayload]:
        return super().process_qgraph(qgraph, *additional_qgraphs)

    def generate_query_term(self, target: str, value: list[str]) -> ESFilterClause:
        """Common utility function to generate a termed query based on key-value pairs."""
        if type(value) is not list:
            raise TypeError("value must be a list")

        adjusted_value = value

        # to match both "category" and "categories"
        if "categor" in target or "predicate" in target:
            adjusted_value = [biolink.rmprefix(cat) for cat in value]
        return {"terms": {f"{target}": adjusted_value}}

    def process_qnode(
        self, qnode: QNodeDict, side: Literal["subject", "object"]
    ) -> list[ESFilterClause]:
        """Provide query terms based on given side and fields of a QNodeDict.

        Example return value: { "terms": { "subject.id": ["NCBIGene:22828"] }},
        """
        query_fields = NODE_FIELDS_MAPPING.copy()

        # bypass categories if id is provided
        if qnode.get("ids", None):
            query_fields.pop("categories")

        return [
            self.generate_query_term(f"{side}.{es_field}", values)
            for qfield, es_field in query_fields.items()
            if (values := qnode.get(qfield))
        ]

    def process_qedge(self, qedge: QEdgeDict) -> list[ESFilterClause]:
        """Provide query terms based on a given QEdgeDict.

        Example return value: { "terms": { "predicates": ["Gene"] }},
        """
        # Check required field
        predicates = qedge.get("predicates")

        if type(predicates) is not list or len(predicates) == 0:
            raise Exception("Invalid predicates values")

        # Scalable to more fields

        return [
            self.generate_query_term(f"{es_field}", values)
            for qfield, es_field in EDGE_FIELDS_MAPPING.items()
            if (values := qedge.get(qfield))
        ]

    def generate_query_for_merged_edges(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
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

        qualifier_constraints = edge.get("qualifier_constraints", None)
        qualifier_terms = process_qualifier_constraints(qualifier_constraints)

        if qualifier_terms:
            # if we have `should` in results, this is a multi-constraint
            if "should" in qualifier_terms:
                query_kwargs["should"] = qualifier_terms["should"]
                query_kwargs["minimum_should_match"] = (
                    1  # ensure `should` array is honored
                )

            # otherwise we have either
            # 0) `ESQueryForSingleQualifierConstraint`, a single constraint with multiple qualifiers, or
            # 1) `ESTermClause`, a single qualifier in a single constraint
            # in both cases, inner payload of one or more qualifier terms can be added
            # as a single or a list of `ESQueryForOneQualifierEntry` to `filter` field
            elif "bool" in qualifier_terms:
                query_kwargs["filter"].extend(qualifier_terms["bool"]["must"])
            elif "term" in qualifier_terms:
                query_kwargs["filter"].append(qualifier_terms)

        # generate constraint terms for edges and associated nodes
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
                constraints = entity.get("attribute_constraints", None)
            else:
                constraints = entity.get("constraints", None)

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

        return ESPayload(query=ESQueryContext(bool=ESBooleanQuery(**query_kwargs)))

    @override
    def convert_triple(self, qgraph: QueryGraphDict) -> ESPayload:
        """Provide an ES query body for given trio of Q-dicts."""
        edge = next(iter(qgraph["edges"].values()), None)
        if edge is None:
            raise ValueError("Query graph must contain exactly one edge.")
        in_node = qgraph["nodes"][edge["subject"]]
        out_node = qgraph["nodes"][edge["object"]]
        return self.generate_query_for_merged_edges(in_node, edge, out_node)

    @override
    def convert_batch_triple(self, qgraphs: list[QueryGraphDict]) -> list[ESPayload]:
        return [self.convert_triple(qgraph) for qgraph in qgraphs]

    def build_attributes(
        self, knowledge: ESEdge | ESNode, special_cases: SpecialCaseDict
    ) -> list[AttributeDict]:
        """Build attributes from the given knowledge."""
        attributes: list[AttributeDict] = []

        for field, value in knowledge.attributes.items():
            if field in special_cases:
                continue
            if value is not None and value not in ([], ""):
                attributes.append(
                    AttributeDict(
                        attribute_type_id=biolink.ensure_prefix(field),
                        value=value,
                    )
                )

        for name, value in special_cases.values():
            if value is not None and value not in ([], ""):
                attributes.append(AttributeDict(attribute_type_id=name, value=value))

        return attributes

    def build_nodes(
        self, edges: list[ESEdge], query_subject: QNodeDict, query_object: QNodeDict
    ) -> dict[CURIE, NodeDict]:
        """Build TRAPI nodes from backend representation."""
        nodes = dict[CURIE, NodeDict]()
        for edge in edges:
            node_ids = dict[str, CURIE]()
            for node_pos in ("subject", "object"):
                node: ESNode = getattr(edge, node_pos)
                node_id = CURIE(node.id)
                node_ids[node_pos] = node_id
                if node_id in nodes:
                    continue
                attributes: list[AttributeDict] = []

                # Cases that require additional formatting to be TRAPI-compliant
                special_cases: SpecialCaseDict = {
                    "equivalent_identifiers": (
                        "biolink:xref",
                        [
                            CURIE(i)
                            for i in node.attributes.get("equivalent_identifiers", [])
                        ],
                    )
                }

                attributes = self.build_attributes(node, special_cases)

                constraints = (
                    query_subject if node_pos == "subject" else query_object
                ).get("constraints", []) or []
                if not attributes_meet_contraints(constraints, attributes):
                    continue

                trapi_node = NodeDict(
                    name=node.name,
                    categories=[
                        BiolinkEntity(biolink.ensure_prefix(cat))
                        for cat in node.category
                    ],
                    attributes=attributes,
                )

                nodes[node_id] = trapi_node

        return nodes

    def build_edges(
        self, edges: list[ESEdge], qedge: QEdgeDict
    ) -> dict[EdgeIdentifier, EdgeDict]:
        """Build TRAPI edges from backend representation."""
        trapi_edges = dict[EdgeIdentifier, EdgeDict]()
        for edge in edges:
            attributes: list[AttributeDict] = []
            qualifiers: list[QualifierDict] = []
            sources: list[RetrievalSourceDict] = []

            # Cases that require additional formatting to be TRAPI-compliant
            special_cases: SpecialCaseDict = {
                "category": (
                    "biolink:category",
                    [
                        BiolinkEntity(biolink.ensure_prefix(cat))
                        for cat in edge.attributes.get("category", [])
                    ],
                ),
            }

            attributes = self.build_attributes(edge, special_cases)

            constraints = qedge.get("attribute_constraints", []) or []
            if not attributes_meet_contraints(constraints, attributes):
                continue

            # Build Qualifiers
            for qtype, qval in edge.qualifiers.items():
                qualifiers.append(
                    QualifierDict(
                        qualifier_type_id=QualifierTypeID(biolink.ensure_prefix(qtype)),
                        qualifier_value=qval,
                    )
                )

            # Build Sources
            for source in edge.sources:
                retrieval_source = RetrievalSourceDict(
                    resource_id=Infores(source["resource_id"]),
                    resource_role=source["resource_role"],
                )
                if upstream_resource_ids := source.get("upstream_resource_ids"):
                    retrieval_source["upstream_resource_ids"] = [
                        Infores(upstream) for upstream in upstream_resource_ids
                    ]
                if source_record_urls := source.get("source_record_urls"):
                    retrieval_source["source_record_urls"] = source_record_urls
                sources.append(retrieval_source)

            # Build Edge
            trapi_edge = EdgeDict(
                predicate=BiolinkPredicate(biolink.ensure_prefix(edge.predicate)),
                subject=CURIE(edge.subject.id),
                object=CURIE(edge.object.id),
                sources=sources,
            )
            if len(attributes) > 0:
                trapi_edge["attributes"] = attributes
            if len(qualifiers) > 0:
                trapi_edge["qualifiers"] = qualifiers

            append_aggregator_source(trapi_edge, Infores(CONFIG.tier1.backend_infores))

            edge_hash = hash_hex(hash_edge(trapi_edge))
            trapi_edges[edge_hash] = trapi_edge

        return trapi_edges

    @override
    def convert_results(
        self, qgraph: QueryGraphDict, results: list[ESEdge] | None
    ) -> BackendResult:
        edge = next(iter(qgraph["edges"].values()))
        sbj = qgraph["nodes"][edge["subject"]]
        obj = qgraph["nodes"][edge["object"]]
        nodes = self.build_nodes(results, sbj, obj) if results is not None else {}
        edges = self.build_edges(results, edge) if results is not None else {}

        return BackendResult(
            results=[],
            knowledge_graph=KnowledgeGraphDict(nodes=nodes, edges=edges),
            auxiliary_graphs={},
        )

    def convert_batch_results(
        self, qgraph_list: list[QueryGraphDict], results: list[list[ESEdge]]
    ) -> list[BackendResult]:
        """Wrapper for converting results for a batch query."""
        return [
            self.convert_results(qgraph, result)
            for qgraph, result in zip(qgraph_list, results, strict=False)
        ]
