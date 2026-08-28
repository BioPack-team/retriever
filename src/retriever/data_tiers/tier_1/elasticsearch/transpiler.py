from collections.abc import Sequence
from typing import Any, Literal, cast, override

import orjson
from translator_tom.v1_6 import (
    CURIE,
    AttributeConstraint,
    Biolink,
    EdgeID,
    Infores,
    QEdge,
    QNode,
    QueryGraph,
)
from translator_tom.v1_6.model_dicts import (
    AttributeConstraintDict,
    AttributeConstraintDictUtil,
    AttributeDict,
    EdgeDict,
    EdgeDictUtil,
    KnowledgeGraphDict,
    NodeDict,
    QualifierConstraintDict,
    QualifierDict,
    RetrievalSourceDict,
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

SpecialCaseDict = dict[str, tuple[str, Any]]


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
                    {"trapi": qgraph.to_dict(), "es": payload},
                    option=orjson.OPT_APPEND_NEWLINE,
                ),
            )

        return payload

    def generate_query_term(self, target: str, value: Sequence[str]) -> ESFilterClause:
        """Common utility function to generate a termed query based on key-value pairs."""
        # to match both "category" and "categories"
        if "categor" in target or "predicate" in target:
            adjusted_value = [Biolink.rmprefix(entry) for entry in value]
        else:
            adjusted_value = list(value)
        return {"terms": {f"{target}": adjusted_value}}

    def process_qnode(
        self, qnode: QNode, side: Literal["subject", "object"]
    ) -> list[ESFilterClause]:
        """Provide query terms based on given side and fields of a QNode.

        Example return value: { "terms": { "subject.id": ["NCBIGene:22828"] }},
        """
        # bypass categories if id is provided
        if ids := qnode.ids:
            return [self.generate_query_term(f"{side}.id", ids)]
        if categories := qnode.categories:
            return [self.generate_query_term(f"{side}.category", categories)]
        return []

    def process_qedge(self, qedge: QEdge) -> list[ESFilterClause]:
        """Provide query terms based on a given QEdge.

        Example return value: { "terms": { "predicate_ancestors": ["biolink:related_to"] }},
        """
        predicates = qedge.predicates
        if not predicates:
            raise Exception("Invalid predicates values")

        return [self.generate_query_term("predicate_ancestors", predicates)]

    def generate_attribute_constraints(
        self,
        in_node: QNode,
        edge: QEdge,
        out_node: QNode,
        query_kwargs: ESBooleanQuery,
    ) -> ESBooleanQuery:
        """Generate attribute constraints based on QNode/QEdge payload."""
        origins: list[tuple[AttributeOrigin, list[AttributeConstraint]]] = [
            ("edge", edge.attribute_constraints_list),
            ("subject", in_node.constraints_list),
            ("object", out_node.constraints_list),
        ]

        all_must: list[AttributeFilterQuery] = []
        all_must_not: list[AttributeFilterQuery] = []

        for origin, raw_constraints in origins:
            if not raw_constraints:
                continue
            constraints = cast(
                "list[AttributeConstraintDict]",
                [constraint.to_dict() for constraint in raw_constraints],
            )
            must, must_not = process_attribute_constraints(constraints, origin)
            all_must.extend(must)
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

        qualifier_constraints = (
            cast(
                "list[QualifierConstraintDict]",
                [constraint.to_dict() for constraint in edge.qualifier_constraints],
            )
            if edge.qualifier_constraints is not None
            else None
        )
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
                        attribute_type_id=Biolink(field),
                        value=value,
                    )
                )

        for name, value in special_cases.values():
            if value is not None and value not in ([], ""):
                attributes.append(AttributeDict(attribute_type_id=name, value=value))

        return attributes

    def build_single_node(
        self, node: ESNode, attributes: list[AttributeDict] | None = None
    ) -> NodeDict:
        """Build a single TRAPI node from the given knowledge."""
        _attributes = [] if attributes is None else attributes

        if attributes is None:
            # Cases that require additional formatting to be TRAPI-compliant
            special_cases: SpecialCaseDict = {}
            _attributes = self.build_attributes(node, special_cases)

        trapi_node = NodeDict(
            name=node.name,
            categories=[Biolink.Entity(Biolink(cat)) for cat in node.category],
            attributes=_attributes,
        )

        return trapi_node

    def build_nodes(
        self, edges: list[ESEdge], query_subject: QNode, query_object: QNode
    ) -> dict[CURIE, NodeDict]:
        """Build TRAPI nodes from backend representation."""
        nodes = dict[CURIE, NodeDict]()
        for edge in edges:
            for node_pos in ("subject", "object"):
                node: ESNode = getattr(edge, node_pos)
                node_id = CURIE(node.id)
                if node_id in nodes:
                    continue
                # Cases that require additional formatting to be TRAPI-compliant
                special_cases: SpecialCaseDict = {}

                attributes = self.build_attributes(node, special_cases)

                qnode = query_subject if node_pos == "subject" else query_object
                constraints = cast(
                    "list[AttributeConstraintDict]",
                    [constraint.to_dict() for constraint in qnode.constraints_list],
                )

                if not AttributeConstraintDictUtil.set_met_by(constraints, attributes):
                    continue

                nodes[node_id] = self.build_single_node(node, attributes)

        return nodes

    def build_edges(self, edges: list[ESEdge], qedge: QEdge) -> dict[EdgeID, EdgeDict]:
        """Build TRAPI edges from backend representation."""
        trapi_edges = dict[EdgeID, EdgeDict]()
        for edge in edges:
            qualifiers: list[QualifierDict] = []
            sources: list[RetrievalSourceDict] = []

            # Cases that require additional formatting to be TRAPI-compliant
            special_cases: SpecialCaseDict = {
                "category": (
                    "biolink:category",
                    [
                        Biolink.Entity(Biolink(cat))
                        for cat in edge.attributes.get("category", [])
                    ],
                ),
            }

            attributes = self.build_attributes(edge, special_cases)

            constraints = cast(
                "list[AttributeConstraintDict]",
                [
                    constraint.to_dict()
                    for constraint in qedge.attribute_constraints_list
                ],
            )
            if not AttributeConstraintDictUtil.set_met_by(constraints, attributes):
                continue

            # Build Qualifiers
            for qtype, qval in edge.qualifiers.items():
                qualifiers.append(
                    QualifierDict(
                        qualifier_type_id=Biolink.Qualifier(Biolink(qtype)),
                        qualifier_value=qval
                        if "qualified_predicate" not in qtype
                        else Biolink(qval),
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
                predicate=Biolink.Predicate(Biolink(edge.predicate)),
                subject=CURIE(edge.subject.id),
                object=CURIE(edge.object.id),
                sources=sources,
            )
            if len(attributes) > 0:
                trapi_edge["attributes"] = attributes
            if len(qualifiers) > 0:
                trapi_edge["qualifiers"] = qualifiers

            EdgeDictUtil.append_aggregator(
                trapi_edge, Infores(CONFIG.tier1.backend_infores)
            )

            trapi_edges[EdgeDictUtil.hash(trapi_edge)] = trapi_edge

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
            knowledge_graph=KnowledgeGraphDict(nodes=nodes, edges=edges),
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
