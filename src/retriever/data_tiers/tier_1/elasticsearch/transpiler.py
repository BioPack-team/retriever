from typing import Any, Literal, override

from retriever.config.general import CONFIG
from retriever.data_tiers.base_transpiler import Tier1Transpiler
from retriever.data_tiers.tier_1.elasticsearch.constraints.attributes.attribute import (
    process_attribute_constraints,
)
from retriever.data_tiers.tier_1.elasticsearch.constraints.qualifiers.qualifier import (
    process_qualifier_constraints,
)
from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESBooleanQuery,
    ESFilterClause,
    ESHit,
    ESPayload,
    ESQueryContext,
)
from retriever.data_tiers.utils import (
    DINGO_KG_EDGE_TOPLEVEL_VALUES,
    DINGO_KG_NODE_TOPLEVEL_VALUES,
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
from retriever.utils.trapi import append_aggregator_source, hash_edge, hash_hex

# TODO: Eventually we can roll this into the Tier2 x-bte transpiler
# And use x-bte annotations either on the SmartAPI for each Tier1 resource
# Or just a built-in annotation


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
        return [
            self.generate_query_term(f"{side}.{es_field}", values)
            for qfield, es_field in NODE_FIELDS_MAPPING.items()
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

        attribute_constraints = edge.get("attribute_constraints", None)

        if attribute_constraints:
            must, must_not = process_attribute_constraints(attribute_constraints)
            if must:
                query_kwargs["must"] = must
            if must_not:
                query_kwargs["must_not"] = must_not

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

    def build_nodes(self, hits: list[ESHit]) -> dict[CURIE, NodeDict]:
        """Build TRAPI nodes from backend representation."""
        nodes = dict[CURIE, NodeDict]()
        for hit in hits:
            node_ids = dict[str, CURIE]()
            for argument in ("subject", "object"):
                node = hit[argument]
                node_id = node["id"]
                node_ids[argument] = node_id
                if node_id in nodes:
                    continue
                attributes: list[AttributeDict] = []

                # Cases that require additional formatting to be TRAPI-compliant
                special_cases: dict[str, tuple[str, Any]] = {
                    "equivalent_identifiers": (
                        "biolink:xref",
                        [CURIE(i) for i in node["equivalent_identifiers"]],
                    )
                }

                for field, value in node.items():
                    if field in DINGO_KG_NODE_TOPLEVEL_VALUES:
                        continue
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
                        attributes.append(
                            AttributeDict(attribute_type_id=name, value=value)
                        )

                trapi_node = NodeDict(
                    name=node["name"],
                    categories=[
                        BiolinkEntity(biolink.ensure_prefix(cat))
                        for cat in node["category"]
                    ],
                    attributes=attributes,
                )

                nodes[node_id] = trapi_node
        return nodes

    def build_edges(self, hits: list[ESHit]) -> dict[EdgeIdentifier, EdgeDict]:
        """Build TRAPI edges from backend representation."""
        edges = dict[EdgeIdentifier, EdgeDict]()
        for hit in hits:
            attributes: list[AttributeDict] = []
            qualifiers: list[QualifierDict] = []
            sources: list[RetrievalSourceDict] = []

            # Cases that require additional formatting to be TRAPI-compliant
            special_cases: dict[str, tuple[str, Any]] = {
                "category": (
                    "biolink:category",
                    [
                        BiolinkEntity(biolink.ensure_prefix(cat))
                        for cat in hit.get("category", [])
                    ],
                ),
            }

            # Build Attributes and Qualifiers
            for field, value in hit.items():
                if field in DINGO_KG_EDGE_TOPLEVEL_VALUES or field in special_cases:
                    continue
                if biolink.is_qualifier(field):
                    qualifiers.append(
                        QualifierDict(
                            qualifier_type_id=QualifierTypeID(
                                biolink.ensure_prefix(field)
                            ),
                            qualifier_value=str(value),
                        )
                    )
                    pass
                elif value is not None and value not in ([], ""):
                    attributes.append(
                        AttributeDict(
                            attribute_type_id=biolink.ensure_prefix(field),
                            value=value,
                        )
                    )

            # Special case attributes
            for name, value in special_cases.values():
                if value is not None and value not in ([], ""):
                    attributes.append(
                        AttributeDict(attribute_type_id=name, value=value)
                    )

            # Build Sources
            for source in hit["sources"]:
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
                predicate=BiolinkPredicate(biolink.ensure_prefix(hit["predicate"])),
                subject=CURIE(hit["subject"]["id"]),
                object=CURIE(hit["object"]["id"]),
                sources=sources,
            )
            if len(attributes) > 0:
                trapi_edge["attributes"] = attributes
            if len(qualifiers) > 0:
                trapi_edge["qualifiers"] = qualifiers

            append_aggregator_source(trapi_edge, Infores(CONFIG.tier1.backend_infores))

            edge_hash = hash_hex(hash_edge(trapi_edge))
            edges[edge_hash] = trapi_edge
        return edges

    @override
    def convert_results(
        self, qgraph: QueryGraphDict, results: list[ESHit] | None
    ) -> BackendResult:
        nodes = self.build_nodes(results) if results is not None else {}
        edges = self.build_edges(results) if results is not None else {}

        return BackendResult(
            results=[],
            knowledge_graph=KnowledgeGraphDict(nodes=nodes, edges=edges),
            auxiliary_graphs={},
        )

    def convert_batch_results(
        self, qgraph_list: list[QueryGraphDict], results: list[list[ESHit]]
    ) -> list[BackendResult]:
        """Wrapper for converting results for a batch query."""
        return [
            self.convert_results(qgraph, result)
            for qgraph, result in zip(qgraph_list, results, strict=False)
        ]
