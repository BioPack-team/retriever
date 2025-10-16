from typing import Literal, override

from retriever.data_tiers.base_transpiler import Tier1Transpiler
from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESBooleanQuery,
    ESFilterClause,
    ESHit,
    ESPayload,
    ESQueryContext,
)
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    CURIE,
    AttributeDict,
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
from retriever.utils.trapi import hash_edge, hash_hex

# TODO: Eventually we can roll this into the Tier2 x-bte transpiler
# And use x-bte annotations either on the SmartAPI for each Tier1 resource
# Or just a built-in annotation


class ElasticsearchTranspiler(Tier1Transpiler):
    """Transpiler for TRAPI to/from Elasticsearch queries."""

    @override
    def process_qgraph(
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> ESPayload:
        return super().process_qgraph(qgraph, *additional_qgraphs)

    def generate_query_term(self, target: str, value: list[str]) -> ESFilterClause:
        """Common utility function to generate a termed query based on key-value pairs."""
        if type(value) is not list:
            raise TypeError("value must be a list")

        adjusted_value = value
        if "categor" in target or "predicate" in target:
            adjusted_value = [biolink.rmprefix(cat) for cat in value]
        return {"terms": {f"{target}": adjusted_value}}

    def process_qnode(
        self, qnode: QNodeDict, side: Literal["subject", "object"]
    ) -> list[ESFilterClause]:
        """Provide query terms based on given side and fields of a QNodeDict.

        Example return value: { "terms": { "subject.id": ["NCBIGene:22828"] }},
        """
        field_mapping = {
            "ids": "id",
            "categories": "all_categories",  # Could be just "category"
        }

        return [
            self.generate_query_term(f"{side}.{es_field}", values)
            for qfield, es_field in field_mapping.items()
            if (values := qnode.get(qfield))
        ]

    def process_qedge(self, qedge: QEdgeDict) -> list[ESFilterClause]:
        """Provide query terms based on a given QEdgeDict.

        Example return value: { "terms": { "predicates": ["biolink:Gene"] }},
        """
        # Check required field
        predicates = qedge.get("predicates")

        if type(predicates) is not list or len(predicates) == 0:
            raise Exception("Invalid predicates values")

        # Scalable to more fields
        field_mapping = {
            "predicates": "all_predicates",
        }

        return [
            self.generate_query_term(f"{es_field}", values)
            for qfield, es_field in field_mapping.items()
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

        return ESPayload(
            query=ESQueryContext(
                bool=ESBooleanQuery(filter=[*subject_terms, *object_terms, *edge_terms])
            )
        )

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
    def convert_batch_triple(self, qgraphs: list[QueryGraphDict]) -> ESPayload:
        raise NotImplementedError("Batching is not supported.")

    def build_nodes(self, hits: list[ESHit]) -> dict[CURIE, NodeDict]:
        """Build TRAPI nodes from backend representation."""
        nodes = dict[CURIE, NodeDict]()
        for hit in hits:
            node_ids = dict[str, CURIE]()
            for argument in ("subject", "object"):
                node = hit[argument]
                node_id = node["id"]
                node_ids[argument] = node_id
                if node_id not in nodes:
                    trapi_node = NodeDict(
                        name=node["name"],
                        categories=node["all_categories"],
                        attributes=[
                            AttributeDict(
                                attribute_type_id="biolink:xref",
                                value=node["equivalent_curies"],
                            ),
                        ],
                    )
                    if synonyms := node.get("all_names"):
                        trapi_node["attributes"].append(
                            AttributeDict(
                                attribute_type_id="biolink:synonym",
                                value=synonyms,
                            )
                        )

                    for attribute_type_id in ("publications",):  # iri?
                        if attribute_type_id not in node:
                            continue
                        trapi_node["attributes"].append(
                            AttributeDict(
                                attribute_type_id=f"biolink:{attribute_type_id}",
                                value=node[attribute_type_id],
                            )
                        )

                    nodes[node_id] = trapi_node
        return nodes

    def populate_edge_attributes(self, hit: ESHit, edge: EdgeDict) -> None:
        """Populate the `attributes` field of an edge using a given hit."""
        if "attributes" not in edge or edge["attributes"] is None:
            edge["attributes"] = []
        edge["attributes"].extend(
            (
                AttributeDict(
                    attribute_type_id="biolink:knowledge_level",
                    value=hit["knowledge_level"] or "not_provided",
                ),
                AttributeDict(
                    attribute_type_id="biolink:agent_type",
                    value=hit["agent_type"] or "not_provided",
                ),
            )
        )
        if "publications" in hit:
            edge["attributes"].append(
                AttributeDict(
                    attribute_type_id="biolink:publications",
                    value=hit["publications"],
                )
            )

        if "publications_info" in hit:
            for info in hit["publications_info"]:
                if info is None:
                    continue
                study_attr = AttributeDict(
                    attribute_type_id="biolink:has_supporting_study_result",
                    value=info["pmid"],
                    attributes=[],
                )
                sub_attrs = list[AttributeDict](
                    (
                        AttributeDict(
                            attribute_type_id="biolink:publications",
                            value=[info["pmid"]],
                        ),
                    )
                )
                if "sentence" in info:
                    sub_attrs.append(
                        AttributeDict(
                            attribute_type_id="biolink:supporting_text",
                            value=info["sentence"],
                        )
                    )
                if "publication_date" in info:
                    sub_attrs.append(
                        AttributeDict(
                            attribute_type_id="biolink:publication_date",
                            value=info["publication_date"],
                        )
                    )
                study_attr["attributes"] = sub_attrs
                edge["attributes"].append(study_attr)

    def build_edges(self, hits: list[ESHit]) -> dict[EdgeIdentifier, EdgeDict]:
        """Build TRAPI edges from backend representation."""
        edges = dict[EdgeIdentifier, EdgeDict]()
        for hit in hits:
            edge = EdgeDict(
                predicate=hit["predicate"],
                subject=hit["subject"]["id"],
                object=hit["object"]["id"],
                sources=[
                    RetrievalSourceDict(
                        resource_id=hit["primary_knowledge_source"],
                        resource_role="primary_knowledge_source",
                    ),
                    RetrievalSourceDict(
                        resource_id=Infores("infores:rtx-kg2"),
                        resource_role="aggregator_knowledge_source",
                        upstream_resource_ids=[hit["primary_knowledge_source"]],
                    ),
                ],
                attributes=[],
            )

            self.populate_edge_attributes(hit, edge)

            for qualifier_type_id in biolink.get_all_qualifiers():
                if qualifier_type_id == "qualifier":
                    continue  # Shouldn't be used (doesn't qualify anything)
                qualifier_key = qualifier_type_id
                if qualifier_type_id != "qualified_predicate":
                    # ES index uses a different naming scheme for most qualifiers
                    qualifier_key = (
                        f"qualified_{qualifier_type_id.replace('_qualifier', '')}"
                    )
                if value := hit.get(qualifier_key):
                    if "qualifiers" not in edge or edge["qualifiers"] is None:
                        edge["qualifiers"] = []
                    edge["qualifiers"].append(
                        QualifierDict(
                            qualifier_type_id=QualifierTypeID(qualifier_type_id),
                            qualifier_value=value,
                        )
                    )
            edge_hash = hash_hex(hash_edge(edge))
            edges[edge_hash] = edge
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
