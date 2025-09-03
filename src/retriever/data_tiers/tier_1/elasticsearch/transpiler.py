from typing import Any, Literal

from typing_extensions import override

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResults
from retriever.types.trapi import QueryGraphDict, QNodeDict, QEdgeDict



class ElasticsearchTranspiler(Transpiler):
    """Transpiler for TRAPI to/from Elasticsearch queries"""


    def generate_query_term(self, target: str, value):
        """ Common utility function to generate a termed query based on key-value pairs"""
        return { "terms": { f"{target}.keyword": value}}

    def process_qnode(self, qnode, side:  Literal["subject", "object"]):
        """
        Provide query terms based on given side and fields of a QNodeDict
        example return value: { "terms": { "subject.id.keyword": ["NCBIGene:22828"] }},
        """

        field_mapping = {
            "ids": "id",
            "categories": "all_categories", # could be just "category
        }

        return [
            self.generate_query_term(f"{side}.{es_field}", values)
            for qfield, es_field in field_mapping.items()
            if (values := qnode.get(qfield))
        ]


    def process_qedge(self, qedge):
        """
        Provide query terms based on a given QEdgeDict

        example return value: { "terms": { "predicates.keyword": ["biolink:Gene"] }},
        """

        # check required field
        predicates = qedge.get('predicates')

        if type(predicates) is not list or len(predicates) == 0:
            raise Exception(f"Invalid predicates values")


        # scalable to more fields
        field_mapping = {
            "predicates": "predicate",
        }

        return [
            self.generate_query_term(f"{es_field}", values)
            for qfield, es_field in field_mapping.items()
            if (values := qedge.get(qfield))
        ]



    def generate_query_for_merged_edges(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> dict:
        """
        Generate query based on merged edges schema on Elasticsearch

        example pyload:

        {
          "query": {
            "bool": {
              "filter": [
                { "terms": { "subject.id.keyword": ["NCBIGene:22828"] }},
                { "terms": { "object.id.keyword": ["NCBIGene:2801"] }}
              ]
            }
          }
        }
        """

        subject_terms = self.process_qnode(in_node, "subject")
        object_terms = self.process_qnode(out_node, "object")
        edge_terms = self.process_qedge(edge)

        return {
                "query": {
                    "bool": {
                        "filter": [*subject_terms, *object_terms, *edge_terms]
                    }
                }
            }


    @override
    def convert_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> tuple[QueryGraphDict, dict]:
        """ Special case of convert_batch_triple"""

        return self.convert_batch_triple(in_node, edge, out_node)

    @override
    def convert_batch_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> tuple[QueryGraphDict, dict]:
        """ Provide an ES query body for given trio of Q-dicts  """

        qgraph = QueryGraphDict(
            nodes={edge["subject"]: in_node, edge["object"]: out_node},
            edges={"edge": edge},
        )

        # use "merged_edges" schema for now
        # todo "adjacency list schema", when updated

        return qgraph, self.generate_query_for_merged_edges(in_node, edge, out_node)


    @override
    def _convert_multihop(self, qgraph: QueryGraphDict) -> Any:
        raise NotImplementedError

    @override
    def _convert_batch_multihop(self, qgraph: QueryGraphDict) -> Any:
        raise NotImplementedError

    @override
    def convert_results(self, qgraph: QueryGraphDict, results: Any) -> BackendResults:
        raise NotImplementedError