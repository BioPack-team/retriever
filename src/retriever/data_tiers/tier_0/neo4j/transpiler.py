from typing import Any, LiteralString, cast, override

from reasoner_transpiler.cypher import (
    get_query,  # pyright:ignore[reportUnknownVariableType]
    transform_result,  # pyright:ignore[reportUnknownVariableType]
)

from retriever.data_tiers.base_transpiler import Transpiler
from retriever.types.general import BackendResult
from retriever.types.trapi import QEdgeDict, QNodeDict, QueryGraphDict


class Neo4jTranspiler(Transpiler):
    """Transpiler for TRAPI -> Neo4j and back."""

    @override
    def process_qgraph(self, qgraph: QueryGraphDict) -> LiteralString:
        return self._convert_batch_multihop(qgraph)

    @override
    def convert_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> tuple[QueryGraphDict, LiteralString]:
        return self.convert_batch_triple(in_node, edge, out_node)

    @override
    def convert_batch_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> tuple[QueryGraphDict, LiteralString]:
        qgraph = QueryGraphDict(
            nodes={edge["subject"]: in_node, edge["object"]: out_node},
            edges={"edge": edge},
        )

        # Special case because we need qgraph for reasoner-transpiler's result conversion
        return qgraph, self._convert_batch_multihop(qgraph, subclass=False)

    @override
    def _convert_multihop(self, qgraph: QueryGraphDict) -> LiteralString:
        return self._convert_batch_multihop(qgraph)

    @override
    def _convert_batch_multihop(
        self, qgraph: QueryGraphDict, subclass: bool = True
    ) -> LiteralString:
        # This is a special case where we don't do any internal logic
        # Soley because reasoner-transpiler exists for this case
        # (This only works for Neo4j and the robokopkg backend)
        return get_query(qgraph, subclass=subclass)

    @override
    def convert_results(self, qgraph: QueryGraphDict, results: Any) -> BackendResult:
        # Have to cast to object, then BackendResults because of some type weirdness
        # Transpiler will always output valid TRAPI so this should always work
        result = cast(
            BackendResult,
            cast(object, transform_result(results, dict(qgraph))),
        )

        # Have to fix the edge source information because reasoner-transpiler
        # Adds itself as an aggregator, which is somewhat incorrect.
        # We also need to ensure edges have the data tier as a source.
        for edge in result["knowledge_graph"]["edges"].values():
            transpiler_source = next(
                iter(
                    source
                    for source in edge["sources"]
                    if source["resource_id"] == "reasoner-transpiler"
                )
            )
            transpiler_source["resource_id"] = "infores:automat-robokopkg"

        return result
