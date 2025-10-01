from typing import Any, LiteralString, cast, override

from reasoner_transpiler.cypher import (
    get_query,  # pyright:ignore[reportUnknownVariableType]
    transform_result,  # pyright:ignore[reportUnknownVariableType]
)

from retriever.data_tiers.base_transpiler import (
    Tier0Transpiler,
    Tier1Transpiler,
)
from retriever.types.general import BackendResult
from retriever.types.trapi import Infores, QueryGraphDict


class Neo4jTranspiler(Tier0Transpiler, Tier1Transpiler):
    """Transpiler for TRAPI -> Neo4j and back."""

    @override
    def process_qgraph(
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> LiteralString:
        return super().process_qgraph(qgraph, *additional_qgraphs)

    @override
    def convert_triple(
        self, qgraph: QueryGraphDict, subclass: bool = True
    ) -> LiteralString:
        return self.convert_multihop(qgraph, subclass)

    @override
    def convert_batch_triple(self, qgraphs: list[QueryGraphDict]) -> LiteralString:
        raise NotImplementedError("Neo4jTranspiler does not support batching.")

    @override
    def convert_multihop(
        self, qgraph: QueryGraphDict, subclass: bool = True
    ) -> LiteralString:
        return get_query(qgraph, subclass=subclass)

    @override
    def convert_batch_multihop(
        self, qgraphs: list[QueryGraphDict], subclass: bool = True
    ) -> LiteralString:
        raise NotImplementedError("Neo4jTranspiler does not support batching.")

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
            transpiler_source["resource_id"] = Infores("infores:automat-robokopkg")

        return result
