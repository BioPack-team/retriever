from typing import cast, override

from opentelemetry import trace
from reasoner_transpiler.cypher import (
    get_query,  # pyright:ignore[reportUnknownVariableType]
    transform_result,  # pyright:ignore[reportUnknownVariableType]
)

from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.neo4j.driver import Neo4jDriver
from retriever.types.general import BackendResults
from retriever.types.trapi import QueryGraphDict

tracer = trace.get_tracer("lookup.execution.tracer")


class Neo4jQuery(Tier0Query):
    """Adapter to querying Plater Neo4j as a Tier 0 backend."""

    @override
    async def get_results(self, qgraph: QueryGraphDict) -> BackendResults:
        neo4j_driver = Neo4jDriver()

        # Transpile to cypher
        query_cypher = get_query(qgraph)
        self.job_log.trace(query_cypher)

        neo4j_record = await neo4j_driver.run_query(query_cypher)

        # Have to cast to object, then BackendResults because of some type weirdness
        # Transpiler will always output valid TRAPI so this should always work
        with tracer.start_as_current_span("transform_results"):
            result = cast(
                BackendResults,
                cast(object, transform_result(neo4j_record, dict(qgraph))),
            )

        for edge in result["knowledge_graph"]["edges"].values():
            transpiler_source = next(
                iter(
                    source
                    for source in edge["sources"]
                    if source["resource_id"] == "reasoner-transpiler"
                )
            )
            transpiler_source["resource_id"] = "infores:dogpark-tier0"

        return result
