from typing import override

from opentelemetry import trace

from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.neo4j.driver import Neo4jDriver
from retriever.data_tiers.tier_0.neo4j.transpiler import Neo4jTranspiler
from retriever.types.general import BackendResult
from retriever.types.trapi import QueryGraphDict

tracer = trace.get_tracer("lookup.execution.tracer")


class Neo4jQuery(Tier0Query):
    """Adapter to querying Plater Neo4j as a Tier 0 backend."""

    @override
    async def get_results(self, qgraph: QueryGraphDict) -> BackendResult:
        neo4j_driver = Neo4jDriver()
        transpiler = Neo4jTranspiler()

        # Transpile to cypher
        query_cypher = transpiler.process_qgraph(qgraph)
        self.job_log.trace(query_cypher)

        neo4j_record = await neo4j_driver.run_query(query_cypher)

        # Convert neo4j record response to TRAPI
        with tracer.start_as_current_span("transform_results"):
            result = transpiler.convert_results(qgraph, neo4j_record)

        return result
