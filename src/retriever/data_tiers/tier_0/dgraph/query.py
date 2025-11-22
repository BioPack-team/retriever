from typing import override

from opentelemetry import trace

from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.dgraph.driver import DgraphGrpcDriver
from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.types.general import BackendResult
from retriever.types.trapi import QueryGraphDict

tracer = trace.get_tracer("lookup.execution.tracer")


class DgraphQuery(Tier0Query):
    """Adapter to querying Dgraph as a Tier 0 backend."""

    @override
    async def get_results(self, qgraph: QueryGraphDict) -> BackendResult:
        backend_driver = DgraphGrpcDriver()
        dgraph_schema_version = await backend_driver.get_active_version()
        transpiler = DgraphTranspiler(version=dgraph_schema_version)

        # Transpile to backend QL
        query_payload = transpiler.process_qgraph(qgraph)
        self.job_log.trace(query_payload)

        backend_record = await backend_driver.run_query(
            query_payload, transpiler=transpiler
        )

        # Convert neo4j record response to TRAPI
        with tracer.start_as_current_span("transform_results"):
            result = transpiler.convert_results(qgraph, backend_record.data["q0"])

        return result
