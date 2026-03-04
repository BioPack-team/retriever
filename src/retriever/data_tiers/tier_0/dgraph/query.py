from typing import override

from opentelemetry import trace

from retriever.config.general import CONFIG
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

        # Build one standalone DQL query per expansion type
        main_query, symmetric_queries, subclassing_queries = (
            transpiler.build_split_queries(qgraph)
        )

        # Run all queries (parallel or sequential depending on config)
        main_and_sym_nodes, subclassing_nodes = await backend_driver.run_split_queries(
            main_query,
            symmetric_queries,
            subclassing_queries,
            transpiler=transpiler,
            parallel=CONFIG.tier0.dgraph.parallel_expansion_queries,
        )

        # Convert results: symmetric nodes merge directly; subclassing nodes are
        # flattened (intermediate A'/B' hops collapsed) before processing
        with tracer.start_as_current_span("transform_results"):
            result = transpiler.convert_split_results(
                qgraph, main_and_sym_nodes, subclassing_nodes
            )

        return result
