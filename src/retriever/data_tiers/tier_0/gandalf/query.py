from typing import override

from opentelemetry import trace
from translator_tom import (
    KnowledgeGraph,
    Message,
    Query,
    QueryGraph,
)

from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.gandalf.driver import GandalfDriver
from retriever.types.general import BackendResult

tracer = trace.get_tracer("lookup.execution.tracer")


class GandalfQuery(Tier0Query):
    """Adapter to querying Dgraph as a Tier 0 backend."""

    @override
    async def get_results(self, qgraph: QueryGraph) -> BackendResult:
        backend_driver = GandalfDriver()
        query_payload = Query(
            message=Message(
                query_graph=qgraph,
            )
        )

        result = await backend_driver.run_query(query_payload)

        return BackendResult(
            results=result.message.results_list,
            knowledge_graph=result.message.knowledge_graph or KnowledgeGraph.new(),
            auxiliary_graphs=result.message.auxiliary_graphs_dict,
        )
