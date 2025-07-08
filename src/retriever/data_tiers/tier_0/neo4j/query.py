from typing import override

from opentelemetry import trace
from reasoner_transpiler.cypher import (
    get_query,  # pyright:ignore[reportUnknownVariableType]
)

from retriever.config.general import CONFIG
from retriever.data_tiers.tier_0.base import Tier0Query
from retriever.data_tiers.tier_0.neo4j.driver import GraphInterface
from retriever.types.trapi import QueryGraphDict, ResultDict
from retriever.utils.trapi import normalize_kgraph, update_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")


class Neo4jQuery(Tier0Query):
    """Adapter to querying Plater Neo4j as a Tier 0 backend."""

    @override
    async def get_results(self) -> list[ResultDict]:
        graph_interface = GraphInterface(
            host=CONFIG.tier0.neo4j.host,
            port=CONFIG.tier0.neo4j.bolt_port,
            auth=(CONFIG.tier0.neo4j.username, CONFIG.tier0.neo4j.password),
        )

        qgraph = QueryGraphDict(**self.qgraph.model_dump())

        query_cypher = get_query(qgraph)
        self.job_log.trace(query_cypher)

        with tracer.start_as_current_span("neo4j_query"):
            result = await graph_interface.run_cypher(query_cypher, qgraph)

        with tracer.start_as_current_span("update_kg"):
            normalize_kgraph(
                result["knowledge_graph"], result["results"], result["auxiliary_graphs"]
            )
            update_kgraph(self.kgraph, result["knowledge_graph"])
        with tracer.start_as_current_span("update_auxgraphs"):
            self.aux_graphs.update(result["auxiliary_graphs"])

        return result["results"]
