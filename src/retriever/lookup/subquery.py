import asyncio
import math
import time

from opentelemetry import trace
from reasoner_pydantic import QueryGraph

from retriever.data_tiers.tier_0.neo4j.driver import Neo4jDriver
from retriever.data_tiers.tier_0.neo4j.transpiler import Neo4jTranspiler
from retriever.lookup.branch import Branch
from retriever.types.trapi import (
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeDict,
    QNodeDict,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import append_aggregator_source, normalize_kgraph

# TODO:
# Use Robokop as a mockup on this level (make a Tier 1 version for query and such)
# Then build the subquery dispatcher around it
# Ignore source selection logic, just give robokop a hash and assume it can answer
# Then hook up as much logic as you can


tracer = trace.get_tracer("lookup.execution.tracer")


@tracer.start_as_current_span("subquery")
async def mock_subquery(
    job_id: str, branch: Branch, qg: QueryGraph
) -> tuple[KnowledgeGraphDict, list[LogEntryDict]]:
    """Placeholder subquery function to mockup its overall behavior.

    Assumptions:
        Returns KEdges in the direction of the QEdge
    """
    try:
        job_log: TRAPILogger = TRAPILogger(job_id)
        start = time.time()

        transpiler = Neo4jTranspiler()
        neo4j_driver = Neo4jDriver()

        # branch comes in execution direction
        # edge it refers to is in query direction

        edge_id = branch.current_edge
        current_edge = QEdgeDict(**qg.edges[edge_id].model_dump())
        input_node = QNodeDict(**qg.nodes[current_edge["subject"]].model_dump())
        input_node["ids"] = [branch.input_curie]
        output_node = QNodeDict(**qg.nodes[current_edge["object"]].model_dump())

        qgraph, query_cypher = transpiler.convert_triple(
            input_node, current_edge, output_node
        )

        job_log.debug(
            f"Subquerying Robokop for {input_node.get('ids', [])} -{current_edge.get('predicates', []) or []}-> {output_node.get('ids', []) or []}..."
        )

        neo4j_record = await neo4j_driver.run_query(query_cypher)

        result = transpiler.convert_results(qgraph, neo4j_record)

        # Add Retriever to the provenance chain
        for edge_id, edge in result["knowledge_graph"]["edges"].items():
            try:
                append_aggregator_source(edge, "infores:retriever")
            except ValueError:
                job_log.warning(f"Edge f{edge_id} has an invalid provenance chain.")

        end = time.time()
        job_log.debug(
            f"Subquery mock got {len(result['knowledge_graph']['edges'])} records in {math.ceil((end - start) * 1000)}ms"
        )

        # Have to do this for each
        normalize_kgraph(
            result["knowledge_graph"], result["results"], result["auxiliary_graphs"]
        )

        return result["knowledge_graph"], job_log.get_logs()

    except asyncio.CancelledError:
        return KnowledgeGraphDict(nodes={}, edges={}), []
