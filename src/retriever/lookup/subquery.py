import asyncio
import math
import time

from opentelemetry import trace
from reasoner_pydantic import QueryGraph

from retriever.data_tiers.tier_1.elasticsearch.driver import ElasticSearchDriver
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.lookup.branch import Branch
from retriever.types.trapi import (
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeDict,
    QNodeDict,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import append_aggregator_source, normalize_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")


@tracer.start_as_current_span("subquery")
async def mock_subquery(
    job_id: str, branch: Branch, qg: QueryGraph
) -> tuple[KnowledgeGraphDict, list[LogEntryDict]]:
    """Placeholder subquery function to mockup retrieving from Tier 1.

    Assumptions:
        Returns KEdges in the direction of the QEdge
    """
    try:
        job_log: TRAPILogger = TRAPILogger(job_id)
        start = time.time()

        transpiler = ElasticsearchTranspiler()
        query_driver = ElasticSearchDriver()

        # branch comes in execution direction
        # edge it refers to is in query direction

        edge_id = branch.current_edge
        current_edge = QEdgeDict(**qg.edges[edge_id].model_dump())
        subject_node = QNodeDict(**qg.nodes[current_edge["subject"]].model_dump())
        object_node = QNodeDict(**qg.nodes[current_edge["object"]].model_dump())
        if not branch.reversed:
            subject_node["ids"] = [branch.input_curie]
        else:
            object_node["ids"] = [branch.input_curie]

        qgraph, query_cypher = transpiler.convert_triple(
            subject_node, current_edge, object_node
        )

        job_log.debug(
            f"Subquerying Tier 1 for {subject_node.get('ids', [])} -{current_edge.get('predicates', []) or []}-> {object_node.get('ids', []) or []}..."
        )

        response_record = await query_driver.run_query(query_cypher)
        print(response_record)

        result = transpiler.convert_results(qgraph, response_record)

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
