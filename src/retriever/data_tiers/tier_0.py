import asyncio
import math
import random
import time

import kuzu
from opentelemetry import trace
from reasoner_pydantic import (
    AuxiliaryGraphs,
    QueryGraph,
    Results,
)
from reasoner_transpiler.cypher import (
    get_query,  # pyright:ignore[reportUnknownVariableType] Not strongly typed for now
)

from retriever.types.general import LookupArtifacts
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import initialize_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")

kuzu_db = kuzu.Database("dbs/kuzu_rtx_kg2", read_only=True)


class Tier0Query:
    """Handler class for running a single Tier 0 query."""

    def __init__(self, qgraph: QueryGraph, job_id: str, _tier: set[int]) -> None:
        """Initialize a Tier 0 Query instance."""
        self.qgraph: QueryGraph = qgraph
        self.job_id: str = job_id
        self.job_log: TRAPILogger = TRAPILogger(job_id)

    @tracer.start_as_current_span("tier0_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute a lookup against Tier 0 and return what is found."""
        start_time = time.time()
        self.job_log.info("Starting lookup against Tier 0...")
        results = Results()
        kgraph = initialize_kgraph(self.qgraph)
        aux_graphs = AuxiliaryGraphs()

        query_cypher = get_query(self.qgraph.model_dump())
        self.job_log.debug(query_cypher)

        # TODO: so, this either happens in a thread or you update kuzu to get async...which breaks db compatibility
        # Oh and transpiler is highly specific to the neo4j schema and plugins
        with kuzu.Connection(kuzu_db) as conn:
            result = conn.execute(query_cypher)
            for item in result:
                print(item)

        await asyncio.sleep(random.random() * 0.5)

        end_time = time.time()
        duration_ms = math.ceil((end_time - start_time) * 1000)
        self.job_log.info(
            f"Tier 0: Retrieved {len(results)} results / {len(kgraph.nodes)} nodes / {len(kgraph.edges)} edges in {duration_ms:}ms."
        )
        return LookupArtifacts(results, kgraph, aux_graphs, self.job_log.get_logs())
