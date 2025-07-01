import asyncio
import math
import random
import time

from opentelemetry import trace
from reasoner_pydantic import (
    AuxiliaryGraphs,
    KnowledgeGraph,
    QueryGraph,
)

from retriever.types.general import LookupArtifacts
from retriever.types.trapi import ResultDict
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import initialize_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")


class Tier0Query:
    """Handler class for running a single Tier 0 query."""

    def __init__(self, qgraph: QueryGraph, job_id: str, _tier: set[int]) -> None:
        """Initialize a Tier 0 Query instance."""
        self.qgraph: QueryGraph = qgraph
        self.job_id: str = job_id
        self.job_log: TRAPILogger = TRAPILogger(job_id)
        self.kgraph: KnowledgeGraph = initialize_kgraph(self.qgraph)
        self.aux_graphs: AuxiliaryGraphs = AuxiliaryGraphs()

    @tracer.start_as_current_span("tier0_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute a lookup against Tier 0 and return what is found."""
        try:
            start_time = time.time()
            self.job_log.info("Starting lookup against Tier 0...")

            await asyncio.sleep(random.random())
            results = await self.get_results()

            end_time = time.time()
            duration_ms = math.ceil((end_time - start_time) * 1000)
            self.job_log.info(
                f"Tier 0: Retrieved {len(results)} results / {len(self.kgraph.nodes)} nodes / {len(self.kgraph.edges)} edges in {duration_ms}ms."
            )

            return LookupArtifacts(
                results, self.kgraph, self.aux_graphs, self.job_log.get_logs()
            )
        except Exception:
            self.job_log.exception(
                "Unhandled exception occurred while processing Tier 0. See logs for details."
            )
            return LookupArtifacts(
                [], self.kgraph, self.aux_graphs, self.job_log.get_logs(), error=True
            )

    async def get_results(self) -> list[ResultDict]:
        """Interface with the Tier 0 backend and retrieve results, converting to ResultDict."""
        raise NotImplementedError("Implemented in subclasses of Tier0Query.")
