import asyncio
import math
import time
from abc import ABC, abstractmethod

from opentelemetry import trace
from reasoner_pydantic import (
    QueryGraph,
)

from retriever.config.general import CONFIG
from retriever.types.general import BackendResults, LookupArtifacts
from retriever.types.trapi import (
    AuxGraphDict,
    KnowledgeGraphDict,
    QueryGraphDict,
    ResultDict,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import initialize_kgraph, normalize_kgraph, update_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")


class Tier0Query(ABC):
    """Handler class for running a single Tier 0 query."""

    def __init__(self, qgraph: QueryGraph, job_id: str, _tier: set[int]) -> None:
        """Initialize a Tier 0 Query instance."""
        self.qgraph: QueryGraph = qgraph
        self.job_id: str = job_id
        self.job_log: TRAPILogger = TRAPILogger(job_id)
        self.kgraph: KnowledgeGraphDict = initialize_kgraph(self.qgraph)
        self.aux_graphs: dict[str, AuxGraphDict] = {}

    @tracer.start_as_current_span("tier0_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute a lookup against Tier 0 and return what is found."""
        try:
            start_time = time.time()
            self.job_log.info("Starting lookup against Tier 0...")

            try:
                timeout = None if CONFIG.job.timeout < 0 else CONFIG.job.timeout - 0.5
                async with asyncio.timeout(timeout):
                    backend_results = await self.get_results(
                        QueryGraphDict(**self.qgraph.model_dump())
                    )

            except TimeoutError:
                self.job_log.error("Tier 0 operation timed out.")
                backend_results = BackendResults(
                    results=list[ResultDict](),
                    knowledge_graph=KnowledgeGraphDict(nodes={}, edges={}),
                    auxiliary_graphs=dict[str, AuxGraphDict](),
                )

            with tracer.start_as_current_span("update_kg"):
                normalize_kgraph(
                    backend_results["knowledge_graph"],
                    backend_results["results"],
                    backend_results["auxiliary_graphs"],
                )
                update_kgraph(self.kgraph, backend_results["knowledge_graph"])
            with tracer.start_as_current_span("update_auxgraphs"):
                self.aux_graphs.update(backend_results["auxiliary_graphs"])

            end_time = time.time()
            duration_ms = math.ceil((end_time - start_time) * 1000)
            self.job_log.info(
                f"Tier 0: Retrieved {len(backend_results['results'])} results / {len(self.kgraph['nodes'])} nodes / {len(self.kgraph['edges'])} edges in {duration_ms}ms."
            )

            return LookupArtifacts(
                backend_results["results"],
                self.kgraph,
                self.aux_graphs,
                self.job_log.get_logs(),
            )
        except Exception:
            self.job_log.exception(
                "Unhandled exception occurred while processing Tier 0. See logs for details."
            )
            return LookupArtifacts(
                [], self.kgraph, self.aux_graphs, self.job_log.get_logs(), error=True
            )

    @abstractmethod
    async def get_results(self, qgraph: QueryGraphDict) -> BackendResults:
        """Interface with the Tier 0 backend and retrieve results, converting to ResultDict."""
        raise NotImplementedError("Implemented in subclasses of Tier0Query.")
