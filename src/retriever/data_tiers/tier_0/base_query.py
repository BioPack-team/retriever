import asyncio
import math
import time
from abc import ABC, abstractmethod

from opentelemetry import trace
from translator_tom.v1_6 import (
    Infores,
    QueryGraph,
)
from translator_tom.v1_6.model_dicts import (
    EdgeDictUtil,
    KnowledgeGraphDictUtil,
)

from retriever.types.general import BackendResult, LookupArtifacts, QueryInfo
from retriever.types.trapi import Parameters
from retriever.utils.logs import TRAPILogger

tracer = trace.get_tracer("lookup.execution.tracer")


class Tier0Query(ABC):
    """Handler class for running a single Tier 0 query."""

    def __init__(self, qgraph: QueryGraph, query_info: QueryInfo) -> None:
        """Initialize a Tier 0 Query instance."""
        self.ctx: QueryInfo = query_info
        self.qgraph: QueryGraph = qgraph
        self.job_log: TRAPILogger = TRAPILogger(self.ctx.job_id)

    @tracer.start_as_current_span("tier0_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute a lookup against Tier 0 and return what is found."""
        try:
            start_time = time.time()

            timeout = None if self.ctx.timeout < 0 else self.ctx.timeout
            self.job_log.debug(
                f"Tier 0 timeout is {'disabled' if timeout is None else f'{timeout}s'}."
            )
            async with asyncio.timeout(timeout):
                backend_results = await self.get_results(self.qgraph)

            results, kgraph, aux_graphs = (
                backend_results.results,
                backend_results.knowledge_graph,
                backend_results.auxiliary_graphs,
            )

            if self.ctx.body is None:
                raise ValueError("Expected query body, got None")

            if logs := backend_results.logs:
                self.job_log.debug("<Begin backend-produced logs>")
                for entry in logs:
                    self.job_log.replay_entry(entry)
                self.job_log.debug("<End of backend logs>")

            parameters = (self.ctx.body).parameters or Parameters.model_construct()

            end_time = time.time()
            duration_ms = math.ceil((end_time - start_time) * 1000)
            self.job_log.info(
                f"Tier 0: Retrieved {len(results)} results / {len(kgraph['nodes'])} nodes / {len(kgraph['edges'])} edges in {duration_ms}ms."
            )

            if not parameters.dehydrated:
                # Add Retriever to the provenance chain
                for edge_id, edge in kgraph["edges"].items():
                    try:
                        EdgeDictUtil.append_aggregator(
                            edge, Infores("infores:retriever")
                        )
                    except ValueError:
                        self.job_log.warning(
                            f"Edge f{edge_id} has an invalid provenance chain."
                        )

            return LookupArtifacts(results, kgraph, aux_graphs, self.job_log.get_logs())
        except Exception as e:
            timed_out = isinstance(e, TimeoutError)
            if timed_out:
                self.job_log.error("Tier 0 operation timed out.")
            elif not isinstance(e, asyncio.exceptions.CancelledError):
                self.job_log.exception(
                    "Unhandled exception occurred while processing Tier 0 query. See logs for details."
                )
            return LookupArtifacts(
                [],
                KnowledgeGraphDictUtil.new(),
                {},
                self.job_log.get_logs(),
                status="TimedOut" if timed_out else "Failed",
            )

    @abstractmethod
    async def get_results(self, qgraph: QueryGraph) -> BackendResult:
        """Interface with the Tier 0 backend and retrieve results, converting to Result.

        Note that this method is responsible for calling the appropriate transpiler,
        running the query against the appropriate driver, tranforming the response,
        and ensuring that all edges have the correct Tier 0 provenance.
        """
        raise NotImplementedError("Implemented in subclasses of Tier0Query.")
