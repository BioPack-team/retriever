import asyncio
import math
import time
from abc import ABC, abstractmethod

from opentelemetry import trace
from reasoner_pydantic import (
    QueryGraph,
)

from retriever.types.general import BackendResult, LookupArtifacts, QueryInfo
from retriever.types.trapi import (
    AuxGraphID,
    AuxiliaryGraphDict,
    Infores,
    KnowledgeGraphDict,
    QueryGraphDict,
    ResultDict,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import (
    append_aggregator_source,
    initialize_kgraph,
    normalize_kgraph,
    update_kgraph,
)

tracer = trace.get_tracer("lookup.execution.tracer")


class Tier0Query(ABC):
    """Handler class for running a single Tier 0 query."""

    def __init__(self, qgraph: QueryGraph, query_info: QueryInfo) -> None:
        """Initialize a Tier 0 Query instance."""
        self.ctx: QueryInfo = query_info
        self.qgraph: QueryGraph = qgraph
        self.job_log: TRAPILogger = TRAPILogger(self.ctx.job_id)
        self.kgraph: KnowledgeGraphDict = initialize_kgraph(self.qgraph)
        self.aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict] = {}

    @tracer.start_as_current_span("tier0_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute a lookup against Tier 0 and return what is found."""
        try:
            start_time = time.time()
            self.job_log.info("Starting lookup against Tier 0...")

            try:
                timeout = None if self.ctx.timeout[0] < 0 else self.ctx.timeout[0]
                self.job_log.debug(
                    f"Tier 0 timeout is {'disabled' if timeout is None else f'{timeout}s'}."
                )
                async with asyncio.timeout(timeout):
                    qgraph_dict = QueryGraphDict(**self.qgraph.model_dump(by_alias=True))

                    # WORKAROUND: Fix the 'not' field using raw JSON if available
                    raw_json = None
                    if (self.ctx.body and
                        hasattr(self.ctx.body, 'model_extra') and
                        self.ctx.body.model_extra and
                        '_raw_json' in self.ctx.body.model_extra):
                        raw_json = self.ctx.body.model_extra['_raw_json']
                        self.job_log.debug(f"[base_query.py] Found raw JSON in model_extra")

                    # Fix the bug: use raw JSON to get correct 'not' values
                    if qgraph_dict.get("edges") and raw_json:
                        raw_qgraph = raw_json.get('message', {}).get('query_graph', {})
                        raw_edges = raw_qgraph.get('edges', {})

                        for edge_id, edge in qgraph_dict["edges"].items():
                            if edge.get("attribute_constraints") and edge_id in raw_edges:
                                raw_edge = raw_edges[edge_id]
                                raw_constraints = raw_edge.get('attribute_constraints', [])

                                for i, constraint in enumerate(edge["attribute_constraints"]):
                                    if i < len(raw_constraints):
                                        # Get the correct 'not' value from raw JSON
                                        constraint["not"] = raw_constraints[i].get("not", False)
                                        self.job_log.debug(f"[base_query.py] Fixed constraint {i} 'not' to {constraint['not']} from raw JSON")

                    backend_results = await self.get_results(qgraph_dict)

            except TimeoutError:
                self.job_log.error("Tier 0 operation timed out.")
                backend_results = BackendResult(
                    results=list[ResultDict](),
                    knowledge_graph=KnowledgeGraphDict(nodes={}, edges={}),
                    auxiliary_graphs=dict[AuxGraphID, AuxiliaryGraphDict](),
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

            # Add Retriever to the provenance chain
            for edge_id, edge in backend_results["knowledge_graph"]["edges"].items():
                try:
                    append_aggregator_source(edge, Infores("infores:retriever"))
                except ValueError:
                    self.job_log.warning(
                        f"Edge f{edge_id} has an invalid provenance chain."
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
    async def get_results(self, qgraph: QueryGraphDict) -> BackendResult:
        """Interface with the Tier 0 backend and retrieve results, converting to ResultDict.

        Note that this method is responsible for calling the appropriate transpiler,
        running the query against the appropriate driver, tranforming the response,
        and ensuring that all edges have the correct Tier 0 provenance.
        """
        raise NotImplementedError("Implemented in subclasses of Tier0Query.")
