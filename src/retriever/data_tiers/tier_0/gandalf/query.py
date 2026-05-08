import json
from typing import override

from opentelemetry import trace

from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.gandalf.driver import GandalfDriver
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    KnowledgeGraphDict,
    MessageDict,
    QueryDict,
    QueryGraphDict,
)

tracer = trace.get_tracer("lookup.execution.tracer")


class GandalfQuery(Tier0Query):
    """Adapter to querying Dgraph as a Tier 0 backend."""

    @override
    async def get_results(self, qgraph: QueryGraphDict) -> BackendResult:
        backend_driver = GandalfDriver()
        query_payload = QueryDict(
            message=MessageDict(
                query_graph=qgraph,
            )
        )
        self.job_log.trace(json.dumps(query_payload))

        result = await backend_driver.run_query(query_payload)

        return BackendResult(
            results=result["message"].get("results") or [],
            knowledge_graph=result["message"].get("knowledge_graph")
            or KnowledgeGraphDict(nodes={}, edges={}),
            auxiliary_graphs=result["message"].get("auxiliary_graphs") or {},
        )
