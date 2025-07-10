import asyncio
import itertools
import math
import random
import time
import uuid

from opentelemetry import trace
from reasoner_pydantic import QueryGraph
from reasoner_pydantic.shared import ResourceRoleEnum

from retriever.tasks.lookup.branch import Branch
from retriever.types.trapi import (
    CURIE,
    EdgeDict,
    KnowledgeGraphDict,
    LogEntryDict,
    NodeDict,
    RetrievalSourceDict,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import hash_edge, hash_hex

CATEGORIES = [
    "biolink:Gene",
    "biolink:Cell",
    "biolink:Disease",
    "biolink:PhenotypicFeature",
    "biolink:Pathway",
    "biolink:NamedThing",
    "biolink:ClinicalFinding",
    "biolink:BehavioralFeature",
]
NODE_COUNT = 10


MOCKUP_NODES = {
    category: [CURIE(f"curie:test:{uuid.uuid4().hex}") for _ in range(NODE_COUNT)]
    for category in CATEGORIES
}


CHOICES = {
    category: itertools.cycle([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    for category in CATEGORIES
}

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

        # Test some random time jitter
        await asyncio.sleep(random.random() * 0.2)

        edge_id = branch.current_edge
        current_edge = qg.edges[edge_id]
        node = qg.nodes[branch.output_node]

        if node.ids:
            curies = list(node.ids)
        else:
            category = (
                str(next(iter(list(node.categories))))
                if node.categories
                else "biolink:NamedThing"
            )
            curies = (
                [
                    CURIE(MOCKUP_NODES[category][next(CHOICES[category])])
                    # random.choice(MOCKUP_NODES[category])
                    # for _ in range(random.randint(0, 10))
                    for _ in range(5)
                ]
                # if category != "biolink:Disease"
                # else []
            )
        output_node = qg.nodes[branch.output_node]
        nodes = {
            curie: NodeDict(
                categories=[str(cat) for cat in output_node.categories]
                if output_node.categories
                else [],
                attributes=[],
            )
            for curie in curies
        }

        edges = [
            EdgeDict(
                subject=(branch.input_curie if not branch.reversed else curie),
                predicate=(
                    str(next(iter(list(current_edge.predicates))))
                    if current_edge.predicates is not None
                    else "biolink:related_to"
                ),
                object=(curie if not branch.reversed else branch.input_curie),
                sources=[
                    RetrievalSourceDict(
                        resource_id=f"SOURCE:{uuid.uuid4().hex}",
                        resource_role=ResourceRoleEnum.primary_knowledge_source,
                    )
                ],
                attributes=[],
            )
            for curie in curies
        ]

        end = time.time()
        job_log.info(
            f"Subquery mock got {len(edges)} records in {math.ceil((end - start) * 1000)}ms"
        )

        return KnowledgeGraphDict(
            nodes=nodes, edges={hash_hex(hash_edge(edge)): edge for edge in edges}
        ), job_log.get_logs()
    except asyncio.CancelledError:
        return (KnowledgeGraphDict(nodes={}, edges={}), [])
