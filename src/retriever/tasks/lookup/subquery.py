import asyncio
import math
import random
import time
import uuid
from collections import deque

from opentelemetry import trace
from reasoner_pydantic import (
    CURIE,
    Edge,
    HashableMapping,
    HashableSet,
    KnowledgeGraph,
    LogEntry,
    Node,
    QueryGraph,
    RetrievalSource,
)
from reasoner_pydantic.shared import EdgeIdentifier, ResourceRoleEnum

from retriever.tasks.lookup.branch import Branch
from retriever.utils.logs import TRAPILogger

CATEGORIES = [
    "biolink:Gene",
    "biolink:Cell",
    "biolink:Disease",
    "biolink:PhenotypicFeature",
    "biolink:Pathway",
    "biolink:NamedThing",
]
NODE_COUNT = 10


MOCKUP_NODES = {
    category: [CURIE(f"curie:test:{uuid.uuid4().hex}") for _ in range(NODE_COUNT)]
    for category in CATEGORIES
}


CHOICES = {
    category: deque([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10000)
    for category in CATEGORIES
}

tracer = trace.get_tracer("lookup.execution.tracer")


@tracer.start_as_current_span("subquery")
async def mock_subquery(
    job_id: str, branch: Branch, qg: QueryGraph
) -> tuple[KnowledgeGraph, list[LogEntry]]:
    """Placeholder subquery function to mockup its overall behavior.

    Assumptions:
        Returns KEdges in the direction of the QEdge
    """
    job_log: TRAPILogger = TRAPILogger(job_id)
    start = time.time()

    # Test some random time jitter
    await asyncio.sleep(random.random() * 0.1)

    edge_id = branch.current_edge
    current_edge = qg.edges[edge_id]
    node = qg.nodes[branch.output_node]

    if node.ids:
        curies = list(node.ids)
    else:
        category = (node.categories or ["biolink:NamedThing"])[0]
        curies = (
            [
                # MOCKUP_NODES[category][CHOICES[category].popleft()]
                random.choice(MOCKUP_NODES[category])
                for _ in range(random.randint(0, 3))
                # for _ in range(2)
            ]
            # if category != "biolink:Disease"
            # else []
        )
    nodes = HashableMapping(
        {
            curie: Node(
                categories=HashableSet(
                    set(qg.nodes[branch.output_node].categories or [])
                ),
                attributes=HashableSet(),
            )
            for curie in curies
        }
    )
    edges = [
        Edge(
            subject=(branch.input_curie if not branch.reversed else curie),
            predicate=(current_edge.predicates or ["biolink:related_to"])[0],
            object=(curie if not branch.reversed else branch.input_curie),
            sources=HashableSet(
                {
                    RetrievalSource(
                        resource_id=CURIE(f"SOURCE:{uuid.uuid4().hex}"),
                        resource_role=ResourceRoleEnum.primary_knowledge_source,
                    )
                }
            ),
            attributes=HashableSet(),
        )
        for curie in curies
    ]

    end = time.time()
    job_log.info(
        f"Subquery mock got {len(edges)} records in {math.ceil((end - start) * 1000)}ms"
    )

    return KnowledgeGraph(
        nodes=nodes,
        edges=HashableMapping(
            {EdgeIdentifier(str(hash(edge))): edge for edge in edges}
        ),
    ), job_log.get_logs()
