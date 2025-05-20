import asyncio
import random
from typing import (
    cast,
)

from reasoner_pydantic import (
    CURIE,
    Edge,
    HashableMapping,
    HashableSet,
    KnowledgeGraph,
    Node,
    QEdge,
    QueryGraph,
    RetrievalSource,
)
from reasoner_pydantic.shared import EdgeIdentifier, ResourceRoleEnum

from retriever.tasks.lookup.branch import Branch
from retriever.type_defs import EdgeIDMap

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
    category: [CURIE(f"curie:test{random.random()}") for _ in range(NODE_COUNT)]
    for category in CATEGORIES
}


async def mock_subquery(
    branch: Branch, qg: QueryGraph, em: EdgeIDMap
) -> KnowledgeGraph:
    """Placeholder subquery function to mockup its overall behavior.

    Assumptions:
        Returns KEdges in the direction of the QEdge
    """
    # Test some random time jitter
    await asyncio.sleep(random.random() * 0.3)

    edge_id = branch.current_edge
    current_edge = cast(QEdge, em[edge_id])
    node = qg.nodes[branch.output_node]

    if node.ids:
        curies = list(node.ids)
    else:
        curies = [
            random.choice(MOCKUP_NODES[(node.categories or "biolink:NamedThing")[0]])
            # for _ in range(random.randint(1, 3))
            for _ in range(2)
        ]
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
                        resource_id=CURIE(f"SOURCE:{random.random()}"),
                        resource_role=ResourceRoleEnum.primary_knowledge_source,
                    )
                }
            ),
            attributes=HashableSet(),
        )
        for curie in curies
    ]

    return KnowledgeGraph(
        nodes=nodes,
        edges=HashableMapping(
            {EdgeIdentifier(str(hash(edge))): edge for edge in edges}
        ),
    )
