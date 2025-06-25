from typing import cast

from opentelemetry import trace
from reasoner_pydantic import (
    CURIE,
    Attribute,
    AuxiliaryGraphs,
    HashableMapping,
    KnowledgeGraph,
    Node,
    QueryGraph,
    Results,
)
from reasoner_pydantic.shared import EdgeIdentifier

from retriever.utils.logs import TRAPILogger

tracer = trace.get_tracer("lookup.execution.tracer")


def initialize_kgraph(qgraph: QueryGraph) -> KnowledgeGraph:
    """Initialize a knowledge graph, using nodes from the query graph."""
    kgraph = KnowledgeGraph()
    for qnode in qgraph.nodes.values():
        if qnode.ids is None:
            continue
        for curie in qnode.ids:
            kgraph.nodes[curie] = Node.model_validate(
                {
                    "categories": qnode.categories or [],
                    "attributes": [],
                }
            )
    return kgraph


@tracer.start_as_current_span("prune_kg")
def prune_kg(
    results: Results,
    kgraph: KnowledgeGraph,
    aux_graphs: AuxiliaryGraphs,
    job_log: TRAPILogger,
) -> None:
    """Use finished results to prune the knowledge graph to only bound knowledge."""
    bound_edges = set[str]()
    bound_nodes = set[str]()
    for result in results:
        for node_binding_set in result.node_bindings.values():
            bound_nodes.update([str(binding.id) for binding in node_binding_set])
        # Only ever one analysis so we can use next(iter())
        for edge_binding_set in next(iter(result.analyses)).edge_bindings.values():
            bound_edges.update([str(binding.id) for binding in edge_binding_set])

    edges_to_check = list(bound_edges)
    while len(edges_to_check) > 0:
        edge = kgraph.edges[EdgeIdentifier(edges_to_check.pop())]
        bound_edges.add(str(hash(edge)))
        bound_nodes.add(str(edge.subject))
        bound_nodes.add(str(edge.object))

        edge_aux_graphs = next(
            (
                attr
                for attr in (edge.attributes or set[Attribute]())
                if attr.attribute_type_id == CURIE("biolink:support_graphs")
            ),
            None,
        )
        if edge_aux_graphs is None:
            continue
        for aux_graph_id in cast(list[str], edge_aux_graphs.value):
            edges_to_check.extend(str(edge) for edge in aux_graphs[aux_graph_id].edges)

    prior_edge_count = len(kgraph.edges)
    prior_node_count = len(kgraph.nodes)

    kgraph.edges = HashableMapping(
        {
            EdgeIdentifier(edge_id): kgraph.edges[EdgeIdentifier(edge_id)]
            for edge_id in bound_edges
        }
    )
    kgraph.nodes = HashableMapping(
        {CURIE(curie): kgraph.nodes[CURIE(curie)] for curie in bound_nodes}
    )

    pruned_edges = prior_edge_count - len(kgraph.edges)
    pruned_nodes = prior_node_count - len(kgraph.nodes)

    job_log.debug(
        f"KG Pruning: {len(kgraph.edges)} (-{pruned_edges}) edges and {len(kgraph.nodes)} (-{pruned_nodes}) nodes remain."
    )
