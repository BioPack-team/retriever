from typing import Literal

from translator_tom import (
    CURIE,
    Attribute,
    AuxGraphID,
    AuxiliaryGraph,
    Biolink,
    Edge,
    EdgeID,
    KnowledgeGraph,
    PathfinderAnalysis,
    Result,
    RetrievalSource,
    tomhash,
)

from retriever.config.general import CONFIG
from retriever.utils.logs import TRAPILogger

SourcelessEdgeKey = tuple[CURIE, Biolink.Predicate, str, CURIE]
SubclassEdgesByCURIE = dict[tuple[CURIE, CURIE], tuple[EdgeID, Edge]]
AuxGraphEdgesByConstruct = dict[SourcelessEdgeKey, tuple[AuxGraphID, set[EdgeID]]]
ConstructEdgesMapping = dict[SourcelessEdgeKey, tuple[EdgeID, Edge]]


def create_subclass_edge(parent: CURIE, descendant: CURIE) -> tuple[EdgeID, Edge]:
    """Create a subclass edge given the parent and its descendant."""
    edge = Edge.model_construct(
        predicate="biolink:subclass_of",
        subject=descendant,
        object=parent,
        sources=[
            RetrievalSource.model_construct(
                resource_id="infores:ubergraph",
                resource_role="primary_knowledge_source",
            ),
            RetrievalSource.model_construct(
                resource_id=CONFIG.tier1.backend_infores,
                resource_role="aggregator_knowledge_source",
                upstream_resource_ids=["infores:ubergraph"],
            ),
            RetrievalSource.model_construct(
                resource_id="infores:retriever",
                resource_role="aggregator_knowledge_source",
                upstream_resource_ids=[CONFIG.tier1.backend_infores],
            ),
        ],
        attributes=[
            Attribute.model_construct(
                attribute_type_id="biolink:knowledge_level",
                value="knowledge_assertion",
            ),
            Attribute.model_construct(
                attribute_type_id="biolink:agent_type",
                value="manual_agent",
            ),
        ],
    )

    return edge.hash(), edge


def build_intermediate_support_graph(
    subclass_backmap: dict[CURIE, CURIE],
    edge_id: EdgeID,
    edge: Edge,
    subclass_edges: dict[tuple[CURIE, CURIE], tuple[EdgeID, Edge]],
) -> tuple[SourcelessEdgeKey, set[EdgeID] | None]:
    """Create a key for the pattern of edge to be replaced, and a support graph for it."""
    sbj_subclass = edge.subject in subclass_backmap and edge.subject
    obj_subclass = edge.object in subclass_backmap and edge.object

    edge_key = (
        subclass_backmap[sbj_subclass] if sbj_subclass else edge.subject,
        edge.predicate,
        tomhash(edge.qualifiers),
        subclass_backmap[obj_subclass] if obj_subclass else edge.object,
    )

    if not (sbj_subclass or obj_subclass):
        return edge_key, None

    support_graph = set[EdgeID]((edge_id,))
    for subclass in list[CURIE | Literal[False]]((sbj_subclass, obj_subclass)):
        if not subclass:
            continue

        if (subclass_backmap[subclass], subclass) not in subclass_edges:
            subclass_edge_hash, subclass_edge = create_subclass_edge(
                subclass_backmap[subclass], subclass
            )
            subclass_edges[subclass_backmap[subclass], subclass] = (
                subclass_edge_hash,
                subclass_edge,
            )
        else:
            subclass_edge_hash = subclass_edges[(subclass_backmap[subclass], subclass)][
                0
            ]
        support_graph.add(subclass_edge_hash)

    return edge_key, support_graph


def build_subclass_construct_edge(edge_key: SourcelessEdgeKey, edge: Edge) -> Edge:
    """Build a Retriever-constructed edge which asserts the subclass-driven knowledge."""
    construct_edge = Edge.model_construct(
        subject=edge_key[0],
        object=edge_key[3],
        predicate=edge.predicate,
        qualifiers=edge.qualifiers,
        # BUG: this breaks 2.0-clarified attribute constraint binding rules
        # Would have to make a new construct for each edge, rather than aggregate
        attributes=[
            Attribute.model_construct(
                attribute_type_id="biolink:support_graphs",
                value=[f"support_{'_'.join(edge_key)}_via_subclass"],
            ),
            Attribute.model_construct(
                attribute_type_id="biolink:knowledge_level",
                value="logical_entailment",
            ),
            Attribute.model_construct(
                attribute_type_id="biolink:agent_type",
                value="automated_agent",
            ),
        ],
        sources=[
            RetrievalSource.model_construct(
                resource_id="infores:retriever",
                resource_role="primary_knowledge_source",
                upstream_resource_ids=["infores:ubergraph"],
            ),
            RetrievalSource.model_construct(
                resource_id="infores:ubergraph",
                resource_role="supporting_data_source",
            ),
        ],
    )

    if edge_source := edge.primary_knowledge_source:
        construct_edge.sources.append(
            RetrievalSource.model_construct(
                resource_id=edge_source.resource_id,
                resource_role="supporting_data_source",
            )
        )
        # Access by index because it's known
        construct_edge.sources[0].upstream_resource_ids = [
            *construct_edge.sources[0].upstream_resource_ids_list,
            edge_source.resource_id,
        ]

    return construct_edge


def insert_constructs(
    subclass_backmap: dict[CURIE, CURIE],
    results: list[Result],
    aux_graphs: dict[AuxGraphID, AuxiliaryGraph],
    edges_to_fix: dict[EdgeID, SourcelessEdgeKey],
    construct_edges: ConstructEdgesMapping,
) -> None:
    """Replace uses of subclassed knowledge edges with their associated constructs.

    This way all instances refer to the support graph containing the subclass edge
    and knowledge.
    """
    # Replace edges with constructs in aux graphs
    for aux_graph in aux_graphs.values():
        aux_graph.edges = [
            edge_id
            if edge_id not in edges_to_fix
            else construct_edges[edges_to_fix[edge_id]][0]
            for edge_id in aux_graph.edges
        ]

    # Replace edges and nodes in results
    merged_results = dict[str, Result]()
    for result in results:
        for node_bindings in result.node_bindings.values():
            for binding in node_bindings:
                if binding.id in subclass_backmap:
                    binding.id = subclass_backmap[binding.id]

        for analysis in result.analyses:
            if isinstance(analysis, PathfinderAnalysis):
                continue
            for edge_bindings in analysis.edge_bindings.values():
                for binding in edge_bindings:
                    if binding.id in edges_to_fix:
                        binding.id = construct_edges[edges_to_fix[binding.id]][0]

        # Merge the result
        result_hash = result.hash()
        if result_hash in merged_results:
            merged_results[result_hash].update(result)
        else:
            merged_results[result_hash] = result
    results.clear()
    results.extend(merged_results.values())


def add_new_knowledge(
    kg: KnowledgeGraph,
    aux_graphs: dict[AuxGraphID, AuxiliaryGraph],
    subclass_edges: SubclassEdgesByCURIE,
    support_graphs: AuxGraphEdgesByConstruct,
    construct_edges: ConstructEdgesMapping,
) -> None:
    """Update the kg/aux with the new format information."""
    # Merge in new edges and aux graphs
    kg.edges.update(dict(subclass_edges.values()))
    kg.edges.update(dict(construct_edges.values()))

    aux_graphs.update(
        {
            support_graph_id: AuxiliaryGraph.model_construct(
                edges=list(support_edges), attributes=[]
            )
            for support_graph_id, support_edges in support_graphs.values()
        }
    )


def solve_subclass_edges(
    subclass_backmap: dict[CURIE, CURIE],
    kg: KnowledgeGraph,
    results: list[Result],
    aux_graphs: dict[AuxGraphID, AuxiliaryGraph],
    job_log: TRAPILogger | None = None,
) -> None:
    """Given the subclass mapping, fix the kg/results/aux to use correct subclass structure.

    WARNING: This implementation is specific to Tier 1/2 use.
    """
    if job_log:
        job_log.debug("Fixing implicit subclass-derived knowledge format...")
    # Keyed to parent, descendant
    subclass_edges = SubclassEdgesByCURIE()

    # Map original edges to non-subclassed sbj/obj
    edges_to_fix = dict[EdgeID, SourcelessEdgeKey]()
    # Map non-subclassed sbj/obj to support graph for merging
    support_graphs = AuxGraphEdgesByConstruct()
    # Map original edges to their construct replacements
    construct_edges = ConstructEdgesMapping()

    for edge_id, edge in kg.edges.items():
        edge_key, support_graph = build_intermediate_support_graph(
            subclass_backmap, edge_id, edge, subclass_edges
        )
        if support_graph is None:  # Edge doesn't rely on subclassing
            continue

        # Update overall support graphs and edge tracking
        support_graph_id = f"support_{'_'.join(edge_key)}_via_subclass"
        edges_to_fix[edge_id] = edge_key
        if edge_key not in support_graphs:
            support_graphs[edge_key] = support_graph_id, set()
        support_graphs[edge_key][1].update(support_graph)

        # Don't build redundant construct edges, just append the supporting source
        if edge_key in construct_edges:
            if edge_source := edge.primary_knowledge_source:
                construct_edge = construct_edges[edge_key][1]
                if edge_source.resource_id not in (
                    construct_edge.sources[0].upstream_resource_ids_list
                ):
                    construct_edge.sources.append(
                        RetrievalSource.model_construct(
                            resource_id=edge_source.resource_id,
                            resource_role="supporting_data_source",
                        )
                    )
                    construct_edge.sources[0].upstream_resource_ids = list(
                        {
                            *(construct_edge.sources[0].upstream_resource_ids_list),
                            edge_source.resource_id,
                        }
                    )
            continue

        construct_edges[edge_key] = (
            f"{'_'.join(edge_key)}_via_subclass",
            build_subclass_construct_edge(edge_key, edge),
        )

    if job_log:
        job_log.debug(
            f"Found and reformated dependents for {len(edges_to_fix)} subclass-derived edges."
        )

    insert_constructs(
        subclass_backmap, results, aux_graphs, edges_to_fix, construct_edges
    )
    add_new_knowledge(kg, aux_graphs, subclass_edges, support_graphs, construct_edges)
