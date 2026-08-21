from typing import Literal

from translator_tom.v1_6 import CURIE, AuxGraphID, Biolink, EdgeID, tomhash
from translator_tom.v1_6.model_dicts import (
    AttributeDict,
    AuxiliaryGraphDict,
    EdgeDict,
    EdgeDictUtil,
    KnowledgeGraphDict,
    QualifierDictUtil,
    ResultDict,
    ResultDictUtil,
    RetrievalSourceDict,
)

from retriever.config.general import CONFIG
from retriever.utils.logs import TRAPILogger

SourcelessEdgeKey = tuple[CURIE, Biolink.Predicate, str, CURIE]
SubclassEdgesByCURIE = dict[tuple[CURIE, CURIE], tuple[EdgeID, EdgeDict]]
AuxGraphEdgesByConstruct = dict[SourcelessEdgeKey, tuple[AuxGraphID, set[EdgeID]]]
ConstructEdgesMapping = dict[SourcelessEdgeKey, tuple[EdgeID, EdgeDict]]


def _primary_knowledge_source(edge: EdgeDict) -> RetrievalSourceDict | None:
    """Get the edge's primary knowledge source, or None if it has none.

    Returns None rather than raising (as `EdgeDictUtil.primary_knowledge_source`
    does) so subclass enrichment can skip PKS-less edges gracefully.
    """
    for source in edge["sources"]:
        if source["resource_role"] == "primary_knowledge_source":
            return source
    return None


def create_subclass_edge(parent: CURIE, descendant: CURIE) -> tuple[EdgeID, EdgeDict]:
    """Create a subclass edge given the parent and its descendant."""
    edge = EdgeDict(
        predicate="biolink:subclass_of",
        subject=descendant,
        object=parent,
        sources=[
            RetrievalSourceDict(
                resource_id="infores:ubergraph",
                resource_role="primary_knowledge_source",
            ),
            RetrievalSourceDict(
                resource_id=CONFIG.tier1.backend_infores,
                resource_role="aggregator_knowledge_source",
                upstream_resource_ids=["infores:ubergraph"],
            ),
            RetrievalSourceDict(
                resource_id="infores:retriever",
                resource_role="aggregator_knowledge_source",
                upstream_resource_ids=[CONFIG.tier1.backend_infores],
            ),
        ],
        attributes=[
            AttributeDict(
                attribute_type_id="biolink:knowledge_level",
                value="knowledge_assertion",
            ),
            AttributeDict(
                attribute_type_id="biolink:agent_type",
                value="manual_agent",
            ),
        ],
    )

    edge_hash = EdgeDictUtil.hash(edge)

    return edge_hash, edge


def build_intermediate_support_graph(
    subclass_backmap: dict[CURIE, CURIE],
    edge_id: EdgeID,
    edge: EdgeDict,
    subclass_edges: dict[tuple[CURIE, CURIE], tuple[EdgeID, EdgeDict]],
) -> tuple[SourcelessEdgeKey, set[EdgeID] | None]:
    """Create a key for the pattern of edge to be replaced, and a support graph for it."""
    sbj_subclass = edge["subject"] in subclass_backmap and edge["subject"]
    obj_subclass = edge["object"] in subclass_backmap and edge["object"]

    edge_key = (
        subclass_backmap[sbj_subclass] if sbj_subclass else edge["subject"],
        edge["predicate"],
        tomhash(
            frozenset(
                QualifierDictUtil.hash(qualifier)
                for qualifier in EdgeDictUtil.qualifiers_list(edge)
            )
        ),
        subclass_backmap[obj_subclass] if obj_subclass else edge["object"],
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


def build_subclass_construct_edge(
    edge_key: SourcelessEdgeKey, edge: EdgeDict
) -> EdgeDict:
    """Build a Retriever-constructed edge which asserts the subclass-driven knowledge."""
    construct_edge = EdgeDict(
        subject=edge_key[0],
        object=edge_key[3],
        predicate=edge["predicate"],
        qualifiers=EdgeDictUtil.qualifiers_list(edge),
        # BUG: this breaks 2.0-clarified attribute constraint binding rules
        # Would have to make a new construct for each edge, rather than aggregate
        attributes=[
            AttributeDict(
                attribute_type_id="biolink:support_graphs",
                value=[f"support_{'_'.join(edge_key)}_via_subclass"],
            ),
            AttributeDict(
                attribute_type_id="biolink:knowledge_level",
                value="logical_entailment",
            ),
            AttributeDict(
                attribute_type_id="biolink:agent_type",
                value="automated_agent",
            ),
        ],
        sources=[
            RetrievalSourceDict(
                resource_id="infores:obie",
                resource_role="primary_knowledge_source",
                upstream_resource_ids=["infores:ubergraph"],
            ),
            RetrievalSourceDict(
                resource_id="infores:ubergraph",
                resource_role="supporting_data_source",
            ),
        ],
    )

    if edge_source := _primary_knowledge_source(edge):
        construct_edge["sources"].append(
            RetrievalSourceDict(
                resource_id=edge_source["resource_id"],
                resource_role="supporting_data_source",
            )
        )
        # Access by index because it's known
        (construct_edge["sources"][0].get("upstream_resource_ids", []) or []).append(
            edge_source["resource_id"]
        )

    return construct_edge


def insert_constructs(
    subclass_backmap: dict[CURIE, CURIE],
    results: list[ResultDict],
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
    edges_to_fix: dict[EdgeID, SourcelessEdgeKey],
    construct_edges: ConstructEdgesMapping,
) -> None:
    """Replace uses of subclassed knowledge edges with their associated constructs.

    This way all instances refer to the support graph containing the subclass edge
    and knowledge.
    """
    # Replace edges with constructs in aux graphs
    for aux_graph in aux_graphs.values():
        aux_graph["edges"] = [
            edge_id
            if edge_id not in edges_to_fix
            else construct_edges[edges_to_fix[edge_id]][0]
            for edge_id in aux_graph["edges"]
        ]

    # Replace edges and nodes in results
    for result in results:
        for node_bindings in result["node_bindings"].values():
            for binding in node_bindings:
                if binding["id"] in subclass_backmap:
                    binding["id"] = subclass_backmap[binding["id"]]

        for analysis in result["analyses"]:
            if "edge_bindings" not in analysis:
                continue
            for edge_bindings in analysis["edge_bindings"].values():
                for binding in edge_bindings:
                    if binding["id"] in edges_to_fix:
                        binding["id"] = construct_edges[edges_to_fix[binding["id"]]][0]

    # Merge the results now that their bindings point at the constructs
    ResultDictUtil.merge_results(results)


def add_new_knowledge(
    kg: KnowledgeGraphDict,
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
    subclass_edges: SubclassEdgesByCURIE,
    support_graphs: AuxGraphEdgesByConstruct,
    construct_edges: ConstructEdgesMapping,
) -> None:
    """Update the kg/aux with the new format information."""
    # Merge in new edges and aux graphs
    kg["edges"].update(dict(subclass_edges.values()))
    kg["edges"].update(dict(construct_edges.values()))

    aux_graphs.update(
        {
            support_graph_id: AuxiliaryGraphDict(
                edges=list(support_edges), attributes=[]
            )
            for support_graph_id, support_edges in support_graphs.values()
        }
    )


def solve_subclass_edges(
    subclass_backmap: dict[CURIE, CURIE],
    kg: KnowledgeGraphDict,
    results: list[ResultDict],
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
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

    for edge_id, edge in kg["edges"].items():
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
            if edge_source := _primary_knowledge_source(edge):
                construct_edge = construct_edges[edge_key][1]
                if edge_source["resource_id"] not in (
                    construct_edge["sources"][0].get("upstream_resource_ids", []) or []
                ):
                    construct_edge["sources"].append(
                        RetrievalSourceDict(
                            resource_id=edge_source["resource_id"],
                            resource_role="supporting_data_source",
                        )
                    )
                    construct_edge["sources"][0]["upstream_resource_ids"] = list(
                        {
                            *(
                                construct_edge["sources"][0].get(
                                    "upstream_resource_ids", []
                                )
                                or []
                            ),
                            edge_source["resource_id"],
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
