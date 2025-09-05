import hashlib
import itertools
from collections.abc import Sequence
from typing import cast

from opentelemetry import trace
from reasoner_pydantic import (
    QueryGraph,
)
from reasoner_pydantic.qgraph import QualifierConstraint
from reasoner_pydantic.utils import make_hashable

from retriever.types.trapi import (
    AttributeDict,
    AuxGraphDict,
    EdgeDict,
    KnowledgeGraphDict,
    MetaAttributeDict,
    NodeBindingDict,
    NodeDict,
    ResultDict,
    RetrievalSourceDict,
)
from retriever.utils.logs import TRAPILogger

tracer = trace.get_tracer("lookup.execution.tracer")


def initialize_kgraph(qgraph: QueryGraph) -> KnowledgeGraphDict:
    """Initialize a knowledge graph, using nodes from the query graph."""
    kgraph = KnowledgeGraphDict(nodes={}, edges={})
    for qnode in qgraph.nodes.values():
        if qnode.ids is None:
            continue
        for curie in qnode.ids:
            kgraph["nodes"][str(curie)] = NodeDict(
                categories=list(qnode.categories or []),
                attributes=[],
            )
    return kgraph


def hash_hex(hashint: int) -> str:
    """Convert a regular hash to the type used in pydantic EdgeIdentifier."""
    return hashlib.blake2b(
        hashint.to_bytes(16, byteorder="big", signed=True), digest_size=6
    ).hexdigest()


def hash_attribute(attr: AttributeDict) -> int:
    """Get a hash of an AttributeDict instance."""
    # Make use of a reasoner_pydantic util for this
    return hash(make_hashable(attr))  # pyright:ignore[reportUnknownArgumentType]


def hash_meta_attribute(attr: MetaAttributeDict) -> int:
    """Get a hash of a MetaAttributeDict instance."""
    return hash(
        tuple(
            (
                key,
                (
                    value
                    if key != "original_attribute_names"
                    else tuple(cast(list[str], value))
                ),
            )
            for key, value in attr.items()
        )
    )


def hash_node_binding(binding: NodeBindingDict) -> int:
    """Get a hash of a NodeBindingDict instance."""
    return hash(
        (
            binding["id"],
            binding.get("query_id"),
            *(hash_attribute(attr) for attr in binding["attributes"]),
        )
    )


def hash_retrieval_source(source: RetrievalSourceDict) -> int:
    """Get a hash of a RetrievalSourceDict instance."""
    return hash((source["resource_id"], source["resource_role"]))


def edge_primary_knowledge_source(edge: EdgeDict) -> RetrievalSourceDict | None:
    """Get the primary source information for a given edge."""
    for source in edge["sources"]:
        if source["resource_role"] == "primary_knowledge_source":
            return source


def append_aggregator_source(
    edge: EdgeDict,
    source: str,
) -> None:
    """Append a aggregator source to an edge's provenance, reference the current most downstream source."""
    upstreams = set(
        itertools.chain(
            *[source.get("upstream_resource_ids", []) for source in edge["sources"]]
        )
    )
    most_downstream_source = next(
        iter(
            source
            for source in edge["sources"]
            if source["resource_id"] not in upstreams
        ),
        None,
    )
    if most_downstream_source is None:
        raise ValueError("Provenance chain is invalid.")
    edge["sources"].append(
        RetrievalSourceDict(
            resource_id=source,
            resource_role="aggregator_knowledge_source",
            upstream_resource_ids=[most_downstream_source["resource_id"]],
        )
    )


def hash_edge(edge: EdgeDict) -> int:
    """Get a hash of an EdgeDict instance."""
    primary_knowledge_source = (
        edge_primary_knowledge_source(edge)
        or RetrievalSourceDict(
            resource_role="primary_knowledge_source", resource_id="err_missing"
        )
    )["resource_id"]
    return hash(
        (edge["subject"], edge["object"], edge["predicate"], primary_knowledge_source)
    )


def hash_result(result: ResultDict) -> int:
    """Get a hash of a ResultDict instance."""
    return hash(
        tuple(
            (qnode_id, *(hash_node_binding(binding) for binding in bindings))
            for qnode_id, bindings in result["node_bindings"].items()
        )
    )


@tracer.start_as_current_span("merge_results")
def merge_results(current: dict[int, ResultDict], new: list[ResultDict]) -> None:
    """Merge ResultDicts in a dict of results by hash."""
    for result in new:
        key = hash_result(result)
        if key not in current:
            current[key] = result
            continue
        # Otherwise, need to merge results
        # We're gonna do a lazy style: no deep analysis merge
        current[key]["analyses"].extend(result["analyses"])


def update_node(node: NodeDict, new: NodeDict) -> None:
    """Update the first node in-place, merging information from the second."""
    new_name = new.get("name")
    if new_name:
        node["name"] = new_name
    if new["categories"]:
        node["categories"] = list(set(node["categories"]) | set(new["categories"]))
    if new["attributes"]:
        old_attributes = {hash_attribute(attr): attr for attr in node["attributes"]}
        new_attributes = {hash_attribute(attr): attr for attr in new["attributes"]}
        node["attributes"] = list({**old_attributes, **new_attributes}.values())


def update_retrieval_source(
    source: RetrievalSourceDict, new: RetrievalSourceDict
) -> None:
    """Update the first source in-place, merging information from the second."""
    if "upstream_resource_ids" in new:
        source["upstream_resource_ids"] = list(
            set(source.get("upstream_resource_ids", []))
            | set(new["upstream_resource_ids"])
        )


def update_edge(edge: EdgeDict, new: EdgeDict) -> None:
    """Update the first edge in-place, merging information from the second."""
    if "attributes" in new:
        old_attributes = {
            hash_attribute(attr): attr for attr in edge.get("attributes", [])
        }
        new_attributes = {
            hash_attribute(attr): attr
            for attr in new.get("attributes", [])
            if attr["attribute_type_id"]  # Don't want multiple KL/AT
            not in ("biolink:knowledge_level", "biolink:agent_type")
        }
        edge["attributes"] = list({**old_attributes, **new_attributes}.values())
    if new["sources"]:
        old_sources = {
            hash_retrieval_source(source): source for source in edge["sources"]
        }
        new_sources = {
            hash_retrieval_source(source): source for source in new["sources"]
        }

        # Roll in upstream_resource_ids from new sources that overlap
        for source_id, source in old_sources.items():
            if new_source := new_sources.get(source_id):
                # Update new source so it overwrites the old source
                update_retrieval_source(new_source, source)
        edge["sources"] = list({**old_sources, **new_sources}.values())


def update_kgraph(kgraph: KnowledgeGraphDict, new: KnowledgeGraphDict) -> None:
    """Update the first kgraph in-place, merging in nodes and edges from the second.

    Requires that both kgraphs have already been normalized.
    """
    for node_id, node in new["nodes"].items():
        if node_id not in kgraph["nodes"]:
            kgraph["nodes"][node_id] = node
            continue
        update_node(kgraph["nodes"][node_id], node)

    for edge_id, edge in new["edges"].items():
        if edge_id not in kgraph["edges"]:
            kgraph["edges"][edge_id] = edge
            continue
        update_edge(kgraph["edges"][edge_id], edge)


def normalize_kgraph(
    kgraph: KnowledgeGraphDict,
    results: list[ResultDict],
    aux_graphs: dict[str, AuxGraphDict],
) -> None:
    """Normalize the kgraph ids to their hashes, updating references in results and auxiliary graphs.

    All work is done in-place. This function mirrors the reasoner_pydantic method to ensure compatibility.
    """
    # Map old IDs to new IDs for reference fixing
    edge_id_mapping = dict[str, str]()

    for edge_id in list(kgraph["edges"].keys()):
        edge = kgraph["edges"].pop(edge_id)
        new_id = hash_hex(hash_edge(edge))
        edge_id_mapping[edge_id] = new_id
        kgraph["edges"][new_id] = edge

    # Skips some of the error detection in reasoner_pydantic
    for aux_graph in aux_graphs.values():
        aux_graph["edges"] = [
            edge_id_mapping.get(edge_id, edge_id) for edge_id in aux_graph["edges"]
        ]

    for result in results:
        for analysis in result["analyses"]:
            for binding_list in analysis["edge_bindings"].values():
                for binding in binding_list:
                    binding["id"] = edge_id_mapping.get(binding["id"], binding["id"])


@tracer.start_as_current_span("prune_kg")
def prune_kg(
    results: list[ResultDict],
    kgraph: KnowledgeGraphDict,
    aux_graphs: dict[str, AuxGraphDict],
    job_log: TRAPILogger,
) -> None:
    """Use finished results to prune the knowledge graph to only bound knowledge."""
    bound_edges = set[str]()
    bound_nodes = set[str]()
    for result in results:
        for node_binding_set in result["node_bindings"].values():
            bound_nodes.update([str(binding["id"]) for binding in node_binding_set])
        for analysis in result["analyses"]:
            for edge_binding_set in analysis["edge_bindings"].values():
                bound_edges.update([str(binding["id"]) for binding in edge_binding_set])

    checked_edges = set[str]()
    edges_to_check = list(bound_edges)
    while len(edges_to_check) > 0:
        edge_id = edges_to_check.pop()

        # Avoid infinite loops if edge and aux graph reference each other
        if edge_id in checked_edges:
            continue
        checked_edges.add(edge_id)

        edge = kgraph["edges"][edge_id]

        bound_edges.add(edge_id)
        bound_nodes.add(str(edge["subject"]))
        bound_nodes.add(str(edge["object"]))

        edge_aux_graphs = next(
            (
                attr
                for attr in edge.get("attributes", [])
                if attr["attribute_type_id"] == "biolink:support_graphs"
            ),
            None,
        )
        if edge_aux_graphs is None:
            continue
        # Have to cast because support graphs always has value of type list[str]
        # But attribute value is generally of type Any
        for aux_graph_id in cast(list[str], edge_aux_graphs["value"]):
            edges_to_check.extend(
                str(edge) for edge in aux_graphs[aux_graph_id]["edges"]
            )

    prior_edge_count = len(kgraph["edges"])
    prior_node_count = len(kgraph["nodes"])

    kgraph["edges"] = {edge_id: kgraph["edges"][edge_id] for edge_id in bound_edges}
    kgraph["nodes"] = {curie: kgraph["nodes"][curie] for curie in bound_nodes}

    pruned_edges = prior_edge_count - len(kgraph["edges"])
    pruned_nodes = prior_node_count - len(kgraph["nodes"])

    job_log.debug(
        f"KG Pruning: {len(kgraph['nodes'])} (-{pruned_nodes}) nodes and {len(kgraph['edges'])} (-{pruned_edges}) edges remain."
    )


def meta_qualifier_meets_constraints(
    meta_qualifiers: dict[str, list[str]] | None,
    constraints: Sequence[QualifierConstraint],
) -> bool:
    """Check if a number of qualifier constraints are met by a meta-qualifier set."""
    if len(constraints) == 0:  # No constraints to meet
        return True
    elif meta_qualifiers is None or len(meta_qualifiers) == 0:  # Can't meet constraints
        return False

    for constraint in constraints:
        qualifiers_met = True
        for qualifier in constraint.qualifier_set:
            q_type, q_val = qualifier.qualifier_type_id, qualifier.qualifier_value
            if q_type in meta_qualifiers or q_val in meta_qualifiers[q_type]:
                continue
            else:
                qualifiers_met = False
                break

        if qualifiers_met:
            return True
    return False
