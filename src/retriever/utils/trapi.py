import hashlib
import itertools
from collections import defaultdict
from collections.abc import Sequence
from typing import cast

from opentelemetry import trace
from reasoner_pydantic import QueryGraph
from reasoner_pydantic.utils import make_hashable

from retriever.types.trapi import (
    CURIE,
    AnalysisDict,
    AttributeDict,
    AuxGraphID,
    AuxiliaryGraphDict,
    BiolinkEntity,
    EdgeBindingDict,
    EdgeDict,
    EdgeIdentifier,
    Infores,
    KnowledgeGraphDict,
    MetaAttributeDict,
    NodeBindingDict,
    NodeDict,
    QNodeID,
    QualifierConstraintDict,
    QualifierTypeID,
    QueryGraphDict,
    ResultDict,
    RetrievalSourceDict,
    SetInterpretationEnum,
)
from retriever.utils import biolink
from retriever.utils.logs import TRAPILogger

tracer = trace.get_tracer("lookup.execution.tracer")


def initialize_kgraph(qgraph: QueryGraphDict | QueryGraph) -> KnowledgeGraphDict:
    """Initialize a knowledge graph, using nodes from the query graph."""
    kgraph = KnowledgeGraphDict(nodes={}, edges={})
    if isinstance(qgraph, QueryGraph):
        for qnode in qgraph.nodes.values():
            if qnode.ids is None:
                continue
            for curie in qnode.ids:
                kgraph["nodes"][CURIE(curie)] = NodeDict(
                    categories=[BiolinkEntity(cat) for cat in (qnode.categories or [])],
                    attributes=[],
                )
    else:
        for qnode in qgraph["nodes"].values():
            if "ids" not in qnode or qnode["ids"] is None:
                continue
            for curie in qnode["ids"]:
                kgraph["nodes"][CURIE(curie)] = NodeDict(
                    categories=[
                        BiolinkEntity(cat)
                        for cat in (qnode.get("categories", []) or [])
                    ],
                    attributes=[],
                )
    return kgraph


def hash_hex(hashint: int) -> EdgeIdentifier:
    """Convert a regular hash to the type used in pydantic EdgeIdentifier."""
    return EdgeIdentifier(
        hashlib.blake2b(
            hashint.to_bytes(16, byteorder="big", signed=True), digest_size=6
        ).hexdigest()
    )


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
    source: Infores,
) -> None:
    """Append a aggregator source to an edge's provenance, reference the current most downstream source."""
    upstreams = set(
        itertools.chain(
            *[
                source.get(Infores("upstream_resource_ids"), [])
                for source in edge["sources"]
            ]
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
            resource_role="primary_knowledge_source", resource_id=Infores("err_missing")
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
            set(source.get("upstream_resource_ids", []) or [])
            | set(new["upstream_resource_ids"] or [])
        )


def update_edge(edge: EdgeDict, new: EdgeDict) -> None:
    """Update the first edge in-place, merging information from the second."""
    if "attributes" in new:
        old_attributes = {
            hash_attribute(attr): attr for attr in (edge.get("attributes", []) or [])
        }
        new_attributes = {
            hash_attribute(attr): attr
            for attr in (new.get("attributes", []) or [])
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
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
) -> None:
    """Normalize the kgraph ids to their hashes, updating references in results and auxiliary graphs.

    All work is done in-place. This function mirrors the reasoner_pydantic method to ensure compatibility.
    """
    # Map old IDs to new IDs for reference fixing
    edge_id_mapping = dict[str, EdgeIdentifier]()

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
            if "edge_bindings" not in analysis:
                continue
            for binding_list in analysis["edge_bindings"].values():
                for binding in binding_list:
                    binding["id"] = edge_id_mapping.get(binding["id"], binding["id"])


@tracer.start_as_current_span("prune_kg")
def prune_kg(
    results: list[ResultDict],
    kgraph: KnowledgeGraphDict,
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
    job_log: TRAPILogger,
) -> None:
    """Use finished results to prune the knowledge graph to only bound knowledge."""
    bound_edges = set[EdgeIdentifier]()
    bound_nodes = set[CURIE]()
    for result in results:
        for node_binding_set in result["node_bindings"].values():
            bound_nodes.update([binding["id"] for binding in node_binding_set])
        for analysis in result["analyses"]:
            if "edge_bindings" not in analysis:
                continue
            for edge_binding_set in analysis["edge_bindings"].values():
                bound_edges.update([binding["id"] for binding in edge_binding_set])

    checked_edges = set[EdgeIdentifier]()
    edges_to_check = list(bound_edges)
    while len(edges_to_check) > 0:
        edge_id = edges_to_check.pop()

        # Avoid infinite loops if edge and aux graph reference each other
        if edge_id in checked_edges:
            continue
        checked_edges.add(edge_id)

        edge = kgraph["edges"][edge_id]

        bound_edges.add(edge_id)
        bound_nodes.add(edge["subject"])
        bound_nodes.add(edge["object"])

        edge_aux_graphs = next(
            (
                attr
                for attr in (edge.get("attributes", []) or [])
                if attr["attribute_type_id"] == "biolink:support_graphs"
            ),
            None,
        )
        if edge_aux_graphs is None:
            continue
        # Have to cast because support graphs always has value of type list[str]
        # But attribute value is generally of type Any
        for aux_graph_id in cast(list[AuxGraphID], edge_aux_graphs["value"]):
            edges_to_check.extend(edge for edge in aux_graphs[aux_graph_id]["edges"])

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
    meta_qualifiers: dict[QualifierTypeID, list[str]] | None,
    constraints: Sequence[QualifierConstraintDict],
) -> bool:
    """Check if a number of qualifier constraints are met by a meta-qualifier set."""
    if len(constraints) == 0:  # No constraints to meet
        return True
    elif meta_qualifiers is None or len(meta_qualifiers) == 0:  # Can't meet constraints
        return False

    for constraint in constraints:
        qualifiers_met = True
        for qualifier in constraint["qualifier_set"]:
            q_type, q_val = qualifier["qualifier_type_id"], qualifier["qualifier_value"]
            if q_type in meta_qualifiers:
                expanded_vals = biolink.get_descendant_values(q_type, q_val)
                if not len(meta_qualifiers[QualifierTypeID(q_type)]):
                    continue
                if expanded_vals & set(meta_qualifiers[QualifierTypeID(q_type)]):
                    continue
                else:
                    qualifiers_met = False
                    break

            else:
                qualifiers_met = False
                break

        if qualifiers_met:
            return True
    return False


def evaluate_set_interpretation(
    qgraph: QueryGraphDict,
    results: list[ResultDict],
    job_log: TRAPILogger,
) -> list[ResultDict]:
    """Handles set interpretation logic from the TRAPI specification.

    Each node in the graph can have 3 different values for set_interpretation

    1. BATCH
    This is the default case if not specified. We can have any pair-wise relationship
    with nodes that have the BATCH value. We will not prune or consolidate the
    results due BATCH nodes

    2. ALL
    With set_interpretation : ALL, we force all edges connected to an ALL node
    to include every possible node connection

    Example 1:
    Node N {
        ids: 000, 001, 002,
        set_interpretation: BATCH
    }
    Node M {
        ids: uuid.uuid4("woodworking"),
        set_interpretation: ALL,
        member_ids: AAA, BBB, CCC
    }
    Edge P { object: Node N, subject: Node M }

    Here we have two nodes and one edge specified in the query graph.
    Node N is a BATCH node so we don't have care about it's connectivity.
    Node M is an ALL node, so every edge connected to M has to have a full
    connectivity

    This can be represented as the cartesian prouct of the ids between the two
    nodes:

    >>> import itertools
    >>> node_n = ["000", "001", "002"]
    >>> node_m = ["AAA", "BBB", "CCC"]
    >>> full_connectivity = tuple(itertools.product(node_n, node_m, repeat=1))
    (
        ('000', 'AAA'), ('000', 'BBB'), ('000', 'CCC'),
        ('001', 'AAA'), ('001', 'BBB'), ('001', 'CCC'),
        ('002', 'AAA'), ('002', 'BBB'), ('002', 'CCC')
    )

    In the case of full connectivity we can consolidate the results graph so
    that each identifier in node N that is fully connected to node M can be one
    result entry that points from node N <-> node M using the uuid provided in
    the ids attribute

    So our final results would be the following:

    Node N(000) -> Node M(uuid.uuid4("woodworking")
    Node N(001) -> Node M(uuid.uuid4("woodworking")
    Node N(002) -> Node M(uuid.uuid4("woodworking")

    In the case of non-full connectivity, we prune these results from the
    results collection

    Example 2:
    Imagine we have the following results
    (
        ('000', 'AAA'), ('000', 'CCC'),
        ('001', 'AAA'), ('001', 'BBB'), ('001', 'CCC'),
        ('002', 'BBB'), ('002', 'CCC')
    )

    We'd prune the results from Node N(000) and Node N(002) as they don't form
    a complete set.

    So the final results would be the following:

    Node N(001) -> Node M(uuid.uuid4("woodworking")

    3. MANY
    This is a less strict version of case 2 with ALL. We still evaluate the
    connectivity, but perform no pruning if we don't have full connectivity.

    Example 3:

    Node N {
        ids: 000, 001, 002,
        set_interpretation: BATCH
    }
    Node M {
        ids: uuid.uuid4("woodworking"),
        set_interpretation: MANY,
        member_ids: AAA, BBB, CCC
    }
    Edge P { object: Node N, subject: Node M }

    Imagine we have the following results:
    (
        ('000', 'AAA'),
        ('001', 'AAA'), ('001', 'BBB')
        ('002', 'AAA'), ('002', 'BBB'), ('002', 'CCC')
    )

    Our final results would be the following:

    Node N(000) -> Node M("AAA")
    Node N(001) -> Node M("AAA")
    Node N(001) -> Node M("BBB")
    Node N(002) -> Node M(uuid.uuid4("woodworking")

    Control Flow:

    We have both nodes and edges we have to consider when analyzing the
    connections. We want to build a bi-directional map that allows use
    identify what node is connected to what and return a set representing
    all connected nodes

    So we first aggregate all nodes in
    """
    node_group_all, node_group_many = _aggregate_node_groupings(qgraph, job_log)

    # Determine if any nodes have a set_interpretation value of ALL or MANY
    group_all: bool = any(len(node_group) > 0 for node_group in node_group_all.values())
    group_many: bool = any(len(node_group) > 0 for node_group in node_group_many.values())

    if group_all or group_many:
        identifier_identifier_lookup_table, identifier_result_index = _build_identifier_lookup_tables(results)

        if group_all:
            results = _evaluate_set_interpretation_all(
                qgraph,
                results,
                node_group_all,
                identifier_identifier_lookup_table,
                identifier_result_index,
                job_log,
            )

        if group_many:
            results = _evaluate_set_interpretation_many(
                qgraph,
                results,
                node_group_many,
                identifier_identifier_lookup_table,
                identifier_result_index,
                job_log,
            )

    return results


def _aggregate_node_groupings(
    qgraph: QueryGraphDict, job_log: TRAPILogger
) -> tuple[defaultdict[QNodeID, set[CURIE]], defaultdict[QNodeID, set[CURIE]]]:
    """Determines whether any set_interpretation : ALL or MANY groups exist.

    Iterates over the query graph nodes to extract the set_interpretation values
    for each node. If the node has either ALL or MANY, we store the node identifiers
    in a dictionary to track the identifiers we require as a set for full connectivity
    """
    node_group_all: defaultdict[QNodeID, set[CURIE]] = defaultdict(set)
    node_group_many: defaultdict[QNodeID, set[CURIE]] = defaultdict(set)

    for node_name, node in qgraph.get("nodes", {}).items():
        node_set_interpretation = node.get(
            "set_interpretation", SetInterpretationEnum.BATCH
        )
        match node_set_interpretation:
            case SetInterpretationEnum.ALL:
                member_identifiers = node.get("member_ids", [])
                if member_identifiers is None or len(member_identifiers) == 0:
                    job_log.error(
                        f"No `member_ids` specified for `set_interpretation`: ALL for node {node}. "
                    )
                else:
                    for node_id in member_identifiers:
                        node_group_all[node_name].add(node_id)

            case SetInterpretationEnum.MANY:
                member_identifiers = node.get("member_ids", [])
                if member_identifiers is None or len(member_identifiers) == 0:
                    job_log.error(
                        f"No `member_ids` specified for `set_interpretation`: MANY for node {node}. "
                    )
                else:
                    for node_id in member_identifiers:
                        node_group_many[node_name].add(node_id)
            case _:
                job_log.error(
                    f"Unknown value provided for set_interpretation within qgraph: {node_set_interpretation}"
                )
    return node_group_all, node_group_many


def _evaluate_set_interpretation_all(
    qgraph: QueryGraphDict,
    results: list[ResultDict],
    node_group: defaultdict[QNodeID, set[CURIE]],
    identifier_identifier_lookup_table: defaultdict[CURIE, set[CURIE]],
    identifier_result_index: defaultdict[CURIE, list[int]],
    job_log: TRAPILogger,
) -> list[ResultDict]:
    """Handles the results graph pruning for `set_interpretation` : ALL."""
    (
        identifier_full_connectivity_mapping,
        missing_identifier_mapping,
        identifier_edge_mapping,
    ) = _evaluate_node_connectivity(
        qgraph,
        node_group,
        identifier_identifier_lookup_table,
    )

    results_prune_mask: list[int] = [1] * len(results)
    collapse_entries: list[ResultDict] = []

    for identifier, fully_connected in identifier_full_connectivity_mapping.items():
        if fully_connected:
            job_log.debug(f"Collapsing fully connected node identifier: {identifier}")
            collapse_entries.append(
                _build_collapsed_result_entry(
                    qgraph,
                    results,
                    identifier,
                    identifier_edge_mapping,
                    identifier_result_index,
                )
            )
            for collapse_index in identifier_result_index[identifier]:
                results_prune_mask[collapse_index] = 0
        else:
            job_log.debug(
                f"[set interpretation: ALL] Pruning partially connected identifier: {identifier}. "
                f"Missing identifiers from full connection: {missing_identifier_mapping[identifier]}"
            )
            for prune_index in identifier_result_index[identifier]:
                results_prune_mask[prune_index] = 0

    results = list(itertools.compress(results, results_prune_mask))
    results.extend(collapse_entries)
    return results


def _evaluate_set_interpretation_many(
    qgraph: QueryGraphDict,
    results: list[ResultDict],
    node_group: defaultdict[QNodeID, set[CURIE]],
    identifier_identifier_lookup_table: defaultdict[CURIE, set[CURIE]],
    identifier_result_index: defaultdict[CURIE, list[int]],
    job_log: TRAPILogger,
) -> list[ResultDict]:
    """Handles the results graph pruning for `set_interpretation` : MANY.

    The main difference here is we won't perform pruning, only collapse any nodes
    that meet the full connectivity requirement
    """
    (
        identifier_full_connectivity_mapping,
        missing_identifier_mapping,
        identifier_edge_mapping,
    ) = _evaluate_node_connectivity(
        qgraph,
        node_group,
        identifier_identifier_lookup_table,
    )

    results_prune_mask = [1] * len(results)
    collapse_entries = []

    for identifier, fully_connected in identifier_full_connectivity_mapping.items():
        if fully_connected:
            job_log.debug(f"Collapsing fully connected node identifier: {identifier}")
            collapse_entries.append(
                _build_collapsed_result_entry(
                    qgraph,
                    results,
                    identifier,
                    identifier_edge_mapping,
                    identifier_result_index,
                )
            )
            for collapse_index in identifier_result_index[identifier]:
                results_prune_mask[collapse_index] = 0
        else:
            job_log.debug(
                f"[set interpretation: MANY] No pruning of partially connected identifier: {identifier}. "
                f"Missing identifiers from full connection: {missing_identifier_mapping[identifier]}"
            )

    results = list(itertools.compress(results, results_prune_mask))
    results.extend(collapse_entries)
    return results


def _evaluate_node_connectivity(
    qgraph: QueryGraphDict,
    node_group: defaultdict[QNodeID, set[CURIE]],
    identifier_identifier_lookup_table: defaultdict[CURIE, set[CURIE]],
) -> tuple[dict[QNodeID, bool], dict[QNodeID, list[CURIE]], dict[QNodeID, dict[str, QNodeID]]]:
    """Evaluates how fully connected a node is to other nodes."""
    node_identifier_lookup_map: dict[QNodeID, list[CURIE | None]] = {}
    for node_name, node in qgraph["nodes"].items():
        match node.get("set_interpretation", SetInterpretationEnum.BATCH):
            case SetInterpretationEnum.BATCH:
                node_identifiers = node.get("ids", None)
            case SetInterpretationEnum.ALL:
                node_identifiers = node.get("member_ids", None)
            case SetInterpretationEnum.MANY:
                node_identifiers = node.get("member_ids", None)
            case _:
                node_identifiers = None

        if node_identifiers is None:
            node_identifiers = []

        node_identifier_lookup_map[node_name] = node_identifiers

    identifier_full_connectivity_mapping: dict[QNodeID, bool] = {}
    missing_identifier_mapping: dict[QNodeID, list[CURIE]] = {}
    identifier_edge_mapping: dict[QNodeID, dict[str, QNodeID]] = {}
    for edge in qgraph["edges"].values():
        subject_node: QNodeID = edge["subject"]
        subject_set: set[CURIE] = node_group.get(subject_node, set())

        object_node: QNodeID = edge["object"]
        object_set: set[CURIE] = node_group.get(object_node, set())

        if len(subject_set) > 0:
            for node_id in node_identifier_lookup_map[object_node]:
                identifier_full_connectivity_mapping[node_id] = object_set.issubset(
                    identifier_identifier_lookup_table[node_id]
                )
                missing_identifier_mapping[node_id] = list(
                    object_set.difference(identifier_identifier_lookup_table[node_id])
                )
                identifier_edge_mapping[node_id] = {
                    "connection": subject_node,
                    "origin": object_node,
                }

        if len(object_set) > 0:
            for node_id in node_identifier_lookup_map[subject_node]:
                identifier_full_connectivity_mapping[node_id] = object_set.issubset(
                    identifier_identifier_lookup_table[node_id]
                )
                missing_identifier_mapping[node_id] = list(
                    object_set.difference(identifier_identifier_lookup_table[node_id])
                )
                identifier_edge_mapping[node_id] = {
                    "origin": subject_node,
                    "connection": object_node,
                }
    return (
        identifier_full_connectivity_mapping,
        missing_identifier_mapping,
        identifier_edge_mapping,
    )


def _build_identifier_lookup_tables(results: list[ResultDict]) -> tuple[defaultdict[CURIE, set[CURIE]], defaultdict[CURIE, list[int]]]:
    """Builds an identifier-identifier lookup table."""
    identifier_identifier_lookup_table: defaultdict[CURIE, set[CURIE]] = defaultdict(set)
    identifier_result_index: defaultdict[CURIE, list[int]] = defaultdict(list)

    for index, result in enumerate(results):
        node_entries = tuple(result["node_bindings"].values())

        # Making assumption that there's only one entry here
        first_identifier = node_entries[0][0]["id"]
        second_identifier = node_entries[1][0]["id"]

        identifier_identifier_lookup_table[first_identifier].add(second_identifier)
        identifier_identifier_lookup_table[second_identifier].add(first_identifier)
        identifier_result_index[first_identifier].append(index)
        identifier_result_index[second_identifier].append(index)

    return identifier_identifier_lookup_table, identifier_result_index


def _build_collapsed_result_entry(
    qgraph: QueryGraphDict,
    results: list[ResultDict],
    identifier: CURIE,
    identifier_edge_mapping: dict[QNodeID, dict[str, QNodeID]],
    identifier_result_index: defaultdict[CURIE, list[int]],
) -> ResultDict:
    """Builds the collapsed entries for fully connected identifiers.

    Extracts the information from the nodes and edges bindings to build
    a new result that represents a merged entry
    """
    edge_identifiers = []
    for location in identifier_result_index[identifier]:
        for edge_data in results[location]["analyses"][0]["edge_bindings"].values():
            edge_identifiers.extend(edge["id"] for edge in edge_data)

    edge_ordering = identifier_edge_mapping[identifier]
    set_identifier = qgraph["nodes"][edge_ordering["connection"]]["ids"]

    return ResultDict(
        node_bindings={
            edge_ordering["origin"]: [NodeBindingDict(id=identifier, attributes=[])],
            edge_ordering["connection"]: [
                NodeBindingDict(id=set_identifier, attributes=[])
            ],
        },
        analyses=[
            AnalysisDict(
                resource_id=Infores("infores:retriever"),
                edge_bindings={
                    "e0": [
                        EdgeBindingDict(id=kedge_id, attributes=[])
                        for kedge_id in edge_identifiers
                    ]
                },
            )
        ],
    )
