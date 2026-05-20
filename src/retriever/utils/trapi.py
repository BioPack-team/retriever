import itertools
import uuid
from collections import defaultdict
from typing import cast

from translator_tom import (
    CURIE,
    Analysis,
    EdgeBinding,
    EdgeID,
    KnowledgeGraph,
    Node,
    NodeBinding,
    PathfinderAnalysis,
    QEdgeID,
    QNode,
    QNodeID,
    QueryGraph,
    Result,
    SetInterpretationEnum,
)

from retriever.utils.logs import TRAPILogger


def initialize_kgraph(qgraph: QueryGraph) -> KnowledgeGraph:
    """Initialize a knowledge graph, using nodes from the query graph."""
    kgraph = KnowledgeGraph.new()
    for qnode in qgraph.nodes.values():
        if qnode.ids is None:
            continue
        for curie in qnode.ids:
            kgraph.nodes[curie] = Node.model_construct(
                name=None,
                categories=[],
                attributes=[],
            )
    return kgraph


def evaluate_set_interpretation(
    qgraph: QueryGraph,
    results: list[Result],
    job_log: TRAPILogger,
) -> list[Result]:
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

    So we first aggregate all nodes that have either ALL or MANY for their
    set interpretation value

    We evaluate all nodes with set interpreation ALL first and then apply any
    additional collapsing to the set interpretation MANY nodes
    """
    node_group_all, node_group_many = _aggregate_node_groupings(qgraph, job_log)

    # Determine if any nodes have a set_interpretation value of ALL or MANY
    group_all: bool = any(len(node_group) > 0 for node_group in node_group_all.values())
    group_many: bool = any(
        len(node_group) > 0 for node_group in node_group_many.values()
    )

    if group_all or group_many:
        if group_all:
            results = _evaluate_set_interpretation_all(
                qgraph,
                results,
                node_group_all,
                job_log,
            )

        if group_many:
            results = _evaluate_set_interpretation_many(
                qgraph,
                results,
                node_group_many,
                job_log,
            )

    return results


def _aggregate_node_groupings(
    qgraph: QueryGraph, job_log: TRAPILogger
) -> tuple[defaultdict[QNodeID, set[CURIE]], defaultdict[QNodeID, set[CURIE]]]:
    """Determines whether any set_interpretation : ALL or MANY groups exist.

    Iterates over the query graph nodes to extract the set_interpretation
    values for each node. If the node has either ALL or MANY, we store the
    node identifiers in a dictionary to track the identifiers we require as
    a set for full connectivity
    """
    node_group_all: defaultdict[QNodeID, set[CURIE]] = defaultdict(set)
    node_group_many: defaultdict[QNodeID, set[CURIE]] = defaultdict(set)

    for node_name, node in qgraph.nodes.items():
        node_set_interpretation = SetInterpretationEnum(
            node.set_interpretation or SetInterpretationEnum.BATCH.value
        )
        match node_set_interpretation:
            case SetInterpretationEnum.BATCH:  # None defaults to BATCH
                pass  # no-opt, but valid value
            case SetInterpretationEnum.ALL:
                member_identifiers = node.member_ids
                if member_identifiers is None or len(member_identifiers) == 0:
                    job_log.error(
                        f"No `member_ids` specified for `set_interpretation`: ALL for node {node}. "
                    )
                else:
                    for node_id in member_identifiers:
                        node_group_all[node_name].add(node_id)

            case SetInterpretationEnum.MANY:
                member_identifiers = node.member_ids
                if member_identifiers is None or len(member_identifiers) == 0:
                    job_log.error(
                        f"No `member_ids` specified for `set_interpretation`: MANY for node {node}. "
                    )
                else:
                    for node_id in member_identifiers:
                        node_group_many[node_name].add(node_id)
            case _:  # pyright:ignore[reportUnnecessaryComparison]
                job_log.error(  # pyright:ignore[reportUnreachable]
                    f"Unknown value provided for set_interpretation within qgraph: {node_set_interpretation}"
                )
    return node_group_all, node_group_many


def _evaluate_set_interpretation_all(
    qgraph: QueryGraph,
    results: list[Result],
    node_group: defaultdict[QNodeID, set[CURIE]],
    job_log: TRAPILogger,
) -> list[Result]:
    """Handles the results graph pruning for `set_interpretation` : ALL.

    We first build two different lookup tables

    1) identifier-identifier lookup table
    Extracted from the results, we create a lookup table where we provide the CURIE as a key
    to return the set of CURIE's that are connected to that key CURIE

    Example:
    defaultdict(
       <class 'set'>,
       {
          'MONDO:0000001': {'MONDO:0000532', 'MONDO:0020644', 'UMLS:C2983716'},
          'MONDO:0000532': {'MONDO:0000001', 'MONDO:0004993', 'MONDO:0008903'},
          'MONDO:0004993': {'MONDO:0000532', 'UMLS:C2983716'},
          'MONDO:0008903': {'MONDO:0000532', 'MONDO:0020644', 'UMLS:C2983716'},
          'MONDO:0020644': {'MONDO:0000001', 'MONDO:0008903'},
          'UMLS:C2983716': {'MONDO:0000001', 'MONDO:0004993', 'MONDO:0008903'}
       }
    )

    In this case, we found 6 entries
    >>> 4 of those entries had 3 connections
    >>> 2 of those entries had 2 connections

    2) identifier-result index
    This is a basic index to tell use where a CURIE is found in the results
    via the integer index within the list

    Example:
    defaultdict(
        <class 'list'>,
      {
         'MONDO:0000001': [5, 6, 7],
         'MONDO:0000532': [0, 4, 5],
         'MONDO:0004993': [3, 4],
         'MONDO:0008903': [0, 1, 2],
         'MONDO:0020644': [1, 6],
         'UMLS:C2983716': [2, 3, 7]
      }
    )

    We build the index at the same time as the identifier-identifier
    lookup table to avoid iterating over the results more than once.
    For smaller result collections this won't matter, but if we have a large
    number of results, we'd prefer to minimize iterating over it needlessly

    After this we evaluate the node connectivity to determine which CURIE
    identifiers have full connectivity with the specified nodes.

    If we find full connectivity for an identifier we collapse it into a single
    entry

    If we find only partial to no connectivity then we prune all instances of
    the CURIE from the results

    We do the pruning in one step via the pruning mask to manipulate the
    results as little as possible

    After collapsing all results and pruning the entries, we add the new
    collapsed entries to results collection and return them
    """
    identifier_identifier_lookup_table, identifier_result_index = (
        _build_identifier_lookup_tables(results)
    )

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
    collapse_entries: list[Result] = []

    for identifier, fully_connected in identifier_full_connectivity_mapping.items():
        if fully_connected:
            job_log.debug(f"Collapsing fully connected node identifier: {identifier}")
            node_bindings = _build_collapsed_result_node_bindings(
                qgraph,
                identifier,
                identifier_edge_mapping,
                job_log,
            )
            if node_bindings is not None:
                analysis_bindings = _build_collapsed_result_analysis(
                    results,
                    identifier,
                    identifier_result_index,
                )
                collapse_result = Result.model_construct(
                    node_bindings=node_bindings, analyses=analysis_bindings
                )

                collapse_entries.append(collapse_result)
                for collapse_index in identifier_result_index[identifier]:
                    results_prune_mask[collapse_index] = 0
        else:
            job_log.debug(
                f"[set interpretation: ALL] Pruning partially connected identifier: {identifier}. Missing identifiers from full connection: {missing_identifier_mapping[identifier]}"
            )
            for prune_index in identifier_result_index[identifier]:
                results_prune_mask[prune_index] = 0

    results = list(itertools.compress(results, results_prune_mask))
    results.extend(collapse_entries)
    return results


def _evaluate_set_interpretation_many(
    qgraph: QueryGraph,
    results: list[Result],
    node_group: defaultdict[QNodeID, set[CURIE]],
    job_log: TRAPILogger,
) -> list[Result]:
    """Handles the results graph pruning for `set_interpretation` : MANY.

    See the _evaluate_set_interpretation_all docstring for more involved
    details

    The main difference here is we won't perform pruning, only collapse any nodes
    that meet the full connectivity requirement. Otherwise this method
    is pretty much identical to _evaluate_set_interpretation_all. Because we
    only have two methods, I think trying to create an abstraction between
    the two would make the logic more complicated than needed for minimal
    code reducation. If more set interpretation values are added in the future,
    this assumption may no longer hold
    """
    identifier_identifier_lookup_table, identifier_result_index = (
        _build_identifier_lookup_tables(results)
    )

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
    collapse_entries: list[Result] = []

    for identifier, fully_connected in identifier_full_connectivity_mapping.items():
        if fully_connected:
            job_log.debug(f"Collapsing fully connected node identifier: {identifier}")
            node_bindings = _build_collapsed_result_node_bindings(
                qgraph,
                identifier,
                identifier_edge_mapping,
                job_log,
            )
            if node_bindings is not None:
                analysis_bindings = _build_collapsed_result_analysis(
                    results,
                    identifier,
                    identifier_result_index,
                )
                collapse_result = Result.model_construct(
                    node_bindings=node_bindings, analyses=analysis_bindings
                )

                collapse_entries.append(collapse_result)
                for collapse_index in identifier_result_index[identifier]:
                    results_prune_mask[collapse_index] = 0
        else:
            job_log.debug(
                f"[set interpretation: MANY] No pruning of partially connected identifier: {identifier}. Missing identifiers from full connection: {missing_identifier_mapping[identifier]}"
            )

    results = list(itertools.compress(results, results_prune_mask))
    results.extend(collapse_entries)
    return results


def _evaluate_node_connectivity(
    qgraph: QueryGraph,
    node_group: defaultdict[QNodeID, set[CURIE]],
    identifier_identifier_lookup_table: defaultdict[CURIE, set[CURIE]],
) -> tuple[
    dict[CURIE, bool], dict[CURIE, list[CURIE]], dict[CURIE, dict[str, QNodeID]]
]:
    """Evaluates how fully connected a CURIE identifier is to other nodes.

    We first build a node-identifier lookup table so that we can provide
    the node name and get all the identifiers associated with the node
    from the query graph

    We then iterate over all edges in the query graph looking to
    see if either the subject or objeect node exist in the provided
    node_group

    If we find the set in the node_group, we then have to look at the
    paired nodes identifiers to see if they're connected

    If the subject node has set interpretation : ALL/MANY -> look at object
    If the object node has set interpretation : ALL/MANY -> look at subject

    We iterate over the paired nodes identifiers and get the identifiers
    connectivity set from the lookup table we created prior to this method

    If it's a subset then we can collapse that node identifiers connections,
    otherwise we prune all instances

    However, for now we just return the data structures we generated that
    represent the connectivity of the identifiers

    Examples:
    >>> identifier_full_connectivity_mapping
        {'MONDO:0008903': True, 'MONDO:0000001': True, 'MONDO:0004993': False}
    >>> missing_identifier_mapping
        {'MONDO:0008903': [], 'MONDO:0000001': [], 'MONDO:0004993': ['MONDO:0020644']}
    >>> identifier_edge_mapping
        {
           'MONDO:0008903': {'origin': 'n0', 'connection': 'n1'},
           'MONDO:0000001': {'origin': 'n0', 'connection': 'n1'},
           'MONDO:0004993': {'origin': 'n0', 'connection': 'n1'}
        }
    """
    node_identifier_lookup_map: dict[QNodeID, list[CURIE]] = {}
    for node_name, node in qgraph.nodes.items():
        match SetInterpretationEnum(
            node.set_interpretation or SetInterpretationEnum.BATCH.value
        ):
            case SetInterpretationEnum.BATCH:
                node_identifiers = node.ids
            case SetInterpretationEnum.ALL:
                node_identifiers = node.member_ids
            case SetInterpretationEnum.MANY:
                node_identifiers = node.member_ids

        if node_identifiers is None:
            node_identifiers = []

        node_identifier_lookup_map[node_name] = node_identifiers

    identifier_full_connectivity_mapping: dict[CURIE, bool] = {}
    missing_identifier_mapping: dict[CURIE, list[CURIE]] = {}
    identifier_edge_mapping: dict[CURIE, dict[str, QNodeID]] = {}
    for edge in qgraph.edges.values():
        subject_node: QNodeID = edge.subject
        subject_set: set[CURIE] = node_group.get(subject_node, set())

        object_node: QNodeID = edge.object
        object_set: set[CURIE] = node_group.get(object_node, set())

        if len(subject_set) > 0:
            for node_id in node_identifier_lookup_map[object_node]:
                identifier_full_connectivity_mapping[node_id] = subject_set.issubset(
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
                    "connection": object_node,
                    "origin": subject_node,
                }

    return (
        identifier_full_connectivity_mapping,
        missing_identifier_mapping,
        identifier_edge_mapping,
    )


def _build_identifier_lookup_tables(
    results: list[Result],
) -> tuple[defaultdict[CURIE, set[CURIE]], defaultdict[CURIE, list[int]]]:
    """Builds an identifier metadata tables for evaluating node connectivity.

    We build the identifier-identifier lookup table to see
    how connected the identifiers are in the results

    We build the identifier-result index to avoid iterating over
    the results
    """
    identifier_identifier_lookup_table: defaultdict[CURIE, set[CURIE]] = defaultdict(
        set
    )
    identifier_result_index: defaultdict[CURIE, list[int]] = defaultdict(list)

    for index, result in enumerate(results):
        node_entries = tuple(result.node_bindings.values())

        # Making assumption that there's only one entry here
        first_identifier = node_entries[0][0].id
        second_identifier = node_entries[1][0].id

        identifier_identifier_lookup_table[first_identifier].add(second_identifier)
        identifier_identifier_lookup_table[second_identifier].add(first_identifier)
        identifier_result_index[first_identifier].append(index)
        identifier_result_index[second_identifier].append(index)

    return identifier_identifier_lookup_table, identifier_result_index


def _build_collapsed_result_node_bindings(
    qgraph: QueryGraph,
    identifier: CURIE,
    identifier_edge_mapping: dict[CURIE, dict[str, QNodeID]],
    job_log: TRAPILogger,
) -> dict[QNodeID, list[NodeBinding]] | None:
    """Generates the node bindings for the collapsed result entry.

    Extracts the relevant information from the query graph nodes
    and from our provided edge mapping to ensure we're collapsing
    the correct edge

    Also verifies that the set identifier provided is correct and is
    a valid UUID before generating the node bindings
    """
    edge_ordering: dict[str, QNodeID] = identifier_edge_mapping[identifier]
    graph_nodes: dict[QNodeID, QNode] = qgraph.nodes
    set_identifier: list[CURIE] | None = graph_nodes[edge_ordering["connection"]].ids

    if set_identifier is None:
        job_log.error(
            "Set identifier not specified in the query graph. Unable to build collapsed result"
        )
        return None
    else:
        # Unsure how typing works in this case because it's overloaded
        # This normally would be a CURIE, but it's actually a uuid
        try:
            uuid_set_identifier = set_identifier[0]
        except IndexError:
            job_log.error(
                "Unable to access the set identifier to build the collapsed result"
            )

            return None

        try:
            uuid.UUID(uuid_set_identifier, version=4)
        except ValueError:
            job_log.error(
                f"Invalid UUID provided for the set identifier {uuid_set_identifier}"
            )
            return None

        node_bindings = {
            edge_ordering["origin"]: [
                NodeBinding.model_construct(id=identifier, attributes=[])
            ],
            edge_ordering["connection"]: [
                NodeBinding.model_construct(id=uuid_set_identifier, attributes=[])
            ],
        }
        return node_bindings


def _build_collapsed_result_analysis(
    results: list[Result],
    identifier: CURIE,
    identifier_result_index: defaultdict[CURIE, list[int]],
) -> list[Analysis | PathfinderAnalysis]:
    """Generates the analysis bindings for the collapsed result entry.

    Leverages the result index to determine where we should extract
    the result entries from and builds the edge bindings for
    the analyses entry
    """
    edge_identifiers: list[EdgeID] = []
    for location in identifier_result_index[identifier]:
        try:
            identifier_result: Result = results[location]
        except IndexError:
            pass
        else:
            result_analysis: list[Analysis | PathfinderAnalysis] = (
                identifier_result.analyses
            )
            for analysis in cast(list[Analysis], result_analysis):
                edge_bindings: dict[QEdgeID, list[EdgeBinding]] = analysis.edge_bindings
                for edge_binding in edge_bindings.values():
                    edge_identifiers.extend(edge.id for edge in edge_binding)

    collapsed_edge_id: QEdgeID = "e0"
    analyses: list[Analysis | PathfinderAnalysis] = [
        Analysis.model_construct(
            resource_id="infores:retriever",
            edge_bindings={
                collapsed_edge_id: [
                    EdgeBinding.model_construct(id=kedge_id, attributes=[])
                    for kedge_id in edge_identifiers
                ]
            },
        )
    ]
    return analyses
