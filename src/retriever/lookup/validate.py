from translator_tom import (
    Biolink,
    PathfinderQueryGraph,
    QEdge,
    QEdgeID,
    QNode,
    QNodeID,
    QueryGraph,
)

from retriever.config.openapi import OPENAPI_CONFIG


def validate(
    qg: QueryGraph | PathfinderQueryGraph,
) -> tuple[list[str], list[str]]:
    """Check that a given query graph is valid.

    Returns:
        A list of warning messages, which do not fail validation but should be logged.
        And a list of messages detailing validation problems.
        If the list is empty, the graph passes validation.
    """
    if isinstance(qg, PathfinderQueryGraph):
        return [], ["Retriever does not support Pathfinder queries."]
    warnings = list[str]()
    problems = dict[str, bool]()  # False means failing
    problems["Query graph must have at least one node"] = len(qg.nodes.values()) > 0
    problems["Query graph must have at least one edge"] = len(qg.edges.values()) > 0
    problems["Query graph must have at least one node with an ID"] = any(
        node for node in qg.nodes.values() if len(node.ids_list) > 0
    )

    # node_pairs = set[str]()
    for qedge_id, qedge in qg.edges.items():
        edge_warnings, edge_problems = validate_qedge(qg, qedge_id, qedge)
        problems.update(edge_problems)
        warnings.extend(edge_warnings)

    for qnode_id, qnode in qg.nodes.items():
        node_warnings, node_problems = validate_qnode(qg, qnode_id, qnode)
        problems.update(node_problems)
        warnings.extend(node_warnings)

    return warnings, [name for name, passed in problems.items() if not passed]


def validate_qedge(
    qg: QueryGraph, qedge_id: QEdgeID, qedge: QEdge
) -> tuple[list[str], dict[str, bool]]:
    """Find and return any problems with a given Query Edge.

    Problems in the dictionary marked False are failing.
    """
    problems = dict[str, bool]()

    if qedge.subject not in qg.nodes:
        problems[
            f"Edge `{qedge_id}` subject `{qedge.subject}` not defined in query graph."
        ] = False

    if qedge.object not in qg.nodes:
        problems[
            f"Edge `{qedge_id}` object `{qedge.object}` not defined in query graph."
        ] = False

    for i, qualifier_constraint in enumerate(qedge.qualifier_constraints_list):
        qualifier_types: set[str] = set()
        for qualifier in qualifier_constraint.qualifier_set:
            if qualifier.qualifier_type_id in qualifier_types:
                problems[
                    f"Edge `{qedge_id}` qualifier constraint {i} has duplicate qualifier_type_id `{qualifier.qualifier_type_id}`"
                ] = False
            qualifier_types.add(qualifier.qualifier_type_id)

    if qedge.knowledge_type == "inferred":
        problems["Retriever does not handle inferred-type queries."] = False

    invalid_predicates = [
        p for p in qedge.predicates_list if not Biolink.is_valid_predicate(p)
    ]
    if len(invalid_predicates) > 0:
        problems[f"Edge `{qedge_id}` has invalid predicates: {invalid_predicates}"] = (
            False
        )

    # if (
    #     f"{qedge.subject}-{qedge.object}" in node_pairs
    #     or f"{qedge.object}-{qedge.subject}" in node_pairs
    # ):
    #     problems["Duplicate qedges not allowed."] = False
    # node_pairs.add(f"{qedge.subject}-{qedge.object}")

    warnings = list[str]()
    unknown_fields = list(qedge.extra_dict.keys())
    if len(unknown_fields) > 0:
        warnings.append(
            f"Edge `{qedge_id}`: skipping unknown fields ({', '.join(unknown_fields)})"
        )

    return warnings, problems


def validate_qnode(
    _qg: QueryGraph, qnode_id: QNodeID, qnode: QNode
) -> tuple[list[str], dict[str, bool]]:
    """Find and return any problems with a given Query Node.

    Problems in the dictionary marked False are failing.
    """
    problems = dict[str, bool]()

    if len(qnode.ids_list) > OPENAPI_CONFIG.x_trapi.batch_size_limit:
        problems[
            f"Node `{qnode_id}` ID count ({len(qnode.ids_list)}) exceeds batch size limit of {OPENAPI_CONFIG.x_trapi.batch_size_limit}"
        ] = False

    invalid_categories = [
        c for c in qnode.categories_list if not Biolink.is_valid_category(c)
    ]
    if len(invalid_categories) > 0:
        problems[f"Node `{qnode_id}` has invalid categories: {invalid_categories}"] = (
            False
        )

    warnings = list[str]()
    unknown_fields = list(qnode.extra_dict.keys())
    if len(unknown_fields) > 0:
        warnings.append(
            f"Node `{qnode_id}`: skipping unknown fields ({', '.join(unknown_fields)})"
        )

    return warnings, problems
