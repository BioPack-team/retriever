from typing import get_type_hints

from reasoner_pydantic.shared import KnowledgeType

from retriever.config.openapi import OPENAPI_CONFIG
from retriever.types.trapi import (
    PathfinderQueryGraphDict,
    QEdgeDict,
    QEdgeID,
    QNodeDict,
    QNodeID,
    QueryGraphDict,
)
from retriever.utils import biolink


def validate(
    qg: QueryGraphDict | PathfinderQueryGraphDict,
) -> tuple[list[str], list[str]]:
    """Check that a given query graph is valid.

    Returns:
        A list of warning messages, which do not fail validation but should be logged.
        And a list of messages detailing validation problems.
        If the list is empty, the graph passes validation.
    """
    if "paths" in qg:
        return [], ["Retriever does not support Pathfinder queries."]
    warnings = list[str]()
    problems = dict[str, bool]()  # False means failing
    problems["Query graph must have at least one node"] = len(qg["nodes"].values()) > 0
    problems["Query graph must have at least one edge"] = len(qg["edges"].values()) > 0
    problems["Query graph must have at least one node with an ID"] = any(
        node
        for node in qg["nodes"].values()
        if "ids" in node and len(node["ids"] or []) > 0
    )

    # node_pairs = set[str]()
    for qedge_id, qedge in qg["edges"].items():
        edge_warnings, edge_problems = validate_qedge(qg, qedge_id, qedge)
        problems.update(edge_problems)
        warnings.extend(edge_warnings)

    for qnode_id, qnode in qg["nodes"].items():
        node_warnings, node_problems = validate_qnode(qg, qnode_id, qnode)
        problems.update(node_problems)
        warnings.extend(node_warnings)

    return warnings, [name for name, passed in problems.items() if not passed]


def validate_qedge(
    qg: QueryGraphDict, qedge_id: QEdgeID, qedge: QEdgeDict
) -> tuple[list[str], dict[str, bool]]:
    """Find and return any problems with a given Query Edge.

    Problems in the dictionary marked False are failing.
    """
    problems = dict[str, bool]()

    if qedge["subject"] not in qg["nodes"]:
        problems[
            f"Edge `{qedge_id}` subject `{qedge['subject']}` not defined in query graph."
        ] = False

    if qedge["object"] not in qg["nodes"]:
        problems[
            f"Edge `{qedge_id}` object `{qedge['object']}` not defined in query graph."
        ] = False

    for i, qualifier_constraint in enumerate(qedge.get("qualifier_constraints", [])):
        qualifier_types: set[str] = set()
        for qualifier in qualifier_constraint["qualifier_set"]:
            if qualifier["qualifier_type_id"] in qualifier_types:
                problems[
                    f"Edge `{qedge_id}` qualifier constraint {i} has duplicate qualifier_type_id `{qualifier['qualifier_type_id']}`"
                ] = False
            qualifier_types.add(qualifier["qualifier_type_id"])

    if qedge.get("knowlqedge_type") == KnowledgeType.inferred:
        problems["Retriever does not handle inferred-type queries."] = False

    invalid_predicates = [
        p for p in (qedge.get("predicates", []) or []) if not biolink.is_predicate(p)
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
    known_fields = get_type_hints(QEdgeDict)
    unknown_fields = [field for field in qedge if field not in known_fields]
    if len(unknown_fields) > 0:
        warnings.append(
            f"Edge `{qedge_id}`: skipping unknown fields ({', '.join(unknown_fields)})"
        )

    return warnings, problems


def validate_qnode(
    _qg: QueryGraphDict, qnode_id: QNodeID, qnode: QNodeDict
) -> tuple[list[str], dict[str, bool]]:
    """Find and return any problems with a given Query Node.

    Problems in the dictionary marked False are failing.
    """
    problems = dict[str, bool]()

    if len(qnode.get("ids", []) or []) > OPENAPI_CONFIG.x_trapi.batch_size_limit:
        problems[
            f"Node `{qnode_id}` ID count ({len(qnode.get('ids', []) or [])}) exceeds batch size limit of {OPENAPI_CONFIG.x_trapi.batch_size_limit}"
        ] = False

    invalid_categories = [
        c for c in (qnode.get("categories", []) or []) if not biolink.is_category(c)
    ]
    if len(invalid_categories) > 0:
        problems[f"Node `{qnode_id}` has invalid categories: {invalid_categories}"] = (
            False
        )

    warnings = list[str]()
    known_fields = get_type_hints(QNodeDict)
    unknown_fields = [field for field in qnode if field not in known_fields]
    if len(unknown_fields) > 0:
        warnings.append(
            f"Node `{qnode_id}`: skipping unknown fields ({', '.join(unknown_fields)})"
        )

    return warnings, problems
