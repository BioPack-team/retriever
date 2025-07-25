from reasoner_pydantic import QueryGraph
from reasoner_pydantic.qgraph import PathfinderQueryGraph
from reasoner_pydantic.shared import KnowledgeType


def validate(qg: QueryGraph | PathfinderQueryGraph) -> list[str]:
    """Check that a given query graph is valid.

    Returns:
        A list of messages detailing validation problems.
        If the list is empty, the graph passes validation.
    """
    if isinstance(qg, PathfinderQueryGraph):
        return ["Retriever does not support Pathfinder queries."]
    problems: dict[str, bool] = {}
    problems["Query graph must have at least one node"] = len(qg.nodes.values()) > 0
    problems["Query graph must have at least one edge"] = len(qg.edges.values()) > 0
    problems["Query graph must have at least one node with an ID"] = any(
        node for node in qg.nodes.values() if node.ids is not None and len(node.ids) > 0
    )

    # node_pairs = set[str]()
    for edge_id, edge in qg.edges.items():
        if edge.subject not in qg.nodes:
            problems[
                f"Edge `{edge_id}` subject `{edge.subject}` not defined in query graph."
            ] = False
        if edge.object not in qg.nodes:
            problems[
                f"Edge `{edge_id}` object `{edge.object}` not defined in query graph."
            ] = False

        for i, qualifier_constraint in enumerate(edge.qualifier_constraints):
            qualifier_types: set[str] = set()
            for qualifier in qualifier_constraint.qualifier_set:
                if qualifier.qualifier_type_id in qualifier_types:
                    problems[
                        f"Edge `{edge_id}` qualifier constraint {i} has duplicate qualifier_type_id `{qualifier.qualifier_type_id}`"
                    ] = False
                qualifier_types.add(qualifier.qualifier_type_id)

        if edge.knowledge_type == KnowledgeType.inferred:
            problems["Retriever does not handle inferred-type queries."] = False

        # if (
        #     f"{edge.subject}-{edge.object}" in node_pairs
        #     or f"{edge.object}-{edge.subject}" in node_pairs
        # ):
        #     problems["Duplicate edges not allowed."] = False
        # node_pairs.add(f"{edge.subject}-{edge.object}")

    return [name for name, passed in problems.items() if not passed]
