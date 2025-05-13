import asyncio
import itertools
import random
from typing import Literal, cast

from reasoner_pydantic import (
    CURIE,
    AsyncQuery,
    Edge,
    HashableSet,
    KnowledgeGraph,
    LogEntry,
    Node,
    QEdge,
    Query,
    QueryGraph,
    Result,
    Results,
)
from reasoner_pydantic.shared import EdgeIdentifier

from retriever.tasks.lookup.branch import Branch
from retriever.tasks.lookup.partial import Partial
from retriever.type_defs import AdjacencyGraph, EdgeIDMap
from retriever.utils.logs import TRAPILogger


def make_mappings(qg: QueryGraph) -> tuple[AdjacencyGraph, EdgeIDMap]:
    """Make an undirected QGraph representation in which edges are presented by their nodes."""
    agraph: AdjacencyGraph = {}
    edge_id_map: EdgeIDMap = {}
    for edge_id, edge in qg.edges.items():
        edge_id_map[edge_id] = edge
        edge_id_map[edge] = edge_id
        if edge.subject not in agraph:
            agraph[edge.subject] = dict[str, QEdge]()
        if edge.object not in agraph:
            agraph[edge.object] = dict[str, QEdge]()
        agraph[edge.subject][edge.object] = edge
        agraph[edge.object][edge.subject] = edge

    return agraph, edge_id_map


async def execute_query_graph(
    query: Query | AsyncQuery, job_id: str
) -> tuple[Results, KnowledgeGraph, list[LogEntry]]:
    """Execute query graph using a quantum-metaphor asynchronous approach.

    Note that this algorithm assumes that between any two nodes, there is at most one edge.
    """
    job_log = TRAPILogger(job_id)
    qgraph = query.message.query_graph
    if qgraph is None:  # This should not occur.
        raise ValueError("Cannot execute nonexistent graph.")

    agraph, edge_id_map = make_mappings(qgraph)
    qedge_claims: dict[str, Branch | None] = dict.fromkeys(qgraph.edges.keys())
    claim_lock = asyncio.Lock()

    kgraph_dict = {
        "nodes": {},
        "edges": {},
    }

    # Add qgraph nodes to kgraph
    for qnode in qgraph.nodes.values():
        if qnode.ids is None:
            continue
        for curie in qnode.ids:
            kgraph_dict["nodes"][curie] = {
                "categories": qnode.categories or [],
                "attributes": [],
            }

    kgraph = KnowledgeGraph.model_validate(kgraph_dict)

    # TODO: async yielding, have to do slightly different stop conditions
    # have to know ahead that branch will stop due to claim or leaf
    # TODO: move as much logic out as possible so this function is clean
    async def qdfs(branch: Branch) -> list[Partial] | Literal[False]:
        edge_id = branch.current_edge

        async with claim_lock:
            if qedge_claims[edge_id] is not None and qedge_claims[edge_id] != branch:
                job_log.debug(
                    f"{branch.superposition_name}: Stopping, edge {edge_id} is claimed by {qedge_claims[edge_id].superposition_name}"
                )
                return False
            qedge_claims[edge_id] = branch

        # Test some random time jitter
        # await asyncio.sleep(random.random())

        if qgraph.nodes[branch.end_node].ids:
            curies = list(qgraph.nodes[branch.end_node].ids or [])
        else:
            curies = [
                CURIE(f"curie:test{random.random()}")
                for i in range(random.randint(1, 3))
            ]
        current_edge = cast(QEdge, edge_id_map[edge_id])
        edges = {
            curie: Edge(
                subject=(
                    branch.curies[-1]
                    if branch.nodes[-1] == current_edge.object
                    else curie
                ),
                predicate=(current_edge.predicates or ["biolink:related_to"])[0],
                object=(
                    curie
                    if branch.nodes[-1] == current_edge.object
                    else branch.curies[-1]
                ),
                sources=HashableSet(),
                attributes=HashableSet(),
            )
            for curie in curies
        }

        # Update KG
        for curie in curies:
            kgraph.nodes[curie] = Node(
                categories=HashableSet(
                    set(qgraph.nodes[branch.end_node].categories or [])
                ),
                attributes=HashableSet(),
            )
        for edge in edges.values():
            kgraph.edges[EdgeIdentifier(str(hash(edge)))] = edge

        next_steps = branch.get_next_steps(curies)
        job_log.debug(
            f"{branch.superposition_name}: Found {len(next_steps)} next step(s)"
        )
        if len(next_steps) == 0:
            return [
                Partial([(branch.nodes[-1], curie)], [(edge_id, edge)])
                for curie, edge in edges.items()
            ]

        partials = dict[CURIE, dict[str, list[Partial]]]()
        for next_branch in next_steps:  # Have to run once ahead to build partials dict
            curie = next_branch.curies[-1]
            next_edge = cast(str, edge_id_map[next_branch.current_edge])

            if curie not in partials:
                partials[curie] = dict[str, list[Partial]]()
            if next_edge not in partials[curie]:
                partials[curie][next_edge] = list[Partial]()
        reconciled_partials = list[Partial]()

        skip_branches = set[int]()
        for next_branch in next_steps:
            curie = next_branch.curies[-1]
            next_edge = cast(str, edge_id_map[next_branch.current_edge])
            next_branch_hash = hash(next_branch)

            if next_branch_hash in skip_branches:
                continue

            new_parts = await qdfs(next_branch)

            if new_parts is False:
                # Branch was made invalid by another branch's edge claim
                skip_branches.add(next_branch_hash)
                continue

            partials[curie][next_edge].extend(new_parts)
            for new in new_parts:
                new.node_bindings.append((branch.end_node, next_branch.curies[-1]))
                new.edge_bindings.append((edge_id, edges[next_branch.curies[-1]]))

            if len(partials[curie]) == 1:  # If there's no reconciliation to be done
                reconciled_partials.extend(new_parts)
                continue
            others = itertools.chain(
                *(parts for edge, parts in partials[curie].items() if edge != next_edge)
            )
            for new, old in itertools.product(new_parts, others):
                if combined := new.combine(old):
                    reconciled_partials.append(combined)

        if len(skip_branches) == len(next_steps):
            return [
                Partial([(branch.nodes[-1], curie)], [(edge_id, edge)])
                for curie, edge in edges.items()
            ]
        return reconciled_partials

        # node_tasks = []
        # superpositions = []

    starting_branches = Branch.get_start_branches(qgraph, agraph, edge_id_map)
    job_log.debug(
        f"Found {len(starting_branches)} starting branches: {', '.join(branch.superposition_name for branch in starting_branches)}"
    )

    partials = {hash(branch): list[Partial]() for branch in starting_branches}
    for branch in starting_branches:
        node_id = branch.start_node
        curie = branch.curies[0]
        edge = cast(str, edge_id_map[branch.current_edge])
        branch_hash = hash(branch)

        parts = await qdfs(branch)
        if parts is False:
            partials.pop(branch_hash, None)
            continue
        for part in parts:
            part.node_bindings.append((node_id, curie))
            partials[branch_hash].append(part)

    job_log.debug(
        f"Got {sum(len(parts) for parts in partials.values())} partial results to reconcile."
    )

    reconciled = list[Partial]()
    for edge, parts in partials.items():
        if len(partials) == 1:  # If there's no reconciliation to be done
            reconciled.extend(parts)
            continue
        others = itertools.chain(
            *(parts for edge_id, parts in partials.items() if edge_id != edge)
        )
        for a, b in itertools.product(parts, others):
            if combined := a.reconcile(b):
                reconciled.append(combined)

    results = Results()
    for raw_result in reconciled:
        result_dict = {
            "node_bindings": {
                qnode: [{"id": curie, "attributes": []}]
                for qnode, curie in raw_result.node_bindings
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        qedge: [{"id": str(hash(edge)), "attributes": []}]
                        for qedge, edge in raw_result.edge_bindings
                    },
                }
            ],
        }
        results.append(Result.model_validate(result_dict))

    for result in results:
        job_log.info(f"Got result {result}")
    # TODO: QDFS should async yield things, this should combine iterators for each call
    # tasks = [qdfs(edge, edge_id) for edge_id, edge in starting_edges]

    # branches = asyncio.gather(tasks)
    # results = reconcile_branches(branches)

    return results, kgraph, job_log.get_logs()
