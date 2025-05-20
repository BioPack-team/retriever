import asyncio
import itertools
from collections.abc import AsyncGenerator

from reasoner_pydantic import (
    CURIE,
    AsyncQuery,
    Edge,
    KnowledgeGraph,
    LogEntry,
    Query,
    Result,
    Results,
)

from retriever.tasks.lookup.branch import Branch
from retriever.tasks.lookup.partial import Partial
from retriever.tasks.lookup.subquery import mock_subquery
from retriever.tasks.lookup.utils import (
    get_subgraph,
    initialize_kgraph,
    make_mappings,
    merge_iterators,
)
from retriever.type_defs import SuperpositionHop
from retriever.utils.logs import TRAPILogger


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
    locks = {
        "claim": asyncio.Lock(),
        "start": asyncio.Lock(),
    }

    # Initialize Knowledge Graph
    kgraph = initialize_kgraph(qgraph)

    # For more detailed tracking
    # Each key represents an input superposition for a given branch
    # (branch's input curie and current edge)
    kedges_by_input = dict[SuperpositionHop, list[Edge]]()
    kedges_agraph = dict[tuple[str, CURIE, CURIE], list[Edge]]()

    # TODO for basically anything being accessed above here, we need a lock

    # TODO: Error handling in the merge_iterators. Currently errors cause forever-execution
    # TODO: broken chain handling
    # TODO: move as much logic out as possible so this function is clean
    async def qdfs(
        current_branch: Branch,
    ) -> AsyncGenerator[tuple[Branch, Partial | None]]:
        qedge_id = current_branch.current_edge

        # Check that this branch may proceed (target edge is unclaimed/claimed for this branch)
        if not await current_branch.has_claim(qedge_claims, locks["claim"]):
            yield current_branch, None
            return

        kedge_in_key = (
            current_branch.input_curie,
            current_branch.current_edge,
        )

        async with locks["start"]:
            if kedge_in_key in kedges_by_input:
                job_log.debug(
                    f"{current_branch.input_curie}-{current_branch.current_edge} already executed, skipping..."
                )
                update_kgraph = False
                subquery_tasks = [
                    asyncio.create_task(get_subgraph(current_branch, kedge_in_key, kedges_by_input, kgraph))
                    for _ in range(2)
                ]
            else:
                # Simulate finding a variable number of operations to perform
                update_kgraph = True
                subquery_tasks = [
                    asyncio.create_task(mock_subquery(current_branch, qgraph, edge_id_map))
                    # for _ in range(random.randint(0, 3))
                    for _ in range(2)
                ]

        branch_tasks = list[AsyncGenerator[tuple[Branch, Partial | None]]]()

        next_branches = set[int]()
        skip_branches = set[int]()
        yielded_partials = 0
        # Each key represents an output superposition for this branch
        # (output curie for this branch and this branch's edge)
        # Each sub dict is then partials by the respective edge their branch executed
        # this allows partials to be branch-reconciled while avoiding
        # reconciling partials of different superpositions on the same branch
        partials = dict[SuperpositionHop, dict[str, list[Partial]]]()
        edges_this_hop = dict[CURIE, list[Edge]]()

        # Create next steps as edges come back from subqueries
        for task in asyncio.as_completed(subquery_tasks):
            new_kgraph = await task
            if len(new_kgraph.edges) == 0:
                # TODO: one part of broken chain handling
                pass

            if update_kgraph:
                kgraph.update(new_kgraph)

            if kedge_in_key not in kedges_by_input:
                kedges_by_input[kedge_in_key] = list[Edge]()

            kedges_by_input[kedge_in_key].extend(new_kgraph.edges.values())
            for edge in new_kgraph.edges.values():
                key = qedge_id, edge.subject, edge.object
                if key not in kedges_agraph:
                    kedges_agraph[key] = list[Edge]()
                kedges_agraph[key].append(edge)

            next_steps = current_branch.get_next_steps(new_kgraph.nodes.keys())
            job_log.debug(
                f"{current_branch.superposition_name}: found {len(next_steps)} next steps for curies {list(new_kgraph.nodes.keys())}."
            )

            if len(next_steps) == 0:
                job_log.debug(
                    f"{current_branch.superposition_name}: found no next steps for curies {list(new_kgraph.nodes.keys())}. Returning {len(new_kgraph.nodes)} Partial results."
                )
                for curie in new_kgraph.nodes:
                    key = qedge_id, current_branch.input_curie, curie
                    yield (
                        current_branch,
                        Partial([(current_branch.output_node, curie)], [key]),
                    )
                    yielded_partials += 1
                continue

            # Build partial dict and fire off next branch tasks
            for next_branch in next_steps:
                partial_key = (next_branch.input_curie, current_branch.current_edge)

                if partial_key not in partials:
                    partials[partial_key] = dict[str, list[Partial]]()
                if next_branch.current_edge not in partials[partial_key]:
                    partials[partial_key][next_branch.current_edge] = list[Partial]()

                next_branch_hash = hash(next_branch)
                next_branches.add(next_branch_hash)
                if next_branch_hash in skip_branches:
                    continue

                branch_tasks.append(qdfs(next_branch))

        if len(next_branches) == 0:
            job_log.debug(
                f"{current_branch.superposition_name}: Found no next steps for any curies. {yielded_partials} partials have been returned."
            )
            return

        # PERF: Next branch tasks currently only fire after all current subquery tasks complete
        # We want them happening simultaneously.
        # Probably fixable just by making them into tasks?

        # Get new partial results from branch tasks and reconcile on the fly
        async for next_branch, partial in merge_iterators(*branch_tasks):
            if partial is None:
                # Branch was made invalid by another branch's edge claim
                skip_branches.add(hash(partial))
                continue

            partials_key = (next_branch.input_curie, current_branch.current_edge)
            partial.node_bindings.append(
                (current_branch.output_node, next_branch.input_curie)
            )
            partial.edge_bindings.add(
                (qedge_id, current_branch.input_curie, next_branch.input_curie)
            )
            partials[partials_key][next_branch.current_edge].append(partial)

            if (
                len(partials[partials_key]) == 1
            ):  # If there's no reconciliation to be done
                yield current_branch, partial
                yielded_partials += 1
                continue

            others = itertools.chain(
                *(
                    parts
                    for edge, parts in partials[partials_key].items()
                    if edge != next_branch.current_edge
                )
            )
            for new, old in itertools.product([partial], others):
                yield current_branch, new.combine(old)
                yielded_partials += 1

        if len(skip_branches) == len(next_branches):
            for curie in edges_this_hop:
                key = qedge_id, current_branch.input_curie, curie
                yield (
                    current_branch,
                    Partial(
                        [(current_branch.output_node, curie)],
                        [key],
                    ),
                )
                yielded_partials += 1
            job_log.debug(
                f"{current_branch.superposition_name}: All next steps are claimed by other branches. Returning {yielded_partials} partial results."
            )
        return

    # ^ def qdfs()

    starting_branches = Branch.get_start_branches(qgraph, agraph, edge_id_map)
    job_log.debug(
        f"Found {len(starting_branches)} starting branches: {', '.join(branch.superposition_name for branch in starting_branches)}"
    )

    partials = {
        branch.superposition_name: list[Partial]() for branch in starting_branches
    }
    reconciled = set[Partial]()
    branch_tasks = list[AsyncGenerator[tuple[Branch, Partial | None]]]()
    for branch in starting_branches:
        branch_tasks.append(qdfs(branch))

    async for branch, partial in merge_iterators(*branch_tasks):
        branch_name = branch.superposition_name
        if partial is None:
            job_log.debug(f"{branch_name}: Starting branch cancelled.")
            partials.pop(branch_name, None)
            continue
        partial.node_bindings.append((branch.start_node, branch.curies[0]))
        partials[branch_name].append(partial)
        others = itertools.chain(
            *(
                parts
                for other_bname, parts in partials.items()
                if other_bname != branch_name
            )
        )
        for other in others:
            if combined := partial.reconcile(other):
                reconciled.add(combined)

    if len(partials) == 1:  # No reconciliation to be done
        reconciled = list(itertools.chain(*partials.values()))

    job_log.info(
        f"Reconciled {sum(len(parts) for parts in partials.values())} partial results into {len(reconciled)} results."
    )
    if len(reconciled) == 0:
        job_log.info("Failed to reconcile any results out of partials.")

    results = Results()
    for filled_partial in reconciled:
        result_dict = {
            "node_bindings": {
                qnode: [{"id": curie, "attributes": []}]
                for qnode, curie in filled_partial.node_bindings
            },
            "analyses": [
                {
                    "resource_id": "infores:retriever",
                    "edge_bindings": {
                        kedge_key[0]: [
                            {"id": str(hash(kedge)), "attributes": []}
                            for kedge in kedges_agraph[kedge_key]
                        ]
                        for kedge_key in filled_partial.edge_bindings
                    },
                }
            ],
        }
        results.append(Result.model_validate(result_dict))

    job_log.info(f"Got {len(results)} results.")

    return results, kgraph, job_log.get_logs()
