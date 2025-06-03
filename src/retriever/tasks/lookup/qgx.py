import asyncio
import itertools
import random
from asyncio.tasks import Task
from collections.abc import AsyncGenerator, Hashable
from typing import cast

from opentelemetry import trace
from reasoner_pydantic import (
    CURIE,
    AsyncQuery,
    Attribute,
    AuxiliaryGraphs,
    Edge,
    HashableMapping,
    KnowledgeGraph,
    LogEntry,
    Query,
    QueryGraph,
    Results,
)
from reasoner_pydantic.shared import EdgeIdentifier

from retriever.tasks.lookup.branch import Branch
from retriever.tasks.lookup.partial import Partial
from retriever.tasks.lookup.subquery import mock_subquery
from retriever.tasks.lookup.utils import (
    get_subgraph,
    initialize_kgraph,
    make_mappings,
    merge_iterators,
)
from retriever.type_defs import (
    AdjacencyGraph,
    KAdjacencyGraph,
    QEdgeIDMap,
    SuperpositionHop,
)
from retriever.utils.logs import TRAPILogger

tracer = trace.get_tracer("lookup.execution.tracer")

# TODO:
# Remaining handling to implement in order of priority:
# DONE: Locks to make access safe
# DONE: Turns out broken chain detection is an emergent behavior of the algorithm and can't be made more efficient
# DONE: KG Pruning
# Time limits
# Set interpretation

# PERF: Optimization steps
# There are a few big culprits, and a few smaller additive ones
# The biggest one is Partial.as_result (25% of slowdown)
# But looking at the flamegraph...a bunch of hashing is going on all the time that can probably be reduced
# So another thing that could be done is reducing use of sets and hashing generally
# Also look for data structure use that's inefficient


class QueryGraphExecutor:
    """Handler class for running the QGX algorithm."""

    def __init__(self, query: Query | AsyncQuery, job_id: str) -> None:
        """Initialize a QueryGraphExecutor, setting up information shared by methods."""
        self.query: Query | AsyncQuery = query
        self.job_id: str = job_id
        self.job_log: TRAPILogger = TRAPILogger(job_id)

        if query.message.query_graph is None:  # This should not occur.
            raise ValueError("Cannot execute nonexistent graph.")
        self.qgraph: QueryGraph = query.message.query_graph

        agraph, qedge_map = make_mappings(self.qgraph)
        self.agraph: AdjacencyGraph = agraph
        self.qedge_map: QEdgeIDMap = qedge_map
        self.qedge_claims: dict[str, Branch | None] = dict.fromkeys(
            self.qgraph.edges.keys()
        )

        self.kgraph: KnowledgeGraph = initialize_kgraph(self.qgraph)
        self.aux_graphs: AuxiliaryGraphs = AuxiliaryGraphs()
        # For more detailed tracking
        # Each key represents an input superposition for a given branch
        # (branch's input curie and current edge)
        self.kedges_by_input: dict[SuperpositionHop, list[Edge]] = {}
        self.k_agraph: KAdjacencyGraph = {
            qedge_id: dict[CURIE, dict[CURIE, list[Edge]]]()
            for qedge_id in self.qgraph.edges
        }

        self.active_superpositions: set[str] = set()
        self.complete_paths: set[str] = set()

        # Initialize locks for accessing some of the above
        self.locks: dict[Hashable, asyncio.Lock] = {
            "claim": asyncio.Lock(),
            "hop_check": asyncio.Lock(),
            "partial_sync": asyncio.Lock(),
            "kgraph": asyncio.Lock(),
        }

    async def execute(self) -> tuple[Results, KnowledgeGraph, list[LogEntry]]:
        """Execute query graph using a quantum-metaphor asynchronous approach.

        Note that this algorithm assumes that between any two nodes, there is at most one edge.
        """
        starting_branches = Branch.get_start_branches(
            self.qgraph, self.agraph, self.qedge_map
        )
        self.job_log.debug(
            f"Found {len(starting_branches)} starting branches: {', '.join(branch.superposition_name for branch in starting_branches)}"
        )

        collected_partials = 0
        partials = {
            branch.superposition_name: list[Partial]() for branch in starting_branches
        }
        reconciled = list[Partial]()

        branch_tasks = [self.qdfs(branch) for branch in starting_branches]

        with tracer.start_as_current_span("run_branches"):
            async for branch, partial in merge_iterators(*branch_tasks):
                # await asyncio.sleep(0)
                branch_name = branch.superposition_name
                if partial is None:
                    self.job_log.debug(f"{branch_name}: Starting branch cancelled.")
                    partials.pop(branch_name, None)
                    continue
                partial.node_bindings.append((branch.start_node, branch.curies[0]))
                partials[branch_name].append(partial)
                collected_partials += 1

                if len(partials) == 1:  # No reconciliation to be done
                    reconciled.append(partial)
                    # self.job_log.debug(f"RESULT: {hash(partial)}")
                    continue

                others = itertools.chain(
                    *(
                        parts
                        for other_bname, parts in partials.items()
                        if other_bname != branch_name
                    )
                )
                for other in others:
                    # await asyncio.sleep(0)
                    if combined := partial.reconcile(other):
                        reconciled.append(combined)
                        # self.job_log.debug(f"RESULT: {hash(combined)}")

        self.job_log.debug(
            f"Collected {collected_partials} partials into {sum(len(parts) for parts in partials.values())} distinct partial results. Reconciled {len(reconciled)} results."
        )

        # PERF: Converting results presently takes a decent amount of time
        # This accounts for about 25% of load testing slowdown
        with tracer.start_as_current_span("convert_results"):
            results = Results()
            for part in reconciled:
                await asyncio.sleep(0)
                results.append(await part.as_result(self.k_agraph))

        await self.prune_kg(results)

        self.job_log.info(f"Got {len(results)} results.")

        return results, self.kgraph, self.job_log.get_logs()

    async def qdfs(
        self, current_branch: Branch
    ) -> AsyncGenerator[tuple[Branch, Partial | None]]:
        """Recursively execute a query graph, traversing in parallel by curie."""
        # Check that this branch may proceed (target edge is unclaimed/claimed for this branch)
        if not await current_branch.has_claim(self.qedge_claims, self.locks["claim"]):
            yield current_branch, None
            return

        # self.job_log.debug(f"{current_branch.superposition_name}")

        # Ensure this SuperpositionHop has a lock to work with
        async with self.locks["hop_check"]:
            if current_branch.hop_name not in self.locks:
                self.locks[current_branch.hop_name] = asyncio.Lock()

        # Execute with the given lock so any other superpositions that cause this hop await the first one's completion
        async with self.locks[current_branch.hop_name]:
            subquery_tasks, update_kgraph = await self.get_subquery_tasks(
                current_branch
            )

            branch_tasks = list[asyncio.Task[list[Partial]]]()
            parallel_tasks = list[
                asyncio.Task[tuple[KnowledgeGraph, list[LogEntry]]]
                | asyncio.Task[list[Partial]]
            ](subquery_tasks)

            # Each key represents an output superposition for this branch
            # (output curie for this branch and this branch's edge)
            # Each sub dict is then partials by the respective edge their branch executed
            # this allows partials to be branch-reconciled while avoiding
            # reconciling partials of different superpositions on the same branch
            partials = dict[SuperpositionHop, dict[str, list[Partial]]]()
            yielded_partials = 0

            # Simultaneously:
            # - Create next steps as edges come back from subqueries
            # - Yield partials from next steps as they complete
            while parallel_tasks:
                done, pending = await asyncio.wait(
                    parallel_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                new_branch_tasks = list[asyncio.Task[list[Partial]]]()

                for task in done:
                    # await asyncio.sleep(0)
                    async for branch, partial in self.consume_parallel_task(
                        task,
                        current_branch,
                        partials,
                        update_kgraph,
                        new_branch_tasks,
                    ):
                        yield branch, partial
                        yielded_partials += 1

                branch_tasks.extend(new_branch_tasks)
                parallel_tasks = [*pending, *new_branch_tasks]

            # if len(current_branch.next_steps) > 0 and len(
            #     current_branch.skipped_steps
            # ) == len(current_branch.next_steps):
            #     self.job_log.debug(
            #         f"{current_branch.superposition_name}: All next steps are claimed by other branches. Yielded {yielded_partials} partials."
            #     )
            # elif len(current_branch.next_steps) == 0:
            #     self.job_log.debug(
            #         f"{current_branch.superposition_name}: Found no next steps for any curies. Yielded {yielded_partials} partials."
            #     )

        return

    @tracer.start_as_current_span("create_subqueries")
    async def get_subquery_tasks(
        self,
        current_branch: Branch,
    ) -> tuple[list[asyncio.Task[tuple[KnowledgeGraph, list[LogEntry]]]], bool]:
        """Create tasks for the subqueries the current branch will generate.

        Avoids creating duplicate tasks when some superposition is already executing the
        specific hop.
        """
        if current_branch.hop_name in self.kedges_by_input:
            # self.job_log.debug(
            #     f"{current_branch.input_curie}-{current_branch.current_edge}: Already executed, referring to KG..."
            # )
            update_kgraph = False
            async with self.locks["kgraph"]:
                subquery_tasks = [
                    asyncio.create_task(
                        get_subgraph(
                            current_branch,
                            current_branch.hop_name,
                            self.kedges_by_input,
                            self.kgraph,
                        )
                    )
                    for _ in range(2)
                ]
        else:
            # Simulate finding a variable number of operations to perform
            update_kgraph = True
            subquery_tasks = [
                asyncio.create_task(
                    mock_subquery(self.job_id, current_branch, self.qgraph)
                )
                for _ in range(random.randint(0, 3))
                # for _ in range(2)
            ]

        return subquery_tasks, update_kgraph

    async def consume_parallel_task(
        self,
        task: asyncio.Task[tuple[KnowledgeGraph, list[LogEntry]]]
        | asyncio.Task[list[Partial]],
        current_branch: Branch,
        partials: dict[SuperpositionHop, dict[str, list[Partial]]],
        update_kgraph: bool,
        new_branch_tasks: list[Task[list[Partial]]],
    ) -> AsyncGenerator[tuple[Branch, Partial]]:
        """Consume either a subquery task or a branch task, handling appropriately."""
        qedge_id = current_branch.current_edge
        task_result = await task
        if isinstance(task_result, tuple):
            new_kgraph, logs = task_result

            async with self.locks["kgraph"]:
                self.job_log.log_deque.extend(logs)
                await self.update_knowledge(current_branch, new_kgraph, update_kgraph)

            next_steps = current_branch.get_next_steps(new_kgraph.nodes.keys())
            # self.job_log.debug(
            #     f"{current_branch.superposition_name}: found {len(next_steps)} next steps for curies {list(new_kgraph.nodes.keys())}."
            # )

            if len(next_steps) == 0:
                # self.job_log.debug(
                #     f"{current_branch.superposition_name}: Returning {len(new_kgraph.nodes)} Partial results."
                # )
                for curie in new_kgraph.nodes:
                    # await asyncio.sleep(0)
                    # self.job_log.debug(
                    #     f"{current_branch.superposition_name[:-2]}{curie}]"
                    # )
                    key = qedge_id, current_branch.input_curie, curie

                    path_name = f"{current_branch.superposition_name[:-2]}{curie}"
                    async with self.locks["partial_sync"]:
                        if path_name in self.complete_paths:
                            continue  # Prevent yielding duplicate partials
                        self.complete_paths.add(path_name)
                        yield (
                            current_branch,
                            Partial([(current_branch.output_node, curie)], [key]),
                        )
                return

            # Build partial dict and fire off next branch tasks
            new_branch_tasks.extend(
                await self.prepare_new_branch_tasks(
                    current_branch,
                    next_steps,
                    partials,
                )
            )
        else:
            new_partials = task_result
            for partial in new_partials:
                yield current_branch, partial
        return

    async def update_knowledge(
        self, current_branch: Branch, new_kgraph: KnowledgeGraph, update_kgraph: bool
    ) -> None:
        """Update knowledge tracking with a new knowledge graph."""
        qedge_id = current_branch.current_edge

        if update_kgraph:
            self.kgraph.update(new_kgraph)

        if current_branch.hop_name not in self.kedges_by_input:
            self.kedges_by_input[current_branch.hop_name] = list[Edge]()
        self.kedges_by_input[current_branch.hop_name].extend(new_kgraph.edges.values())

        # Update the k_agraph
        for edge in new_kgraph.edges.values():
            # await asyncio.sleep(0)
            in_node, out_node = (
                (edge.subject, edge.object)
                if not current_branch.reversed
                else (edge.object, edge.subject)
            )
            if in_node not in self.k_agraph[qedge_id]:
                self.k_agraph[qedge_id][in_node] = dict[CURIE, list[Edge]]()
            if out_node not in self.k_agraph[qedge_id][in_node]:
                self.k_agraph[qedge_id][in_node][out_node] = list[Edge]()

            self.k_agraph[qedge_id][in_node][out_node].append(edge)

    @tracer.start_as_current_span("prepare_branch_tasks")
    async def prepare_new_branch_tasks(
        self,
        current_branch: Branch,
        next_steps: list[Branch],
        partials: dict[SuperpositionHop, dict[str, list[Partial]]],
    ) -> list[asyncio.Task[list[Partial]]]:
        """Make new branch tasks given a list of next-step branches.

        Updates tracking information as well.
        """
        new_branch_tasks = list[asyncio.Task[list[Partial]]]()
        for next_branch in next_steps:
            # await asyncio.sleep(0)
            partial_key = (next_branch.input_curie, current_branch.current_edge)

            if partial_key not in partials:
                self.locks[partial_key] = asyncio.Lock()
                partials[partial_key] = dict[str, list[Partial]]()
            async with self.locks[partial_key]:
                if next_branch.current_edge not in partials[partial_key]:
                    partials[partial_key][next_branch.current_edge] = list[Partial]()

            current_branch.next_steps.add(next_branch.branch_name)
            if next_branch.branch_name in current_branch.skipped_steps:
                continue

            # TODO needs a lock probably
            if next_branch.superposition_name in self.active_superpositions:
                continue
            self.active_superpositions.add(next_branch.superposition_name)

            new_branch_tasks.append(
                asyncio.create_task(
                    self.reconcile_branches(
                        current_branch,
                        partials,
                        self.qdfs(next_branch),
                    )
                )
            )

        return new_branch_tasks

    @tracer.start_as_current_span("reconcile_branches")
    async def reconcile_branches(
        self,
        current_branch: Branch,
        partials: dict[SuperpositionHop, dict[str, list[Partial]]],
        branch_task: AsyncGenerator[tuple[Branch, Partial | None]],
    ) -> list[Partial]:
        """Consume the output of a qdfs call, returning reconciled partials."""
        qedge_id = current_branch.current_edge

        task_partials = list[Partial]()
        async for next_branch, partial in branch_task:
            # await asyncio.sleep(0)
            node_bind = (current_branch.output_node, next_branch.input_curie)
            edge_bind = (qedge_id, current_branch.input_curie, next_branch.input_curie)

            if partial is None:
                # Branch was made invalid by another branch's edge claim
                # self.job_log.debug(
                #     f"{current_branch.superposition_name[:-2]}{next_branch.input_curie}]"
                # )
                current_branch.skipped_steps.add(next_branch.branch_name)
                task_partials.append(Partial([node_bind], [edge_bind]))
                continue

            partials_key = (next_branch.input_curie, current_branch.current_edge)
            partial.node_bindings.append(node_bind)
            partial.edge_bindings.add(edge_bind)

            async with self.locks[partials_key]:
                partials[partials_key][next_branch.current_edge].append(partial)

            if len(current_branch.next_edges) == 1:
                # There's no reconciliation to be done
                task_partials.append(partial)
                continue

            async with self.locks[partials_key]:
                others = itertools.chain(
                    *(
                        parts
                        for qedge_id, parts in partials[partials_key].items()
                        if qedge_id != next_branch.current_edge
                    )
                )
                for new, old in itertools.product([partial], others):
                    task_partials.append(new.combine(old))

        return task_partials

    @tracer.start_as_current_span("prune_kg")
    async def prune_kg(self, results: Results) -> None:
        """Use finished results to prune the knowledge graph to only bound knowledge."""
        bound_edges = set[str]()
        bound_nodes = set[str]()
        for result in results:
            # await asyncio.sleep(0)
            for node_binding_set in result.node_bindings.values():
                bound_nodes.update([str(binding.id) for binding in node_binding_set])
            # Only ever one analysis so we can use next(iter())
            for edge_binding_set in next(iter(result.analyses)).edge_bindings.values():
                bound_edges.update([str(binding.id) for binding in edge_binding_set])

        edges_to_check = list(bound_edges)
        while len(edges_to_check) > 0:
            # await asyncio.sleep(0)
            edge = self.kgraph.edges[EdgeIdentifier(edges_to_check.pop())]
            bound_edges.add(str(hash(edge)))
            bound_nodes.add(str(edge.subject))
            bound_nodes.add(str(edge.object))

            aux_graphs = next(
                (
                    attr
                    for attr in (edge.attributes or set[Attribute]())
                    if attr.attribute_type_id == CURIE("biolink:support_graphs")
                ),
                None,
            )
            if aux_graphs is None:
                continue
            for aux_graph_id in cast(list[str], aux_graphs.value):
                edges_to_check.extend(
                    str(edge) for edge in self.aux_graphs[aux_graph_id].edges
                )

        prior_edge_count = len(self.kgraph.edges)
        prior_node_count = len(self.kgraph.nodes)

        self.kgraph.edges = HashableMapping(
            {EdgeIdentifier(edge_id): self.kgraph.edges[EdgeIdentifier(edge_id)] for edge_id in bound_edges}
        )
        self.kgraph.nodes = HashableMapping(
            {CURIE(curie): self.kgraph.nodes[CURIE(curie)] for curie in bound_nodes}
        )

        pruned_edges = prior_edge_count - len(self.kgraph.edges)
        pruned_nodes = prior_node_count - len(self.kgraph.nodes)

        self.job_log.debug(
            f"KG Pruning: {len(self.kgraph.edges)} (-{pruned_edges}) edges and {len(self.kgraph.nodes)} (-{pruned_nodes}) nodes remain."
        )
