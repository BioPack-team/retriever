import asyncio
import itertools
import math
import time
from asyncio.tasks import Task
from collections.abc import AsyncGenerator, Hashable

from opentelemetry import trace
from reasoner_pydantic import (
    CURIE,
    QueryGraph,
)

from retriever.lookup.branch import Branch, SuperpositionName
from retriever.lookup.partial import Partial
from retriever.lookup.subquery import mock_subquery
from retriever.lookup.utils import get_subgraph, make_mappings
from retriever.types.general import (
    AdjacencyGraph,
    KAdjacencyGraph,
    LookupArtifacts,
    QEdgeIDMap,
    QueryInfo,
    SuperpositionHop,
)
from retriever.types.trapi import (
    AuxGraphDict,
    AuxGraphID,
    EdgeDict,
    EdgeIdentifier,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeID,
    ResultDict,
)
from retriever.utils.general import merge_iterators
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import initialize_kgraph, update_kgraph

tracer = trace.get_tracer("lookup.execution.tracer")

# TODO:
# Set interpretation
# Subclassing

CompletePathName = str


class QueryGraphExecutor:
    """Handler class for running the QGX algorithm."""

    def __init__(self, qgraph: QueryGraph, query_info: QueryInfo) -> None:
        """Initialize a QueryGraphExecutor, setting up information shared by methods."""
        self.ctx: QueryInfo = query_info
        self.qgraph: QueryGraph = qgraph
        self.job_log: TRAPILogger = TRAPILogger(self.ctx.job_id)

        q_agraph, qedge_map = make_mappings(self.qgraph)
        self.q_agraph: AdjacencyGraph = q_agraph
        self.qedge_map: QEdgeIDMap = qedge_map
        self.qedge_claims: dict[QEdgeID, Branch | None] = dict.fromkeys(
            self.qgraph.edges.keys()
        )

        self.kgraph: KnowledgeGraphDict = initialize_kgraph(self.qgraph)
        self.aux_graphs: dict[AuxGraphID, AuxGraphDict] = {}
        # For more detailed tracking
        # Each key represents an input superposition for a given branch
        # (branch's input curie and current edge)
        self.kedges_by_input: dict[SuperpositionHop, list[EdgeDict]] = {}
        self.k_agraph: KAdjacencyGraph = {
            qedge_id: dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]()
            for qedge_id in self.qgraph.edges
        }

        self.active_superpositions: set[SuperpositionName] = set()
        self.complete_paths: set[CompletePathName] = set()

        # Initialize locks for accessing some of the above
        self.locks: dict[Hashable, asyncio.Lock] = {
            "claim": asyncio.Lock(),
            "hop_check": asyncio.Lock(),
            "partial_sync": asyncio.Lock(),
            "kgraph": asyncio.Lock(),
        }

        self.terminate: bool = False
        self.start_time: float = 0  # replaced on execute start

    @tracer.start_as_current_span("qgx_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute query graph using a quantum-metaphor asynchronous approach.

        Note that this algorithm assumes that between any two nodes, there is at most one edge.
        """
        timeout_task = asyncio.create_task(self.timeout())
        try:
            self.start_time = time.time()
            self.job_log.info(
                f"Starting lookup against Tier {', '.join(str(t) for t in self.ctx.tiers if t > 0)}..."
            )
            starting_branches = await Branch.get_start_branches(
                self.qgraph,
                self.q_agraph,
                self.qedge_map,
                self.qedge_claims,
                self.locks["claim"],
            )
            self.job_log.debug(
                f"Found {len(starting_branches)} starting branches: {', '.join(branch.superposition_name for branch in starting_branches)}"
            )

            partials, reconciled = await self.run_starting_branches(starting_branches)

            self.job_log.debug(
                f"Reconciled {sum(len(parts) for parts in partials.values())} partials into {len(reconciled)} results."
            )

            with tracer.start_as_current_span("convert_results"):
                results = list[ResultDict]()
                for part in reconciled:
                    await asyncio.sleep(0)
                    results.append(await part.as_result(self.k_agraph))

            timeout_task.cancel()
            end_time = time.time()
            duration_ms = math.ceil((end_time - self.start_time) * 1000)
            self.job_log.info(
                "Tier {}: Retrieved {} results / {} nodes / {} edges in {}ms.".format(
                    ", ".join(str(t) for t in self.ctx.tiers if t > 0),
                    len(results),
                    len(self.kgraph["nodes"]),
                    len(self.kgraph["edges"]),
                    duration_ms,
                )
            )

            # TODO: cleanup (subclass, is_set)

            return LookupArtifacts(
                results, self.kgraph, self.aux_graphs, self.job_log.get_logs()
            )
        except Exception:
            self.job_log.exception(
                f"Unhandled exception occured in QGX while processing Tier {', '.join(str(t) for t in self.ctx.tiers)}. See logs for details."
            )
            timeout_task.cancel()
            return LookupArtifacts(
                [], self.kgraph, self.aux_graphs, self.job_log.get_logs(), error=True
            )

    @tracer.start_as_current_span("run_branches")
    async def run_starting_branches(
        self, starting_branches: list[Branch]
    ) -> tuple[dict[SuperpositionName, list[Partial]], list[Partial]]:
        """Run a given set of starting branches and reconcile the partial results they return."""
        partials = {
            branch.superposition_name: list[Partial]() for branch in starting_branches
        }
        reconciled = list[Partial]()

        branch_tasks = [self.execute_branch(branch) for branch in starting_branches]

        async for branch, partial in merge_iterators(*branch_tasks):
            # await asyncio.sleep(0)
            super_name = branch.superposition_name
            partial.node_bindings.append((branch.start_node, branch.curies[0]))
            partials[super_name].append(partial)

            if len(partials) == 1:  # No reconciliation to be done
                reconciled.append(partial)
                # self.job_log.trace(f"RESULT: {hash(partial)}")
                continue

            # Only attempt reconciliation if all branches have representatives
            other_branches = [
                parts for s_name, parts in partials.items() if s_name != super_name
            ]
            if any(len(parts) == 0 for parts in other_branches):
                self.job_log.trace("No other branches ready.")
                continue

            self.job_log.trace("Reconciliation started!")
            # Find valid reconciliations of each branch combined
            for combo in itertools.product(*other_branches):
                reconcile_attempt = partial
                for part in combo:
                    reconcile_attempt = reconcile_attempt.reconcile(part)
                    if reconcile_attempt is None:
                        break
                if reconcile_attempt is not None:
                    reconciled.append(reconcile_attempt)
            self.job_log.trace("Reconciliation finished!")

        return partials, reconciled

    async def execute_branch(
        self, current_branch: Branch
    ) -> AsyncGenerator[tuple[Branch, Partial]]:
        """Recursively execute a query graph, traversing in parallel by curie.

        Yields:
            A tuple of the branch which was executed, and a partial result backtracking on that branch.
        """
        self.job_log.trace(f"{current_branch.superposition_name}")

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
                asyncio.Task[tuple[KnowledgeGraphDict, list[LogEntryDict]]]
                | asyncio.Task[list[Partial]]
            ](subquery_tasks)

            # Each key represents an output superposition for this branch
            # (output curie for this branch and this branch's edge)
            # Each sub dict is then partials by the respective edge their branch executed
            # this allows partials to be branch-reconciled while avoiding
            # reconciling partials of different superpositions on the same branch
            partials = dict[SuperpositionHop, dict[QEdgeID, list[Partial]]]()
            yielded_partials = 0

            # Simultaneously:
            # - Create next steps as edges come back from subqueries
            # - Yield partials from next steps as they complete
            while parallel_tasks:
                timeout = (
                    max(
                        self.ctx.timeout - (time.time() - self.start_time) - 0.5,
                        0,
                    )
                    if self.ctx.timeout >= 0
                    else None
                )
                done, pending = await asyncio.wait(
                    parallel_tasks,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if len(done) == 0 or self.terminate:  # Timed out, cancel subqueries
                    for task in pending:
                        if task.get_name() == "subquery":
                            task.cancel()

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

        return

    @tracer.start_as_current_span("create_subqueries")
    async def get_subquery_tasks(
        self,
        current_branch: Branch,
    ) -> tuple[list[asyncio.Task[tuple[KnowledgeGraphDict, list[LogEntryDict]]]], bool]:
        """Create tasks for the subqueries the current branch will generate.

        Avoids creating duplicate tasks when some superposition is already executing the
        specific hop.
        """
        if current_branch.hop_name in self.kedges_by_input:
            self.job_log.trace(
                f"{current_branch.input_curie}-{current_branch.current_edge}: Already executed, referring to KG..."
            )
            update_kgraph = False
            # If another superposition has already executed this hop, we don't want to
            # do it again, so instead of running a subquery just grab the subgraph represented by this hop
            async with self.locks["kgraph"]:
                subquery_tasks = [
                    asyncio.create_task(
                        get_subgraph(
                            current_branch,
                            current_branch.hop_name,
                            self.kedges_by_input,
                            self.kgraph,
                        ),
                        name="get_subgraph",
                    )
                ]
        else:
            update_kgraph = True
            if self.terminate:
                return [], update_kgraph
            # Simulate finding a variable number of operations to perform
            # TODO: replace this with looking up operations from the metakg
            subquery_tasks = [
                asyncio.create_task(
                    mock_subquery(self.ctx.job_id, current_branch, self.qgraph),
                    name="subquery",
                )
                # for _ in range(random.randint(0, 3))
                # for _ in range(2)
            ]

        return subquery_tasks, update_kgraph

    async def consume_parallel_task(
        self,
        task: asyncio.Task[tuple[KnowledgeGraphDict, list[LogEntryDict]]]
        | asyncio.Task[list[Partial]],
        current_branch: Branch,
        partials: dict[SuperpositionHop, dict[QEdgeID, list[Partial]]],
        update_kgraph: bool,
        new_branch_tasks: list[Task[list[Partial]]],
    ) -> AsyncGenerator[tuple[Branch, Partial]]:
        """Consume either a subquery task or a branch task, handling appropriately."""
        task_result = await task
        if isinstance(
            task_result, tuple
        ):  # Task is a subquery task returning new knowledge
            async for partial in self.consume_subquery_task(
                task_result, current_branch, partials, update_kgraph, new_branch_tasks
            ):
                yield current_branch, partial
        else:  # Task is a branch task returning partials
            new_partials = task_result
            for partial in new_partials:
                yield current_branch, partial
        return

    async def consume_subquery_task(
        self,
        task_result: tuple[KnowledgeGraphDict, list[LogEntryDict]],
        current_branch: Branch,
        partials: dict[SuperpositionHop, dict[QEdgeID, list[Partial]]],
        update_kgraph: bool,
        new_branch_tasks: list[Task[list[Partial]]],
    ) -> AsyncGenerator[Partial]:
        """Consume the result of a subquery task.

        Updates the kgraph and fires off next steps, yielding partials if there are none.
        """
        qedge_id = current_branch.current_edge
        new_kgraph, logs = task_result

        async with self.locks["kgraph"]:
            self.job_log.log_deque.extend(logs)
            await self.update_knowledge(current_branch, new_kgraph, update_kgraph)

        # Ensure "backwards" edges don't cause input curie to propogate
        input_categories = set(
            self.qgraph.nodes[current_branch.input_node].categories or []
        )
        output_categories = set(
            self.qgraph.nodes[current_branch.output_node].categories or []
        )
        qnodes_allow_self_edge = not input_categories.isdisjoint(output_categories)

        # Don't iterate through all kedges unless the QEdge has the potential to
        # cause node self-edges (otherwise input curie can show up as output due to
        # "backwards" edges)
        use_input_for_next = qnodes_allow_self_edge and any(
            edge["subject"] == current_branch.input_curie
            and edge["object"] == current_branch.input_curie
            for edge in new_kgraph["edges"].values()
        )

        next_curies = new_kgraph["nodes"].keys()
        if not use_input_for_next:
            next_curies = (
                curie for curie in next_curies if curie != current_branch.input_curie
            )

        next_steps = await current_branch.get_next_steps(
            next_curies, self.qedge_claims, self.locks["claim"]
        )

        self.job_log.trace(
            f"{current_branch.superposition_name}: found {len(next_steps)} next steps for curies {list(new_kgraph['nodes'].keys())}."
        )

        if len(next_steps) == 0:
            self.job_log.trace(
                f"{current_branch.superposition_name}: Returning {len(new_kgraph['nodes'])} Partial results."
            )
            for edge in new_kgraph["edges"].values():
                if edge["subject"] == current_branch.input_curie:
                    next_hop_curie = edge["object"]
                else:
                    next_hop_curie = edge["subject"]

                partial_binding = qedge_id, current_branch.input_curie, next_hop_curie

                path_name = f"{current_branch.superposition_name[:-2]}{next_hop_curie}"
                async with self.locks["partial_sync"]:
                    if path_name in self.complete_paths:
                        continue  # Prevent yielding duplicate partials
                    self.complete_paths.add(path_name)
                    self.job_log.trace(
                        f"{current_branch.superposition_name[:-2]}{next_hop_curie}]"
                    )
                    yield Partial(
                        [(current_branch.output_node, next_hop_curie)],
                        [partial_binding],
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

    async def update_knowledge(
        self,
        current_branch: Branch,
        new_kgraph: KnowledgeGraphDict,
        do_update_kgraph: bool,
    ) -> None:
        """Update knowledge tracking with a new knowledge graph."""
        qedge_id = current_branch.current_edge

        if do_update_kgraph:
            update_kgraph(self.kgraph, new_kgraph)

        if current_branch.hop_name not in self.kedges_by_input:
            self.kedges_by_input[current_branch.hop_name] = list[EdgeDict]()
        self.kedges_by_input[current_branch.hop_name].extend(
            new_kgraph["edges"].values()
        )

        # Update the k_agraph
        for edge_id, edge in new_kgraph["edges"].items():
            if edge["subject"] == current_branch.input_curie:
                in_node, out_node = edge["subject"], edge["object"]
            else:
                in_node, out_node = edge["object"], edge["subject"]
            if in_node not in self.k_agraph[qedge_id]:
                self.k_agraph[qedge_id][in_node] = dict[CURIE, list[EdgeIdentifier]]()
            if out_node not in self.k_agraph[qedge_id][in_node]:
                self.k_agraph[qedge_id][in_node][out_node] = list[EdgeIdentifier]()

            self.k_agraph[qedge_id][in_node][out_node].append(edge_id)

    @tracer.start_as_current_span("prepare_branch_tasks")
    async def prepare_new_branch_tasks(
        self,
        current_branch: Branch,
        next_steps: list[Branch],
        partials: dict[SuperpositionHop, dict[QEdgeID, list[Partial]]],
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
                partials[partial_key] = dict[QEdgeID, list[Partial]]()
            async with self.locks[partial_key]:
                if next_branch.current_edge not in partials[partial_key]:
                    partials[partial_key][next_branch.current_edge] = list[Partial]()

            current_branch.next_steps.add(next_branch.branch_name)
            if next_branch.branch_name in current_branch.skipped_steps:
                continue

            # Check if superposition has already been created by some other branch
            async with self.locks["hop_check"]:
                if next_branch.superposition_name in self.active_superpositions:
                    continue
                self.active_superpositions.add(next_branch.superposition_name)

            new_branch_tasks.append(
                asyncio.create_task(
                    self.reconcile_branches(
                        current_branch,
                        partials,
                        self.execute_branch(next_branch),
                    )
                )
            )

        return new_branch_tasks

    @tracer.start_as_current_span("reconcile_branches")
    async def reconcile_branches(
        self,
        current_branch: Branch,
        partials: dict[SuperpositionHop, dict[QEdgeID, list[Partial]]],
        branch_task: AsyncGenerator[tuple[Branch, Partial | None]],
    ) -> list[Partial]:
        """Consume the output of a qdfs call, returning reconciled partials."""
        qedge_id = current_branch.current_edge

        task_partials = list[Partial]()
        async for next_branch, partial in branch_task:
            # await asyncio.sleep(0)
            node_bind = current_branch.output_node, next_branch.input_curie
            edge_bind = qedge_id, current_branch.input_curie, next_branch.input_curie

            if partial is None:
                # Branch was made invalid by another branch's edge claim
                self.job_log.trace(
                    f"{current_branch.superposition_name[:-2]}{next_branch.input_curie}]"
                )
                current_branch.skipped_steps.add(next_branch.branch_name)
                task_partials.append(Partial([node_bind], [edge_bind]))
                continue

            partials_key = (next_branch.input_curie, current_branch.current_edge)
            partial.node_bindings.append(node_bind)
            partial.edge_bindings.add(edge_bind)

            async with self.locks[partials_key]:
                partials[partials_key][next_branch.current_edge].append(partial)

            if len(partials) == 1:
                # There's no reconciliation to be done
                task_partials.append(partial)
                continue

            async with self.locks[partials_key]:
                other_branches = [
                    parts
                    for qedge_id, parts in partials[partials_key].items()
                    if qedge_id != next_branch.current_edge
                ]
                if any(len(parts) == 0 for parts in other_branches):
                    continue

                for combo in itertools.product(*other_branches):
                    combined = partial
                    for part in combo:
                        combined = combined.combine(part)
                    task_partials.append(combined)

        return task_partials

    async def timeout(self) -> None:
        """Set work to terminate after the configured timeout."""
        try:
            if self.ctx.timeout < 0:
                return
            await asyncio.sleep(max(self.ctx.timeout - 0.5, 0))
            self.job_log.error("QGX hit timeout, attempting wrapup...")
            self.terminate = True
        except asyncio.CancelledError:
            return
