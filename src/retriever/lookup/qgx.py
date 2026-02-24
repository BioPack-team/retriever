import asyncio
import itertools
import math
import time
from asyncio.tasks import Task
from collections.abc import AsyncGenerator, Hashable, Iterable
from typing import Literal, cast

from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.lookup.branch import (
    Branch,
    BranchID,
    SuperpositionHop,
    SuperpositionID,
)
from retriever.lookup.partial import Partial
from retriever.lookup.subclass import SubclassMapping
from retriever.lookup.subquery import get_subgraph, subquery
from retriever.lookup.utils import make_mappings
from retriever.metadata.optable import OP_TABLE_MANAGER, OperationPlan
from retriever.types.general import (
    AdjacencyGraph,
    KAdjacencyGraph,
    LookupArtifacts,
    QEdgeIDMap,
    QueryInfo,
)
from retriever.types.trapi import (
    CURIE,
    AttributeDict,
    AuxGraphID,
    AuxiliaryGraphDict,
    BiolinkPredicate,
    EdgeDict,
    EdgeIdentifier,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeID,
    QNodeID,
    QueryGraphDict,
    ResultDict,
    RetrievalSourceDict,
)
from retriever.utils import biolink
from retriever.utils.general import EmptyIteratorError, merge_iterators
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import (
    hash_edge,
    hash_hex,
    hash_qualifier_set,
    initialize_kgraph,
    merge_results,
    update_kgraph,
)

tracer = trace.get_tracer("lookup.execution.tracer")

# TODO: Set interpretation

CompletePathName = str

SourcelessEdgeKey = tuple[CURIE, BiolinkPredicate, str, CURIE]
SubclassEdgesByCURIE = dict[tuple[CURIE, CURIE], tuple[EdgeIdentifier, EdgeDict]]
AuxGraphEdgesByConstruct = dict[
    SourcelessEdgeKey, tuple[AuxGraphID, set[EdgeIdentifier]]
]
ConstructEdgesMapping = dict[SourcelessEdgeKey, tuple[EdgeIdentifier, EdgeDict]]


class QueryGraphExecutor:
    """Handler class for running the QGX algorithm."""

    ctx: QueryInfo
    qgraph: QueryGraphDict
    job_log: TRAPILogger

    q_agraph: AdjacencyGraph
    qedge_map: QEdgeIDMap
    qedge_claims: dict[QEdgeID, Branch | None]
    kgraph: KnowledgeGraphDict
    aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict]
    kedges_by_input: dict[SuperpositionHop, list[EdgeDict]]
    k_agraph: KAdjacencyGraph

    active_branches: set[BranchID]
    active_superpositions: set[SuperpositionID]
    dead_superpositions: set[SuperpositionID]
    complete_paths: set[CompletePathName]

    subclass_backmap: dict[CURIE, CURIE]

    locks: dict[Hashable, asyncio.Lock]
    terminate: bool
    start_time: float
    timeout: float

    def __init__(self, qgraph: QueryGraphDict, query_info: QueryInfo) -> None:
        """Initialize a QueryGraphExecutor, setting up information shared by methods."""
        self.ctx = query_info
        self.qgraph = qgraph
        self.job_log = TRAPILogger(self.ctx.job_id)

        q_agraph, qedge_map = make_mappings(self.qgraph)
        self.q_agraph = q_agraph
        self.qedge_map = qedge_map
        self.qedge_claims = {
            QEdgeID(qedge_id): None for qedge_id in self.qgraph["edges"]
        }

        self.kgraph = initialize_kgraph(self.qgraph)
        self.aux_graphs = {}
        # For more detailed tracking
        # Each key represents an input superposition for a given branch
        # (branch's input curie and current edge)
        self.kedges_by_input = {}
        self.k_agraph = {
            QEdgeID(qedge_id): dict[CURIE, dict[CURIE, list[EdgeIdentifier]]]()
            for qedge_id in self.qgraph["edges"]
        }

        self.active_branches = set()
        self.active_superpositions = set()
        self.dead_superpositions = set()
        self.complete_paths = set()

        self.subclass_backmap = {}

        # Initialize locks for accessing some of the above
        self.locks = {
            "claim": asyncio.Lock(),
            "hop_check": asyncio.Lock(),
            "partial_sync": asyncio.Lock(),
            "kgraph": asyncio.Lock(),
        }

        self.terminate = False
        self.start_time = 0  # replaced on execute start

        # Because timeouts are configured on a per-tier basis, we take the longest timeout
        # of the tiers we are accessing. If any of them are -1, we consider the timeout
        # disabled.
        timeouts = [
            timeout
            for tier, timeout in self.ctx.timeout.items()
            if tier in self.ctx.tiers
        ]
        if -1 in timeouts:
            self.timeout = -1
        self.timeout = max(timeouts)

    @tracer.start_as_current_span("qgx_execute")
    async def execute(self) -> LookupArtifacts:
        """Execute query graph using a quantum-metaphor asynchronous approach.

        Note that this algorithm assumes that between any two nodes, there is at most one edge.
        """
        timeout_task = asyncio.create_task(self.start_timeout_clock())
        try:
            self.start_time = time.time()
            self.job_log.info(
                f"Starting lookup against Tier {', '.join(str(t) for t in self.ctx.tiers if t > 0)}..."
            )
            supported, operation_plan = await OP_TABLE_MANAGER.create_operation_plan(
                self.qgraph, {t for t in self.ctx.tiers if t > 0}
            )
            if not supported:
                self.job_log.warning(
                    f"Failed for find operations supporting query edge(s): {list(operation_plan.keys())}"
                )
                return LookupArtifacts(
                    [], self.kgraph, self.aux_graphs, self.job_log.get_logs()
                )

            await self.expand_initial_subclasses()

            starting_branches = await Branch.get_start_branches(
                self.qedge_claims,
                self.locks["claim"],
                (
                    self.qgraph,
                    self.q_agraph,
                    self.qedge_map,
                    cast(
                        OperationPlan, operation_plan
                    ),  # Should never be the alternative
                ),
                self.job_log,
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
                    results.append(part.as_result(self.k_agraph))

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

            # TODO: cleanup (set_interpretation)
            self.solve_subclass_edges(self.kgraph, results, self.aux_graphs)

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

    async def expand_initial_subclasses(self) -> None:
        """Check if any pinned nodes have subclasses and expand them accordingly."""
        for qnode_id, node in self.qgraph["nodes"].items():
            # Verify that no edges connected to this node use subclass_of

            expanded_curies = await self.expand_subclasses(
                qnode_id, (node.get("ids", []) or [])
            )

            node["ids"] = expanded_curies

    async def expand_subclasses(
        self, qnode_id: QNodeID, curies: Iterable[CURIE]
    ) -> list[CURIE]:
        """Given a set of CURIEs, return them and any subclasses they may have."""
        if not CONFIG.job.lookup.implicit_subclassing:
            return list(curies)

        subclass_mapping = await SubclassMapping().get()
        if subclass_mapping is None:
            return list(curies)

        # If the qnode is going into a predicate that can return "subclass_of",
        # Then we can't proceed
        if any(
            any(
                set(edge.get("predicates", []) or []).intersection(
                    biolink.SUBCLASS_SKIP_PREDICATES
                )
                for edge in edges
            )
            for edges in self.q_agraph[qnode_id].values()
        ):
            return list(curies)

        new_curies = set[CURIE](curies)
        for curie in curies:
            descendants = subclass_mapping.get(curie, [])
            if not descendants:
                self.job_log.trace(
                    f"Found no descendants for {curie} on QNode {qnode_id}"
                )
                continue

            self.subclass_backmap.update(dict.fromkeys(descendants, curie))
            new_curies.update(descendants)
            self.job_log.info(
                f"Found {len(descendants)} descendants for {curie} on QNode {qnode_id}: {descendants}"
            )

        return list(new_curies)

    @tracer.start_as_current_span("run_branches")
    async def run_starting_branches(
        self, starting_branches: list[Branch]
    ) -> tuple[dict[BranchID, list[Partial]], list[Partial]]:
        """Run a given set of starting branches and reconcile the partial results they return."""
        partials = {branch.branch_id: list[Partial]() for branch in starting_branches}
        reconciled = list[Partial]()

        branch_tasks = [self.execute_branch(branch) for branch in starting_branches]

        try:
            await self.reconcile_starting_branches(branch_tasks, partials, reconciled)
        except EmptyIteratorError as e:
            # Longest branch matching the starting branch will always be the one that failed
            initial_id = starting_branches[e.iterator].branch_id
            self.job_log.warning(
                f"Branch {Branch.branch_id_to_name(initial_id)} terminated in an unexpected manner."
            )
            self.terminate = True

        return partials, reconciled

    async def reconcile_starting_branches(
        self,
        branch_tasks: list[AsyncGenerator[tuple[Branch, Partial | None]]],
        partials: dict[BranchID, list[Partial]],
        reconciled: list[Partial],
    ) -> None:
        """Iterate over a number of starting branch tasks and reconcile their partials."""
        async for branch, partial in merge_iterators(
            *branch_tasks, raise_on_empty=True
        ):
            if partial is None:  # A branch terminated with nothing
                if self.terminate:  # Termination case, no handling needed
                    self.job_log.debug(
                        f"Branch {branch.superposition_name} terminated due to QGX wrapup."
                    )
                    break

                # Either this is just one of several on the same starting node
                # Meaning we can continue, business as usual
                self.job_log.debug(
                    f"Branch {branch.superposition_name} terminated with no partials."
                )
                self.dead_superpositions.add(branch.superposition_id)

                starting_hop = branch.branch_id[0]
                if any(
                    s_id not in self.dead_superpositions
                    and (s_id[0][0], s_id[0][2]) == starting_hop
                    for s_id in self.active_superpositions
                ):
                    continue

                # ...Or it's the last of several on the starting node
                # Meaning we have to terminate, because there'll be nothing to reconcile to
                matching_branches = [
                    br
                    for br in self.active_branches
                    if str(branch.branch_id) in str(br)
                ]
                branch_lengths = [len(br) for br in matching_branches]
                longest_branch = matching_branches[
                    branch_lengths.index(max(branch_lengths))
                ]
                self.job_log.warning(
                    f"QEdge {longest_branch[-2][1]} found no supporting knowledge. No results can be reconciled. Query terminates."
                )
                self.terminate = True
                break

            # await asyncio.sleep(0)
            branch_id = branch.branch_id
            partial.node_bindings.append((branch.start_node, branch.curies[0]))
            partials[branch_id].append(partial)

            if len(partials) == 1:  # No reconciliation to be done
                reconciled.append(partial)
                # self.job_log.trace(f"RESULT: {hash(partial)}")
                continue

            other_branches = [
                parts for s_name, parts in partials.items() if s_name != branch_id
            ]
            if any(len(parts) == 0 for parts in other_branches):
                self.job_log.trace("No other branches ready.")
                continue

            # Find valid reconciliations of each branch combined
            for combo in itertools.product(*other_branches):
                reconcile_attempt = partial
                for part in combo:
                    reconcile_attempt = reconcile_attempt.reconcile(part)
                    if reconcile_attempt is None:
                        break
                if reconcile_attempt is not None:
                    reconciled.append(reconcile_attempt)

    async def execute_branch(
        self, current_branch: Branch
    ) -> AsyncGenerator[tuple[Branch, Partial | None]]:
        """Recursively execute a query graph, traversing in parallel by curie.

        Yields:
            A tuple of the branch which was executed, and a partial result backtracking on that branch.
        """
        self.job_log.trace(f"{current_branch.superposition_name}")

        # Ensure this SuperpositionHop has a lock to work with
        async with self.locks["hop_check"]:
            if current_branch.hop_id not in self.locks:
                self.locks[current_branch.hop_id] = asyncio.Lock()

        # Execute with the given lock so any other superpositions that cause this hop await the first one's completion
        async with self.locks[current_branch.hop_id]:
            self.active_branches.add(current_branch.branch_id)
            async with self.locks["hop_check"]:
                self.active_superpositions.add(current_branch.superposition_id)
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
                        self.timeout - (time.time() - self.start_time),
                        0,
                    )
                    if self.timeout >= 0
                    else None
                )
                done, pending = await asyncio.wait(
                    parallel_tasks,  # pyright:ignore[reportUnknownArgumentType] pyright being weird
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

            if yielded_partials == 0:
                yield current_branch, None

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
        if current_branch.hop_id in self.kedges_by_input:
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
                            current_branch.hop_id,
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

            subquery_tasks = [
                asyncio.create_task(
                    subquery(self.ctx.job_id, current_branch, self.qgraph),
                    name="subquery",
                )
                # for operation in current_branch.operations
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
            self.qgraph["nodes"][current_branch.input_node].get("categories", []) or []
        )
        output_categories = set(
            self.qgraph["nodes"][current_branch.output_node].get("categories", []) or []
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

        expanded_next_curies = await self.expand_subclasses(
            current_branch.output_node, next_curies
        )

        next_steps = await current_branch.get_next_steps(
            expanded_next_curies, self.qedge_claims, self.locks["claim"]
        )

        self.job_log.trace(
            f"{current_branch.superposition_name}: found {len(next_steps)} next steps for curies {list(next_curies)}."
        )

        if len(next_steps) == 0:
            self.job_log.trace(
                f"{current_branch.superposition_name}: Returning {len(list(next_curies))} Partial results."
            )
            for edge in new_kgraph["edges"].values():
                if edge["subject"] == current_branch.input_curie:
                    next_hop_curie = edge["object"]
                else:
                    next_hop_curie = edge["subject"]

                partial_binding = qedge_id, current_branch.input_curie, next_hop_curie

                path_name = f"{current_branch.superposition_name}{next_hop_curie}"
                async with self.locks["partial_sync"]:
                    if path_name in self.complete_paths:
                        continue  # Prevent yielding duplicate partials
                    self.complete_paths.add(path_name)
                    self.job_log.trace(
                        f"{current_branch.superposition_name}{next_hop_curie}]"
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

        if current_branch.hop_id not in self.kedges_by_input:
            self.kedges_by_input[current_branch.hop_id] = list[EdgeDict]()
        self.kedges_by_input[current_branch.hop_id].extend(new_kgraph["edges"].values())

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
            partial_key: SuperpositionHop = (
                next_branch.input_node,
                next_branch.input_curie,
                current_branch.current_edge,
            )

            if partial_key not in partials:
                self.locks[partial_key] = asyncio.Lock()
                partials[partial_key] = dict[QEdgeID, list[Partial]]()
            async with self.locks[partial_key]:
                if next_branch.current_edge not in partials[partial_key]:
                    partials[partial_key][next_branch.current_edge] = list[Partial]()

            # Check if superposition has already been created by some other branch
            async with self.locks["hop_check"]:
                if next_branch.superposition_id in self.active_superpositions:
                    continue
                self.active_superpositions.add(next_branch.superposition_id)

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
                # All superpositions on next branch failed
                # Not necessarily a broken chain since other superpositions may be running
                continue

            partials_key: SuperpositionHop = (
                next_branch.input_node,
                next_branch.input_curie,
                current_branch.current_edge,
            )
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

    async def start_timeout_clock(self) -> None:
        """Set work to terminate after the configured timeout."""
        try:
            self.job_log.info(
                f"QGX timeout is {'disabled' if self.timeout < 0 else f'{self.timeout}s'}."
            )
            if self.timeout < 0:
                return
            await asyncio.sleep(self.timeout)
            self.job_log.error("QGX hit timeout, attempting wrapup...")
            self.terminate = True
        except asyncio.CancelledError:
            return

    def create_subclass_edge(
        self, parent: CURIE, descendant: CURIE
    ) -> tuple[EdgeIdentifier, EdgeDict]:
        """Create a subclass edge given the parent and its descendant."""
        edge = EdgeDict(
            predicate="biolink:subclass_of",
            subject=descendant,
            object=parent,
            sources=[
                RetrievalSourceDict(
                    resource_id="infores:ubergraph",
                    resource_role="primary_knowledge_source",
                ),
                RetrievalSourceDict(
                    resource_id=CONFIG.tier1.backend_infores,
                    resource_role="aggregator_knowledge_source",
                    upstream_resource_ids=["infores:ubergraph"],
                ),
            ],
            attributes=[
                AttributeDict(
                    attribute_type_id="biolink:knowledge_level",
                    value="knowledge_assertion",
                ),
                AttributeDict(
                    attribute_type_id="biolink:agent_type",
                    value="manual_agent",
                ),
            ],
        )

        edge_hash = hash_hex(hash_edge(edge))

        return edge_hash, edge

    def build_intermediate_support_graph(
        self,
        edge_id: EdgeIdentifier,
        edge: EdgeDict,
        subclass_edges: dict[tuple[CURIE, CURIE], tuple[EdgeIdentifier, EdgeDict]],
    ) -> tuple[SourcelessEdgeKey, set[EdgeIdentifier] | None]:
        """Create a key for the pattern of edge to be replaced, and a support graph for it."""
        sbj_subclass = edge["subject"] in self.subclass_backmap and edge["subject"]
        obj_subclass = edge["object"] in self.subclass_backmap and edge["object"]

        edge_key = (
            self.subclass_backmap[sbj_subclass] if sbj_subclass else edge["subject"],
            edge["predicate"],
            hash_hex(hash_qualifier_set(edge.get("qualifiers", []) or [])),
            self.subclass_backmap[obj_subclass] if obj_subclass else edge["object"],
        )

        if not (sbj_subclass or obj_subclass):
            return edge_key, None

        support_graph = set[EdgeIdentifier]((edge_id,))
        for subclass in list[CURIE | Literal[False]]((sbj_subclass, obj_subclass)):
            if not subclass:
                continue

            if (self.subclass_backmap[subclass], subclass) not in subclass_edges:
                subclass_edge_hash, subclass_edge = self.create_subclass_edge(
                    self.subclass_backmap[subclass], subclass
                )
                subclass_edges[self.subclass_backmap[subclass], subclass] = (
                    subclass_edge_hash,
                    subclass_edge,
                )
            else:
                subclass_edge_hash = subclass_edges[
                    (self.subclass_backmap[subclass], subclass)
                ][0]
            support_graph.add(subclass_edge_hash)

        return edge_key, support_graph

    def build_subclass_construct_edge(
        self, edge_key: SourcelessEdgeKey, edge: EdgeDict
    ) -> EdgeDict:
        """Build a Retriever-constructed edge which asserts the subclass-driven knowledge."""
        return EdgeDict(
            subject=edge_key[0],
            object=edge_key[3],
            predicate=edge["predicate"],
            qualifiers=edge.get("qualifiers", []) or [],
            # BUG: this breaks 2.0-clarified attribute constraint binding rules
            # Would have to make a new construct for each edge, rather than aggregate
            attributes=[
                AttributeDict(
                    attribute_type_id="biolink:support_graphs",
                    value=[f"support_{'_'.join(edge_key)}_via_subclass"],
                ),
                AttributeDict(
                    attribute_type_id="biolink:knowledge_level",
                    value="logical_entailment",
                ),
                AttributeDict(
                    attribute_type_id="biolink:agent_type",
                    value="automated_agent",
                ),
            ],
            sources=[
                RetrievalSourceDict(
                    resource_id="infores:retriever",
                    resource_role="primary_knowledge_source",
                ),
            ],
        )

    def insert_constructs(
        self,
        results: list[ResultDict],
        aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
        edges_to_fix: dict[EdgeIdentifier, SourcelessEdgeKey],
        construct_edges: ConstructEdgesMapping,
    ) -> None:
        """Replace uses of subclassed knowledge edges with their associated constructs.

        This way all instances refer to the support graph containing the subclass edge
        and knowledge.
        """
        # Replace edges with constructs in aux graphs
        for aux_graph in aux_graphs.values():
            aux_graph["edges"] = [
                edge_id
                if edge_id not in edges_to_fix
                else construct_edges[edges_to_fix[edge_id]][0]
                for edge_id in aux_graph["edges"]
            ]

        # Replace edges and nodes in results
        merged_results = dict[int, ResultDict]()
        for result in results:
            for node_bindings in result["node_bindings"].values():
                for binding in node_bindings:
                    if binding["id"] in self.subclass_backmap:
                        binding["id"] = self.subclass_backmap[binding["id"]]

            for analysis in result["analyses"]:
                if "edge_bindings" not in analysis:
                    continue
                for edge_bindings in analysis["edge_bindings"].values():
                    for binding in edge_bindings:
                        if binding["id"] in edges_to_fix:
                            binding["id"] = construct_edges[
                                edges_to_fix[binding["id"]]
                            ][0]

            # Merge the result
            merge_results(merged_results, [result])
        results.clear()
        results.extend(merged_results.values())

    def add_new_knowledge(
        self,
        kg: KnowledgeGraphDict,
        aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
        subclass_edges: SubclassEdgesByCURIE,
        support_graphs: AuxGraphEdgesByConstruct,
        construct_edges: ConstructEdgesMapping,
    ) -> None:
        """Update the kg/aux with the new format information."""
        # Merge in new edges and aux graphs
        kg["edges"].update(dict(subclass_edges.values()))
        kg["edges"].update(dict(construct_edges.values()))

        aux_graphs.update(
            {
                support_graph_id: AuxiliaryGraphDict(
                    edges=list(support_edges), attributes=[]
                )
                for support_graph_id, support_edges in support_graphs.values()
            }
        )

    def solve_subclass_edges(
        self,
        kg: KnowledgeGraphDict,
        results: list[ResultDict],
        aux_graphs: dict[AuxGraphID, AuxiliaryGraphDict],
    ) -> None:
        """Given the subclass mapping, fix the kg/results/aux to use correct subclass structure.

        WARNING: This implementation is specific to Tier 1/2 use.
        """
        self.job_log.debug("Fixing implicit subclass-derived knowledge format...")
        # Keyed to parent, descendant
        subclass_edges = SubclassEdgesByCURIE()

        # Map original edges to non-subclassed sbj/obj
        edges_to_fix = dict[EdgeIdentifier, SourcelessEdgeKey]()
        # Map non-subclassed sbj/obj to support graph for merging
        support_graphs = AuxGraphEdgesByConstruct()
        # Map original edges to their construct replacements
        construct_edges = ConstructEdgesMapping()

        for edge_id, edge in kg["edges"].items():
            edge_key, support_graph = self.build_intermediate_support_graph(
                edge_id, edge, subclass_edges
            )
            if support_graph is None:  # Edge doesn't rely on subclassing
                continue

            # Update overall support graphs and edge tracking
            support_graph_id = f"support_{'_'.join(edge_key)}_via_subclass"
            edges_to_fix[edge_id] = edge_key
            if edge_key not in support_graphs:
                support_graphs[edge_key] = support_graph_id, set()
            support_graphs[edge_key][1].update(support_graph)

            # Don't build redundant construct edges
            if edge_key in construct_edges:
                continue

            construct_edges[edge_key] = (
                f"{'_'.join(edge_key)}_via_subclass",
                self.build_subclass_construct_edge(edge_key, edge),
            )

        self.job_log.debug(
            f"Found and reformated dependents for {len(edges_to_fix)} edges."
        )

        self.insert_constructs(results, aux_graphs, edges_to_fix, construct_edges)
        self.add_new_knowledge(
            kg, aux_graphs, subclass_edges, support_graphs, construct_edges
        )
