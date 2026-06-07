import asyncio
import contextlib
import itertools
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import NamedTuple, override

import ormsgpack
from loguru import logger
from opentelemetry import trace

from retriever.config.general import CONFIG
from retriever.data_tiers import tier_manager
from retriever.types.dingo import DINGOMetadata
from retriever.types.metakg import (
    FlatOperations,
    Operation,
    OperationNode,
    OperationTable,
    SortedOperations,
)
from retriever.types.trapi import (
    BiolinkEntity,
    BiolinkPredicate,
    MetaAttributeDict,
    MetaEdgeDict,
    MetaKnowledgeGraphDict,
    MetaNodeDict,
    MetaQualifierDict,
    QEdgeDict,
    QEdgeID,
    QNodeID,
    QualifierTypeID,
    QueryGraphDict,
)
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import biolink
from retriever.utils.biolink import expand
from retriever.utils.general import AsyncDaemon
from retriever.utils.redis import (
    OP_TABLE_KEY,
    OP_TABLE_META_KEY,
    OP_TABLE_UPDATE_CHANNEL,
    TIER_RECOVERED_CHANNEL,
    RedisClient,
)
from retriever.utils.trapi import (
    hash_meta_attribute,
    meta_qualifier_meets_constraints,
)

tracer = trace.get_tracer("lookup.execution.tracer")
REDIS_CLIENT = RedisClient()

METAKG_GET_ATTEMPTS = 3

OperationPlan = dict[QEdgeID, list[Operation]]

SPO = tuple[BiolinkEntity, BiolinkPredicate, BiolinkEntity]


class DINGOMetaKGInfo(NamedTuple):
    """Basic info about a given MetaKG resource."""

    metadata: DINGOMetadata
    tier: TierNumber
    infores: str


class TRAPIMetaKGInfo(NamedTuple):
    """Basic info about a given MetaKG resource."""

    metadata: MetaKnowledgeGraphDict
    tier: TierNumber
    infores: str


class QueryNotTraversable(Exception):
    """An exception that implies the edge is not supported by a MetaEdge."""


class UnsupportedConstraint(Exception):
    """An exception that implies the edge's constraints are not totally supported."""

    unmet: list[str]

    def __init__(self, unmet: Iterable[str], *args: object) -> None:
        """Initialize an instance."""
        super().__init__(*args)
        self.unmet = list(unmet)


class OpTableManager(AsyncDaemon):
    """Utility class that keeps an up-to-date OperationTable."""

    _operation_table: OperationTable | None = None
    update_lock: asyncio.Lock
    _refresh_lock: asyncio.Lock
    _pending_refresh: bool = False
    is_leader: bool = False

    def __init__(self) -> None:
        """Initialize without leader role; call `promote_to_leader()` to flip the flag."""
        self.update_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._pending_refresh = False
        super().__init__()

    def promote_to_leader(self) -> None:
        """Flip this instance to leader mode. Must be called before `initialize()`."""
        self.is_leader = True

    @override
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        tasks = list[Callable[[], Coroutine[None, None, None]]]()
        if self.is_leader and CONFIG.job.metakg.build_time > -1:
            tasks.append(self.periodic_build_op_table)
        return tasks

    @override
    async def initialize(self) -> None:
        """Start the appropriate tasks for a given process."""
        if self.is_leader:
            # Register hooks before the initial refresh so a startup
            # against a down dependency still recovers later.
            REDIS_CLIENT.on_recover(self.refresh)
            for tier in range(0, 2):
                tier_manager.get_driver(tier).on_recover(self.refresh)
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.subscribe(
                    TIER_RECOVERED_CHANNEL, self._on_remote_tier_recover
                )
            try:
                await self.refresh()
            except Exception:
                logger.exception(
                    "Initial OpTable refresh failed; on_recover will retry once dependencies are back."
                )
        else:
            with contextlib.suppress(Exception):
                await self.pull_op_table("")
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.subscribe(
                    OP_TABLE_UPDATE_CHANNEL, self.pull_op_table
                )
            REDIS_CLIENT.on_recover(self._on_redis_recover)
            for tier_idx in range(0, 2):
                driver = tier_manager.get_driver(tier_idx)
                driver.on_recover(self._on_tier_recover)
                # Tell the leader so it rebuilds without waiting on its periodic ping.
                driver.on_recover(self._make_remote_publisher(tier_idx))
        return await super().initialize()

    def _make_remote_publisher(self, tier: int) -> Callable[[], Awaitable[None]]:
        """Build a tier-specific on_recover hook that broadcasts to Redis."""

        async def _publish() -> None:
            if not REDIS_CLIENT.up:
                return
            try:
                await REDIS_CLIENT.publish(TIER_RECOVERED_CHANNEL, str(tier))
            except Exception:
                logger.debug(f"Failed to publish tier {tier} recovery notice to Redis.")

        return _publish

    async def _on_remote_tier_recover(self, _message: str) -> None:
        """Leader-side subscriber callback for cross-process tier recovery."""
        await self.refresh()

    async def refresh(self) -> None:
        """Rebuild and publish the OpTable; concurrent calls collapse to one trailing rebuild."""
        self._pending_refresh = True
        if self._refresh_lock.locked():
            logger.debug(
                "OpTable rebuild already in progress; trailing rebuild queued."
            )
            return
        async with self._refresh_lock:
            while self._pending_refresh:
                self._pending_refresh = False
                await self.build_operation_table()

    async def _on_redis_recover(self) -> None:
        """Worker hook: re-pull the published OpTable on Redis recovery."""
        try:
            await self.pull_op_table("")
        except Exception:
            logger.exception("Failed to re-pull OpTable on Redis recovery.")

    async def _on_tier_recover(self) -> None:
        """Worker hook: locally rebuild on tier recovery, but only while Redis is down."""
        if REDIS_CLIENT.up:
            return
        try:
            await self.degraded_local_build()
        except Exception:
            logger.exception("Local OpTable rebuild on tier recovery failed.")

    async def degraded_local_build(self) -> None:
        """Worker-side local OpTable build when Redis is unavailable; not published."""
        logger.info(
            "Building local OpTable from available tiers (Redis unavailable)..."
        )
        op_table = await self._collect_tier_ops(bypass_cache=True)
        async with self.update_lock:
            if REDIS_CLIENT.up:
                logger.debug(
                    "Redis recovered mid-local-build; discarding local OpTable."
                )
                return
            self._operation_table = op_table
        logger.success(
            f"Local OpTable built with {len(op_table.operations_flat)} operations / {len(op_table.nodes)} nodes."
        )

    @override
    async def wrapup(self) -> None:
        """Cancel running tasks so connections can close."""
        if self.is_leader:
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.unsubscribe(
                    TIER_RECOVERED_CHANNEL, self._on_remote_tier_recover
                )
        else:
            with contextlib.suppress(Exception):
                await REDIS_CLIENT.unsubscribe(
                    OP_TABLE_UPDATE_CHANNEL, self.pull_op_table
                )
        return await super().wrapup()

    async def store_operation_table(self, op_table: OperationTable) -> None:
        """Update the stored OpTable."""
        op_table_json = ormsgpack.packb(
            {
                "operations_flat": [
                    op._asdict() for op in op_table.operations_flat.values()
                ],
                "nodes": {
                    cat: {
                        str(tier): node._asdict() for tier, node in tier_nodes.items()
                    }
                    for cat, tier_nodes in op_table.nodes.items()
                },
            }
        )

        await REDIS_CLIENT.set(OP_TABLE_KEY, op_table_json, compress=True)
        await REDIS_CLIENT.write_freshness(
            OP_TABLE_META_KEY,
            count=len(op_table.operations_flat),
        )
        await REDIS_CLIENT.publish(OP_TABLE_UPDATE_CHANNEL, 1)

    async def retrieve_stored_operation_table(self) -> OperationTable | None:
        """Retrieve the stored OpTable."""
        stored = await REDIS_CLIENT.get(OP_TABLE_KEY, compressed=True)
        if stored is None:
            return None
        op_table_json = ormsgpack.unpackb(stored)

        operations_sorted = SortedOperations()
        operations_flat = FlatOperations()

        for op_dict in op_table_json["operations_flat"]:
            op = Operation(**op_dict)
            operations_flat[op.hash] = op
            if op.subject not in operations_sorted:
                operations_sorted[op.subject] = {}
            if op.predicate not in operations_sorted[op.subject]:
                operations_sorted[op.subject][op.predicate] = {}
            if op.object not in operations_sorted[op.subject][op.predicate]:
                operations_sorted[op.subject][op.predicate][op.object] = []
            operations_sorted[op.subject][op.predicate][op.object].append(op)

        return OperationTable(
            operations_sorted=operations_sorted,
            operations_flat=operations_flat,
            nodes={
                category: {
                    int(tier): OperationNode(**node)
                    for tier, node in tier_nodes.items()
                }
                for category, tier_nodes in op_table_json["nodes"].items()
            },
        )

    def merge_operations(
        self,
        operations_flat: FlatOperations,
        operations_sorted: SortedOperations,
        new_operations: list[Operation],
    ) -> None:
        """Merge new operations into the existing operations."""
        for op in new_operations:
            if attributes := op.attributes:
                for attr in attributes:
                    attr["constraint_use"] = True
                    attr["constraint_name"] = biolink.rmprefix(
                        attr["attribute_type_id"]
                    )
            operations_flat[op.hash] = op
            if op.subject not in operations_sorted:
                operations_sorted[op.subject] = {}
            if op.predicate not in operations_sorted[op.subject]:
                operations_sorted[op.subject][op.predicate] = {}
            if op.object not in operations_sorted[op.subject][op.predicate]:
                operations_sorted[op.subject][op.predicate][op.object] = []
            operations_sorted[op.subject][op.predicate][op.object].append(op)

    def merge_nodes(
        self,
        nodes: dict[BiolinkEntity, dict[TierNumber, OperationNode]],
        new_nodes: dict[BiolinkEntity, OperationNode],
        tier: TierNumber,
    ) -> None:
        """Merge new nodes into the existing nodes."""
        for entity, node in new_nodes.items():
            if entity not in nodes:
                nodes[entity] = {}
            if tier not in nodes[entity]:
                nodes[entity][tier] = node
            # Merge nodes
            # APIs won't overlap so just pull in info from new API
            nodes[entity][tier].prefixes.update(node.prefixes)
            nodes[entity][tier].attributes.update(node.attributes)

        for tier_nodes in nodes.values():
            for node in tier_nodes.values():
                for attr in itertools.chain(*node.attributes.values()):
                    attr["constraint_use"] = True
                    attr["constraint_name"] = biolink.rmprefix(
                        attr["attribute_type_id"]
                    )

    async def _collect_tier_ops(self, *, bypass_cache: bool = False) -> OperationTable:
        """Collect operations from all implemented tiers concurrently.

        Per-tier failures are logged and skipped so a single tier outage
        doesn't waste the work of the others. Raises `ValueError` only
        when no tier contributed - callers should treat that as "preserve
        the previous OpTable" rather than overwrite with an empty one.

        `bypass_cache=True` propagates to each driver so periodic rebuilds
        pull fresh upstream metadata; drivers fall back to their own
        cache when the live fetch fails.
        """
        results = await asyncio.gather(
            *(
                tier_manager.get_driver(tier).get_operations(bypass_cache=bypass_cache)
                for tier in range(0, 2)
            ),
            return_exceptions=True,
        )

        operations_flat = FlatOperations()
        operations_sorted = SortedOperations()
        nodes = dict[BiolinkEntity, dict[TierNumber, OperationNode]]()
        succeeded = 0
        for tier, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"OpTable build: Tier {tier} get_operations failed; skipping. Error: {result!r}"
                )
                continue
            new_operations, new_nodes = result
            self.merge_operations(operations_flat, operations_sorted, new_operations)
            self.merge_nodes(nodes, new_nodes, tier)
            succeeded += 1

        if succeeded == 0:
            raise ValueError("No tier drivers succeeded; preserving previous OpTable.")
        return OperationTable(operations_sorted, operations_flat, nodes)

    async def build_operation_table(self) -> None:
        """Build Retriever's internal OperationTable and store it to Redis."""
        if CONFIG.instance_idx != 0:
            return

        logger.info("Building Operation Table...")
        op_table = await self._collect_tier_ops(bypass_cache=True)
        async with self.update_lock:
            self._operation_table = op_table

        await self.store_operation_table(self._operation_table)
        logger.success(
            f"Built Operation Table containing {len(op_table.operations_flat)} operations / {len(op_table.nodes)} nodes."
        )
        # The leader never reads _operation_table back - it only exists to push to
        # Redis. Drop the reference so the snapshot doesn't sit in process memory.
        if self.is_leader:
            async with self.update_lock:
                self._operation_table = None

    async def periodic_build_op_table(self) -> None:
        """Periodically rebuild the operation table; build failures log and retry next interval."""
        try:
            while True:
                try:
                    await self.build_operation_table()
                except ValueError:
                    logger.warning(
                        "OpTable rebuild had no successful tiers; will retry next interval."
                    )
                except Exception:
                    logger.exception(
                        "OpTable rebuild failed; will retry next interval."
                    )
                await asyncio.sleep(CONFIG.job.metakg.build_time)
        except asyncio.CancelledError:
            return

    async def pull_op_table(self, _message: str) -> None:
        """Start a subscriber that updates the local operation table."""
        logger.info("Pulling OpTable...")
        async with self.update_lock:
            self._operation_table = await self.retrieve_stored_operation_table()
        logger.success("In-memory OpTable updated.")

    async def get_op_table(self) -> OperationTable:
        """Return the currently-stored Operation Table; worker-builds locally if Redis is down."""
        # Phase 1: try pulling from Redis up to 3 times.
        # Phase 2: build then pull up to 3 more times.
        # Worker fallback (Redis down): build locally and re-check.
        for phase in range(2):
            for _ in range(3):
                async with self.update_lock:
                    op_table = self._operation_table
                if op_table is not None:
                    return op_table
                if not REDIS_CLIENT.up and not self.is_leader:
                    # Worker can't pull the published copy; build from
                    # available tiers and re-check.
                    await self.degraded_local_build()
                    continue
                if phase == 1:
                    await self.build_operation_table()
                await self.pull_op_table("")
        raise ValueError("Failed to retrieve or build a valid OpTable!")

    async def find_operations(
        self, edge: QEdgeDict, qgraph: QueryGraphDict, tier: TierNumber
    ) -> list[Operation]:
        """Find a list of operations that match a given Branch.

        Raises either QueryNotTraversable or UnsupportedConstraint if no appropriate
        operations could be found.
        """
        input_node = qgraph["nodes"][edge["subject"]]
        output_node = qgraph["nodes"][edge["object"]]

        input_categories = expand(
            set(
                input_node.get("categories", ["biolink:NamedThing"])
                or ["biolink:NamedThing"]
            )
        )
        output_categories = expand(
            set(
                output_node.get("categories", ["biolink:NamedThing"])
                or ["biolink:NamedThing"]
            )
        )
        predicates = expand(set(edge.get("predicates") or ["biolink:related_to"]))

        op_table = await self.get_op_table()

        predicate_tables = [
            op_table.operations_sorted[BiolinkEntity(sbj_cat)]
            for sbj_cat in input_categories
            if BiolinkEntity(sbj_cat) in op_table.operations_sorted
        ]
        object_tables: list[dict[BiolinkEntity, list[Operation]]] = []
        for predicate in predicates:
            object_tables.extend(
                table[BiolinkPredicate(predicate)]
                for table in predicate_tables
                if BiolinkPredicate(predicate) in table
            )
        operations = list[Operation]()

        unmet_constraints = defaultdict[str, int](int)
        for obj_cat in output_categories:
            for table in object_tables:
                op_list = table.get(BiolinkEntity(obj_cat))
                if op_list is None:
                    continue
                for op in op_list:
                    if op.tier != tier or not meta_qualifier_meets_constraints(
                        op.qualifiers, edge.get("qualifier_constraints", [])
                    ):
                        continue
                    op_attr_types = {
                        mattr["attribute_type_id"]
                        for mattr in (op.attributes or [])
                        if mattr.get("constraint_use", False)
                    }
                    attr_constraints_met = True
                    for constr in edge.get("attribute_constraints", []) or []:
                        if constr["id"] not in op_attr_types:
                            unmet_constraints[constr["name"]] += 1
                            attr_constraints_met = False
                    if attr_constraints_met:
                        operations.append(op)

        if len(operations) == 0:
            if len(unmet_constraints) > 0:
                raise UnsupportedConstraint(unmet=unmet_constraints.keys())
            else:
                raise QueryNotTraversable
        return operations

    @tracer.start_as_current_span("operation_plan")
    async def create_operation_plan(
        self, qgraph: QueryGraphDict, tier: TierNumber
    ) -> tuple[
        bool, OperationPlan | dict[QEdgeID, UnsupportedConstraint | QueryNotTraversable]
    ]:
        """Obtain a list of supporting operations for each edge in the query graph.

        If any qedge is unsupported, instead returns a dict of unsupported edges and the relevant error code.
        """
        plan = OperationPlan()
        unsupported_qedges = dict[
            QEdgeID, UnsupportedConstraint | QueryNotTraversable
        ]()
        for qedge_id, qedge in qgraph["edges"].items():
            operations = []
            try:
                operations = await self.find_operations(qedge, qgraph, tier)
            except (UnsupportedConstraint, QueryNotTraversable) as e:
                unsupported_qedges[qedge_id] = e
            plan[QEdgeID(qedge_id)] = operations

        if len(unsupported_qedges) > 0:
            return False, unsupported_qedges
        return True, plan

    async def qnodes_supported(
        self, qgraph: QueryGraphDict, tier: TierNumber
    ) -> None | dict[QNodeID, UnsupportedConstraint]:
        """Check if any nodes contain unsupported constraints, returning a dictionary of any that are unsupported."""
        unmet_nodes = defaultdict[QNodeID, set[str]](set)
        op_table = await self.get_op_table()
        nodes_met = dict.fromkeys(qgraph["nodes"], False)
        for qnode_id, node in qgraph["nodes"].items():
            constraints = node.get("constraints", []) or []
            if len(constraints) == 0:
                nodes_met[qnode_id] = True
                continue
            categories = expand(
                set(
                    node.get("categories", ["biolink:NamedThing"])
                    or ["biolink:NamedThing"]
                )
            )
            for category in categories:
                op_tier_nodes = op_table.nodes.get(category)
                if op_tier_nodes is None:
                    continue
                for supported_tier, op_node in op_tier_nodes.items():
                    if supported_tier != tier:
                        continue

                    available_attrs = itertools.chain(
                        *(
                            (
                                attr["attribute_type_id"]
                                for attr in attrs
                                if attr.get("constraint_use", False)
                            )
                            for attrs in op_node.attributes.values()
                        )
                    )
                    met = True
                    for constr in constraints:
                        if constr["id"] not in available_attrs:
                            unmet_nodes[qnode_id].add(constr["name"])
                            met = False
                    if met:
                        nodes_met[qnode_id] = True
                        break
                if nodes_met[qnode_id]:
                    break

        if all(nodes_met.values()):
            return None
        return {
            qnode_id: UnsupportedConstraint(unmet=unmet)
            for qnode_id, unmet in unmet_nodes.items()
        }

    async def build_edges(
        self,
        op_table: OperationTable,
        tier: TierNumber | None,
    ) -> tuple[
        dict[SPO, MetaEdgeDict],
        dict[SPO, dict[str, set[str]]],
        dict[SPO, dict[int, MetaAttributeDict]],
        set[BiolinkEntity],
    ]:
        """Build merged TRAPI MetaEdges from the operation table."""
        edges = dict[SPO, MetaEdgeDict]()
        edge_qualifiers = dict[SPO, dict[str, set[str]]]()
        edge_attributes = dict[SPO, dict[int, MetaAttributeDict]]()
        mentioned_nodes = set[BiolinkEntity]()
        for op in op_table.operations_flat.values():
            if tier is not None and op.tier != tier:
                continue

            sbj, obj, pred = op.subject, op.object, op.predicate
            mentioned_nodes.update((sbj, obj))

            spo = (sbj, pred, obj)
            if spo in edges:
                meta_edge = edges[spo]
                qualifiers = edge_qualifiers[spo]
                attributes = edge_attributes[spo]
            else:
                meta_edge = MetaEdgeDict(
                    subject=sbj, predicate=pred, object=obj, knowledge_types=["lookup"]
                )
                qualifiers = dict[str, set[str]]()
                attributes = dict[int, MetaAttributeDict]()

            # Merge qualifiers
            if op.qualifiers is not None:
                for qual_type, values in op.qualifiers.items():
                    if qual_type not in qualifiers:
                        qualifiers[qual_type] = set[str]()
                    qualifiers[qual_type].update(values)

            # Merge attributes
            if op.attributes is not None:
                attributes.update(
                    {hash_meta_attribute(attr): attr for attr in op.attributes}
                )

            if spo not in edges:
                edges[spo] = meta_edge
                edge_qualifiers[spo] = qualifiers
                edge_attributes[spo] = attributes

        return edges, edge_qualifiers, edge_attributes, mentioned_nodes

    async def get_trapi_metakg(self, tier: TierNumber | None) -> MetaKnowledgeGraphDict:
        """Convert an OperationTable to a TRAPI MetaKG dict.

        Because it depends on OP_TABLE_MANAGER, it can't be used with the lead manager.
        This shouldn't be a problem because the lead manager isn't used to answer API calls.
        """
        op_table = await self.get_op_table()
        (
            edges,
            edge_qualifiers,
            edge_attributes,
            mentioned_nodes,
        ) = await self.build_edges(op_table, tier)
        nodes = dict[BiolinkEntity, MetaNodeDict]()

        for spo, edge in edges.items():
            qualifiers = list[MetaQualifierDict]()
            for qual_type, values in edge_qualifiers[spo].items():
                qualifier = MetaQualifierDict(
                    qualifier_type_id=QualifierTypeID(qual_type),
                )
                if len(values):
                    qualifier["applicable_values"] = list(values)
                qualifiers.append(qualifier)
            if len(qualifiers):
                edge["qualifiers"] = qualifiers
            if len(edge_attributes[spo]):
                edge["attributes"] = list(edge_attributes[spo].values())

        for category, tier_nodes in op_table.nodes.items():
            if category not in mentioned_nodes:
                continue
            id_prefixes = set[str]()
            attributes = dict[int, MetaAttributeDict]()
            for supported_tier, node in tier_nodes.items():
                if supported_tier != tier:
                    continue
                id_prefixes.update(itertools.chain(*node.prefixes.values()))
                attributes.update(
                    {
                        hash_meta_attribute(attr): attr
                        for attr in itertools.chain(*node.attributes.values())
                    }
                )
            nodes[category] = MetaNodeDict(
                id_prefixes=list(id_prefixes),
                attributes=list(attributes.values()),
            )

        return MetaKnowledgeGraphDict(nodes=nodes, edges=list(edges.values()))
