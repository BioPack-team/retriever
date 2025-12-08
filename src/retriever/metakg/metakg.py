import asyncio
import itertools
from typing import NamedTuple

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
    QualifierTypeID,
    QueryGraphDict,
)
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils.biolink import expand
from retriever.utils.redis import OP_TABLE_KEY, OP_TABLE_UPDATE_CHANNEL, REDIS_CLIENT
from retriever.utils.trapi import (
    hash_meta_attribute,
    meta_qualifier_meets_constraints,
)

tracer = trace.get_tracer("lookup.execution.tracer")

METAKG_GET_ATTEMPTS = 3

OperationPlan = dict[QEdgeID, list[Operation]]


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


class MetaKGManager:
    """Utility class that keeps an up-to-date metakg."""

    def __init__(self, leader: bool = False) -> None:
        """Initialize a MetaKGManager instance."""
        self.task: asyncio.Task[None] | None = None
        self._operation_table: OperationTable | None = None
        self.update_lock: asyncio.Lock = asyncio.Lock()
        self.is_leader: bool = leader

    async def store_operation_table(self, metakg: OperationTable) -> None:
        """Update the stored MetaKG."""
        metakg_json = ormsgpack.packb(
            {
                "operations_flat": [
                    op._asdict() for op in metakg.operations_flat.values()
                ],
                "nodes": {cat: node._asdict() for cat, node in metakg.nodes.items()},
            }
        )

        await REDIS_CLIENT.set(OP_TABLE_KEY, metakg_json, compress=True)
        await REDIS_CLIENT.publish(OP_TABLE_UPDATE_CHANNEL, 1)

    async def retrieve_stored_operation_table(self) -> OperationTable | None:
        """Retrieve the stored MetaKG."""
        stored = await REDIS_CLIENT.get(OP_TABLE_KEY, compressed=True)
        if stored is None:
            return None
        metakg_json = ormsgpack.unpackb(stored)

        operations_sorted = SortedOperations()
        operations_flat = FlatOperations()

        for op_dict in metakg_json["operations_flat"]:
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
                spo: OperationNode(**node) for spo, node in metakg_json["nodes"].items()
            },
        )

    async def build_operation_table(self) -> None:
        """Build Retriever's internal MetaKG and store it to Redis."""
        if CONFIG.instance_idx != 0:
            return

        logger.info("Building Operation Table...")

        operations_flat = FlatOperations()
        operations_sorted = SortedOperations()
        nodes = dict[BiolinkEntity, OperationNode]()

        driver_ops = [
            await tier_manager.get_driver(tier).get_operations() for tier in range(0, 2)
        ]

        for new_operations, new_nodes in driver_ops:
            for op in new_operations:
                operations_flat[op.hash] = op
                if op.subject not in operations_sorted:
                    operations_sorted[op.subject] = {}
                if op.predicate not in operations_sorted[op.subject]:
                    operations_sorted[op.subject][op.predicate] = {}
                if op.object not in operations_sorted[op.subject][op.predicate]:
                    operations_sorted[op.subject][op.predicate][op.object] = []
                operations_sorted[op.subject][op.predicate][op.object].append(op)
            for entity, node in new_nodes.items():
                if entity in nodes:  # Have to merge nodes
                    # APIs won't overlap so just pull in info from new API
                    nodes[entity].prefixes.update(node.prefixes)
                    nodes[entity].attributes.update(node.attributes)
                    continue
                nodes[entity] = node

        async with self.update_lock:
            self._operation_table = OperationTable(
                operations_sorted, operations_flat, nodes
            )

        await self.store_operation_table(self._operation_table)
        logger.success(
            f"Built Operation Table containing {len(operations_flat)} operations / {len(nodes)} nodes."
        )

    async def periodic_build_op_table(self) -> None:
        """Periodically rebuild the operation table."""
        while True:
            try:
                await self.build_operation_table()
                await asyncio.sleep(CONFIG.job.metakg.build_time)
            except (ValueError, asyncio.CancelledError):
                break

    async def pull_op_table(self, _message: str) -> None:
        """Start a subscriber that updates the local operation table."""
        logger.info("Pulling OpTable...")
        async with self.update_lock:
            self._operation_table = await self.retrieve_stored_operation_table()
        logger.success("In-memory OpTable updated.")

    async def initialize(self) -> None:
        """Start the appropriate tasks for a given process."""
        if self.is_leader:
            await self.build_operation_table()
            if CONFIG.job.metakg.build_time > -1:
                self.task = asyncio.create_task(
                    self.periodic_build_op_table(), name="build_op_table_task"
                )
        else:
            await self.pull_op_table("")
            await REDIS_CLIENT.subscribe(OP_TABLE_UPDATE_CHANNEL, self.pull_op_table)

    async def get_op_table(self, retries: int = 0) -> OperationTable:
        """Return the currently-stored Operation Table."""
        async with self.update_lock:
            op_table = self._operation_table
        if op_table is None:
            if retries >= 3:  # noqa: PLR2004
                raise ValueError("Failed to retrieve a built OpTable!")
            await self.pull_op_table("")
            return await self.get_op_table(retries + 1)
        return op_table

    async def wrapup(self) -> None:
        """Cancel running tasks so connections can close."""
        try:
            await REDIS_CLIENT.unsubscribe(OP_TABLE_UPDATE_CHANNEL, self.pull_op_table)
        except Exception:
            logger.exception("Exception occurred stopping OpTable task.")

    async def find_operations(
        self, edge: QEdgeDict, qgraph: QueryGraphDict, tiers: set[TierNumber]
    ) -> list[Operation]:
        """Find a list of operations that match a given Branch."""
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
        for obj_cat in output_categories:
            for table in object_tables:
                if op_list := table.get(BiolinkEntity(obj_cat)):
                    # TODO: filtering by attribute constraints?
                    operations.extend(
                        op
                        for op in op_list
                        if op.tier in tiers
                        and meta_qualifier_meets_constraints(
                            op.qualifiers, edge.get("qualifier_constraints", [])
                        )
                    )

        return operations

    @tracer.start_as_current_span("operation_plan")
    async def create_operation_plan(
        self, qgraph: QueryGraphDict, tiers: set[TierNumber]
    ) -> OperationPlan | list[QEdgeID]:
        """Obtain a list of supporting operations for each edge in the query graph.

        Returns None if any QEdge is unsupported by the current operation table.
        """
        plan = OperationPlan()
        unsupported_qedges = list[QEdgeID]()
        for qedge_id, qedge in qgraph["edges"].items():
            operations = await self.find_operations(qedge, qgraph, tiers)
            if len(operations) == 0:
                unsupported_qedges.append(QEdgeID(qedge_id))
            plan[QEdgeID(qedge_id)] = operations

        if len(unsupported_qedges) > 0:
            return unsupported_qedges
        return plan


METAKG_MANAGER = MetaKGManager()


async def build_edges(
    op_table: OperationTable,
    tiers: tuple[TierNumber, ...],
) -> tuple[
    dict[str, MetaEdgeDict],
    dict[str, dict[str, set[str]]],
    dict[str, dict[int, MetaAttributeDict]],
    set[BiolinkEntity],
]:
    """Build merged TRAPI MetaEdges from the operation table."""
    edges = dict[str, MetaEdgeDict]()
    edge_qualifiers = dict[str, dict[str, set[str]]]()
    edge_attributes = dict[str, dict[int, MetaAttributeDict]]()
    mentioned_nodes = set[BiolinkEntity]()
    for op in op_table.operations_flat.values():
        if op.tier not in tiers:
            continue

        sbj, obj, pred = op.subject, op.object, op.predicate
        mentioned_nodes.update((sbj, obj))

        spo = f"{sbj} {pred} {obj}"
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


async def get_trapi_metakg(tiers: tuple[TierNumber, ...]) -> MetaKnowledgeGraphDict:
    """Convert an OperationTable to a TRAPI MetaKG dict.

    Because it depends on METAKG_MANAGER, it can't be used with the lead manager.
    This shouldn't be a problem because the lead manager isn't used to answer API calls.
    """
    op_table = await METAKG_MANAGER.get_op_table()
    edges, edge_qualifiers, edge_attributes, mentioned_nodes = await build_edges(
        op_table, tiers
    )
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

    for category, node in op_table.nodes.items():
        if category not in mentioned_nodes:
            continue
        attributes = {
            hash_meta_attribute(attr): attr
            for attr in itertools.chain(*node.attributes.values())
        }
        nodes[category] = MetaNodeDict(
            id_prefixes=list(set(itertools.chain(*node.prefixes.values()))),
            attributes=list(attributes.values()),
        )

    return MetaKnowledgeGraphDict(nodes=nodes, edges=list(edges.values()))
