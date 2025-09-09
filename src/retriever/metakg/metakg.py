import asyncio
import itertools
from pathlib import Path
from typing import NamedTuple

import aiofiles
import orjson
from loguru import logger
from reasoner_pydantic import QEdge, QueryGraph

from retriever.config.general import CONFIG
from retriever.types.metakg import Operation, OperationNode, OperationTable
from retriever.types.trapi import (
    BiolinkCategory,
    MetaAttributeDict,
    MetaEdgeDict,
    MetaKnowledgeGraphDict,
    MetaNodeDict,
    MetaQualifierDict,
    QEdgeID,
)
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils.biolink import expand
from retriever.utils.redis import METAKG_UPDATE_CHANNEL, REDIS_CLIENT
from retriever.utils.trapi import hash_meta_attribute, meta_qualifier_meets_constraints

METAKG_GET_ATTEMPTS = 3

OperationPlan = dict[QEdgeID, list[Operation]]


class TRAPIMetaKGInfo(NamedTuple):
    """Basic info about a given MetaKG resource."""

    metakg: MetaKnowledgeGraphDict
    tier: TierNumber
    infores: str


class MetaKGManager:
    """Utility class that keeps an up-to-date metakg."""

    def __init__(self, leader: bool = False) -> None:
        """Initialize a MetaKGManager instance."""
        self.task: asyncio.Task[None] | None = None
        self._metakg: OperationTable | None = None
        self.metakg_lock: asyncio.Lock = asyncio.Lock()
        self.is_leader: bool = leader

    def parse_trapi_metakg(
        self,
        metakg_info: TRAPIMetaKGInfo,
        operations: dict[str, list[Operation]],
        nodes: dict[BiolinkCategory, OperationNode],
    ) -> None:
        """Parse a TRAPI MetaKG to build operations."""
        metakg, tier, infores = metakg_info
        for edge in metakg["edges"]:
            edge_dict = MetaEdgeDict(**edge)
            spo = (
                f"{edge_dict['subject']} {edge_dict['predicate']} {edge_dict['object']}"
            )
            if spo not in operations:
                operations[spo] = list[Operation]()
            operations[spo].append(
                Operation(
                    subject=edge_dict["subject"],
                    predicate=edge_dict["predicate"],
                    object=edge_dict["object"],
                    api=infores,
                    tier=tier,
                    attributes=edge_dict.get("attributes"),
                    qualifiers={
                        qualifier["qualifier_type_id"]: qualifier["applicable_values"]
                        for qualifier in edge_dict.get("qualifiers", [])
                    },
                )
            )
        for category, node in metakg["nodes"].items():
            node_dict = MetaNodeDict(**node)
            nodes[category] = OperationNode(
                prefixes={infores: node_dict.get("id_prefixes", [])},
                attributes={infores: node_dict.get("attributes", [])},
            )

    async def build_metakg(self) -> None:
        """Build Retriever's internal MetaKG and store it to Redis."""
        if CONFIG.instance_idx != 0:
            return

        logger.info("Building MetaKG...")

        operations = dict[str, list[Operation]]()
        nodes = dict[BiolinkCategory, OperationNode]()

        metakg_files = {
            0: {
                "infores:automat-robokop": Path("data/robokop-metakg.json"),
            },
            1: {
                "infores:rtx-kg2": Path("data/rtx-kg2-metakg.json"),
            },
        }

        for tier, sources in metakg_files.items():
            for infores, path in sources.items():
                async with aiofiles.open(path) as file:
                    trapi_metakg = MetaKnowledgeGraphDict(
                        orjson.loads(await file.read())
                    )
                self.parse_trapi_metakg(
                    TRAPIMetaKGInfo(trapi_metakg, tier, infores),
                    operations,
                    nodes,
                )
                logger.success(f"Parsed {infores} as a Tier {tier} resource.")

        async with self.metakg_lock:
            self._metakg = OperationTable(operations, nodes)

        await REDIS_CLIENT.update_metakg(self._metakg)
        logger.success(
            f"Built MetaKG containing {sum(len(ops) for ops in operations.values())} operations / {len(nodes)} nodes."
        )

    async def periodic_build_metakg(self) -> None:
        """Periodically rebuild the metakg."""
        while True:
            try:
                await self.build_metakg()
                await asyncio.sleep(CONFIG.job.metakg.build_time)
            except (ValueError, asyncio.CancelledError):
                break

    async def pull_metakg(self, _message: str) -> None:
        """Start a subscriber that updates the local metakg."""
        logger.info("Pulling MetaKG...")
        async with self.metakg_lock:
            self._metakg = await REDIS_CLIENT.get_metakg()
        logger.success("In-memory MetaKG updated.")

    async def initialize(self) -> None:
        """Start the appropriate tasks for a given process."""
        if self.is_leader:
            await self.build_metakg()
            if CONFIG.job.metakg.build_time > -1:
                self.task = asyncio.create_task(
                    self.periodic_build_metakg(), name="build_metakg_task"
                )
        else:
            await self.pull_metakg("")
            await REDIS_CLIENT.subscribe(METAKG_UPDATE_CHANNEL, self.pull_metakg)

    async def get_metakg(self, retries: int = 0) -> OperationTable:
        """Return the currently-stored MetaKG."""
        async with self.metakg_lock:
            metakg = self._metakg
        if metakg is None:
            if retries >= 3:  # noqa: PLR2004
                raise ValueError("Failed to retrieve a built metakg!")
            await self.pull_metakg("")
            return await self.get_metakg(retries + 1)
        return metakg

    async def wrapup(self) -> None:
        """Cancel running tasks so connections can close."""
        try:
            await REDIS_CLIENT.unsubscribe(METAKG_UPDATE_CHANNEL, self.pull_metakg)
        except Exception:
            logger.exception("Exception occurred stopping MetaKG task.")

    async def find_operations(
        self, edge: QEdge, qgraph: QueryGraph, tiers: set[TierNumber]
    ) -> list[Operation]:
        """Find a list of operations that match a given Branch."""
        input_node = qgraph.nodes[edge.subject]
        output_node = qgraph.nodes[edge.object]

        input_categories = expand(set(input_node.categories or ["biolink:NamedThing"]))
        output_categories = expand(
            set(output_node.categories or ["biolink:NamedThing"])
        )
        predicates = expand(set(edge.predicates or ["biolink:related_to"]))

        spo_combos = (
            f"{s} {p} {o}"
            for s, p, o in itertools.product(
                input_categories, predicates, output_categories
            )
        )

        operations = list[Operation]()

        metakg = await self.get_metakg()

        for spo in spo_combos:
            if spo not in metakg.operations:
                continue
            for operation in metakg.operations[spo]:
                if operation.tier not in tiers:
                    continue
                # TODO: filter by API (once TRAPI source filtering exists)
                if not meta_qualifier_meets_constraints(
                    operation.qualifiers, edge.qualifier_constraints
                ):
                    continue
                operations.append(operation)

        return operations

    async def create_operation_plan(
        self, qgraph: QueryGraph, tiers: set[TierNumber]
    ) -> OperationPlan | list[QEdgeID]:
        """Obtain a list of supporting operations for each edge in the query graph.

        Returns None if any QEdge is unsupported by the current operation table.
        """
        plan = OperationPlan()
        unsupported_qedges = list[str]()
        for qedge_id, qedge in qgraph.edges.items():
            operations = await self.find_operations(qedge, qgraph, tiers)
            if len(operations) == 0:
                unsupported_qedges.append(qedge_id)
            plan[qedge_id] = operations

        if len(unsupported_qedges) > 0:
            return unsupported_qedges
        return plan


METAKG_MANAGER = MetaKGManager()


async def get_trapi_metakg(tiers: tuple[TierNumber, ...]) -> MetaKnowledgeGraphDict:
    """Convert an OperationTable to a TRAPI MetaKG dict.

    Because it depends on METAKG_MANAGER, it can't be used with the lead manager.
    This shouldn't be a problem because the lead manager isn't used to answer API calls.
    """
    edges = list[MetaEdgeDict]()
    mentioned_nodes = set[BiolinkCategory]()
    nodes = dict[BiolinkCategory, MetaNodeDict]()
    op_table = await METAKG_MANAGER.get_metakg()
    for op_list in op_table.operations.values():
        add_edge = False
        # op_list should always be of length >= 1
        sbj, obj, pred = op_list[0].subject, op_list[0].object, op_list[0].predicate
        meta_edge = MetaEdgeDict(subject=sbj, predicate=pred, object=obj)
        qualifiers = dict[str, set[str]]()
        attributes = dict[int, MetaAttributeDict]()

        for op in op_list:
            if op.tier in tiers:
                add_edge = True
            else:
                continue

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

        if add_edge:
            meta_edge["qualifiers"] = [
                MetaQualifierDict(
                    qualifier_type_id=qual_type, applicable_values=list(values)
                )
                for qual_type, values in qualifiers.items()
            ]
            meta_edge["attributes"] = list(attributes.values())

            edges.append(meta_edge)
            mentioned_nodes.update((sbj, obj))

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

    return MetaKnowledgeGraphDict(nodes=nodes, edges=edges)
