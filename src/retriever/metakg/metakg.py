import asyncio
import itertools
from typing import NamedTuple

import ormsgpack
from loguru import logger
from reasoner_pydantic import QEdge, QueryGraph

from retriever.config.general import CONFIG
from retriever.types.dingo import DINGO_ADAPTER, DINGOMetadata
from retriever.types.metakg import Operation, OperationNode, OperationTable
from retriever.types.trapi import (
    BiolinkEntity,
    MetaAttributeDict,
    MetaEdgeDict,
    MetaKnowledgeGraphDict,
    MetaNodeDict,
    MetaQualifierDict,
    QEdgeID,
    QualifierTypeID,
)
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils.biolink import expand
from retriever.utils.calls import get_metadata_client
from retriever.utils.redis import METAKG_UPDATE_CHANNEL, REDIS_CLIENT
from retriever.utils.trapi import (
    hash_hex,
    hash_meta_attribute,
    meta_qualifier_meets_constraints,
)

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
        self._metakg: OperationTable | None = None
        self.metakg_lock: asyncio.Lock = asyncio.Lock()
        self.is_leader: bool = leader

    def parse_trapi_metakg(
        self,
        metakg_info: TRAPIMetaKGInfo,
        operations: list[Operation],
        nodes: dict[BiolinkEntity, OperationNode],
    ) -> None:
        """Parse a TRAPI MetaKG to build operations."""
        metakg, tier, infores = metakg_info
        for edge in metakg["edges"]:
            edge_dict = MetaEdgeDict(**edge)
            operations.append(
                Operation(
                    subject=edge_dict["subject"],
                    predicate=edge_dict["predicate"],
                    object=edge_dict["object"],
                    api=infores,
                    tier=tier,
                    attributes=edge_dict.get("attributes"),
                    qualifiers={
                        qualifier["qualifier_type_id"]: qualifier.get(
                            "applicable_values", []
                        )
                        for qualifier in (edge_dict.get("qualifiers", []) or [])
                    },
                )
            )
        for category, node in metakg["nodes"].items():
            node_dict = MetaNodeDict(**node)
            nodes[category] = OperationNode(
                prefixes={infores: node_dict.get("id_prefixes", [])},
                attributes={infores: (node_dict.get("attributes", []) or [])},
            )

    def parse_dingo_metadata(
        self,
        metakg_info: DINGOMetaKGInfo,
        operations: list[Operation],
        nodes: dict[BiolinkEntity, OperationNode],
    ) -> None:
        """Parse a DINGO Metadata object to build operations."""
        metadata, tier, infores = metakg_info
        for edge in metadata["schema"]["edges"]:
            for sbj, obj in itertools.product(
                edge["subject_category"], edge["object_category"]
            ):
                operations.append(
                    Operation(
                        subject=sbj,
                        predicate=edge["predicate"],
                        object=obj,
                        api=infores,
                        tier=tier,
                        attributes=[
                            MetaAttributeDict(attribute_type_id=attr_type)
                            for attr_type in edge["attributes"]
                        ],
                        qualifiers={qual_type: [] for qual_type in edge["qualifiers"]},
                    )
                )
        for node in metadata["schema"]["nodes"]:
            for category in node["category"]:
                nodes[category] = OperationNode(
                    prefixes={infores: list(node["id_prefixes"].keys())},
                    attributes={
                        infores: [
                            MetaAttributeDict(attribute_type_id=attr_type)
                            for attr_type in node["attributes"]
                        ]
                    },
                )

    async def pull_metadata(self, url: str) -> None:
        """Update metadata for a given DINGO ingest."""
        logger.info(f"Pulling DINGO Metadata from {url}...")
        client = get_metadata_client()
        response = await client.get(url)
        response.raise_for_status()

        raw_data = response.json()
        metadata = DINGO_ADAPTER.validate_python(raw_data)

        async with self.metakg_lock:
            await REDIS_CLIENT.set(
                hash_hex(hash(url)),
                ormsgpack.packb(metadata),
                compress=True,
                ttl=CONFIG.job.metakg.build_time,
            )
        logger.success("DINGO Metadata retrieved!")

    async def get_metadata(self, url: str, retries: int = 0) -> DINGOMetadata:
        """Obtain metadata for a given DINGO ingest."""
        metadata_pack = await REDIS_CLIENT.get(hash_hex(hash(url)), compressed=True)

        if metadata_pack is None:
            await self.pull_metadata(url)
            if retries >= 3:  # noqa: PLR2004
                raise ValueError("Failed to retrieve a built metakg!")
            return await self.get_metadata(url, retries + 1)

        # Don't validate because if we've gotten it at this stage, it's already been validated
        metadata = ormsgpack.unpackb(metadata_pack)
        return DINGOMetadata(**metadata)

    async def build_metakg(self) -> None:
        """Build Retriever's internal MetaKG and store it to Redis."""
        if CONFIG.instance_idx != 0:
            return

        logger.info("Building MetaKG...")

        operations = list[Operation]()
        nodes = dict[BiolinkEntity, OperationNode]()

        metakg_locations = {
            0: {
                CONFIG.tier0.backend_infores: CONFIG.tier0.metakg_url,
            },
            1: {
                CONFIG.tier1.backend_infores: CONFIG.tier1.metakg_url,
            },
        }

        for tier, sources in metakg_locations.items():
            for infores, url in sources.items():
                metadata = await self.get_metadata(url)
                self.parse_dingo_metadata(
                    DINGOMetaKGInfo(metadata, tier, infores),
                    operations,
                    nodes,
                )
                logger.success(f"Parsed {infores} as a Tier {tier} resource.")

        async with self.metakg_lock:
            self._metakg = OperationTable(operations, nodes)

        await REDIS_CLIENT.update_metakg(self._metakg)
        logger.success(
            f"Built MetaKG containing {len(operations)} operations / {len(nodes)} nodes."
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

        operations = list[Operation]()

        metakg = await self.get_metakg()

        for operation in metakg.operations:
            if (
                operation.tier in tiers
                and operation.subject in input_categories
                and operation.predicate in predicates
                and operation.object in output_categories
                and meta_qualifier_meets_constraints(
                    operation.qualifiers, edge.qualifier_constraints
                )
            ):
                operations.append(operation)

        return operations

    async def create_operation_plan(
        self, qgraph: QueryGraph, tiers: set[TierNumber]
    ) -> OperationPlan | list[QEdgeID]:
        """Obtain a list of supporting operations for each edge in the query graph.

        Returns None if any QEdge is unsupported by the current operation table.
        """
        plan = OperationPlan()
        unsupported_qedges = list[QEdgeID]()
        for qedge_id, qedge in qgraph.edges.items():
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
    for op in op_table.operations:
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
            meta_edge = MetaEdgeDict(subject=sbj, predicate=pred, object=obj)
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
    op_table = await METAKG_MANAGER.get_metakg()
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
