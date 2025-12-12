from collections import defaultdict
from typing import Any

import ormsgpack
from elasticsearch import AsyncElasticsearch
from loguru import logger as log

from retriever.config.general import CONFIG
from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.types.dingo import DINGOMetadata
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import BiolinkEntity, Infores
from retriever.utils.redis import REDIS_CLIENT
from retriever.utils.trapi import hash_hex

TIER1_INDICES = [
    "diseases",
    "gene2phenotype",
    "go_cam",
    "goa",
    "hpoa",
    "sider",
    "ctd",
    "panther",
    "ubergraph",
    "ttd",
    "alliance",
]

T1MetaData = dict[str, Any]

# probably needs some versioning info to purge redis?
CACHE_KEY = "TIER1_META"


async def save_metadata_cache(key: str, payload: T1MetaData) -> None:
    """Wrapper for persist metadata in Redis."""
    await REDIS_CLIENT.set(
        hash_hex(hash(key)),
        ormsgpack.packb(payload),
        compress=True,
        ttl=CONFIG.job.metakg.build_time,
    )


async def read_metadata_cache(key: str) -> T1MetaData | None:
    """Wrapper for retrieving persisted metadata in Redis."""
    metadata_pack = await REDIS_CLIENT.get(hash_hex(hash(key)), compressed=True)
    if metadata_pack is not None:
        return ormsgpack.unpackb(metadata_pack)

    return None


def extract_metadata_entries_from_blob(blob: T1MetaData) -> list[T1MetaData]:
    """Extract a list of metadata entries from raw blob."""
    meta_entries: list[T1MetaData] = list(
        filter(
            None,
            [blob[index_name].get("graph") for index_name in TIER1_INDICES],
        )
    )

    return meta_entries


async def retrieve_metadata_from_es(
    es_connection: AsyncElasticsearch, indices_alias: str
) -> T1MetaData:
    """Method to retrieve prefetched metadata from Elasticsearch."""
    mappings = await es_connection.indices.get_mapping(index=indices_alias)

    # here we pull an array of metadata, instead of 1

    meta: T1MetaData = defaultdict(dict)
    for index_name in TIER1_INDICES:
        raw = mappings[index_name]["mappings"]["_meta"]
        keys = ["graph", "release"]

        for key in keys:
            blob = raw.get(key)
            if blob:
                meta[index_name].update({key: blob})

    if not meta:
        raise ValueError("No metadata retrieved from Elasticsearch.")

    return meta


RETRY_LIMIT = 3


async def get_t1_metadata(
    es_connection: AsyncElasticsearch | None, indices_alias: str, retries: int = 0
) -> T1MetaData | None:
    """Caller to orchestrate retrieving t1 metadata."""
    meta_blob = await read_metadata_cache(CACHE_KEY)
    if not meta_blob:
        try:
            if es_connection is None:
                raise ValueError(
                    "Invalid Elasticsearch connection. Driver must be initialized and connected."
                )
            meta_blob = await retrieve_metadata_from_es(es_connection, indices_alias)
            await save_metadata_cache(CACHE_KEY, meta_blob)
        except ValueError as e:
            # if exceeds retries or ES connection is invalid, return None
            if retries == RETRY_LIMIT or str(e).startswith(
                "Invalid Elasticsearch connection"
            ):
                return None
            return await get_t1_metadata(es_connection, indices_alias, retries + 1)

    log.success("DINGO Metadata retrieved!")
    return meta_blob


async def generate_operations(
    meta_entries: list[T1MetaData],
) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
    """Generate operations and associated nodes based on metadata provided."""
    infores = Infores(CONFIG.tier1.backend_infores)

    operations: list[Operation] = []
    nodes: dict[BiolinkEntity, OperationNode] = {}

    for meta_entry in meta_entries:
        curr_ops, curr_nodes = parse_dingo_metadata(
            DINGOMetadata(**meta_entry), 1, infores
        )
        operations.extend(curr_ops)
        nodes.update(curr_nodes)

    log.success(f"Parsed {infores} as a Tier 1 resource.")
    return operations, nodes
