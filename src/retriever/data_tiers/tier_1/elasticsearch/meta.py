import base64
import hashlib
import zlib
from collections import defaultdict
from copy import deepcopy
from typing import Any

import msgpack
import ormsgpack
from elasticsearch import AsyncElasticsearch
from loguru import logger as log

from retriever.config.general import CONFIG
from retriever.data_tiers.tier_1.elasticsearch.types import UbergraphNodeInfo
from retriever.data_tiers.utils import (
    generate_operation,
    get_simple_op_hash,
    parse_dingo_metadata_unhashed,
)
from retriever.types.dingo import DINGOMetadata
from retriever.types.metakg import Operation, OperationNode, UnhashedOperation
from retriever.types.trapi import BiolinkEntity, Infores, MetaAttributeDict
from retriever.utils.redis import REDIS_CLIENT

T1MetaData = dict[str, Any]

CACHE_KEY = "TIER1_META"


async def get_t1_indices(
    client: AsyncElasticsearch,
) -> list[str]:
    """For fetch a list of indices from ES."""
    resp = await client.indices.resolve_index(
        name=CONFIG.tier1.elasticsearch.index_name
    )
    if "aliases" not in resp:
        raise Exception(
            f"Failed to get indices from ES: {CONFIG.tier1.elasticsearch.index_name}"
        )

    backing_indices: list[str] = []
    for a in resp.get("aliases", []):
        if a["name"] == "dingo":
            backing_indices.extend(a["indices"])

    return backing_indices


def get_stable_hash(key: str) -> str:
    """Get a stable SHA-256 hash from a key."""
    return hashlib.sha256(key.encode()).hexdigest()


async def save_metadata_cache(key: str, payload: T1MetaData) -> None:
    """Wrapper for persist metadata in Redis."""
    await REDIS_CLIENT.set(
        get_stable_hash(key),
        ormsgpack.packb(payload),
        compress=True,
        ttl=CONFIG.job.metakg.build_time,
    )


async def read_metadata_cache(key: str) -> T1MetaData | None:
    """Wrapper for retrieving persisted metadata in Redis."""
    redis_key = get_stable_hash(key)
    metadata_pack = await REDIS_CLIENT.get(redis_key, compressed=True)
    if metadata_pack is not None:
        return ormsgpack.unpackb(metadata_pack)

    return None


def extract_metadata_entries_from_blob(
    blob: T1MetaData, indices: list[str]
) -> list[T1MetaData]:
    """Extract a list of metadata entries from raw blob."""
    meta_entries: list[T1MetaData] = list(
        filter(
            None,
            [blob[index_name].get("graph") for index_name in indices],
        )
    )

    return meta_entries


async def retrieve_metadata_from_es(
    es_connection: AsyncElasticsearch, indices_alias: str
) -> T1MetaData:
    """Method to retrieve prefetched metadata from Elasticsearch."""
    mappings = await es_connection.indices.get_mapping(index=indices_alias)
    tier1_indices = await get_t1_indices(es_connection)

    # here we pull an array of metadata, instead of 1

    meta: T1MetaData = defaultdict(dict)
    for index_name in tier1_indices:
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
    es_connection: AsyncElasticsearch | None,
    indices_alias: str,
    bypass_cache: bool,
    retries: int = 0,
) -> T1MetaData | None:
    """Caller to orchestrate retrieving t1 metadata."""
    meta_blob = None if bypass_cache else await read_metadata_cache(CACHE_KEY)
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
            return await get_t1_metadata(
                es_connection, indices_alias, bypass_cache=True, retries=retries + 1
            )

    log.success("DINGO Metadata retrieved!")
    return meta_blob


def hash_meta_attribute(attr: MetaAttributeDict) -> int:
    """Method to hash MetaAttributeDict."""
    keys = [
        "attribute_type_id",
        "attribute_source",
        "original_attribute_names",
        "constraint_use",
        "constraint_name",
    ]
    values: list[Any] = []
    for key in keys:
        val: list[str] | None = attr.get(key)
        if isinstance(val, list):
            values.append(tuple(val))
        else:
            values.append(val)
    return hash(tuple(values))


def merge_nodes(
    nodes: dict[BiolinkEntity, OperationNode],
    curr_nodes: dict[BiolinkEntity, OperationNode],
    infores: Infores,
) -> dict[BiolinkEntity, OperationNode]:
    """Merge OperationNodes generated."""
    for category, node in curr_nodes.items():
        # Category not seen before → initialize
        if category not in nodes:
            nodes[category] = deepcopy(node)
            continue

        existing = nodes[category]
        # Merge prefixes
        existing.prefixes[infores].extend(node.prefixes[infores])
        # Merge attributes
        existing.attributes[infores].extend(node.attributes[infores])

    return nodes


def dedupe_nodes(
    nodes: dict[BiolinkEntity, OperationNode], infores: Infores
) -> dict[BiolinkEntity, OperationNode]:
    """De-duplicate OperationNodes generated."""
    for current in nodes.values():
        current.prefixes[infores] = list(set(current.prefixes[infores]))

        seen_attr: set[int] = set()
        attrs: list[MetaAttributeDict] = current.attributes[infores]
        deduped: list[MetaAttributeDict] = []
        for attr in attrs:
            hash_code = hash_meta_attribute(attr)
            if hash_code not in seen_attr:
                deduped.append(attr)
                seen_attr.add(hash_code)

        current.attributes[infores] = deduped

    return nodes


def merge_operations(ops_unhashed: list[UnhashedOperation]) -> list[Operation]:
    """Merge duplicate operations."""
    seen_op = dict[str, Operation]()
    operations = list[Operation]()

    for op in ops_unhashed:
        op_hash = get_simple_op_hash(op)
        if op_hash not in seen_op:
            operation = generate_operation(op, op_hash)
            operations.append(operation)
            seen_op[op_hash] = operation
        # needs merging if seen
        else:
            hashed_op = seen_op[op_hash]
            if hashed_op.attributes is not None and op.attributes is not None:
                hashed_op.attributes.extend(op.attributes)
            if hashed_op.qualifiers is not None and op.qualifiers is not None:
                hashed_op.qualifiers.update(op.qualifiers)

            # ignoring access_metadata for now

    return operations


async def generate_operations(
    meta_entries: list[T1MetaData],
) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
    """Generate operations and associated nodes based on metadata provided."""
    infores = Infores(CONFIG.tier1.backend_infores)

    operations_unhashed: list[UnhashedOperation] = []
    nodes: dict[BiolinkEntity, OperationNode] = {}

    for meta_entry in meta_entries:
        curr_ops, curr_nodes = parse_dingo_metadata_unhashed(
            DINGOMetadata(**meta_entry), 1, infores
        )
        operations_unhashed.extend(curr_ops)
        nodes = merge_nodes(nodes, curr_nodes, infores)

    operations = merge_operations(operations_unhashed)
    nodes = dedupe_nodes(nodes, infores)

    log.success(f"Parsed {infores} as a Tier 1 resource.")
    return operations, nodes


async def retrieve_ubergraph_info_from_es(
    es_connection: AsyncElasticsearch,
) -> UbergraphNodeInfo:
    """Retrieve Ubergraph info from Elasticsearch."""
    index_name = "ubergraph_nodes_mapping"
    resp = await es_connection.search(
        index=index_name,
        size=10000,
        query={"match_all": {}},
        sort=[{"chunk_index": {"order": "asc"}}],
    )

    b64 = "".join(hit["_source"]["value"] for hit in resp["hits"]["hits"])

    obj = msgpack.unpackb(zlib.decompress(base64.b64decode(b64)), raw=False)
    return obj


def to_ubergraph_info(data: T1MetaData) -> UbergraphNodeInfo:
    """Casting method to satisfy our linter overlord."""
    return UbergraphNodeInfo(mapping=data.get("mapping", {}))


def from_ubergraph_info(info: UbergraphNodeInfo) -> T1MetaData:
    """Reverse of `to_ubergraph_info`."""
    return {
        "mapping": info["mapping"],
    }


async def get_ubergraph_info(
    es_connection: AsyncElasticsearch | None, retries: int = 0
) -> UbergraphNodeInfo:
    """Assemble ubergraph related info from ES."""
    ubergraph_info_cache_key = "TIER1_UBERGRAPH_INFO"

    cached_info = await read_metadata_cache(ubergraph_info_cache_key)

    if cached_info is not None:
        return to_ubergraph_info(cached_info)

    try:
        if es_connection is None:
            raise ValueError(
                "Invalid Elasticsearch connection. Driver must be initialized and connected."
            )
        cached_info = await retrieve_ubergraph_info_from_es(es_connection)
        await save_metadata_cache(
            ubergraph_info_cache_key, from_ubergraph_info(cached_info)
        )
        log.success("ubergraph info saved!")
    except ValueError as e:
        # if exceeds retries or ES connection is invalid, return None
        if retries == RETRY_LIMIT:
            raise ValueError(
                "Failed to retrieve UBERGRAPH info from Elasticsearch due to retries exceeded."
            ) from e
        if str(e).startswith("Invalid Elasticsearch connection"):
            raise ValueError(
                "Failed to retrieve UBERGRAPH info from Elasticsearch due to invalid ES connection."
            ) from e
        return await get_ubergraph_info(es_connection, retries + 1)

    log.success("ubergraph info retrieved!")
    return cached_info
