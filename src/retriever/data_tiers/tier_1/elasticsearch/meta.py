import asyncio
import base64
import hashlib
import zlib
from collections import defaultdict
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import msgpack
import ormsgpack
from elasticsearch import AsyncElasticsearch
from loguru import logger as log

from retriever.config.general import CONFIG
from retriever.data_tiers.utils import (
    generate_operation,
    get_simple_op_hash,
    parse_dingo_metadata_unhashed,
)
from retriever.types.dingo import DINGOMetadata
from retriever.types.metakg import Operation, OperationNode, UnhashedOperation
from retriever.types.trapi import CURIE, BiolinkEntity, Infores, MetaAttributeDict
from retriever.utils.redis import RedisClient

T1MetaData = dict[str, Any]

CACHE_KEY = "TIER1_META"

_LOCAL_CACHE: dict[str, T1MetaData] = {}
"""Per-process metadata cache. Fronts Redis so the worker still serves
hits during an outage; saves populate both layers."""


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
    """Cache `payload` locally; best-effort Redis write skipped while Redis is down."""
    _LOCAL_CACHE[key] = payload
    if not RedisClient().up:
        return
    try:
        await RedisClient().set(
            get_stable_hash(key),
            ormsgpack.packb(payload),
            compress=True,
        )
    except Exception:
        log.debug(f"Redis write for {key} failed; serving from local cache.")


async def read_metadata_cache(key: str) -> T1MetaData | None:
    """Cached metadata preferring local then Redis; `None` on miss or Redis down."""
    cached = _LOCAL_CACHE.get(key)
    if cached is not None:
        return cached
    if not RedisClient().up:
        return None
    try:
        redis_key = get_stable_hash(key)
        metadata_pack = await RedisClient().get(redis_key, compressed=True)
    except Exception:
        log.debug(f"Redis read for {key} failed; treating as cache miss.")
        return None
    if metadata_pack is None:
        return None
    payload: T1MetaData = ormsgpack.unpackb(metadata_pack)
    _LOCAL_CACHE[key] = payload
    return payload


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
    """T1 metadata, preferring cache; `None` if neither cache nor backend can serve."""
    meta_blob = None if bypass_cache else await read_metadata_cache(CACHE_KEY)
    if not meta_blob:
        if es_connection is None:
            return await _cached_fallback(bypass_cache)
        try:
            meta_blob = await retrieve_metadata_from_es(es_connection, indices_alias)
            await save_metadata_cache(CACHE_KEY, meta_blob)
        except Exception as exc:
            if retries >= RETRY_LIMIT:
                log.warning(
                    f"Failed to retrieve T1 metadata after {RETRY_LIMIT} retries: {exc}"
                )
                return await _cached_fallback(bypass_cache)
            return await get_t1_metadata(
                es_connection, indices_alias, bypass_cache=True, retries=retries + 1
            )

    log.success("DINGO Metadata retrieved!")
    return meta_blob


async def _cached_fallback(bypass_cache: bool) -> T1MetaData | None:
    """When `bypass_cache=True` exhausted live retries, fall back to the cached copy."""
    if not bypass_cache:
        return None
    cached = await read_metadata_cache(CACHE_KEY)
    if cached is not None:
        log.warning("Live T1 metadata fetch failed; falling back to cached metadata.")
    return cached


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


UBERGRAPH_MAPPING_INDEX = "ubergraph_nodes_mapping"
"""ES index holding the base64/zlib/msgpack-chunked UBERGRAPH subclass mapping."""

UBERGRAPH_CHUNK_PAGE = 1000
"""Chunks fetched per `search_after` page when reading the mapping."""


async def iter_ubergraph_chunks(
    es_connection: AsyncElasticsearch,
    page_size: int = UBERGRAPH_CHUNK_PAGE,
) -> AsyncIterator[str]:
    """Yield the mapping's base64 chunks in `chunk_index` order.

    Pages with `search_after` so the fetch can't silently truncate the way a fixed
    `size=` search does once the chunk count exceeds the page cap.
    """
    search_after: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            "size": page_size,
            "query": {"match_all": {}},
            "sort": [{"chunk_index": {"order": "asc"}}],
        }
        if search_after is not None:
            body["search_after"] = search_after
        resp = await es_connection.search(index=UBERGRAPH_MAPPING_INDEX, body=body)
        hits = resp["hits"]["hits"]
        for hit in hits:
            yield hit["_source"]["value"]
        if len(hits) < page_size:
            break
        search_after = hits[-1]["sort"]


async def stream_ubergraph_mapping(
    es_connection: AsyncElasticsearch,
    cutoff: int,
) -> AsyncIterator[tuple[CURIE, list[CURIE]]]:
    """Stream the CURIE->descendants mapping, dropping entries over `cutoff`.

    Walks the chunked, base64/zlib/msgpack-encoded blob with a streaming
    `msgpack.Unpacker`, materializing one descendant list at a time so the full
    mapping is never resident (peak stays in the low hundreds of MB). Entries whose
    descendant count exceeds `cutoff` are skipped; `cutoff <= 0` keeps everything.
    Raises `ValueError` if the chunk stream ends mid-object (a truncated blob).
    """
    unpacker = msgpack.Unpacker(raw=False)
    inflator = zlib.decompressobj()
    chunks = iter_ubergraph_chunks(es_connection)
    b64_tail = ""

    async def pump() -> bool:
        """Decode+inflate the next chunk into the unpacker; False once exhausted."""
        nonlocal b64_tail
        try:
            text = await chunks.__anext__()
        except StopAsyncIteration:
            if b64_tail:
                unpacker.feed(inflator.decompress(base64.b64decode(b64_tail)))
                b64_tail = ""
            unpacker.feed(inflator.flush())
            return False
        # base64 decodes in 4-char groups; carry any partial group to the next chunk.
        b64_tail += text
        aligned = len(b64_tail) - (len(b64_tail) % 4)
        raw = base64.b64decode(b64_tail[:aligned])
        b64_tail = b64_tail[aligned:]
        if raw:
            unpacker.feed(inflator.decompress(raw))
        return True

    async def take() -> Any:
        while True:
            try:
                return unpacker.unpack()
            except msgpack.OutOfData:
                if not await pump():
                    raise ValueError(
                        "UBERGRAPH mapping ended mid-object; blob is truncated."
                    ) from None

    async def take_map_header() -> int:
        while True:
            try:
                return unpacker.read_map_header()
            except msgpack.OutOfData:
                if not await pump():
                    raise ValueError(
                        "UBERGRAPH mapping ended before a map header; blob is empty or truncated."
                    ) from None

    for _ in range(await take_map_header()):  # top-level {"mapping": ..., "size": ...}
        if await take() != "mapping":
            _ = await take()  # consume the other value (e.g. "size")
            continue
        for index in range(await take_map_header()):
            curie = await take()
            descendants = await take()
            if cutoff <= 0 or len(descendants) <= cutoff:
                yield curie, descendants
            if index % 10000 == 0:
                await asyncio.sleep(
                    0
                )  # keep the event loop responsive on the long walk

    while await pump():  # drain any trailing chunks so the zlib trailer is validated
        pass
    if not inflator.eof:
        raise ValueError(
            "UBERGRAPH mapping compressed stream is incomplete; blob is truncated."
        )
