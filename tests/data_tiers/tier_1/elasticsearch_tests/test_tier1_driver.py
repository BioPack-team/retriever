import base64
import importlib
import zlib
from collections.abc import Iterator
from typing import Any, cast

import msgpack
import pytest
from payload.trapi_qgraphs import (
    DINGO_QGRAPH,
    EXPANDED_QUALIFIER_QGRAPH,
    ID_BYPASS_PAYLOAD,
    INVALID_REGEX_QGRAPHS,
    VALID_REGEX_QGRAPHS,
)
from test_tier1_transpiler import _convert_batch_triple, _convert_triple

import retriever.config.general as general_mod
import retriever.data_tiers.tier_1.elasticsearch.driver as driver_mod
from retriever.data_tiers.tier_1.elasticsearch.meta import (
    extract_metadata_entries_from_blob,
    get_t1_indices,
    iter_ubergraph_chunks,
    stream_ubergraph_mapping,
)
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.data_tiers.tier_1.elasticsearch.types import ESEdge, ESNode, ESPayload
from retriever.utils.redis import RedisClient


def esp(d: dict[str, Any]) -> ESPayload:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(ESPayload, cast(object, d))


@pytest.fixture
def mock_elasticsearch_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # These tests should use localhost and SKIP
    monkeypatch.setenv("TIER1__ELASTICSEARCH__HOST", "localhost")
    # monkeypatch.setenv("TIER0__DGRAPH__HTTP_PORT", "8080")
    # monkeypatch.setenv("TIER0__DGRAPH__GRPC_PORT", "9080")
    # monkeypatch.setenv("TIER0__DGRAPH__USE_TLS", "false")
    # monkeypatch.setenv("TIER0__DGRAPH__QUERY_TIMEOUT", "3")
    monkeypatch.setenv("TIER1__ELASTICSEARCH__CONNECT_TIMEOUT", "1")

    # Rebuild CONFIG from env and reload driver so classes bind to the new CONFIG
    importlib.reload(general_mod)
    importlib.reload(driver_mod)
    # Ensure the driver module uses the same CONFIG instance
    monkeypatch.setattr(driver_mod, "CONFIG", general_mod.CONFIG, raising=False)
    yield


PAYLOAD_0: ESPayload = esp({
    "query": {
        "bool":
            {"filter": [
                {"terms": {"subject.category": ["Protein", "Gene"]}},
                {"terms": {"object.id": ["MONDO:0012507"]}},
                {"terms": {"object.category": ["disease"]}},
                {"terms": {"predicate_ancestors": ["causes"]}}
            ]
            }
    }
}
)
PAYLOAD_1: ESPayload = esp({
    "query": {
        "bool": {
            "filter": [
                {"terms": {"object.id": ["MONDO:0005233"]}},
                {"terms": {"subject.id": ["CHEBI:70839", "UMLS:C1872686"]}}
            ]
        }
    }
})

PAYLOAD_2: ESPayload = esp({
    "query": {
        "bool": {
            "filter": [
                {
                    "terms": {
                        "subject.id": [
                            "MONDO:0030010",
                            "MONDO:0011766",
                            "MONDO:0009890"
                        ]
                    }
                }
            ],
            "must": [
                {"range": {"has_total": {"gt": 0}}},
                {"range": {"has_total": {"lte": 45}}}
            ],
            "should": [
                {"term": {"sex_qualifier": "PATO:0000383"}},
                {"term": {"frequency_qualifier": "HP:0040280"}}
            ],
            "minimum_should_match": 1
        }
    }
})


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected",
    [
        (PAYLOAD_0, 0),
        (PAYLOAD_1, 2),
        (PAYLOAD_2, 32),
        (
                [PAYLOAD_0, PAYLOAD_1, PAYLOAD_2],
                [0, 2, 32]
        )
    ],
    ids=[
        "single payload 1",
        "single payload 2",
        "single payload 3",
        "batch payload",
    ]
)
async def test_elasticsearch_driver(payload: ESPayload | list[ESPayload], expected: int | list[int]):
    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()
    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    hits: list[ESEdge] | list[list[ESEdge]] = await driver.run_query(payload)

    def assert_single_result(res, expected_result_num: int):
        if not res:
            if expected_result_num != 0:
                raise AssertionError(f"Expected empty result, got {type(res)}")
            else:
                return
        if not isinstance(res, list):
            raise AssertionError(f"Expected results to be list, got {type(res)}")
        if not len(res) == expected_result_num:
            raise AssertionError(f"Expected {expected_result_num} results, got {len(res)}")

    # check batch result
    if len(payload) > 1:
        assert len(hits) == len(payload)
        assert isinstance(hits[0], list)

        for index, result in enumerate(cast(list[list[ESEdge]], hits)):
            assert_single_result(result, expected[index])
    else:
        assert_single_result(hits, expected)

    await driver.wrapup()


@pytest.mark.parametrize(
    "qgraph",
    INVALID_REGEX_QGRAPHS,
    ids=[qgraph["edges"]["e0"]["attribute_constraints"][0]["value"] for qgraph in INVALID_REGEX_QGRAPHS]
)
def test_invalid_regex_qgraph(qgraph):
    transpiler = ElasticsearchTranspiler()
    with pytest.raises(ValueError):
        _convert_triple(transpiler, qgraph)


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_valid_regex_query():
    transpiler = ElasticsearchTranspiler()

    qgraphs_with_valid_regex = _convert_batch_triple(transpiler, VALID_REGEX_QGRAPHS)

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    for payload in qgraphs_with_valid_regex:
        hits: list[ESEdge] = await driver.run_query(payload)

    await driver.wrapup()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_metadata_retrieval():
    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    meta = await driver.get_metadata(bypass_cache=True)

    # make sure each index has metadata extracted
    indices = await get_t1_indices(driver.es_connection)
    assert len(extract_metadata_entries_from_blob(meta, indices)) == len(indices)

    ops, nodes = await driver.get_operations()

    # with open("output_new.json", "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2)

    # _ops, _nodes = await driver.legacy_get_operations()


    # assert len(nodes) == 23

    await driver.wrapup()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_fetch_single_node():
    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping fetch_single_node test: cannot connect")

    node = await driver.fetch_single_node("CHEBI:48927")

    assert isinstance(node, ESNode)
    assert node is not None
    assert node.id == "CHEBI:48927"
    assert len(node.category) > 0
    assert node.name == "N-acyl-L-alpha-amino acid"

    await driver.wrapup()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "qgraph, min_hits",
    [
        # Lower-bound floors instead of exact counts so the test survives
        # data growth in the live ES index.
        (DINGO_QGRAPH, 8),
        (ID_BYPASS_PAYLOAD, 4000),
        (EXPANDED_QUALIFIER_QGRAPH, 1),
    ],
)
async def test_end_to_end(qgraph, min_hits):
    transpiler = ElasticsearchTranspiler()
    payload = _convert_triple(transpiler, qgraph)

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    hits: list[ESEdge] = await driver.run_query(payload)

    assert len(hits) >= min_hits

    await driver.wrapup()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_cache_bypass():
    """Test bypass_cache=True with single payload - should apply enforce_timestamp.

    Uses real ES connection to verify actual query execution behavior.
    Compares results with and without bypass_cache to ensure timestamp filtering works.
    """
    transpiler = ElasticsearchTranspiler()
    payload = _convert_triple(transpiler, DINGO_QGRAPH)

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping bypass_cache test: cannot connect to elasticsearch")

    # Execute query with bypass_cache=True (should enforce timestamp)
    hits_with_bypass = await driver.run_query(payload, bypass_cache=True)

    # Execute same query with bypass_cache=False (should not enforce timestamp)
    hits_without_bypass = await driver.run_query(payload, bypass_cache=False)

    # Both should return lists
    assert isinstance(hits_with_bypass, list)
    assert isinstance(hits_without_bypass, list)

    # Results with bypass should be a subset or equal to results without bypass
    # (since adding timestamp constraint can only reduce or maintain result count)
    assert len(hits_with_bypass) == len(hits_without_bypass)

    await driver.wrapup()

@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_cache_bypass_batch_query():
    """Test bypass_cache=True with batch payloads - should apply enforce_timestamp to each.

    Uses real ES connection to verify batch query execution behavior.
    Compares results with and without bypass_cache for batch processing.
    """
    transpiler = ElasticsearchTranspiler()
    batch_payloads = _convert_batch_triple(transpiler, [DINGO_QGRAPH, ID_BYPASS_PAYLOAD])

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping batch bypass_cache test: cannot connect to elasticsearch")

    # Execute batch query with bypass_cache=True (should enforce timestamp on all)
    hits_with_bypass: list[list[ESEdge]] = await driver.run_query(batch_payloads, bypass_cache=True)

    # Execute same batch query with bypass_cache=False (should not enforce timestamp)
    hits_without_bypass: list[list[ESEdge]] = await driver.run_query(batch_payloads, bypass_cache=False)

    # Both should return list of lists with same structure
    assert isinstance(hits_with_bypass, list)
    assert isinstance(hits_without_bypass, list)
    assert len(hits_with_bypass) == 2
    assert len(hits_without_bypass) == 2

    # Results should be identical since timestamp filtering should not affect these results
    assert len(hits_with_bypass[0]) == len(hits_without_bypass[0])
    assert len(hits_with_bypass[1]) == len(hits_without_bypass[1])

    await driver.wrapup()



@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_cache_bypass_timestamp_structure(monkeypatch: pytest.MonkeyPatch):
    """Verify the timestamp structure is correctly added to queries when bypass_cache=True."""
    from unittest.mock import AsyncMock

    transpiler = ElasticsearchTranspiler()
    payload = _convert_triple(transpiler, DINGO_QGRAPH)

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    # Inject a mock ES connection
    mock_es = AsyncMock()
    driver.es_connection = mock_es

    # Capture the actual query passed to run_single_query
    captured_query = None

    async def mock_run_single(es_connection, index_name, query):
        nonlocal captured_query
        captured_query = query
        return []

    mock_run_single_obj = AsyncMock(side_effect=mock_run_single)
    monkeypatch.setattr(driver_mod, "run_single_query", mock_run_single_obj)

    await driver.run_query(payload, bypass_cache=True)

    # Verify timestamp was added to filters (proof that enforce_timestamp was called)
    if captured_query is None:
        pytest.fail("Query was not captured")

    filters = captured_query["query"]["bool"].get("filter", [])

    # Check that timestamp filter was added
    timestamp_filter_found = False
    for f in filters:
        if isinstance(f, dict) and "bool" in f:
            bool_filter = f["bool"]
            if "should" in bool_filter:
                should_clauses = bool_filter["should"]
                # Check for timestamp range and null check clauses
                has_range = any("range" in clause and "update_date" in clause.get("range", {}) for clause in should_clauses)
                has_null_check = any("bool" in clause and "must_not" in clause.get("bool", {}) for clause in should_clauses)
                if has_range and has_null_check:
                    timestamp_filter_found = True
                    break

    assert timestamp_filter_found, "Timestamp filter not correctly added to query"

    await driver.wrapup()


@pytest.mark.live
@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_ubergraph_info_retrieval():
    # --- MANUAL LOOP RESET ---
    # Since REDIS_CLIENT can retain stale connections/loop references from previous tests,
    # we need to force-reset the Redis connection pool to prevent 'Event loop is closed' errors.
    RedisClient().client.connection_pool.reset()

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()
    try:
        await driver.initialize()
        assert driver.up
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    # Ubergraph grows over time; assert a non-trivial mapping streamed through
    # rather than locking to a moment-in-time count.
    streamed = 0
    async for _curie, _descendants in driver.stream_subclass_mapping(cutoff=-1):
        streamed += 1
    assert streamed > 100_000

    await driver.wrapup()


def _chunk_ubergraph(mapping: dict[str, list[str]], chunk_len: int = 16) -> list[str]:
    """Encode a mapping the way ES stores it: base64(zlib(msgpack)) split into chunks.

    `chunk_len` is a multiple of 4 so chunk boundaries land on base64 quartets.
    """
    blob = base64.b64encode(
        zlib.compress(msgpack.packb({"mapping": mapping, "size": len(mapping)}))
    ).decode()
    return [blob[i : i + chunk_len] for i in range(0, len(blob), chunk_len)] or [""]


class _FakeUbergraphES:
    """Minimal AsyncElasticsearch stand-in serving chunk docs with `search_after`."""

    def __init__(self, chunks: list[str]) -> None:
        self.docs = [
            {"_source": {"value": c}, "chunk_index": i, "sort": [i]}
            for i, c in enumerate(chunks)
        ]
        self.search_calls = 0

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_calls += 1
        size = body["size"]
        after = body.get("search_after")
        start = (after[0] + 1) if after is not None else 0
        return {"hits": {"hits": self.docs[start : start + size]}}


async def _drain(es: Any, cutoff: int) -> dict[str, list[str]]:
    return {curie: desc async for curie, desc in stream_ubergraph_mapping(es, cutoff)}


@pytest.mark.asyncio
async def test_stream_ubergraph_mapping_drops_over_cutoff():
    mapping = {
        "small:1": ["d1", "d2"],
        "small:2": ["d3"],
        "root:huge": [f"x{i}" for i in range(50)],
    }
    result = await _drain(_FakeUbergraphES(_chunk_ubergraph(mapping)), cutoff=5)
    assert set(result) == {"small:1", "small:2"}  # over-cutoff entry dropped
    assert result["small:1"] == ["d1", "d2"]


@pytest.mark.asyncio
async def test_stream_ubergraph_mapping_cutoff_disabled_keeps_all():
    mapping = {"a": [str(i) for i in range(100)], "b": ["x"]}
    result = await _drain(_FakeUbergraphES(_chunk_ubergraph(mapping)), cutoff=-1)
    assert set(result) == {"a", "b"}
    assert len(result["a"]) == 100


@pytest.mark.asyncio
async def test_stream_ubergraph_mapping_truncated_blob_raises():
    mapping = {f"k{i}": [f"d{i}_{j}" for j in range(i % 5)] for i in range(120)}
    chunks = _chunk_ubergraph(mapping, chunk_len=16)
    # Dropping the tail truncates the compressed stream; the stream must raise
    # rather than silently yield a partial mapping.
    with pytest.raises(ValueError):
        await _drain(_FakeUbergraphES(chunks[:-1]), cutoff=-1)


@pytest.mark.asyncio
async def test_iter_ubergraph_chunks_paginates_without_truncating():
    mapping = {f"k{i}": ["d"] for i in range(40)}
    chunks = _chunk_ubergraph(mapping, chunk_len=8)
    es = _FakeUbergraphES(chunks)
    fetched = [chunk async for chunk in iter_ubergraph_chunks(es, page_size=4)]
    assert fetched == chunks  # every chunk, in order (no size= truncation)
    assert es.search_calls > 1  # actually paged via search_after


class _FakeStreamDriver:
    """Driver stand-in whose `stream_subclass_mapping` yields fixed pairs."""

    def __init__(self, pairs: list[tuple[str, list[str]]]) -> None:
        self._pairs = pairs

    def stream_subclass_mapping(self, cutoff: int) -> Any:
        async def gen() -> Any:
            for pair in self._pairs:
                yield pair

        return gen()


class _FakeRedis:
    """In-memory stand-in modeling the hash ops `_reload_mapping` uses."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, bytes]] = {}
        self.freshness_writes: list[int] = []
        self.hset_calls: list[int] = []  # batch sizes, to assert batching

    async def hset(self, key: str, *, mapping: dict[str, bytes]) -> None:
        self.hset_calls.append(len(mapping))
        self.hashes.setdefault(key, {}).update(mapping)

    async def delete(self, key: str) -> bool:
        return self.hashes.pop(key, None) is not None

    async def rename(self, src: str, dst: str) -> None:
        self.hashes[dst] = self.hashes.pop(src)  # KeyError if src missing, as in Redis

    async def write_freshness(self, key: str, count: int, ttl: int = 0) -> None:
        self.freshness_writes.append(count)


def _install_reload(monkeypatch: pytest.MonkeyPatch, fake_redis: _FakeRedis) -> Any:
    """Wire a SubclassMapping to the fake Redis (small batch size); return the instance."""
    import retriever.lookup.subclass as subclass_mod

    sm = subclass_mod.SubclassMapping()
    monkeypatch.setattr(subclass_mod, "REDIS_CLIENT", fake_redis)
    monkeypatch.setattr(sm, "redis_setup_batch_size", 2)
    return sm


@pytest.mark.asyncio
async def test_reload_mapping_streams_kept_entries_to_redis(
    monkeypatch: pytest.MonkeyPatch,
):
    import retriever.lookup.subclass as subclass_mod

    fake_redis = _FakeRedis()
    sm = _install_reload(monkeypatch, fake_redis)
    pairs = [(f"c{i}", [f"d{i}"]) for i in range(5)]
    monkeypatch.setattr(
        subclass_mod.tier_manager, "get_driver", lambda _tier: _FakeStreamDriver(pairs)
    )

    await sm._reload_mapping()

    live = fake_redis.hashes[subclass_mod.MAPPING_ID]
    assert set(live) == {f"c{i}" for i in range(5)}  # every kept entry, swapped in
    assert all(size <= 2 for size in fake_redis.hset_calls)  # batched under the cap
    assert fake_redis.freshness_writes == [5]  # freshness = kept count
    assert subclass_mod.MAPPING_BUILD_ID not in fake_redis.hashes  # temp consumed by rename


@pytest.mark.asyncio
async def test_reload_mapping_swaps_not_merges(monkeypatch: pytest.MonkeyPatch):
    import retriever.lookup.subclass as subclass_mod

    fake_redis = _FakeRedis()
    sm = _install_reload(monkeypatch, fake_redis)

    monkeypatch.setattr(
        subclass_mod.tier_manager,
        "get_driver",
        lambda _tier: _FakeStreamDriver([("a", ["1"]), ("b", ["2"]), ("c", ["3"])]),
    )
    await sm._reload_mapping()
    assert set(fake_redis.hashes[subclass_mod.MAPPING_ID]) == {"a", "b", "c"}

    # Build 2 omits "b" (vanished upstream / now over cutoff): it must not linger.
    monkeypatch.setattr(
        subclass_mod.tier_manager,
        "get_driver",
        lambda _tier: _FakeStreamDriver([("a", ["1"]), ("c", ["3"])]),
    )
    await sm._reload_mapping()
    assert set(fake_redis.hashes[subclass_mod.MAPPING_ID]) == {"a", "c"}


@pytest.mark.asyncio
async def test_reload_mapping_zero_entries_preserves_previous(
    monkeypatch: pytest.MonkeyPatch,
):
    import retriever.lookup.subclass as subclass_mod

    fake_redis = _FakeRedis()
    sm = _install_reload(monkeypatch, fake_redis)
    fake_redis.hashes[subclass_mod.MAPPING_ID] = {"a": b"x"}  # seed a prior good map
    monkeypatch.setattr(
        subclass_mod.tier_manager, "get_driver", lambda _tier: _FakeStreamDriver([])
    )

    await sm._reload_mapping()

    # A stream that yields nothing must not clobber the prior map or its freshness.
    assert fake_redis.hashes[subclass_mod.MAPPING_ID] == {"a": b"x"}
    assert fake_redis.freshness_writes == []
    assert subclass_mod.MAPPING_BUILD_ID not in fake_redis.hashes  # temp cleaned up
