import importlib
from collections.abc import Iterator
from typing import Any, cast

import pytest
import retriever.config.general as general_mod
import retriever.data_tiers.tier_1.elasticsearch.driver as driver_mod
from retriever.data_tiers.tier_1.elasticsearch.meta import extract_metadata_entries_from_blob, get_t1_indices
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.data_tiers.tier_1.elasticsearch.types import ESPayload, ESEdge, ESNode
from payload.trapi_qgraphs import DINGO_QGRAPH, VALID_REGEX_QGRAPHS, INVALID_REGEX_QGRAPHS, ID_BYPASS_PAYLOAD
from retriever.utils.redis import RedisClient
from test_tier1_transpiler import _convert_triple, _convert_batch_triple


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
    monkeypatch.setenv("TIER1__ELASTICSEARCH__CONNECT_RETRIES", "0")

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
        await driver.connect()
        assert driver.es_connection is not None
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

    await driver.close()


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
        await driver.connect()
        assert driver.es_connection is not None
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    for payload in qgraphs_with_valid_regex:
        hits: list[ESEdge] = await driver.run_query(payload)

    await driver.close()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_metadata_retrieval():
    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.connect()
        assert driver.es_connection is not None
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

    await driver.close()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_fetch_single_node():
    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.connect()
        assert driver.es_connection is not None
    except Exception:
        pytest.skip("skipping fetch_single_node test: cannot connect")

    node = await driver.fetch_single_node("CHEBI:48927")

    assert isinstance(node, ESNode)
    assert node is not None
    assert node.id == "CHEBI:48927"
    assert len(node.category) > 0
    assert node.name == "N-acyl-L-alpha-amino acid"

    await driver.close()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "qgraph, expected_hits",
    [
        (DINGO_QGRAPH, 8),
        (ID_BYPASS_PAYLOAD, 4176),  # <-- adjust to the real number
    ],
)
async def test_end_to_end(qgraph, expected_hits):
    transpiler = ElasticsearchTranspiler()
    payload = _convert_triple(transpiler, qgraph)

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.connect()
        assert driver.es_connection is not None
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    hits: list[ESEdge] = await driver.run_query(payload)

    assert len(hits) == expected_hits

    await driver.close()


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
        await driver.connect()
        assert driver.es_connection is not None
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

    await driver.close()

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
        await driver.connect()
        assert driver.es_connection is not None
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

    await driver.close()



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

    await driver.close()


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_ubergraph_info_retrieval():
    # --- MANUAL LOOP RESET ---
    # Since REDIS_CLIENT can retain stale connections/loop references from previous tests,
    # we need to force-reset the Redis connection pool to prevent 'Event loop is closed' errors.
    RedisClient().client.connection_pool.reset()

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()
    try:
        await driver.connect()
        assert driver.es_connection is not None
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    info = await driver.get_subclass_mapping(bypass_cache=True)

    # print("total nodes", len(info["nodes"]))
    # print("adj list sample:")
    # for k, v in islice(info['mapping'].items(), 5):
    #     print(k, v)

    # assert "mapping" in info
    assert len(info) == 122176

    await driver.close()
