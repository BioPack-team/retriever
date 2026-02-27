import importlib
from typing import Iterator, cast, Any

import pytest
import retriever.config.general as general_mod
import retriever.data_tiers.tier_1.elasticsearch.driver as driver_mod
from retriever.data_tiers.tier_1.elasticsearch.meta import extract_metadata_entries_from_blob, get_t1_indices
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.data_tiers.tier_1.elasticsearch.types import ESPayload, ESEdge
from payload.trapi_qgraphs import DINGO_QGRAPH, VALID_REGEX_QGRAPHS, INVALID_REGEX_QGRAPHS, ID_BYPASS_PAYLOAD
from retriever.utils.redis import REDIS_CLIENT
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
        (PAYLOAD_1, 4),
        (PAYLOAD_2, 32),
        (
                [PAYLOAD_0, PAYLOAD_1, PAYLOAD_2],
                [0, 4, 32]
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
        print(len(hits))



@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_metadata_retrieval():
    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()

    try:
        await driver.connect()
        assert driver.es_connection is not None
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    meta = await driver.get_metadata()

    # make sure each index has metadata extracted
    indices = await get_t1_indices(driver.es_connection)
    assert len(extract_metadata_entries_from_blob(meta, indices)) == len(indices)

    ops, nodes = await driver.get_operations()

    # with open("output_new.json", "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2)

    # _ops, _nodes = await driver.legacy_get_operations()


    # assert len(nodes) == 23


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "qgraph, expected_hits",
    [
        (DINGO_QGRAPH, 8),
        (ID_BYPASS_PAYLOAD, 6776),  # <-- adjust to the real number
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


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
async def test_ubergraph_info_retrieval():
    # --- MANUAL LOOP RESET ---
    # Since REDIS_CLIENT can retain stale connections/loop references from previous tests,
    # we need to force-reset the Redis connection pool to prevent 'Event loop is closed' errors.
    REDIS_CLIENT.client.connection_pool.reset()

    driver: driver_mod.ElasticSearchDriver = driver_mod.ElasticSearchDriver()
    try:
        await driver.connect()
        assert driver.es_connection is not None
    except Exception:
        pytest.skip("skipping es driver connection test: cannot connect")

    info = await driver.get_subclass_mapping()

    # print("total nodes", len(info["nodes"]))
    # print("adj list sample:")
    # for k, v in islice(info['mapping'].items(), 5):
    #     print(k, v)

    assert "mapping" in info
    assert len(info["mapping"]) == 122707
