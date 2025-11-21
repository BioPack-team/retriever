import importlib
from typing import Iterator, cast, Any

import pytest
import retriever.config.general as general_mod
import retriever.data_tiers.tier_1.elasticsearch.driver as driver_mod
from retriever.data_tiers.tier_1.elasticsearch.types import ESPayload, ESHit


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
    'query': {
        'bool':
            {'filter': [
                {'terms': {'subject.all_categories': ['Protein', 'Gene']}},
                {'terms': {'object.id': ['UMLS:C0011847']}},
                {'terms': {'object.all_categories': ['Disease']}},
                {'terms': {'all_predicates': ['causes']}}]
            }
        }
    }
)
PAYLOAD_1: ESPayload = esp({
    "query":{
        "bool":{
            "filter":[
                {"terms":{"subject.id":["NCBITaxon:9606"]}}
            ]
        }
    }
})


PAYLOAD_2: ESPayload = esp( {'query': {'bool': {'filter': [{'terms': {'subject.all_categories': ['Gene']}}, {'terms': {'object.id': ['UMLS:C0011847']}}, {'terms': {'object.all_categories': ['Disease']}}, {'terms': {'all_predicates': ['causes']}}]}}})


@pytest.mark.usefixtures("mock_elasticsearch_config")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected",
    [
        (PAYLOAD_0, 42),
        (PAYLOAD_1, 85),
        (PAYLOAD_2, 22),
        (
            [PAYLOAD_0, PAYLOAD_1,PAYLOAD_2],
            [42,85,22]
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

    hits: list[ESHit] | list[ESHit]  = await driver.run_query(payload)

    def assert_single_result(res, expected_result_num: int):
        if not isinstance(res, list):
            raise AssertionError(f"Expected results to be list, got {type(res)}")
        if not len(res) == expected_result_num:
            raise AssertionError(f"Expected {expected_result_num} results, got {len(res)}")

    # check batch result
    if len(payload) > 1:
        assert len(hits) == len(payload)
        assert isinstance(hits[0], list)

        for index, result in enumerate(cast(list[list[ESHit]], hits)):
            assert_single_result(result, expected[index])
    else:
        assert_single_result(hits, expected)

    await driver.close()



