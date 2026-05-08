# check each index
# for existence of seq_
from collections.abc import Iterator
from typing import Any

import elastic_transport
import pytest
from elasticsearch import Elasticsearch

from retriever.config.general import CONFIG

query: dict[str, Any] = {
    "size": 1,
    "query": {"function_score": {"functions": [{"random_score": {}}]}},
}



@pytest.fixture(scope="module")
def es_client() -> Iterator[Elasticsearch]:
    # should only be run in local setup, duh
    es_url = "http://localhost:9200"
    try:
        client = Elasticsearch(es_url)
        # quick ping to check connection
        if not client.ping():
            pytest.skip("Elasticsearch is not available")
    except (ConnectionError, elastic_transport.ConnectionError):
        pytest.skip("Cannot connect to Elasticsearch")

    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="module")
def tier1_indices(es_client: Elasticsearch) -> list[str]:
    resp = es_client.indices.resolve_index(name=CONFIG.tier1.elasticsearch.index_name)
    if "aliases" not in resp:
        raise Exception(f"Failed to get indices from ES: {CONFIG.tier1.elasticsearch.index_name}")

    backing_indices: list[str] = []
    for a in resp.get("aliases", []):
        if a["name"] == "dingo":
            backing_indices.extend(a["indices"])

    return backing_indices


def test_required_fields(es_client: Elasticsearch, tier1_indices: list[str]) -> None:
    for index_name in tier1_indices:
        resp = es_client.search(index=index_name, body=query)

        hits = resp["hits"]["hits"]

        # check no empty index
        assert len(hits) != 0

        doc = hits[0]["_source"]
        # check not empty doc
        assert doc

        # check required fields
        required_fields = [
            "seq_"
        ]

        for field in required_fields:
            assert field in doc

            if field == "seq_":
                assert isinstance(doc[field], int)
