# check each index
# for existence of seq_
import pytest
from elasticsearch import Elasticsearch

from retriever.config.general import CONFIG
from retriever.data_tiers.tier_1.elasticsearch.meta import get_t1_indices

query = {
  "size": 1,
  "query": {
    "function_score": {
      "functions": [
        { "random_score": {} }
      ]
    }
  }
}





@pytest.fixture(scope="module")
def es_client():
    try:
        # should only be run in local setup, duh
        es_url = f"http://localhost:9200"
        client = Elasticsearch(es_url)
        # quick ping to check connection
        if not client.ping():
            pytest.skip("Elasticsearch is not available")
        yield client
    except ConnectionError:
        pytest.skip("Cannot connect to Elasticsearch")
    finally:
        if 'client' in locals():
            client.close()

@pytest.fixture(scope="module")
def tier1_indices(es_client):
    resp = es_client.indices.resolve_index(name=CONFIG.tier1.elasticsearch.index_name)
    if 'aliases' not in resp:
        raise Exception(f"Failed to get indices from ES: {CONFIG.tier1.elasticsearch.index_name}")

    backing_indices = []
    for a in resp.get("aliases", []):
        if a["name"] == "dingo":
            backing_indices.extend(a["indices"])

    return backing_indices

def test_required_fields(es_client, tier1_indices):
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
