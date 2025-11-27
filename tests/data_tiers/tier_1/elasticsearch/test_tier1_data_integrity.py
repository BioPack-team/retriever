# check each index
# for existence of seq_
import pytest
from elasticsearch import Elasticsearch

from retriever.config.general import CONFIG

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


indices = [
    "ctd",
    "diseases",
    "gene2phenotype",
    "go_cam",
    "goa",
    "hpoa",
    "sider"
]


@pytest.fixture(scope="module")
def es_client():
    try:
        # should only be run in local setup, duh
        es_url = f"http://{CONFIG.tier1.elasticsearch.host}:{CONFIG.tier1.elasticsearch.port}"
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

@pytest.mark.parametrize("index_name", indices)
def test_required_fields(es_client, index_name):
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


