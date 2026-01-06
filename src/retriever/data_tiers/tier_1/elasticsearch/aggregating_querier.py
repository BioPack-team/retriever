from typing import Any, NotRequired

from elastic_transport import ObjectApiResponse
from elasticsearch import AsyncElasticsearch

from retriever.data_tiers.tier_1.elasticsearch.types import (
    ESDocument,
    ESEdge,
    ESPayload,
    ESResponse,
)


class QueryInfo(ESPayload):
    """Query info needed to generate full query body."""

    search_after: NotRequired[list[Any] | None]


class QueryBody(QueryInfo):
    """Full payload body for a paginated query."""

    sort: list[Any]
    size: int


async def parse_response(
    response: ObjectApiResponse[ESResponse], page_size: int
) -> tuple[list[ESEdge], list[Any] | None]:
    """Parse an ES response and for 0) list of hits, and 1) search_after i.e. the pagination anchor for next query."""
    if "hits" not in response:
        raise RuntimeError(f"Invalid ES response: no hits in response body: {response}")

    fetched_documents: list[ESDocument] = response["hits"]["hits"]

    search_after = None

    # next page exists
    if len(fetched_documents) == page_size:
        search_after = fetched_documents[-1]["sort"]

    hits = [ESEdge.from_dict(hit) for hit in fetched_documents]

    return hits, search_after


def generate_query_body(query_info: QueryInfo, page_size: int) -> QueryBody:
    """Generate a paginated query body for ES search/msearch endpoints."""
    query = query_info.get("query")
    search_after = query_info.get("search_after", None)

    body: QueryBody = {
        "size": page_size,
        "query": query,
        "sort": [{"seq_": "asc"}, {"_index": "asc"}],
    }

    if search_after is not None:
        body["search_after"] = search_after

    return body


async def run_single_query(
    es_connection: AsyncElasticsearch,
    index_name: str,
    query: ESPayload,
    page_size: int = 1000,
) -> list[ESEdge]:
    """Adapter for running single query through _search and aggregating all hits."""
    query_info: QueryInfo = {
        "query": query["query"],
    }

    results = list[ESEdge]()

    while True:
        query_body = generate_query_body(query_info, page_size)
        response = await es_connection.search(index=index_name, body=dict(query_body))
        hits, search_after = await parse_response(response, page_size)

        results.extend(hits)

        if search_after is None:
            break

        query_info["search_after"] = search_after

    return results


async def run_batch_query(
    es_connection: AsyncElasticsearch,
    index_name: str,
    queries: list[ESPayload],
    page_size: int = 1000,
) -> list[list[ESEdge]]:
    """Adapter for running batch queries through _msearch and aggregating all hits."""
    query_collection: list[QueryInfo] = [
        {
            "query": query["query"],
        }
        for query in queries
    ]

    results: list[list[ESEdge]] = [[] for _ in query_collection]

    current_query_indices = range(0, len(query_collection))

    while current_query_indices:
        next_query_indices: list[int] = []
        query_body_list: list[QueryBody | dict[Any, Any]] = []

        for i in current_query_indices:
            query_body_list.extend(
                [
                    {},
                    generate_query_body(query_collection[i], page_size),
                ]
            )

        responses = await es_connection.msearch(index=index_name, body=query_body_list)

        for current_query_order, collection_index in enumerate(current_query_indices):
            response = responses["responses"][current_query_order]
            hits, search_after = await parse_response(response, page_size)

            # update results
            results[collection_index].extend(hits)
            # update query_collection
            query_collection[collection_index]["search_after"] = search_after

            if search_after is not None:
                next_query_indices.append(collection_index)

        current_query_indices = next_query_indices

    return results
