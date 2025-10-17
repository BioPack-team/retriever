from typing import NotRequired

from elasticsearch import AsyncElasticsearch

from retriever.data_tiers.tier_1.elasticsearch.types import ESHit, ESPayload

class QueryInfo(ESPayload):
    """
    Query info needed to generate full query body
    """
    search_after: NotRequired[str | None]

class QueryBody(QueryInfo):
    """
    Full payload body for a paginated query
    """
    sort: list
    size: int


async def parse_response(response, page_size: int) -> tuple[list[ESHit], str | None]:
    """
    Parse an ES response and for 0) list of hits, and 1) search_after i.e. the pagination achor for next query
    """
    if 'hits' not in response:
        raise RuntimeError(f"Invalid ES response: no hits in response body: {response}")

    hits = response['hits']['hits']

    search_after = None

    # next page exists
    if len(hits) == page_size:
        search_after = hits[-1]['sort']

    for hit in hits:
        hit.pop('sort', None)

    return hits, search_after

def generate_query_body(query_info: QueryInfo, page_size: int) -> QueryBody:
    query = query_info.get('query')
    search_after = query_info.get('search_after', None)

    return {
            "size": page_size,
            "query": query,
            "sort": [{"id": "asc"}],
            **({
                   "search_after": search_after,
               } if search_after is not None else {})
    }

async def run_single_query(es_connection: AsyncElasticsearch, index_name: str, query: ESPayload, page_size=1000) -> list[ESHit]:
    """
    Adapter for running single query through _search and aggregating all hits
    """

    query_info: QueryInfo = {
        "query": query['query'],
        "search_after":None,
    }

    results = []

    while True:
        query_body = generate_query_body(query_info, page_size)
        response = await es_connection.search(index=index_name, body=dict(query_body))
        hits, search_after = await parse_response(response, page_size)

        results.extend(hits)

        if search_after is None:
            break

        query_info["search_after"] = search_after

    return results

async def run_batch_query(es_connection: AsyncElasticsearch, index_name: str, queries: list[ESPayload], page_size=1000) -> list[list[ESHit]]:
    """
    Adapter for running batch queries through _msearch and aggregating all hits
    """

    query_collection : list[QueryInfo] = [
        {
            "query": query['query'],
            "search_after": None,
        } for query in queries
    ]

    results = [[] for _ in query_collection]

    current_query_indices = range(0, len(query_collection))

    while current_query_indices:
        next_query_indices = []

        query_body_list = []
        for i in current_query_indices:
            query_body_list.extend([
                {},
                generate_query_body(query_collection[i], page_size),
            ])

        responses = await es_connection.msearch(index=index_name, body=query_body_list)

        for current_query_order, collection_index in enumerate(current_query_indices):
            response = responses["responses"][current_query_order]
            hits, search_after = parse_response(response, page_size)

            # update results
            results[collection_index].extend(hits)
            # update query_collection
            query_collection[collection_index]['search_after'] = search_after

            if search_after is not None:
                next_query_indices.append(collection_index)

        current_query_indices = next_query_indices

    return results