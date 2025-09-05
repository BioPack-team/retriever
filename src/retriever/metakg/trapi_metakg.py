from retriever.metakg.metakg import get_trapi_metakg
from retriever.types.general import QueryInfo
from retriever.types.trapi import (
    MetaKnowledgeGraphDict,
)


async def trapi_metakg(query: QueryInfo) -> tuple[int, MetaKnowledgeGraphDict]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    metakg = await get_trapi_metakg(tuple(query.tiers))

    return 200, metakg
