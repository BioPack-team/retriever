import asyncio

from retriever.metakg.metakg import get_trapi_metakg
from retriever.types.general import ErrorDetail, QueryInfo
from retriever.types.trapi import (
    MetaKnowledgeGraphDict,
)


async def trapi_metakg(
    query: QueryInfo,
) -> tuple[int, MetaKnowledgeGraphDict | ErrorDetail]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    try:
        async with asyncio.timeout(query.timeout[-1]):
            metakg = await get_trapi_metakg(tuple(query.tiers))
    except TimeoutError:
        return 500, ErrorDetail(detail="Building TRAPI MetaKG timed out.")

    return 200, metakg
