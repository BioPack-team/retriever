import asyncio

from retriever.metadata.optable import get_trapi_metakg
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
        async with asyncio.timeout(
            query.timeout[-1] if query.timeout[-1] is not -1 else None
        ):
            metakg = await get_trapi_metakg(tuple(query.tiers))
            return 200, metakg
    except TimeoutError:
        return 500, ErrorDetail(detail="Building TRAPI MetaKG timed out.")
