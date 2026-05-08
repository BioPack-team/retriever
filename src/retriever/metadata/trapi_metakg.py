import asyncio

from translator_tom import MetaKnowledgeGraph

from retriever.metadata.optable import OpTableManager
from retriever.types.general import ErrorDetail, QueryInfo

OP_TABLE_MANAGER = OpTableManager()


async def trapi_metakg(
    query: QueryInfo,
) -> tuple[int, MetaKnowledgeGraph | ErrorDetail]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    try:
        async with asyncio.timeout(
            query.timeout[-1] if query.timeout[-1] is not -1 else None
        ):
            metakg = await OP_TABLE_MANAGER.get_trapi_metakg(tuple(query.tiers))
            return 200, metakg
    except TimeoutError:
        return 500, ErrorDetail(detail="Building TRAPI MetaKG timed out.")
