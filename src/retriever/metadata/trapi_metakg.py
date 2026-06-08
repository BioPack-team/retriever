import asyncio
from http import HTTPStatus

from retriever.metadata.optable import OpTableManager
from retriever.types.general import ErrorDetail, QueryInfo
from retriever.types.trapi import (
    MetaKnowledgeGraphDict,
)

OP_TABLE_MANAGER = OpTableManager()


async def trapi_metakg(
    query: QueryInfo,
) -> tuple[HTTPStatus, MetaKnowledgeGraphDict | ErrorDetail]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    try:
        async with asyncio.timeout(query.timeout if query.timeout is not -1 else None):
            metakg = await OP_TABLE_MANAGER.get_trapi_metakg(query.tier)
            return HTTPStatus.OK, metakg
    except TimeoutError:
        return HTTPStatus.INTERNAL_SERVER_ERROR, ErrorDetail(
            detail="Building TRAPI MetaKG timed out."
        )
