import asyncio
from http import HTTPStatus

from translator_tom.v1_6 import MetaKnowledgeGraph

from retriever.metadata.optable import OpTableManager
from retriever.types.general import ErrorDetail, QueryInfo

OP_TABLE_MANAGER = OpTableManager()


async def trapi_metakg(
    query: QueryInfo,
) -> tuple[HTTPStatus, MetaKnowledgeGraph | ErrorDetail]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    try:
        async with asyncio.timeout(query.timeout if query.timeout != -1 else None):
            metakg = await OP_TABLE_MANAGER.get_trapi_metakg(query.tier)
            return HTTPStatus.OK, metakg
    except TimeoutError:
        return HTTPStatus.INTERNAL_SERVER_ERROR, ErrorDetail(
            detail="Building TRAPI MetaKG timed out."
        )
