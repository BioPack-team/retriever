from reasoner_pydantic import (
    HashableMapping,
    HashableSequence,
    MetaEdge,
    MetaKnowledgeGraph,
    MetaNode,
)

from retriever.type_defs import QueryInfo


async def metakg(
    query: QueryInfo,  # pyright: ignore[reportUnusedParameter] Will be used in the future
) -> tuple[int, MetaKnowledgeGraph]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    # TODO: Actual metakg checking, take into account query.tier
    return 200, MetaKnowledgeGraph(
        nodes=HashableMapping[str, MetaNode](), edges=HashableSequence[MetaEdge]()
    )
