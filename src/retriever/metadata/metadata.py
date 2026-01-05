import asyncio

from retriever.data_tiers.tier_manager import get_driver
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import ErrorDetail, QueryInfo


async def get_metadata(
    query: QueryInfo,
) -> tuple[int, DINGOMetadata | ErrorDetail]:
    """Obtain tier-specific metadata.

    Returns:
        A tuple of HTTP status code, response body.
    """
    try:
        async with asyncio.timeout(
            query.timeout[-1] if query.timeout[-1] is not -1 else None
        ):
            driver = get_driver(next(iter(query.tiers), 0))
            metadata = await driver.get_metadata()
            if metadata is None:
                return 500, ErrorDetail(detail="Metadata could not be retrieved.")
            return 200, DINGOMetadata(**metadata)
    except TimeoutError:
        return 500, ErrorDetail(detail="Retrieving backend metadata timed out.")
