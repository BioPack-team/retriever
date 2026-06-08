import asyncio
from http import HTTPStatus

from retriever.data_tiers.tier_manager import get_driver
from retriever.types.dingo import DINGOMetadata
from retriever.types.general import ErrorDetail, QueryInfo
from retriever.utils import service_health


async def get_metadata(
    query: QueryInfo,
) -> tuple[HTTPStatus, DINGOMetadata | ErrorDetail]:
    """Obtain tier-specific metadata.

    Returns:
        A tuple of HTTP status code, response body. None from the
        driver indicates the tier backend is unreachable and no cache
        layer can serve - surface as 424.
    """
    driver = get_driver(query.tier or 0)
    try:
        async with asyncio.timeout(query.timeout if query.timeout != -1 else None):
            metadata = await driver.get_metadata()
            if metadata is None:
                return HTTPStatus.FAILED_DEPENDENCY, service_health.outage_detail(
                    f"Tier {query.tier or 0} is unavailable; metadata cannot be retrieved.",
                    driver,
                )
            return HTTPStatus.OK, DINGOMetadata(**metadata)
    except TimeoutError:
        return HTTPStatus.INTERNAL_SERVER_ERROR, ErrorDetail(
            detail="Retrieving backend metadata timed out."
        )
