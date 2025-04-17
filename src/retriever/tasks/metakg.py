from typing import Any

from retriever.type_defs import Query


async def metakg(
    query: Query,  # pyright: ignore[reportUnusedParameter] Will be used in the future
) -> tuple[int, dict[str, Any]]:
    """Obtain a TRAPI-formatted meta-kg.

    Returns:
        A tuple of HTTP status code, response body.
    """
    return 200, {}
