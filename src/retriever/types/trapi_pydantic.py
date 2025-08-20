from typing import Annotated

from pydantic import BaseModel, Field
from reasoner_pydantic import AsyncQuery as TRAPIAsyncQuery
from reasoner_pydantic import Query as TRAPIQuery
from reasoner_pydantic import Response as TRAPIResponse

TierNumber = Annotated[
    int,
    Field(ge=0, le=2, description="Data Tiers (0-2) to use. Defaults to 0 if unset."),
]


class Parameters(BaseModel):
    """Parameters that govern some elements of query execution behavior."""

    timeout: Annotated[
        float | None,
        Field(
            description="Custom query timeout in seconds. Defaults to server default if not set. Set to -1 to disable timeout entirely."
        ),
    ] = None
    tiers: list[TierNumber] | None = None


class Query(TRAPIQuery):
    """Request."""

    parameters: Parameters | None = None


class AsyncQuery(TRAPIAsyncQuery):
    """AsyncQuery."""

    parameters: Parameters | None = None


class Response(TRAPIResponse):
    """Response."""

    parameters: Annotated[
        Parameters, Field(description="Parameters used while executing the query.")
    ]
