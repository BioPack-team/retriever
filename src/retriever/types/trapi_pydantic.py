from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from reasoner_pydantic import AsyncQuery as TRAPIAsyncQuery
from reasoner_pydantic import Query as TRAPIQuery
from reasoner_pydantic import Response as TRAPIResponse

TierNumber = Annotated[
    int,
    Field(ge=0, le=2, description="Data Tiers (0-2) to use. Defaults to 0 if unset."),
]


class Parameters(BaseModel):
    """Parameters that govern some elements of query execution behavior."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    timeout: Annotated[
        float | None,
        Field(
            description="Custom query timeout in seconds. Defaults to server default if not set. Set to -1 to disable timeout entirely."
        ),
    ] = None
    tiers: (
        Annotated[
            list[TierNumber],
            Field(
                max_length=1,
                deprecated=True,
                description="Which tier to use. Only supports 1 tier at a time. DEPRECATED: Use `tier` instead.",
            ),
        ]
        | None
    ) = None
    tier: TierNumber | None = None
    dehydrated: Annotated[
        bool | None,
        Field(
            description="Respond without node/edge properties for faster response. Currently only supported for Tier 0."
        ),
    ] = None


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
