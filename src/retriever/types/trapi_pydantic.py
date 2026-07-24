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
    tier_fallback: Annotated[
        bool,
        Field(
            description="When the requested tier is down, fall back to the other implemented tier (T0 ↔ T1). Set False to require the requested tier and 424 if it's unavailable. Default True. Has no effect on T2 (no fallback peer)."
        ),
    ] = True
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


class DataReleaseVersions(BaseModel):
    """Release versions of the knowledge used to answer the query."""

    translator_kg: Annotated[
        str | None,
        Field(description="Release version of the Tier 0 translator_kg."),
    ] = None


class Response(TRAPIResponse):
    """Response."""

    parameters: Annotated[
        Parameters, Field(description="Parameters used while executing the query.")
    ]
    data_release_versions: Annotated[
        DataReleaseVersions | None,
        Field(
            description="Release versions of knowledge sources used to answer the query."
        ),
    ] = None
