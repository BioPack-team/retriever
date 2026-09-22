from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field
from translator_tom.v2_0 import AsyncQuery as TRAPIAsyncQuery
from translator_tom.v2_0 import Query as TRAPIQuery
from translator_tom.v2_0 import QueryParameters as TRAPIQueryParameters
from translator_tom.v2_0 import Response as TRAPIResponse
from translator_tom.v2_0 import TOMBase

TierNumber = Annotated[
    int,
    Field(ge=0, le=2, description="Data Tiers (0-2) to use. Defaults to 0 if unset."),
]


class Parameters(TRAPIQueryParameters):
    """Parameters that govern some elements of query execution behavior.

    Extends the TRAPI 2.0 `QueryParameters` (timeout/log_level/bypass_cache) with
    Retriever-specific execution controls.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    tiers: Annotated[
        list[TierNumber] | None,
        Field(
            max_length=1,
            deprecated=True,
            description="Which tier to use. Only supports 1 tier at a time. DEPRECATED: Use `tier` instead.",
        ),
    ] = None
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

    parameters: Parameters | None = None  # pyright:ignore[reportIncompatibleVariableOverride] Retriever extends QueryParameters


class AsyncQuery(TRAPIAsyncQuery):
    """AsyncQuery."""

    parameters: Parameters | None = None  # pyright:ignore[reportIncompatibleVariableOverride] Retriever extends QueryParameters


class DataReleaseVersions(TOMBase):
    """Release versions of the knowledge used to answer the query."""

    translator_kg: Annotated[
        str | None,
        Field(description="Release version of the Tier 0 translator_kg."),
    ] = None


class Response(TRAPIResponse):
    """Response."""

    parameters: Annotated[  # pyright:ignore[reportIncompatibleVariableOverride] Retriever extends QueryParameters
        Parameters | None,
        Field(description="Parameters used while executing the query."),
    ] = None
    data_release_versions: Annotated[  # pyright:ignore[reportIncompatibleVariableOverride] Structured form of the base dict
        DataReleaseVersions | None,
        Field(
            description="Release versions of knowledge sources used to answer the query."
        ),
    ] = None
