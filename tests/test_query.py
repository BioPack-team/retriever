"""Tests for query dispatch in retriever.query.

Focused on the timeout and tier resolution that the per-endpoint
make_*_query functions perform before handing a QueryInfo off to the
relevant query function.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.datastructures import Headers

from retriever import query as query_module
from retriever.config.general import CONFIG
from retriever.types.general import APIInfo, QueryInfo
from retriever.types.trapi_pydantic import Parameters
from retriever.types.trapi_pydantic import Query as TRAPIQuery


def _ctx(method: str = "GET", path: str = "/metadata/tier_1") -> APIInfo:
    """Build a minimal APIInfo for a synchronous (non-background) request."""
    request = SimpleNamespace(
        url=SimpleNamespace(path=path), method=method, headers=Headers()
    )
    response = SimpleNamespace()
    return APIInfo(request=request, response=response, background_tasks=None)  # pyright: ignore[reportArgumentType]


def _lookup_body(parameters: Parameters | None = None) -> TRAPIQuery:
    """Build a minimal valid TRAPI query body."""
    return TRAPIQuery.model_validate(
        {
            "message": {"query_graph": {"nodes": {}, "edges": {}}},
            **({"parameters": parameters.model_dump()} if parameters else {}),
        }
    )


async def _run(func, ctx, **kwargs) -> QueryInfo:
    """Run the relevant make_*_query function with all query functions mocked,
    returning the QueryInfo that was dispatched so its resolved timeout/tier
    can be asserted."""
    captured: dict[str, QueryInfo] = {}

    async def capture(query: QueryInfo):
        captured["query"] = query
        return 200, {}

    stub = AsyncMock(side_effect=capture)
    with (
        patch.object(query_module, "lookup", stub),
        patch.object(query_module, "trapi_metakg", stub),
        patch.object(query_module, "get_metadata", stub),
        patch.object(query_module.MONGO_QUEUE, "put", Mock()),
    ):
        if func == "lookup":
            await query_module.make_lookup_query(ctx, body=kwargs["body"])
        elif func == "metakg":
            await query_module.make_metakg_query(ctx, tier=kwargs.get("tier"))
        else:  # metadata
            await query_module.make_metadata_query(ctx, tier=kwargs["tier"])
    return captured["query"]


# --- Timeout resolution ---------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_timeout_uses_metakg_default():
    """Regression: a bodyless /metadata GET must fall back to the metakg
    timeout, not the boolean False (== 0) that caused instant timeouts."""
    query = await _run("metadata", _ctx(), tier=1)
    assert query.timeout == CONFIG.job.metakg.timeout
    assert query.timeout is not False


@pytest.mark.asyncio
async def test_metakg_timeout_uses_metakg_default():
    """A bodyless /meta_knowledge_graph GET must use the metakg timeout."""
    query = await _run("metakg", _ctx(path="/meta_knowledge_graph"), tier=None)
    assert query.timeout == CONFIG.job.metakg.timeout


@pytest.mark.asyncio
async def test_lookup_timeout_uses_tier_default():
    """A lookup without a custom timeout uses the per-tier default."""
    body = _lookup_body(Parameters(tier=1))
    query = await _run("lookup", _ctx("POST", "/query"), body=body)
    assert query.timeout == CONFIG.job.lookup.tier1_timeout


@pytest.mark.asyncio
async def test_custom_timeout_overrides_default():
    """An explicit parameters.timeout takes precedence over defaults."""
    body = _lookup_body(Parameters(timeout=42))
    query = await _run("lookup", _ctx("POST", "/query"), body=body)
    assert query.timeout == 42


@pytest.mark.asyncio
async def test_disabled_timeout_passed_through():
    """A custom timeout of -1 (disabled) is passed through unchanged."""
    body = _lookup_body(Parameters(timeout=-1))
    query = await _run("lookup", _ctx("POST", "/query"), body=body)
    assert query.timeout == -1


# --- Tier resolution ------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_tier_passed_through():
    """The tier from the path is used directly for metadata."""
    query = await _run("metadata", _ctx(path="/metadata/tier_2"), tier=2)
    assert query.tier == 2


@pytest.mark.asyncio
async def test_metakg_tier_stays_none():
    """metakg leaves tier unset (None) when none is provided."""
    query = await _run("metakg", _ctx(path="/meta_knowledge_graph"), tier=None)
    assert query.tier is None


@pytest.mark.asyncio
async def test_custom_tier_param_overrides():
    """parameters.tier overrides the default/path tier."""
    body = _lookup_body(Parameters(tier=1))
    query = await _run("lookup", _ctx("POST", "/query"), body=body)
    assert query.tier == 1


@pytest.mark.asyncio
async def test_deprecated_tiers_param_applied():
    """The deprecated parameters.tiers list selects its single tier."""
    body = _lookup_body(Parameters(tiers=[1]))
    query = await _run("lookup", _ctx("POST", "/query"), body=body)
    assert query.tier == 1


@pytest.mark.asyncio
async def test_custom_tier_takes_precedence_over_deprecated_tiers():
    """parameters.tier wins over the deprecated parameters.tiers."""
    body = _lookup_body(Parameters(tiers=[0], tier=1))
    query = await _run("lookup", _ctx("POST", "/query"), body=body)
    assert query.tier == 1


# --- Async telemetry contextualization -----------------------------------


@pytest.mark.asyncio
async def test_async_lookup_reapplies_data_tier_tag():
    """The async background task runs in its own Sentry transaction, so the
    request handler's tags don't reach it. async_lookup must re-apply them;
    here we assert data_tier lands in the Sentry tags for the async path.

    lookup() is stubbed to raise so the assertion is reached before the
    callback machinery - contextualize_query runs first, at the top of the try.
    """
    from retriever.lookup import lookup as lookup_module

    query = QueryInfo(
        endpoint="/asyncquery",
        headers=Headers(),
        method="POST",
        body={
            "message": {"query_graph": {"nodes": {}, "edges": {}}},
            "callback": "https://example.test/callback",
        },
        job_id="job123",
        tier=2,
        timeout=60.0,
    )

    with (
        patch("sentry_sdk.set_tags") as set_tags,
        patch.object(
            lookup_module, "lookup", AsyncMock(side_effect=RuntimeError("stop"))
        ),
        pytest.raises(RuntimeError),
    ):
        await lookup_module.async_lookup(query=query, ctx={})

    assert any(
        call.args and call.args[0].get("data_tier") == 2
        for call in set_tags.call_args_list
    )
