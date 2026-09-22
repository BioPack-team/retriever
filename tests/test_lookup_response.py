"""Tests for response construction in initialize_lookup (parameters, data_release_versions)."""

from contextlib import AbstractContextManager
from unittest.mock import Mock, patch

from starlette.datastructures import Headers

from retriever.lookup import lookup as lookup_module
from retriever.types.general import QueryInfo
from retriever.types.trapi import Query


def _query() -> QueryInfo:
    """A minimal Tier 0 lookup QueryInfo."""
    return QueryInfo(
        endpoint="/query",
        method="POST",
        headers=Headers(),
        body=Query.model_validate(
            {
                "message": {
                    "query_graph": {
                        "nodes": {"n0": {"ids": ["CHEBI:1"]}, "n1": {}},
                        "edges": {"e0": {"subject": "n0", "object": "n1"}},
                    }
                }
            }
        ),
        job_id="job123",
        tier=0,
        timeout=60.0,
    )


def _patch_release_version(version: str | None) -> AbstractContextManager[Mock]:
    """Patch the Tier 0 driver's cached release version as seen by initialize_lookup."""
    driver = Mock()
    driver.get_release_version.return_value = version
    return patch("retriever.data_tiers.tier_manager.get_driver", return_value=driver)


def test_initialize_lookup_includes_data_release_versions() -> None:
    """A known Tier 0 release version is surfaced under translator_kg."""
    with _patch_release_version("2025-07-01"):
        _, _, response = lookup_module.initialize_lookup(_query())
    assert response.get("data_release_versions") == {"translator_kg": "2025-07-01"}


def test_initialize_lookup_omits_when_version_unknown() -> None:
    """An unknown release version omits data_release_versions entirely."""
    with _patch_release_version(None):
        _, _, response = lookup_module.initialize_lookup(_query())
    assert "data_release_versions" not in response


def _query_with_parameters(parameters: dict) -> QueryInfo:
    """A Tier 0 lookup QueryInfo carrying client-submitted parameters."""
    return QueryInfo(
        endpoint="/query",
        method="POST",
        headers=Headers(),
        body=Query.model_validate(
            {
                "message": {
                    "query_graph": {
                        "nodes": {"n0": {"ids": ["CHEBI:1"]}, "n1": {}},
                        "edges": {"e0": {"subject": "n0", "object": "n1"}},
                    }
                },
                "parameters": parameters,
            }
        ),
        job_id="job123",
        tier=0,
        timeout=60.0,
    )


def test_initialize_lookup_echoes_client_parameters() -> None:
    """Client-submitted parameters are echoed back in Response.parameters."""
    query = _query_with_parameters(
        {"log_level": "ERROR", "bypass_cache": True, "tier_fallback": False}
    )
    with _patch_release_version(None):
        _, _, response = lookup_module.initialize_lookup(query)

    echoed = response.get("parameters")
    assert echoed is not None
    assert echoed["log_level"] == "ERROR"
    assert echoed["bypass_cache"] is True
    assert echoed["tier_fallback"] is False
    assert echoed["tier"] == 0  # resolved tier reflected alongside the echo


def test_initialize_lookup_parameters_default_to_resolved_tier() -> None:
    """With no client parameters, Response.parameters still conveys the resolved tier."""
    with _patch_release_version(None):
        _, _, response = lookup_module.initialize_lookup(_query())

    assert response.get("parameters") == {"tier": 0}
