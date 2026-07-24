"""Tests for data_release_versions population in initialize_lookup."""

from contextlib import AbstractContextManager
from unittest.mock import Mock, patch

from starlette.datastructures import Headers

from retriever.lookup import lookup as lookup_module
from retriever.types.general import QueryInfo


def _query() -> QueryInfo:
    """A minimal Tier 0 lookup QueryInfo."""
    return QueryInfo(
        endpoint="/query",
        method="POST",
        headers=Headers(),
        body={"message": {"query_graph": {"nodes": {}, "edges": {}}}},
        job_id="job123",
        tier=0,
        timeout=60.0,
    )


def _patch_release_version(version: str | None) -> AbstractContextManager[Mock]:
    """Patch the Tier 0 driver's cached release version as seen by initialize_lookup."""
    driver = Mock()
    driver.get_release_version.return_value = version
    return patch(
        "retriever.data_tiers.tier_manager.get_driver", return_value=driver
    )


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
