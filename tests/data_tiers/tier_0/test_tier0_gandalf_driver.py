"""Tests for GandalfDriver.get_release_version, the cached DINGO version accessor."""

from collections.abc import Iterator

import pytest

from retriever.data_tiers.tier_0.gandalf.driver import GandalfDriver


@pytest.fixture
def driver() -> Iterator[GandalfDriver]:
    """The Gandalf singleton with its cached metadata snapshotted and restored."""
    instance = GandalfDriver()
    original = instance.metadata
    try:
        yield instance
    finally:
        instance.metadata = original


def test_release_version_from_metadata(driver: GandalfDriver) -> None:
    """The top-level `version` string is returned when present."""
    driver.metadata = {"version": "2025-07-01"}
    assert driver.get_release_version() == "2025-07-01"


def test_release_version_none_without_metadata(driver: GandalfDriver) -> None:
    """No cached metadata yields no version."""
    driver.metadata = None
    assert driver.get_release_version() is None


def test_release_version_none_when_absent(driver: GandalfDriver) -> None:
    """Metadata lacking a `version` key yields no version."""
    driver.metadata = {"dateCreated": "2025-07-01"}
    assert driver.get_release_version() is None


def test_release_version_none_when_not_str(driver: GandalfDriver) -> None:
    """A non-string `version` is treated as unknown."""
    driver.metadata = {"version": 123}
    assert driver.get_release_version() is None
