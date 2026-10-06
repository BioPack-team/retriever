"""Tests for GandalfDriver.get_release_version, the cached DINGO version accessor."""

from collections.abc import Iterator
from unittest.mock import AsyncMock

import orjson
import pytest

import retriever.data_tiers.tier_0.gandalf.driver as driver_mod
from retriever.data_tiers.tier_0.gandalf.driver import GandalfDriver


class _FakeRedis:
    """Redis stand-in with async set/get for the metadata publish/adopt paths."""

    def __init__(self, stored: bytes | None = None) -> None:
        self.set = AsyncMock()
        self.get = AsyncMock(return_value=stored)


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


def test_no_recovery_metadata_fetch(driver: GandalfDriver) -> None:
    """Recovery registers no callback, so workers never self-fetch on recovery.

    Regression: fetching on every recovery fanned N processes out onto Gandalf's
    /metadata; metadata now arrives via the builder's Redis publish instead.
    """
    assert driver._recovery_callback() is None


@pytest.mark.asyncio
async def test_publish_metadata_writes_redis(
    driver: GandalfDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The builder publishes the cached metadata compressed under the shared key."""
    fake = _FakeRedis()
    monkeypatch.setattr(driver_mod, "RedisClient", lambda: fake)
    driver.up = True
    driver.metadata = {"version": "2025-07-01"}

    await driver.publish_metadata()

    key, payload = fake.set.await_args.args
    assert key == driver_mod.GANDALF_METADATA_KEY
    assert orjson.loads(payload) == {"version": "2025-07-01"}
    assert fake.set.await_args.kwargs["compress"] is True


@pytest.mark.asyncio
async def test_publish_metadata_skips_when_down(
    driver: GandalfDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A down tier 0 doesn't publish, so the last-good Redis copy is preserved."""
    fake = _FakeRedis()
    monkeypatch.setattr(driver_mod, "RedisClient", lambda: fake)
    driver.up = False
    driver.metadata = {"version": "x"}

    await driver.publish_metadata()

    fake.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_adopt_metadata_loads_from_redis(
    driver: GandalfDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker adopts the builder's published copy into its own cache."""
    monkeypatch.setattr(
        driver_mod,
        "RedisClient",
        lambda: _FakeRedis(orjson.dumps({"version": "2025-07-01"})),
    )
    driver.metadata = None

    await driver.sync_metadata_from_cache()

    assert driver.get_release_version() == "2025-07-01"


@pytest.mark.asyncio
async def test_adopt_metadata_missing_key_keeps_copy(
    driver: GandalfDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No published copy (cold cluster) leaves the bootstrap-fetched metadata intact."""
    monkeypatch.setattr(driver_mod, "RedisClient", lambda: _FakeRedis(None))
    driver.metadata = {"version": "bootstrap"}

    await driver.sync_metadata_from_cache()

    assert driver.get_release_version() == "bootstrap"
