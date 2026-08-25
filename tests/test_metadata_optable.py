"""Tests for OpTableManager.get_op_table's leader-wait / no-worker-publish policy.

A worker must wait for the leader-published OpTable and, only on timeout, build
locally *without* publishing -- it must never call the publishing
`build_operation_table`, which is reserved for the elected leader.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock

import pytest

from retriever.metadata import optable as optable_module
from retriever.metadata.optable import OpTableManager
from retriever.types.metakg import OperationTable
from retriever.utils.general import Singleton

_SENTINEL = cast(OperationTable, object())
"""Stand-in table compared by identity; get_op_table only checks `is not None`."""


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Iterator[OpTableManager]:
    """A fresh worker-role OpTableManager, isolated from the Singleton cache."""
    _ = Singleton._instances.pop(OpTableManager, None)
    mgr = OpTableManager()
    mgr.is_builder = False
    mgr._operation_table = None
    monkeypatch.setattr(optable_module, "OP_TABLE_WAIT_POLL_SECONDS", 0.001)
    monkeypatch.setattr(optable_module.REDIS_CLIENT, "up", True)
    # A worker publishing is the bug under test; fail loudly if it tries.
    mgr.build_operation_table = AsyncMock(  # pyright: ignore[reportAttributeAccessIssue]
        side_effect=AssertionError("worker must not build+publish the OpTable")
    )
    yield mgr
    _ = Singleton._instances.pop(OpTableManager, None)


@pytest.mark.asyncio
async def test_returns_present_table_without_pulling(worker: OpTableManager) -> None:
    """An already-populated table returns immediately, without pull or build."""
    worker._operation_table = _SENTINEL
    worker.pull_op_table = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue]
    worker.degraded_local_build = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue]

    assert await worker.get_op_table() is _SENTINEL
    worker.pull_op_table.assert_not_awaited()
    worker.degraded_local_build.assert_not_awaited()


@pytest.mark.asyncio
async def test_waits_for_leader_publish_then_returns(worker: OpTableManager) -> None:
    """The worker returns the leader-published table via pull, never building."""

    async def _publish(_msg: str = "") -> None:
        worker._operation_table = _SENTINEL

    worker.pull_op_table = AsyncMock(side_effect=_publish)  # pyright: ignore[reportAttributeAccessIssue]
    worker.degraded_local_build = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue]

    assert await worker.get_op_table() is _SENTINEL
    worker.pull_op_table.assert_awaited()
    worker.degraded_local_build.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_falls_back_to_unpublished_local_build(
    worker: OpTableManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the leader never publishes, the worker force-builds locally, unpublished."""
    monkeypatch.setattr(optable_module.CONFIG.job.metakg, "acquire_timeout", 0.02)
    worker.pull_op_table = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue]

    async def _local(*, force: bool = False) -> None:
        assert force is True
        worker._operation_table = _SENTINEL

    worker.degraded_local_build = AsyncMock(side_effect=_local)  # pyright: ignore[reportAttributeAccessIssue]

    assert await worker.get_op_table() is _SENTINEL
    worker.degraded_local_build.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_disabled_timeout_caps_wait_instead_of_hanging(
    worker: OpTableManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled acquire_timeout (-1) caps the wait, then force-builds locally."""
    monkeypatch.setattr(optable_module.CONFIG.job.metakg, "acquire_timeout", -1)
    monkeypatch.setattr(optable_module, "OP_TABLE_WAIT_CAP_SECONDS", 0.02)
    worker.pull_op_table = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue]

    async def _local(*, force: bool = False) -> None:
        assert force is True
        worker._operation_table = _SENTINEL

    worker.degraded_local_build = AsyncMock(side_effect=_local)  # pyright: ignore[reportAttributeAccessIssue]

    assert await worker.get_op_table() is _SENTINEL
    worker.degraded_local_build.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_redis_down_builds_locally_without_force(
    worker: OpTableManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Redis down there is no published copy to await; build locally at once."""
    monkeypatch.setattr(optable_module.REDIS_CLIENT, "up", False)
    worker.pull_op_table = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue]

    async def _local(*, force: bool = False) -> None:
        assert force is False
        worker._operation_table = _SENTINEL

    worker.degraded_local_build = AsyncMock(side_effect=_local)  # pyright: ignore[reportAttributeAccessIssue]

    assert await worker.get_op_table() is _SENTINEL
    worker.degraded_local_build.assert_awaited_once_with()
    worker.pull_op_table.assert_not_awaited()
