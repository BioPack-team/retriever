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
from translator_tom.v1_6 import MetaQualifier, QEdge, Qualifier, QualifierConstraint

from retriever.data_tiers.utils import parse_dingo_metadata
from retriever.metadata import optable as optable_module
from retriever.metadata.optable import OpTableManager
from retriever.types.metakg import (
    FlatOperations,
    Operation,
    OperationTable,
    SortedOperations,
)
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


class _FakeDriver:
    """Minimal stand-in for a tier driver: an `up` flag and a mocked fetch."""

    def __init__(self, *, up: bool) -> None:
        self.up = up
        self.get_operations = AsyncMock(return_value=([], {}))


@pytest.fixture
def manager() -> Iterator[OpTableManager]:
    """A fresh OpTableManager isolated from the Singleton cache."""
    _ = Singleton._instances.pop(OpTableManager, None)
    yield OpTableManager()
    _ = Singleton._instances.pop(OpTableManager, None)


def _patch_drivers(
    monkeypatch: pytest.MonkeyPatch, drivers: dict[int, _FakeDriver]
) -> None:
    monkeypatch.setattr(optable_module.tier_manager, "get_driver", lambda t: drivers[t])


@pytest.mark.asyncio
async def test_collect_tier_ops_skips_down_driver(
    manager: OpTableManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A down tier is skipped without a doomed live fetch; up tiers still build."""
    up, down = _FakeDriver(up=True), _FakeDriver(up=False)
    _patch_drivers(monkeypatch, {0: up, 1: down})

    table = await manager._collect_tier_ops(bypass_cache=True)

    up.get_operations.assert_awaited_once()
    down.get_operations.assert_not_awaited()  # never issue the doomed fetch
    assert isinstance(table, OperationTable)


@pytest.mark.asyncio
async def test_collect_tier_ops_all_down_raises_without_fetch(
    manager: OpTableManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tier down: no fetch is issued and the build preserves the prior table."""
    down0, down1 = _FakeDriver(up=False), _FakeDriver(up=False)
    _patch_drivers(monkeypatch, {0: down0, 1: down1})

    with pytest.raises(ValueError, match="No tier drivers succeeded"):
        _ = await manager._collect_tier_ops(bypass_cache=True)

    down0.get_operations.assert_not_awaited()
    down1.get_operations.assert_not_awaited()


_QUAL_TYPE = "biolink:object_direction_qualifier"
_QUAL_VALUE = "increased"


def _dingo_metadata(qualifiers: dict[str, int]) -> dict:
    """Minimal DINGO metadata: one edge advertising the given qualifier types."""
    return {
        "schema": {
            "edges": [
                {
                    "subject_category": ["biolink:ChemicalEntity"],
                    "object_category": ["biolink:Gene"],
                    "predicate": "biolink:affects",
                    "attributes": [],
                    "qualifiers": qualifiers,  # DINGO gives type -> count, never values
                }
            ],
            "nodes": [],
        }
    }


def _qualified_edge() -> QEdge:
    """A query edge constraining the qualifier to a specific value."""
    return QEdge.model_construct(
        subject="n0",
        object="n1",
        predicates=["biolink:affects"],
        qualifier_constraints=[
            QualifierConstraint(
                qualifier_set=[
                    Qualifier(qualifier_type_id=_QUAL_TYPE, qualifier_value=_QUAL_VALUE)
                ]
            )
        ],
    )


def test_dingo_qualifier_type_means_all_values(manager: OpTableManager) -> None:
    """A type-only DINGO qualifier serves a qualified edge (values unspecified = all).

    Regression: parsing it as applicable_values=[] read as "serves zero values" and
    made _operation_applies reject every qualified query edge.
    """
    ops, _ = parse_dingo_metadata(
        _dingo_metadata({_QUAL_TYPE: 5}), 0, "infores:test-kp"
    )
    (op,) = ops

    assert op.qualifiers is not None
    assert op.qualifiers[0].applicable_values is None  # None = all, not [] = none

    kept, unmet = manager._operation_applies(op, _qualified_edge(), tier=0)
    assert kept is True
    assert unmet == []


def test_op_without_qualifiers_rejects_qualified_edge(manager: OpTableManager) -> None:
    """Guard against over-correction: an op advertising no qualifiers still can't
    serve a qualified edge."""
    ops, _ = parse_dingo_metadata(_dingo_metadata({}), 0, "infores:test-kp")
    (op,) = ops

    kept, _ = manager._operation_applies(op, _qualified_edge(), tier=0)
    assert kept is False


def _op(op_hash: str, applicable_values: list[str] | None) -> Operation:
    """An op on the ChemicalEntity-affects-Gene edge advertising one qualifier."""
    return Operation(
        hash=op_hash,
        tier=0,
        subject="biolink:ChemicalEntity",
        predicate="biolink:affects",
        object="biolink:Gene",
        api="infores:test-kp",
        attributes=[],
        qualifiers=[
            MetaQualifier.model_construct(
                qualifier_type_id=_QUAL_TYPE, applicable_values=applicable_values
            )
        ],
    )


@pytest.mark.asyncio
async def test_metakg_merge_all_values_dominates_enumerated(
    manager: OpTableManager,
) -> None:
    """Two ops on one edge for the same qualifier: one serves all values (None), one
    enumerates a subset. The merged MetaKG qualifier must stay 'all values' (None) --
    the unconstrained op must not be shrunk to the other's subset."""
    flat = FlatOperations()
    flat["h_all"] = _op("h_all", None)
    flat["h_sub"] = _op("h_sub", ["increased"])
    op_table = OperationTable(SortedOperations(), flat, {})
    manager.get_op_table = AsyncMock(return_value=op_table)  # pyright: ignore[reportAttributeAccessIssue]

    mkg = await manager.get_trapi_metakg(0)

    (edge,) = mkg.edges
    assert edge.qualifiers is not None
    (qualifier,) = edge.qualifiers
    assert qualifier.applicable_values is None
