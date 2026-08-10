"""Unit tests for BackendClient's remote-apply API and broadcast observer.

These exercise the state machine that HealthCoordinator drives: locally
detected transitions fire the observer (and get broadcast), while
remote-applied ones mutate state without re-broadcasting.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retriever.utils.backend_client import BackendClient
from retriever.utils.general import Singleton


class _FakeClient(BackendClient):
    """Minimal concrete BackendClient with a no-op ping for state-machine tests."""

    async def ping(self) -> None:
        return None


@pytest.fixture
def client() -> _FakeClient:
    """A fresh _FakeClient, bypassing the Singleton cache for test isolation."""
    _ = Singleton._instances.pop(_FakeClient, None)
    return _FakeClient()


def test_health_key_defaults_to_class_name(client: _FakeClient) -> None:
    assert client.health_key == "_FakeClient"


def test_remote_outage_marks_down_without_broadcast(client: _FakeClient) -> None:
    observer = MagicMock()
    client.set_transition_observer(observer)

    client.note_remote_outage("boom")

    assert client.up is False
    assert client.last_error == "boom"
    assert client.last_outage is not None
    assert not client._up_event.is_set()
    assert client._check_event.is_set()  # nudged so the health loop takes over
    observer.assert_not_called()  # remote-applied transitions never re-broadcast


def test_remote_outage_is_idempotent(client: _FakeClient) -> None:
    observer = MagicMock()
    client.set_transition_observer(observer)

    client.note_remote_outage("first")
    first_outage = client.last_outage
    client.note_remote_outage("second")

    assert client.last_outage == first_outage  # still down; timestamp unchanged
    assert client.last_error == "second"  # error still refreshes
    observer.assert_not_called()


def test_remote_recovery_is_optimistic(client: _FakeClient) -> None:
    client.note_remote_outage("down")
    observer = MagicMock()
    client.set_transition_observer(observer)
    client._check_event.clear()

    client.note_remote_recovery()

    assert client.up is True
    assert client._up_event.is_set()
    assert client.last_recovery is not None
    assert client.last_error is None
    assert client._check_event.is_set()  # nudged to confirm promptly
    observer.assert_not_called()


def test_remote_recovery_noop_when_already_up(client: _FakeClient) -> None:
    observer = MagicMock()
    client.set_transition_observer(observer)

    assert client.up is True
    client.note_remote_recovery()

    assert client.up is True
    assert client.last_recovery is None  # no transition occurred
    observer.assert_not_called()


def test_local_transitions_fire_observer(client: _FakeClient) -> None:
    observer = MagicMock()
    client.set_transition_observer(observer)

    client._handle_ping_failure(RuntimeError("down"))
    observer.assert_called_once_with("outage")

    observer.reset_mock()
    client._handle_ping_success()
    observer.assert_called_once_with("recovery")


def test_local_transitions_idempotent_observer(client: _FakeClient) -> None:
    observer = MagicMock()
    client.set_transition_observer(observer)

    client._handle_ping_failure(RuntimeError("down"))
    client._handle_ping_failure(RuntimeError("still down"))
    observer.assert_called_once_with("outage")  # only the first flip broadcasts
