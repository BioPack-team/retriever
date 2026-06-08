"""Unit tests for the job-status vocabulary helpers."""

from retriever.utils.job_status import (
    NON_TERMINAL,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
    resolve_status_filter,
)


def test_status_sets_are_disjoint() -> None:
    """No status string should appear in more than one of the three sets."""
    assert NON_TERMINAL.isdisjoint(TERMINAL_SUCCESS)
    assert NON_TERMINAL.isdisjoint(TERMINAL_FAILURE)
    assert TERMINAL_SUCCESS.isdisjoint(TERMINAL_FAILURE)


def test_failure_set_excludes_error() -> None:
    """`Error` is a TRAPI response status, not a persisted job status."""
    assert "Error" not in TERMINAL_FAILURE


def test_resolve_status_filter_none_passthrough() -> None:
    assert resolve_status_filter(None) is None


def test_resolve_status_filter_failed_alias() -> None:
    resolved = resolve_status_filter("failed")
    assert isinstance(resolved, list)
    assert set(resolved) == TERMINAL_FAILURE


def test_resolve_status_filter_specific_status_passthrough() -> None:
    assert resolve_status_filter("Complete") == "Complete"
    assert resolve_status_filter("Running") == "Running"
