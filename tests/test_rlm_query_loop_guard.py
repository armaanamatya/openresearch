"""Tests for ClaudeLlmClient.complete() loop-safe guard.

Verifies that calling complete() from any context — plain sync, inside a
running event loop — works correctly and propagates exceptions, matching the
pattern in hermes_audit/providers.py::ClaudeCodeSdkProvider.call.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client():
    """Return a ClaudeLlmClient with minimal construction (no real SDK needed)."""
    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

    return ClaudeLlmClient()


# ---------------------------------------------------------------------------
# Test 1 — sync context, no event loop
# ---------------------------------------------------------------------------


def test_complete_works_from_sync_context_no_loop():
    """complete() from a plain sync context uses asyncio.run directly."""
    client = _make_client()
    expected = "hello from sync"

    # _async_complete returns (result_text, usage_dict); complete() unpacks both.
    mock_coro = AsyncMock(return_value=(expected, {}))
    with patch.object(client, "_async_complete", mock_coro):
        result = client.complete(system="sys", user="usr")

    assert result == expected
    mock_coro.assert_awaited_once_with(system="sys", user="usr")


# ---------------------------------------------------------------------------
# Test 2 — called from inside a running event loop
# ---------------------------------------------------------------------------


def test_complete_works_from_inside_running_loop():
    """complete() from inside a running loop must NOT raise RuntimeError."""
    client = _make_client()
    expected = "hello from loop"

    # _async_complete returns (result_text, usage_dict); complete() unpacks both.
    mock_coro = AsyncMock(return_value=(expected, {}))

    async def _inner():
        with patch.object(client, "_async_complete", mock_coro):
            return client.complete(system="s", user="u")

    # asyncio.run() starts an event loop; _inner calls complete() from inside it.
    result = asyncio.run(_inner())

    assert result == expected
    mock_coro.assert_awaited_once_with(system="s", user="u")


# ---------------------------------------------------------------------------
# Test 3 — exception propagation from running-loop path
# ---------------------------------------------------------------------------


def test_complete_propagates_exceptions_from_running_loop_path():
    """Exceptions raised by _async_complete bubble through the loop path."""
    client = _make_client()

    async def _raise(*, system: str, user: str) -> str:
        raise ValueError("synthetic")

    async def _inner():
        with patch.object(client, "_async_complete", _raise):
            return client.complete(system="s", user="u")

    with pytest.raises(ValueError, match="synthetic"):
        asyncio.run(_inner())


# ---------------------------------------------------------------------------
# Test 4 — exception propagation from sync path
# ---------------------------------------------------------------------------


def test_complete_propagates_exceptions_from_sync_path():
    """Exceptions raised by _async_complete bubble through the sync path."""
    client = _make_client()

    async def _raise(*, system: str, user: str) -> str:
        raise ValueError("synthetic")

    with patch.object(client, "_async_complete", _raise):
        with pytest.raises(ValueError, match="synthetic"):
            client.complete(system="s", user="u")


# ---------------------------------------------------------------------------
# Test 5 — shutdown(wait=False) is called even on hung worker
# ---------------------------------------------------------------------------


def test_complete_does_not_block_on_hung_worker_in_loop_path():
    """A hung worker (future.result times out) returns empty — not blocks, not crashes.

    Contract (2026-05-29): complete() now passes a timeout to future.result() and,
    on TimeoutError, abandons the worker via shutdown(wait=False) and returns ""
    so the RLM loop continues (an empty completion is a no-op iteration) rather
    than the whole run wedging forever or crashing on a propagated TimeoutError.
    """
    client = _make_client()

    # Track whether shutdown(wait=False) was called.
    shutdown_calls: list[bool] = []

    # Build a mock executor whose submit().result() raises TimeoutError
    # immediately, simulating a hung worker.
    mock_future = MagicMock()
    mock_future.result.side_effect = concurrent.futures.TimeoutError("timed out")

    class _MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn):  # noqa: ANN001
            return mock_future

        def shutdown(self, wait: bool = True) -> None:
            shutdown_calls.append(wait)

    async def _inner():
        with patch(
            "concurrent.futures.ThreadPoolExecutor",
            _MockExecutor,
        ):
            return client.complete(system="s", user="u")

    result = asyncio.run(_inner())
    assert result == "", "hung worker (timeout) must return empty, not raise/block"

    # The finally block must have called shutdown(wait=False).
    assert shutdown_calls, "shutdown() was never called"
    assert shutdown_calls[-1] is False, (
        f"Expected shutdown(wait=False), got wait={shutdown_calls[-1]}"
    )
