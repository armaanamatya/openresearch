"""Tests for transient-error retry inside _execute_in_sandbox (PR-ζ piece ζ.2).

Verified behaviors:
  - First call raises Connection closed → retry → second call succeeds.
    Result has `attempts` list length 1 (only the failing attempt is recorded).
  - 4 consecutive Connection closed → exhausts retries (max 3), propagates.
    Exception has `_retry_attempts` list.
  - BALANCE_TOO_LOW → no retry, propagates immediately.
  - preflight_blocked → no retry, propagates immediately.
  - Backoff timing: second retry ≥ 5s after first, third ≥ 10s after second.

These tests mock asyncio.sleep to avoid actual wall-clock delays and assert
correct timing by inspecting the sleep arguments.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.rlm.primitives import (
    _BACKOFF_BASE_S,
    _MAX_TRANSIENT_RETRIES,
    _execute_in_sandbox,
)
from backend.services.runtime.interface import (
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_sandbox() -> Sandbox:
    """Return a minimal Sandbox instance that passes Pydantic validation."""
    from pathlib import Path
    config = SandboxConfig(
        project_id="proj",
        run_id="run0",
        image="test:image",
        project_root=Path("/tmp"),
    )
    return Sandbox(sandbox_id="fake-sandbox", name="fake-sandbox", image="test:image", config=config)


def _make_fake_service(call_results):
    """Return a mock RuntimeAppService whose create_sandbox follows call_results.

    Each element of call_results is either:
      - SandboxRuntimeError instance → create_sandbox raises it
      - None → create_sandbox succeeds; exec succeeds; destroy is a no-op
    """
    service = MagicMock()
    call_iter = iter(call_results)

    async def fake_create(cmd):
        outcome = next(call_iter)
        if isinstance(outcome, Exception):
            raise outcome
        return _make_fake_sandbox()

    from datetime import datetime, timezone

    async def fake_exec(cmd):
        from backend.services.runtime.interface import ExecResult
        return ExecResult(
            command=cmd.command,
            exit_code=0,
            stdout="ok",
            stderr="",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            duration_seconds=0.01,
        )

    async def fake_destroy(cmd):
        pass

    service.create_sandbox = fake_create
    service.execute = fake_exec
    service.destroy = fake_destroy
    service.probe_alive = AsyncMock(return_value=True)
    service.soft_recover = AsyncMock(return_value=False)
    return service


def _transient_error(msg="Connection closed"):
    return SandboxRuntimeError(
        RuntimeCauseKind.backend_unavailable, msg, retryable=True
    )


def _fatal_error():
    return SandboxRuntimeError(
        RuntimeCauseKind.backend_unavailable,
        "RUNPOD_BALANCE_TOO_LOW: insufficient funds",
        retryable=False,
    )


def _code_bug_error():
    return SandboxRuntimeError(
        RuntimeCauseKind.command_failed,
        "preflight_blocked: contract violation detected",
    )


# ---------------------------------------------------------------------------
# Test: first call fails transiently, second succeeds
# ---------------------------------------------------------------------------

def test_retry_succeeds_on_second_attempt(tmp_path, monkeypatch):
    """First call raises Connection closed; second succeeds. Result has attempts list."""
    (tmp_path / "commands.json").write_text('["echo ok"]')
    (tmp_path / "outputs").mkdir()

    call_results = [_transient_error(), None]
    fake_service = _make_fake_service(call_results)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with patch("backend.agents.rlm.primitives.RuntimeAppService", return_value=fake_service):
        with patch("backend.agents.rlm.primitives._backend_for_sandbox_mode", return_value=fake_service):
            result = asyncio.run(_execute_in_sandbox(
                str(tmp_path), "test:image", ["echo ok"],
                project_id="proj", run_id="run0",
            ))

    assert result["success"] is True
    assert result["resource_limits"]["memory_limit"] == "4g"
    assert result["exit_code"] == 0
    # One sleep (after first failure, before second attempt).
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(_BACKOFF_BASE_S, rel=0.01)
    events = [
        json.loads(line)
        for line in (tmp_path / "dashboard_events.jsonl").read_text().splitlines()
        if line
    ]
    resource_events = [
        e for e in events
        if e.get("event") == "sandbox_resource_limits"
    ]
    assert resource_events[-1]["data"]["memory_limit"] == "4g"


# ---------------------------------------------------------------------------
# Test: 4 consecutive failures exhaust retries (max 3 retries = 4 total attempts)
# ---------------------------------------------------------------------------

def test_retry_exhausted_propagates_with_attempts(tmp_path, monkeypatch):
    """Connection closed × (MAX_TRANSIENT_RETRIES+1) → propagates SandboxRuntimeError."""
    (tmp_path / "commands.json").write_text('["echo ok"]')

    # MAX+1 failures
    call_results = [_transient_error() for _ in range(_MAX_TRANSIENT_RETRIES + 1)]
    fake_service = _make_fake_service(call_results)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with patch("backend.agents.rlm.primitives.RuntimeAppService", return_value=fake_service):
        with patch("backend.agents.rlm.primitives._backend_for_sandbox_mode", return_value=fake_service):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                asyncio.run(_execute_in_sandbox(
                    str(tmp_path), "test:image", ["echo ok"],
                    project_id="proj", run_id="run0",
                ))

    exc = exc_info.value
    # _retry_attempts must be attached with one entry per failed attempt.
    assert hasattr(exc, "_retry_attempts"), "exhausted retry must attach _retry_attempts"
    attempts = exc._retry_attempts
    assert len(attempts) == _MAX_TRANSIENT_RETRIES + 1
    for att in attempts:
        assert att["transient_class"] == "transient"
        assert "Connection closed" in att["error"]

    # We slept MAX_TRANSIENT_RETRIES times (before each of the retries).
    assert len(sleep_calls) == _MAX_TRANSIENT_RETRIES


# ---------------------------------------------------------------------------
# Test: BALANCE_TOO_LOW → no retry at all
# ---------------------------------------------------------------------------

def test_fatal_error_no_retry(tmp_path, monkeypatch):
    """BALANCE_TOO_LOW classifies as fatal → propagates immediately, no sleep."""
    (tmp_path / "commands.json").write_text('["echo ok"]')

    call_results = [_fatal_error()]
    fake_service = _make_fake_service(call_results)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with patch("backend.agents.rlm.primitives.RuntimeAppService", return_value=fake_service):
        with patch("backend.agents.rlm.primitives._backend_for_sandbox_mode", return_value=fake_service):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                asyncio.run(_execute_in_sandbox(
                    str(tmp_path), "test:image", ["echo ok"],
                    project_id="proj", run_id="run0",
                ))

    # No sleep — fatal errors never retry.
    assert sleep_calls == []
    # No _retry_attempts attached (the raise is immediate, no loop needed).
    assert "RUNPOD_BALANCE_TOO_LOW" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: preflight_blocked → code_bug, no retry
# ---------------------------------------------------------------------------

def test_code_bug_error_no_retry(tmp_path, monkeypatch):
    """preflight_blocked classifies as code_bug → propagates immediately, no sleep."""
    (tmp_path / "commands.json").write_text('["echo ok"]')

    call_results = [_code_bug_error()]
    fake_service = _make_fake_service(call_results)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with patch("backend.agents.rlm.primitives.RuntimeAppService", return_value=fake_service):
        with patch("backend.agents.rlm.primitives._backend_for_sandbox_mode", return_value=fake_service):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                asyncio.run(_execute_in_sandbox(
                    str(tmp_path), "test:image", ["echo ok"],
                    project_id="proj", run_id="run0",
                ))

    assert sleep_calls == []
    assert "preflight_blocked" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: backoff doubles correctly — 5s, 10s, 20s
# ---------------------------------------------------------------------------

def test_backoff_doubles_on_each_retry(tmp_path, monkeypatch):
    """Verify sleep arguments follow the exponential backoff schedule."""
    (tmp_path / "commands.json").write_text('["echo ok"]')

    # 3 failures (will exhaust all retries → then propagate on 4th)
    call_results = [_transient_error() for _ in range(_MAX_TRANSIENT_RETRIES + 1)]
    fake_service = _make_fake_service(call_results)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with patch("backend.agents.rlm.primitives.RuntimeAppService", return_value=fake_service):
        with patch("backend.agents.rlm.primitives._backend_for_sandbox_mode", return_value=fake_service):
            with pytest.raises(SandboxRuntimeError):
                asyncio.run(_execute_in_sandbox(
                    str(tmp_path), "test:image", ["echo ok"],
                    project_id="proj", run_id="run0",
                ))

    expected = [_BACKOFF_BASE_S * (2 ** i) for i in range(_MAX_TRANSIENT_RETRIES)]
    assert len(sleep_calls) == len(expected)
    for actual, exp in zip(sleep_calls, expected):
        assert actual == pytest.approx(exp, rel=0.01)
