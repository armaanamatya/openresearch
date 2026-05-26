"""Tests for PR-γ.2 — per-primitive wall-clock enforcement.

Covers:
  * A primitive that exceeds its timeout returns outcome=retryable within budget.
  * A primitive that returns instantly is unaffected.
  * implement_baseline and run_experiment are NOT in PRIMITIVE_TIMEOUT_S (regression).
  * An unknown primitive name uses the 1800s default.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.binding import (
    PRIMITIVE_TIMEOUT_S,
    _DEFAULT_PRIMITIVE_TIMEOUT_S,  # noqa: PLC2701 — tested private constant
    wrap_primitive,
)


# ---------------------------------------------------------------------------
# Minimal ctx stub that satisfies wrap_primitive
# ---------------------------------------------------------------------------


class _FakeDashboard:
    def primitive_call(self, *a: Any, **kw: Any) -> None:
        pass


class _FakeCostLedger:
    def append(self, *a: Any) -> None:
        pass

    def flush(self) -> None:
        pass


def _make_ctx(tmp_path: Path) -> Any:
    ctx = MagicMock()
    ctx.dashboard = _FakeDashboard()
    ctx.cost_ledger = _FakeCostLedger()
    ctx.emit = None
    ctx.project_dir = tmp_path
    ctx.project_id = "test_proj"
    ctx.provider = "anthropic"
    ctx.model = "test-model"
    ctx.llm_client = None
    return ctx


# ---------------------------------------------------------------------------
# Test 1: primitive that sleeps > timeout → retryable within budget
# ---------------------------------------------------------------------------


def test_slow_primitive_returns_retryable_within_budget(tmp_path: Path) -> None:
    """A primitive that would sleep 600s wrapped with a 2s timeout returns
    outcome='retryable' in under 5s (accounting for thread overhead)."""

    def _slow_primitive(*args: Any, ctx: Any = None, **kwargs: Any) -> dict:
        time.sleep(600)
        return {"success": True}

    ctx = _make_ctx(tmp_path)
    wrapped = wrap_primitive("understand_section", _slow_primitive, ctx)

    # Temporarily override the timeout to 2s for test speed.
    with patch.dict(PRIMITIVE_TIMEOUT_S, {"understand_section": 2}):
        start = time.monotonic()
        result = wrapped("some text")
        elapsed = time.monotonic() - start

    assert elapsed < 6, f"Should time out in under 6s, took {elapsed:.2f}s"
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("outcome") == "retryable", f"Expected retryable, got: {result}"
    assert result.get("error") == "primitive_hung", f"Expected primitive_hung, got: {result}"
    assert result.get("primitive") == "understand_section", (
        f"Expected primitive name in result, got: {result}"
    )
    assert result.get("wall_clock_s") == 2, f"Expected wall_clock_s=2, got: {result}"


# ---------------------------------------------------------------------------
# Test 2: primitive that returns instantly — no timeout interference
# ---------------------------------------------------------------------------


def test_fast_primitive_returns_normally(tmp_path: Path) -> None:
    """A primitive that returns instantly is unaffected by the timeout wrapper."""

    def _fast_primitive(*args: Any, ctx: Any = None, **kwargs: Any) -> dict:
        return {"success": True, "result": "ok"}

    ctx = _make_ctx(tmp_path)
    wrapped = wrap_primitive("check_user_messages", _fast_primitive, ctx)

    result = wrapped()

    assert isinstance(result, dict)
    assert result.get("success") is True
    assert result.get("result") == "ok"


# ---------------------------------------------------------------------------
# Test 3: implement_baseline NOT in PRIMITIVE_TIMEOUT_S (regression guard)
# ---------------------------------------------------------------------------


def test_implement_baseline_not_in_timeout_table() -> None:
    """implement_baseline must NOT appear in PRIMITIVE_TIMEOUT_S — it has its own cap."""
    assert "implement_baseline" not in PRIMITIVE_TIMEOUT_S, (
        "implement_baseline must not be in PRIMITIVE_TIMEOUT_S — it has a separate 4h cap"
    )


def test_run_experiment_not_in_timeout_table() -> None:
    """run_experiment must NOT appear in PRIMITIVE_TIMEOUT_S — it has its own cap."""
    assert "run_experiment" not in PRIMITIVE_TIMEOUT_S, (
        "run_experiment must not be in PRIMITIVE_TIMEOUT_S — it has a separate cap"
    )


# ---------------------------------------------------------------------------
# Test 4: unknown primitive name uses the 1800s default
# ---------------------------------------------------------------------------


def test_unknown_primitive_uses_default_timeout(tmp_path: Path) -> None:
    """An unrecognised primitive name uses _DEFAULT_PRIMITIVE_TIMEOUT_S = 1800s."""
    # We verify the lookup logic, not the actual sleep (too slow for a unit test).
    unknown = "some_new_future_primitive_xyz"
    assert unknown not in PRIMITIVE_TIMEOUT_S, "Precondition: name not in table"

    # The wrap_primitive wrapper uses PRIMITIVE_TIMEOUT_S.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S).
    # Verify the default is 1800.
    assert _DEFAULT_PRIMITIVE_TIMEOUT_S == 1800, (
        f"Default timeout should be 1800s, got {_DEFAULT_PRIMITIVE_TIMEOUT_S}"
    )

    # Also verify the lookup returns the default:
    timeout = PRIMITIVE_TIMEOUT_S.get(unknown, _DEFAULT_PRIMITIVE_TIMEOUT_S)
    assert timeout == 1800, f"Unknown primitive should get 1800s default, got {timeout}"


# ---------------------------------------------------------------------------
# Test 5: verify well-known primitives ARE in the table with correct values
# ---------------------------------------------------------------------------


def test_known_primitive_timeouts_are_correct() -> None:
    """Spot-check the timeout table for a few key primitives."""
    expected = {
        "understand_section": 300,
        "extract_hyperparameters": 300,
        "detect_environment": 600,
        "check_user_messages": 30,
        "respond_to_user": 30,
        "heartbeat": 30,
    }
    for name, secs in expected.items():
        assert PRIMITIVE_TIMEOUT_S[name] == secs, (
            f"PRIMITIVE_TIMEOUT_S[{name!r}] expected {secs}, got {PRIMITIVE_TIMEOUT_S[name]}"
        )


# ---------------------------------------------------------------------------
# Test 6: timeout result carries primitive name (not wrong name)
# ---------------------------------------------------------------------------


def test_timeout_result_carries_correct_primitive_name(tmp_path: Path) -> None:
    """The retryable result's 'primitive' field matches the primitive's name."""

    def _slow(*args: Any, ctx: Any = None, **kwargs: Any) -> dict:
        time.sleep(600)
        return {}

    ctx = _make_ctx(tmp_path)
    # Use a name that IS in the table so we know the timeout applies.
    wrapped = wrap_primitive("extract_hyperparameters", _slow, ctx)

    with patch.dict(PRIMITIVE_TIMEOUT_S, {"extract_hyperparameters": 2}):
        result = wrapped("some text")

    assert result.get("primitive") == "extract_hyperparameters", (
        f"Expected primitive='extract_hyperparameters', got: {result}"
    )
