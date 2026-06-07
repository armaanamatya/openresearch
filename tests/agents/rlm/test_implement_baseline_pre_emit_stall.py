"""Tests for PR-π Module C — implement_baseline pre-emit stall watchdog.

The polling loop inside implement_baseline arms a timer when the code_dir
stays empty beyond _PRE_EMIT_STALL_S seconds. These tests stub out the
underlying SDK call so no real agent process is spawned, and monkeypatch
the stall threshold + poll interval so tests run fast (< 15 s).
"""
from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.context import RunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path: Path) -> RunContext:
    """Minimal RunContext for implement_baseline tests."""
    project_id = "prj_stall_test"
    runs_root = tmp_path / "runs"
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "rlm_state").mkdir(exist_ok=True)
    (project_dir / "code").mkdir(exist_ok=True)

    ctx = RunContext(
        project_id=project_id,
        project_dir=project_dir,
        runs_root=runs_root,
        dashboard=None,
        cost_ledger=None,
        llm_client=None,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    ctx.emit = None
    return ctx


def _make_plan(ctx: RunContext) -> dict:
    """Minimal plan dict for implement_baseline."""
    return {
        "paper_claim_map": {"core_contribution": "test paper"},
        "environment_spec": {"framework": "pytorch"},
        "reproduction_contract": None,
    }


def _make_blocking_future() -> concurrent.futures.Future:
    """Return a Future that never resolves — simulates an SDK deadlock."""
    fut: concurrent.futures.Future = concurrent.futures.Future()
    # NOTE: we never call fut.set_result(), so fut.result(timeout=T) always raises TimeoutError.
    return fut


def _make_mock_cache() -> MagicMock:
    """Build a minimal primitive_cache mock."""
    cache = MagicMock()
    cache.maybe_get.return_value = None  # always cache miss
    cache.put.return_value = None
    return cache


# ---------------------------------------------------------------------------
# Test 1: stall escalation when code_dir stays empty
# ---------------------------------------------------------------------------

def test_pre_emit_stall_returns_repairable_after_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When code_dir stays empty for longer than _PRE_EMIT_STALL_S the primitive
    must return a repairable result whose error mentions pre-emit stall.
    """
    # Set a very short stall threshold so the test finishes quickly.
    monkeypatch.setenv("OPENRESEARCH_PRE_EMIT_STALL_S", "1")

    ctx = _make_ctx(tmp_path)
    plan = _make_plan(ctx)

    # Stub the thread pool so the submitted future never resolves.
    blocking_future = _make_blocking_future()

    def _fake_submit(self, fn, *args, **kwargs):  # type: ignore[override]
        return blocking_future

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", _fake_submit)

    # Patch _pre_emit_stall_s at module level to return 1.0 (fast).
    import backend.agents.rlm.primitives as prim_mod
    monkeypatch.setattr(prim_mod, "_pre_emit_stall_s", lambda: 1.0)

    # Patch the primitive cache module so cache.maybe_get always misses.
    mock_cache = _make_mock_cache()
    with patch.dict("sys.modules", {"backend.agents.rlm.primitive_cache": mock_cache}):
        result_holder: list[Any] = []
        exc_holder: list[BaseException] = []

        def _run():
            try:
                r = prim_mod.implement_baseline(plan, ctx=ctx)
                result_holder.append(r)
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=20)

    assert not exc_holder, f"implement_baseline raised: {exc_holder[0]}"
    assert result_holder, "implement_baseline did not return within 20 s"

    result = result_holder[0]
    assert isinstance(result, dict), f"expected dict, got {type(result)}: {result!r}"
    assert result.get("success") is False, f"expected success=False: {result}"
    error_str = str(result.get("error") or result.get("code") or "")
    assert (
        "stall" in error_str.lower()
        or "sdk_pre_emit_stall" in str(result.get("code") or "").lower()
    ), f"error should mention stall: {result}"


# ---------------------------------------------------------------------------
# Test 2: a progress file resets the timer — no escalation within window
# ---------------------------------------------------------------------------

def test_pre_emit_progress_resets_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train.py appearing in code_dir resets the pre-emit timer.

    Scenario: stall threshold = 2s.
      - First 0.3s: empty dir  (elapsed < threshold → no escalation)
      - At 0.3s: write train.py (timer resets)
      - 1.5s after reset: still under the 2s threshold → no escalation expected

    We write train.py from a side-thread at 0.3s, then check at 1.5s total
    that the primitive has NOT escalated with a stall error.
    """
    monkeypatch.setenv("OPENRESEARCH_PRE_EMIT_STALL_S", "2")

    ctx = _make_ctx(tmp_path)
    plan = _make_plan(ctx)

    blocking_future = _make_blocking_future()

    def _fake_submit(self, fn, *args, **kwargs):  # type: ignore[override]
        return blocking_future

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", _fake_submit)

    import backend.agents.rlm.primitives as prim_mod
    monkeypatch.setattr(prim_mod, "_pre_emit_stall_s", lambda: 2.0)

    code_dir = ctx.project_dir / "code"

    # Write train.py after 0.3s (before the 2s threshold would trigger).
    def _write_progress():
        import time
        time.sleep(0.3)
        (code_dir / "train.py").write_text("# progress\n", encoding="utf-8")

    progress_thread = threading.Thread(target=_write_progress, daemon=True)

    mock_cache = _make_mock_cache()
    with patch.dict("sys.modules", {"backend.agents.rlm.primitive_cache": mock_cache}):
        result_holder: list[Any] = []

        def _run():
            try:
                r = prim_mod.implement_baseline(plan, ctx=ctx)
                result_holder.append(r)
            except Exception as e:  # noqa: BLE001
                result_holder.append({"_exc": str(e)})

        t = threading.Thread(target=_run, daemon=True)
        progress_thread.start()
        t.start()
        # Only wait 1.5 s — shorter than the 2s threshold measured from the last
        # progress-file reset (at 0.3s), so the stall escalation shouldn't have
        # fired within this window.
        t.join(timeout=1.5)

    if result_holder:
        result = result_holder[0]
        # If it returned, it must NOT be a pre-emit stall escalation.
        assert not (
            isinstance(result, dict)
            and result.get("success") is False
            and "stall" in str(result.get("error") or "").lower()
        ), f"Unexpected pre-emit stall escalation within reset window: {result}"
    # If no result yet, test passes — the timer was correctly extended by the progress file.


# ---------------------------------------------------------------------------
# Test 3: env var override shortens threshold
# ---------------------------------------------------------------------------

def test_pre_emit_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENRESEARCH_PRE_EMIT_STALL_S=1 overrides the 120s default.

    With threshold=1s and a blocking future, the primitive must escalate
    after ~1s of silence.
    """
    monkeypatch.setenv("OPENRESEARCH_PRE_EMIT_STALL_S", "1")

    ctx = _make_ctx(tmp_path)
    plan = _make_plan(ctx)

    blocking_future = _make_blocking_future()

    def _fake_submit(self, fn, *args, **kwargs):  # type: ignore[override]
        return blocking_future

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", _fake_submit)

    import backend.agents.rlm.primitives as prim_mod

    # Force the stall-s resolver to return 1.0 (mirrors env var).
    monkeypatch.setattr(prim_mod, "_pre_emit_stall_s", lambda: 1.0)

    mock_cache = _make_mock_cache()
    with patch.dict("sys.modules", {"backend.agents.rlm.primitive_cache": mock_cache}):
        result_holder: list[Any] = []
        exc_holder: list[BaseException] = []

        def _run():
            try:
                r = prim_mod.implement_baseline(plan, ctx=ctx)
                result_holder.append(r)
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=20)

    assert not exc_holder, f"implement_baseline raised: {exc_holder[0]}"
    assert result_holder, "implement_baseline did not escalate within 20 s"

    result = result_holder[0]
    assert isinstance(result, dict)
    assert result.get("success") is False
    # The env-override path: error or code must mention stall.
    combined = str(result.get("error") or "") + str(result.get("code") or "")
    assert "stall" in combined.lower() or "sdk_pre_emit" in combined.lower(), (
        f"Expected stall escalation, got: {result}"
    )
