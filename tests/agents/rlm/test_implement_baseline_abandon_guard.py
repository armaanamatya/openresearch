"""Tests for OPENRESEARCH_IMPL_ABANDON_GUARD (BUG A: abandon-and-harvest race).

implement_baseline's SDK aclose-stall watchdog (the ``elif _time.time() -
_stall_start > _ACLOSE_STALL_S`` branch) gives up on a writer thread/process
that is not confirmed to have stopped — ``pool.shutdown(wait=False,
cancel_futures=True)`` cannot cancel an already-running task — and, today,
harvests code_dir and can report ``ok=True`` even though the abandoned writer
may still be mutating it in the background. These tests pin:

1. Flag OFF (default): byte-identical to today — harvest-and-report-ok.
2. Flag ON: the give-up instead returns a distinguishable, never-"ok",
   repairable result (``failure_class="implement_timeout_abandoned"``).

Mirrors the stubbing idiom of
``tests/agents/rlm/test_implement_baseline_pre_emit_stall.py`` (fake
never-resolving Future via a patched ``ThreadPoolExecutor.submit``, cache
bypass). ``_ACLOSE_STALL_S`` lives at module scope in primitives.py
specifically so it can be monkeypatched down here instead of requiring a
real 120s wait.
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.context import RunContext


# ---------------------------------------------------------------------------
# Helpers (mirrors test_implement_baseline_pre_emit_stall.py)
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path: Path) -> RunContext:
    """Minimal RunContext for implement_baseline tests."""
    project_id = "prj_abandon_test"
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
    return {
        "paper_claim_map": {"core_contribution": "test paper"},
        "environment_spec": {"framework": "pytorch"},
        "reproduction_contract": None,
    }


def _make_blocking_future() -> concurrent.futures.Future:
    """A Future that never resolves — simulates a writer that is not confirmed dead."""
    fut: concurrent.futures.Future = concurrent.futures.Future()
    return fut


def _make_mock_cache() -> MagicMock:
    cache = MagicMock()
    cache.maybe_get.return_value = None  # always cache miss
    cache.put.return_value = None
    return cache


def _seed_harvestable_code(code_dir: Path) -> None:
    """Write a valid commands.json + a runnable file — the SAME shape a
    sub-agent's mid-write output can look like, which is exactly what makes
    give-up-and-harvest risky: the harvest looks complete even though the
    writer may not actually be finished.
    """
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]), encoding="utf-8")
    (code_dir / "train.py").write_text("# training script\n", encoding="utf-8")


def _run_implement_baseline_blocking(
    ctx: RunContext,
    plan: dict,
    *,
    monkeypatch: pytest.MonkeyPatch,
    join_timeout: float = 60.0,
) -> dict:
    """Drive implement_baseline on a background thread against a submitted
    future that never resolves, and return whatever it eventually returns.
    """
    blocking_future = _make_blocking_future()

    def _fake_submit(self, fn, *args, **kwargs):  # type: ignore[override]
        return blocking_future

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", _fake_submit)

    import backend.agents.rlm.primitives as prim_mod

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
        # The aclose-stall branch needs >= 2 real _POLL_S(10s)-bounded poll
        # iterations to arm + trip the (patched, tiny) _ACLOSE_STALL_S — give
        # it generous wall-clock, matching the sibling pre-emit-stall suite's
        # "CPU-starved under -n auto" budget.
        t.join(timeout=join_timeout)

    assert not exc_holder, f"implement_baseline raised: {exc_holder[0] if exc_holder else None}"
    assert result_holder, "implement_baseline did not return within the join timeout"
    return result_holder[0]


# ---------------------------------------------------------------------------
# Test 1: flag OFF — byte-identical to today (harvest + report ok)
# ---------------------------------------------------------------------------

def test_abandon_guard_off_harvests_and_reports_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF (default): the aclose-stall give-up still harvests code_dir
    and, since valid artifacts are present, reports ok=True — unchanged.
    """
    monkeypatch.delenv("OPENRESEARCH_IMPL_ABANDON_GUARD", raising=False)

    import backend.agents.rlm.primitives as prim_mod
    monkeypatch.setattr(prim_mod, "_ACLOSE_STALL_S", 0.5)

    ctx = _make_ctx(tmp_path)
    plan = _make_plan(ctx)
    _seed_harvestable_code(ctx.project_dir / "code")

    result = _run_implement_baseline_blocking(ctx, plan, monkeypatch=monkeypatch)

    assert isinstance(result, dict), f"expected dict, got {type(result)}: {result!r}"
    assert result.get("ok") is True, f"expected the pre-existing harvest-ok behavior: {result}"
    assert result.get("outcome") == "ok"


# ---------------------------------------------------------------------------
# Test 2: flag ON — never "ok", distinguishable repairable failure
# ---------------------------------------------------------------------------

def test_abandon_guard_on_returns_repairable_never_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag ON: the same give-up must NOT report ok=True — it must return a
    distinguishable repairable failure instead of harvesting a possibly
    -still-mutating code_dir.
    """
    monkeypatch.setenv("OPENRESEARCH_IMPL_ABANDON_GUARD", "1")

    import backend.agents.rlm.primitives as prim_mod
    monkeypatch.setattr(prim_mod, "_ACLOSE_STALL_S", 0.5)

    ctx = _make_ctx(tmp_path)
    plan = _make_plan(ctx)
    _seed_harvestable_code(ctx.project_dir / "code")

    result = _run_implement_baseline_blocking(ctx, plan, monkeypatch=monkeypatch)

    assert isinstance(result, dict), f"expected dict, got {type(result)}: {result!r}"
    assert result.get("ok") is False, f"guard must never report ok=True: {result}"
    assert result.get("success") is False, f"guard must never report success=True: {result}"
    assert result.get("outcome") == "repairable", f"expected outcome=repairable: {result}"
    assert result.get("repairable") is True
    assert result.get("failure_class") == "implement_timeout_abandoned", (
        f"expected a distinguishable failure_class: {result}"
    )
    # Never cached: caching a give-up result would cascade the same abandoned
    # state into every subsequent implement_baseline call in this run.
    assert "error" in result and result["error"], f"expected a human-readable error: {result}"
