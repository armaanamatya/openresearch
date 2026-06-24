"""Tests for the flag-gated lifecycle-drive branch in _make_degenerate_loop_callback.

``OPENRESEARCH_LIFECYCLE_DRIVE=1`` (default OFF) enables the full lifecycle chain
driver on degenerate-root detection.  When the driven chain successfully calls
``run_experiment``, the callback RETURNS (handback) without marking a terminal
stop.  When ``run_experiment`` is NOT in the driven list, it falls through to the
existing early-abort.

Byte-identical-off contract: when ``drive_enabled=False`` (the default), the
``drive_lifecycle_chain`` callable is NEVER called and the existing Task-4
early-abort behaves unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from backend.agents.rlm.forced_iteration import ForcedIterationPolicy
from backend.agents.rlm.forced_iteration import _WALL_CLOCK_FLOOR_S
from backend.agents.rlm.run import _make_degenerate_loop_callback


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PAPER_TEXT = "PAPER"
_RUBRIC_SPEC = {"areas": [{"name": "A", "weight": 1.0, "leaves": []}]}


def _fake_ctx(*, remaining_s: float = 3600.0) -> Any:
    ctx = SimpleNamespace()
    ctx.remaining_s = lambda: remaining_s
    ctx._terminal_stop_reason = None
    ctx.project_dir = MagicMock()
    return ctx


def _fake_tools() -> dict:
    """Return a minimal tools dict with a no-op recommend_next_tool."""
    def _noop(*args, **kwargs):
        return {"ok": True}

    return {
        "recommend_next_tool": {"tool": _noop},
        "implement_baseline": {"tool": _noop},
        "run_experiment": {"tool": _noop},
    }


def _payload(stage: str = "need_baseline") -> dict:
    return {"signature": "no_progress", "count": 3, "required_stage": stage}


def _make_cb(
    *,
    emit,
    ctx,
    policy=None,
    drive_enabled: bool,
    paper_text: str | None = _PAPER_TEXT,
    rubric_spec: dict | None = None,
    tools=None,
):
    if policy is None:
        policy = ForcedIterationPolicy(min_iterations=2)
    if rubric_spec is None:
        rubric_spec = _RUBRIC_SPEC
    if tools is None:
        tools = _fake_tools()
    return _make_degenerate_loop_callback(
        emit=emit,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=False,
        tools=tools,
        oauth_root=True,
        drive_enabled=drive_enabled,
        paper_text=paper_text,
        rubric_spec=rubric_spec,
    )


# ---------------------------------------------------------------------------
# 1. Successful drive (run_experiment in driven) → HANDBACK, no terminal stop
# ---------------------------------------------------------------------------

def test_drive_enabled_run_experiment_handback() -> None:
    """drive_enabled=True + mock returns run_experiment in driven → handback."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    drive_calls: list = []

    summary = {
        "driven": ["understand_section", "implement_baseline", "run_experiment", "verify_against_rubric"],
        "stopped_at": None,
        "stopped_reason": None,
        "final_result": {},
        "rubric_score": 0.75,
    }

    def _mock_drive(*, tools, ctx, paper_text, rubric_spec, start_stage, emit, **kw):
        drive_calls.append({
            "start_stage": start_stage,
            "paper_text": paper_text,
            "rubric_spec": rubric_spec,
        })
        return summary

    cb = _make_cb(emit=emitted.append, ctx=ctx, drive_enabled=True)

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain", _mock_drive):
        cb(_payload("need_baseline"))

    # Handback: no terminal stop, mock called exactly once with correct args.
    assert ctx._terminal_stop_reason is None, "terminal stop must NOT be set on handback"
    assert len(drive_calls) == 1
    assert drive_calls[0]["start_stage"] == "need_baseline"
    assert drive_calls[0]["paper_text"] == _PAPER_TEXT
    # lifecycle_drive warning event must be emitted.
    codes = [e.get("data", {}).get("code") or e.get("code") for e in emitted]
    assert any("lifecycle_drive" in str(c) for c in codes), f"events: {emitted}"


def test_drive_enabled_run_experiment_handback_lifecycle_drive_count() -> None:
    """After a successful handback ctx._lifecycle_drive_count must be 1 (cap set)."""
    emitted: list[dict] = []
    ctx = _fake_ctx()

    summary = {
        "driven": ["run_experiment"],
        "stopped_at": None,
        "stopped_reason": None,
        "final_result": {},
        "rubric_score": 0.5,
    }

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain", return_value=summary):
        cb = _make_cb(emit=emitted.append, ctx=ctx, drive_enabled=True)
        cb(_payload("need_baseline"))

    assert getattr(ctx, "_lifecycle_drive_count", 0) == 1


# ---------------------------------------------------------------------------
# 2. Drive fires but run_experiment NOT in driven → falls through to early-abort
# ---------------------------------------------------------------------------

def test_drive_enabled_no_run_experiment_falls_to_early_abort() -> None:
    """mock returns summary WITHOUT run_experiment → ctx._terminal_stop_reason set."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    policy = ForcedIterationPolicy(min_iterations=2)

    summary = {
        "driven": ["understand_section", "implement_baseline"],
        "stopped_at": "implement_baseline",
        "stopped_reason": "ok=False",
        "final_result": {"ok": False},
        "rubric_score": None,
    }

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain", return_value=summary):
        cb = _make_cb(emit=emitted.append, ctx=ctx, policy=policy, drive_enabled=True)
        cb(_payload("need_baseline"))

    assert ctx._terminal_stop_reason is not None, "must early-abort when run_experiment not driven"
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"


# ---------------------------------------------------------------------------
# 3. drive_enabled=False → drive_lifecycle_chain NOT called; early-abort runs
# ---------------------------------------------------------------------------

def test_drive_disabled_no_drive_early_abort() -> None:
    """drive_enabled=False → drive_lifecycle_chain NOT called, existing abort path."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    drive_calls: list = []

    def _should_not_be_called(**kw):
        drive_calls.append(kw)
        return {}

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain", _should_not_be_called):
        cb = _make_cb(emit=emitted.append, ctx=ctx, drive_enabled=False)
        cb(_payload("need_baseline"))

    assert drive_calls == [], "drive must NOT be called when drive_enabled=False"
    assert ctx._terminal_stop_reason is not None
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"


# ---------------------------------------------------------------------------
# 4. Once-per-run cap: _lifecycle_drive_count=1 pre-set → drive NOT called
# ---------------------------------------------------------------------------

def test_once_per_run_cap_respected() -> None:
    """Pre-set ctx._lifecycle_drive_count=1 → drive NOT called → early-abort."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    ctx._lifecycle_drive_count = 1  # cap already spent
    drive_calls: list = []

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain",
               side_effect=lambda **kw: drive_calls.append(kw) or {}):
        cb = _make_cb(emit=emitted.append, ctx=ctx, drive_enabled=True)
        cb(_payload("need_baseline"))

    assert drive_calls == [], "drive must NOT fire when cap already spent"
    assert ctx._terminal_stop_reason is not None
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"


# ---------------------------------------------------------------------------
# 5. near_wall_clock → drive NOT called; existing wall-clock guard takes precedence
# ---------------------------------------------------------------------------

def test_near_wall_clock_drive_not_called() -> None:
    """remaining_s below _WALL_CLOCK_FLOOR_S → drive NOT called, no terminal stop."""
    emitted: list[dict] = []
    ctx = _fake_ctx(remaining_s=_WALL_CLOCK_FLOOR_S - 1.0)
    drive_calls: list = []

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain",
               side_effect=lambda **kw: drive_calls.append(kw) or {}):
        cb = _make_cb(emit=emitted.append, ctx=ctx, drive_enabled=True)
        cb(_payload("need_baseline"))

    assert drive_calls == [], "drive must NOT fire near wall clock"
    # The existing wall-clock guard returns without setting terminal stop.
    assert ctx._terminal_stop_reason is None, (
        "wall-clock guard returns without setting terminal (existing behaviour)"
    )


# ---------------------------------------------------------------------------
# 6. already_terminal → drive NOT called
# ---------------------------------------------------------------------------

def test_already_terminal_drive_not_called() -> None:
    """ctx._terminal_stop_reason pre-set → drive NOT called."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    ctx._terminal_stop_reason = {"kind": "something_else"}
    drive_calls: list = []

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain",
               side_effect=lambda **kw: drive_calls.append(kw) or {}):
        cb = _make_cb(emit=emitted.append, ctx=ctx, drive_enabled=True)
        cb(_payload("need_baseline"))

    assert drive_calls == [], "drive must NOT fire when terminal already set"
    # The existing already_terminal guard returns without overwriting.
    assert ctx._terminal_stop_reason == {"kind": "something_else"}


# ---------------------------------------------------------------------------
# 7. Byte-identical-off: existing autodrive tests are unaffected (spot-check)
# ---------------------------------------------------------------------------

def test_byte_identical_off_autodrive_still_works() -> None:
    """autodrive_enabled=True, drive_enabled=False → existing autodrive path unchanged."""
    emitted: list[dict] = []
    drive_calls: list = []

    ctx = _fake_ctx()
    policy = ForcedIterationPolicy(min_iterations=2)

    recommend_calls: list = []
    tools = {
        "recommend_next_tool": {"tool": lambda situation: recommend_calls.append(situation) or {"ok": True}},
        "implement_baseline": {"tool": lambda *a, **kw: {"ok": True}},
        "run_experiment": {"tool": lambda *a, **kw: {"ok": True}},
    }

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain",
               side_effect=lambda **kw: drive_calls.append(kw) or {}):
        cb = _make_degenerate_loop_callback(
            emit=emitted.append,
            ctx=ctx,
            policy=policy,
            autodrive_enabled=True,
            tools=tools,
            oauth_root=True,
            drive_enabled=False,
            paper_text=_PAPER_TEXT,
            rubric_spec=_RUBRIC_SPEC,
        )
        cb(_payload("need_baseline"))

    # drive_lifecycle_chain NOT called (drive_enabled=False).
    assert drive_calls == []
    # Existing autodrive path fired (recommend_next_tool was called).
    assert len(recommend_calls) == 1
    assert "implement_baseline" in recommend_calls[0]
    # No terminal stop (autodrive returned early).
    assert ctx._terminal_stop_reason is None


# ---------------------------------------------------------------------------
# 8. Handback calls policy.reset_repair_state() when run_experiment was driven
# ---------------------------------------------------------------------------

def test_handback_calls_reset_repair_state() -> None:
    """When run_experiment is in the driven list, handback must call reset_repair_state.

    This ensures the policy's stale repairable marker is cleared so the root's
    next FINAL_VAR is not bounced by an unmet repair floor.
    """
    emitted: list[dict] = []
    ctx = _fake_ctx()

    reset_calls: list = []

    policy = ForcedIterationPolicy(min_iterations=2)
    # Spy on reset_repair_state without replacing it
    _orig_reset = policy.reset_repair_state

    def _spy_reset():
        reset_calls.append(True)
        _orig_reset()

    policy.reset_repair_state = _spy_reset  # type: ignore[method-assign]

    summary = {
        "driven": ["run_experiment", "verify_against_rubric"],
        "stopped_at": None,
        "stopped_reason": None,
        "final_result": {},
        "rubric_score": 0.7,
    }

    with patch("backend.agents.rlm.lifecycle_driver.drive_lifecycle_chain", return_value=summary):
        cb = _make_cb(emit=emitted.append, ctx=ctx, policy=policy, drive_enabled=True)
        cb(_payload("need_experiment"))

    assert len(reset_calls) == 1, (
        f"reset_repair_state must be called exactly once on handback, got {len(reset_calls)}"
    )
    # Also confirm handback: no terminal stop set
    assert ctx._terminal_stop_reason is None
