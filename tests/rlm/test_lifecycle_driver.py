"""Unit tests for backend/agents/rlm/lifecycle_driver.py

All tests use MOCK tools — no real primitives are imported or called.
A FakeCtx provides the RunContext interface (project_dir + remaining_s).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


import pytest

from backend.agents.rlm.lifecycle_driver import drive_lifecycle_chain


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx(
    tmp_path: Path,
    remaining: float | None = None,
) -> SimpleNamespace:
    """Minimal fake RunContext with project_dir and remaining_s()."""
    project_dir = tmp_path / "test_proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_dir=project_dir,
        remaining_s=lambda: remaining,
    )


def _make_tools(
    *,
    understand_ret: dict | None = None,
    detect_ret: dict | None = None,
    plan_ret: dict | None = None,
    implement_ret: dict | None = None,
    run_ret: dict | None = None,
    verify_ret: dict | None = None,
) -> tuple[dict, dict]:
    """Build a tools dict and a parallel dict of MagicMocks for call inspection.

    Each mock's return_value is set to the supplied canned return, or a sensible
    default.  The mocks are accessible under the same key in the second dict.
    """
    mocks: dict[str, MagicMock] = {}

    def _mk(name: str, ret: Any) -> MagicMock:
        m = MagicMock(name=name)
        m.return_value = ret
        mocks[name] = m
        return m

    understand_default = {"sections": ["intro", "methods"]}
    detect_default = {"environment": "conda", "python": "3.10"}
    plan_default = {"method_spec": "SDAR", "paper_claim_map": {}}
    implement_default = {"ok": True, "code_path": "/fake/code"}
    run_default = {"success": True, "metrics": {"reward": 0.72}}
    verify_default = {"overall_score": 0.85, "meets_target": True}

    tools = {
        "understand_section": {
            "tool": _mk("understand_section", understand_ret or understand_default),
        },
        "detect_environment": {
            "tool": _mk("detect_environment", detect_ret or detect_default),
        },
        "plan_reproduction": {
            "tool": _mk("plan_reproduction", plan_ret or plan_default),
        },
        "implement_baseline": {
            "tool": _mk("implement_baseline", implement_ret or implement_default),
        },
        "run_experiment": {
            "tool": _mk("run_experiment", run_ret or run_default),
        },
        "verify_against_rubric": {
            "tool": _mk("verify_against_rubric", verify_ret or verify_default),
        },
    }
    return tools, mocks


def _no_emit(event: dict) -> None:  # noqa: ARG001
    pass


PAPER = "A long paper about self-distilled agentic RL."
RUBRIC = {"target_score": 0.7}


# ---------------------------------------------------------------------------
# 1. need_baseline → full chain called in order
# ---------------------------------------------------------------------------


def test_need_baseline_full_chain_called_in_order(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
    )

    expected_order = [
        "understand_section",
        "detect_environment",
        "plan_reproduction",
        "implement_baseline",
        "run_experiment",
        "verify_against_rubric",
    ]
    assert result["driven"] == expected_order, result["driven"]
    assert result["stopped_at"] is None
    assert result["stopped_reason"] is None
    # All mocks should have been called once.
    for name in expected_order:
        mocks[name].assert_called_once()


# ---------------------------------------------------------------------------
# 2. implement_baseline received the correct plan dict
# ---------------------------------------------------------------------------


def test_implement_baseline_receives_correct_plan(tmp_path):
    understand_ret = {"key": "understand_result"}
    detect_ret = {"key": "detect_result"}
    plan_ret = {"key": "plan_result"}
    impl_ret = {"ok": True, "code_path": "/synthetic/code"}

    tools, mocks = _make_tools(
        understand_ret=understand_ret,
        detect_ret=detect_ret,
        plan_ret=plan_ret,
        implement_ret=impl_ret,
    )
    ctx = _make_ctx(tmp_path)

    drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
    )

    mocks["implement_baseline"].assert_called_once()
    (plan_arg,), _ = mocks["implement_baseline"].call_args
    assert plan_arg == {
        "paper_claim_map": understand_ret,
        "environment_spec": detect_ret,
        "reproduction_contract": plan_ret,
    }, plan_arg


# ---------------------------------------------------------------------------
# 3. run_experiment received (code_path_from_impl, "")
# ---------------------------------------------------------------------------


def test_run_experiment_receives_code_path_from_impl(tmp_path):
    expected_code_path = "/impl/returned/code"
    tools, mocks = _make_tools(
        implement_ret={"ok": True, "code_path": expected_code_path},
    )
    ctx = _make_ctx(tmp_path)

    drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
    )

    mocks["run_experiment"].assert_called_once_with(expected_code_path, "")


# ---------------------------------------------------------------------------
# 4. need_experiment → implement NOT called; run_experiment uses ctx.project_dir/code
# ---------------------------------------------------------------------------


def test_need_experiment_skips_implement_calls_run_with_project_dir_code(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)
    expected_code_path = str(ctx.project_dir / "code")

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    mocks["implement_baseline"].assert_not_called()
    mocks["understand_section"].assert_not_called()
    mocks["detect_environment"].assert_not_called()
    mocks["plan_reproduction"].assert_not_called()

    mocks["run_experiment"].assert_called_once_with(expected_code_path, "")
    mocks["verify_against_rubric"].assert_called_once()

    assert "run_experiment" in result["driven"]
    assert "verify_against_rubric" in result["driven"]
    assert "implement_baseline" not in result["driven"]


# ---------------------------------------------------------------------------
# 5. need_verification → ONLY verify_against_rubric called
# ---------------------------------------------------------------------------


def test_need_verification_only_verify_called(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_verification",
        emit=_no_emit,
    )

    # Only verify should run.
    mocks["verify_against_rubric"].assert_called_once()
    for name in ("understand_section", "detect_environment", "plan_reproduction",
                 "implement_baseline", "run_experiment"):
        mocks[name].assert_not_called()

    assert result["driven"] == ["verify_against_rubric"]
    assert result["stopped_at"] is None


# ---------------------------------------------------------------------------
# 6. Fail-soft: implement_baseline ok=False → stop, run_experiment NOT called
# ---------------------------------------------------------------------------


def test_failsoft_impl_ok_false_stops_chain(tmp_path):
    tools, mocks = _make_tools(
        implement_ret={"ok": False, "error": "boom"},
    )
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
    )

    assert result["stopped_at"] == "implement_baseline"
    assert "boom" in result["stopped_reason"]
    mocks["run_experiment"].assert_not_called()
    mocks["verify_against_rubric"].assert_not_called()
    # No exception escaped.
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 7. Fail-soft on raise: run_experiment raises → stops, no exception escapes
# ---------------------------------------------------------------------------


def test_failsoft_run_experiment_raises_no_exception(tmp_path):
    tools, mocks = _make_tools()
    mocks["run_experiment"].side_effect = RuntimeError("GPU exploded")
    ctx = _make_ctx(tmp_path)

    # Must not raise.
    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    assert result["stopped_at"] == "run_experiment"
    assert "GPU exploded" in result["stopped_reason"]
    mocks["verify_against_rubric"].assert_not_called()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 8. Wall-clock gate: remaining_s() < min_remaining_s → stop before first step
# ---------------------------------------------------------------------------


def test_wallclock_stops_before_first_step(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path, remaining=100.0)  # 100s remaining

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
        min_remaining_s=300.0,  # need 300s → 100s is not enough
    )

    assert result["stopped_reason"] == "low_wallclock"
    assert result["stopped_at"] == "understand_section"  # stopped at first step
    # No primitives may have been called.
    assert result["driven"] == []
    for name in ("understand_section", "detect_environment", "plan_reproduction",
                 "implement_baseline", "run_experiment", "verify_against_rubric"):
        mocks[name].assert_not_called()


# ---------------------------------------------------------------------------
# 9. Summary dict has all keys; rubric_score reflects verify's "overall_score"
# ---------------------------------------------------------------------------


def test_summary_keys_and_rubric_score(tmp_path):
    verify_ret = {"overall_score": 0.73, "meets_target": False}
    tools, _ = _make_tools(verify_ret=verify_ret)
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
    )

    required_keys = {"driven", "stopped_at", "stopped_reason", "final_result", "rubric_score"}
    assert required_keys <= result.keys(), f"Missing keys: {required_keys - result.keys()}"
    assert result["rubric_score"] == pytest.approx(0.73)


# ---------------------------------------------------------------------------
# 10. can_finalize → no-op summary with already_finalizable reason
# ---------------------------------------------------------------------------


def test_can_finalize_noop(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="can_finalize",
        emit=_no_emit,
    )

    assert result["driven"] == []
    assert result["stopped_at"] is None
    assert result["stopped_reason"] == "already_finalizable"
    for name in ("understand_section", "detect_environment", "plan_reproduction",
                 "implement_baseline", "run_experiment", "verify_against_rubric"):
        mocks[name].assert_not_called()


# ---------------------------------------------------------------------------
# 11. Unknown stage → treated as already_finalizable
# ---------------------------------------------------------------------------


def test_unknown_stage_noop(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="some_future_stage",
        emit=_no_emit,
    )

    assert result["stopped_reason"] == "already_finalizable"
    assert result["driven"] == []


# ---------------------------------------------------------------------------
# 12. need_environment stage → same as need_experiment (no implement, run+verify)
# ---------------------------------------------------------------------------


def test_need_environment_skips_implement(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_environment",
        emit=_no_emit,
    )

    mocks["implement_baseline"].assert_not_called()
    mocks["run_experiment"].assert_called_once()
    mocks["verify_against_rubric"].assert_called_once()
    assert result["stopped_at"] is None


# ---------------------------------------------------------------------------
# 13. Emit is called for each step with correct event shape
# ---------------------------------------------------------------------------


def test_emit_called_per_step(tmp_path):
    tools, _ = _make_tools()
    ctx = _make_ctx(tmp_path)
    emitted_events: list[dict] = []

    def _collect_emit(event: dict) -> None:
        emitted_events.append(event)

    drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_verification",
        emit=_collect_emit,
    )

    assert len(emitted_events) == 1
    ev = emitted_events[0]
    assert ev["event"] == "lifecycle_drive_step"
    assert ev["primitive"] == "verify_against_rubric"
    assert ev["stage"] == "need_verification"


# ---------------------------------------------------------------------------
# 14. Emit that raises never stops the driver
# ---------------------------------------------------------------------------


def test_emit_exception_does_not_stop_driver(tmp_path):
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    def _bad_emit(_event: dict) -> None:
        raise RuntimeError("emit exploded")

    # Should not raise; full chain should complete.
    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_verification",
        emit=_bad_emit,
    )

    assert result["stopped_at"] is None
    mocks["verify_against_rubric"].assert_called_once()


# ---------------------------------------------------------------------------
# 15. Missing tool (KeyError equivalent) → fail-soft stop
# ---------------------------------------------------------------------------


def test_missing_tool_failsoft(tmp_path):
    tools, _ = _make_tools()
    # Remove verify_against_rubric from tools.
    del tools["verify_against_rubric"]
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_verification",
        emit=_no_emit,
    )

    assert result["stopped_at"] == "verify_against_rubric"
    assert "missing_tool" in result["stopped_reason"]
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 16. verify_against_rubric receives ({}, rubric_spec)
# ---------------------------------------------------------------------------


def test_verify_receives_correct_args(tmp_path):
    rubric = {"target_score": 0.9, "leaves": []}
    tools, mocks = _make_tools()
    ctx = _make_ctx(tmp_path)

    drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=rubric,
        start_stage="need_verification",
        emit=_no_emit,
    )

    mocks["verify_against_rubric"].assert_called_once_with({}, rubric)


# ---------------------------------------------------------------------------
# 17. run_experiment success with no ok key is NOT treated as a failure
# ---------------------------------------------------------------------------


def test_run_experiment_no_ok_key_is_success(tmp_path):
    """run_experiment returns {success, metrics, ...} — no 'ok' key.
    The driver must NOT treat this as ok=False."""
    tools, mocks = _make_tools(
        run_ret={"success": True, "metrics": {"reward": 0.5}},
    )
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    # run_experiment should not have stopped the chain.
    mocks["verify_against_rubric"].assert_called_once()
    assert result["stopped_at"] is None
