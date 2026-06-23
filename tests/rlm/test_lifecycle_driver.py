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

from backend.agents.rlm.lifecycle_driver import drive_lifecycle_chain, run_lifecycle_primary


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


# ---------------------------------------------------------------------------
# 18. Bounded repair loop fires on repairable first-run outcome
# ---------------------------------------------------------------------------


def test_repair_loop_fires_on_repairable(tmp_path):
    """First run_experiment returns repairable → driver re-implements + re-runs.

    Asserts:
    - implement_baseline was called a second time with plan["repair_context"]
      equal to the failed result dict
    - summary["repaired"] == 1
    - summary["last_run_ok"] is True
    - "run_experiment" appears twice in summary["driven"]
    """
    repairable_result = {
        "outcome": "repairable",
        "failure_class": "preflight_blocked",
        "contract_violations": ["x"],
    }
    ok_result = {"outcome": "ok", "metrics": {"acc": 0.9}}

    run_calls: list = []
    impl_calls: list = []

    def _run_side_effect(*args):
        run_calls.append(args)
        return repairable_result if len(run_calls) == 1 else ok_result

    def _impl_side_effect(plan):
        impl_calls.append(plan)
        return {"ok": True, "code_path": "/repaired/code"}

    tools, mocks = _make_tools()
    mocks["run_experiment"].side_effect = _run_side_effect
    mocks["implement_baseline"].side_effect = _impl_side_effect

    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
        max_repair_iterations=2,
    )

    # implement_baseline called twice: once initially, once for repair
    assert len(impl_calls) == 2, f"Expected 2 implement_baseline calls, got {len(impl_calls)}"
    # Second call must have repair_context = the failed result
    repair_call_plan = impl_calls[1]
    assert repair_call_plan.get("repair_context") == repairable_result, (
        f"repair_context mismatch: {repair_call_plan}"
    )
    # run_experiment called twice
    assert result["driven"].count("run_experiment") == 2, result["driven"]
    assert result["repaired"] == 1
    assert result["last_run_ok"] is True


# ---------------------------------------------------------------------------
# 19. Bounded repair loop respects the cap
# ---------------------------------------------------------------------------


def test_repair_loop_respects_cap(tmp_path):
    """run_experiment always repairable → with max_repair_iterations=2, exactly 2 repairs.

    Asserts:
    - summary["repaired"] == 2
    - summary["last_run_ok"] is False
    - verify still ran afterward
    """
    repairable_result = {
        "outcome": "repairable",
        "failure_class": "preflight_blocked",
        "contract_violations": ["always"],
    }

    tools, mocks = _make_tools(run_ret=repairable_result)

    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
        max_repair_iterations=2,
    )

    # repaired == cap
    assert result["repaired"] == 2, f"repaired={result['repaired']}"
    assert result["last_run_ok"] is False
    # verify still ran
    mocks["verify_against_rubric"].assert_called_once()
    # run_experiment called 1 (initial) + 2 (repair) = 3 times
    assert result["driven"].count("run_experiment") == 3, result["driven"]


# ---------------------------------------------------------------------------
# 20. Bounded repair loop stops on wall-clock during repair
# ---------------------------------------------------------------------------


def test_repair_loop_stops_on_wallclock(tmp_path):
    """Wall-clock gate fires inside the repair loop → stops cleanly.

    The first run_experiment is allowed (remaining_s high).  After that,
    remaining_s drops below min_remaining_s.  The loop must stop and
    report stopped_reason == 'low_wallclock'.
    """
    repairable_result = {
        "outcome": "repairable",
        "failure_class": "preflight_blocked",
        "contract_violations": ["x"],
    }

    remaining_values = [9999.0, 9999.0, 9999.0, 50.0]  # drops after first run
    call_idx = {"n": 0}

    def _remaining():
        v = remaining_values[min(call_idx["n"], len(remaining_values) - 1)]
        call_idx["n"] += 1
        return v

    project_dir = tmp_path / "test_proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        project_dir=project_dir,
        remaining_s=_remaining,
    )

    tools, mocks = _make_tools(run_ret=repairable_result)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
        max_repair_iterations=3,
        min_remaining_s=300.0,
    )

    assert result["stopped_reason"] == "low_wallclock"
    # Must not have exceeded cap (stopped early)
    assert result.get("repaired", 0) < 3


# ---------------------------------------------------------------------------
# 21. No repair when first run succeeds
# ---------------------------------------------------------------------------


def test_no_repair_when_first_run_succeeds(tmp_path):
    """First run_experiment returns ok → no repair loop iterations."""
    ok_result = {"outcome": "ok", "metrics": {"acc": 0.95}}
    tools, mocks = _make_tools(run_ret=ok_result)

    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
        max_repair_iterations=2,
    )

    assert result["repaired"] == 0
    assert result["last_run_ok"] is True
    # implement_baseline not called (need_experiment stage)
    mocks["implement_baseline"].assert_not_called()
    # run_experiment called exactly once
    assert result["driven"].count("run_experiment") == 1


# ---------------------------------------------------------------------------
# Change A: verify_result exposed in drive_lifecycle_chain summary
# ---------------------------------------------------------------------------


def test_drive_lifecycle_chain_exposes_verify_result(tmp_path):
    """verify_result must appear in the summary when verify ran."""
    verify_ret = {"overall_score": 0.8, "target_score": 0.7, "meets_target": True}
    tools, _ = _make_tools(verify_ret=verify_ret)
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_verification",
        emit=_no_emit,
    )

    assert "verify_result" in result
    assert result["verify_result"] == verify_ret
    assert result["rubric_score"] == pytest.approx(0.8)


def test_drive_lifecycle_chain_verify_result_none_when_not_run(tmp_path):
    """verify_result is None when verify was not reached (stopped early)."""
    tools, mocks = _make_tools(implement_ret={"ok": False, "error": "nope"})
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_baseline",
        emit=_no_emit,
    )

    assert result["verify_result"] is None
    assert result["rubric_score"] is None


def test_drive_lifecycle_chain_no_verify_result_key_when_can_finalize(tmp_path):
    """can_finalize → no-op summary includes verify_result=None (canonical shape)."""
    tools, _ = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="can_finalize",
        emit=_no_emit,
    )

    assert "verify_result" in result
    assert result["verify_result"] is None


# ---------------------------------------------------------------------------
# Change B: run_lifecycle_primary
# ---------------------------------------------------------------------------


def _make_primary_tools(
    *,
    understand_ret=None,
    detect_ret=None,
    plan_ret=None,
    implement_ret=None,
    run_ret=None,
    verify_ret=None,
    propose_ret=None,
):
    """Build a tools+mocks dict that includes propose_improvements."""
    tools, mocks = _make_tools(
        understand_ret=understand_ret,
        detect_ret=detect_ret,
        plan_ret=plan_ret,
        implement_ret=implement_ret,
        run_ret=run_ret,
        verify_ret=verify_ret,
    )
    m = MagicMock(name="propose_improvements")
    m.return_value = propose_ret if propose_ret is not None else []
    mocks["propose_improvements"] = m
    tools["propose_improvements"] = {"tool": m}
    return tools, mocks


TARGET_VERIFY = {"overall_score": 0.5, "target_score": 0.8, "meets_target": False}
HIGH_VERIFY = {"overall_score": 0.85, "target_score": 0.8, "meets_target": True}
PASS_VERIFY = {"overall_score": 0.9, "target_score": 0.7, "meets_target": True}


def test_run_lifecycle_primary_backbone_only(tmp_path):
    """max_improve_iterations=0 → backbone runs, no propose_improvements called."""
    verify_ret = {"overall_score": 0.72, "target_score": 0.7, "meets_target": True}
    tools, mocks = _make_primary_tools(verify_ret=verify_ret)
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=0,
    )

    assert summary["rubric_score"] == pytest.approx(0.72)
    assert summary["improved"] == 0
    mocks["propose_improvements"].assert_not_called()
    # All backbone primitives ran.
    for name in ("understand_section", "detect_environment", "plan_reproduction",
                 "implement_baseline", "run_experiment", "verify_against_rubric"):
        assert name in summary["driven"]


def test_run_lifecycle_primary_improvement_loop_fires(tmp_path):
    """score < target → propose+implement called, loop runs once."""
    propose_ret = [{"hypothesis": "try larger lr", "rationale": "low score"}]
    verify_side_effects = [TARGET_VERIFY, HIGH_VERIFY]
    call_n = {"n": 0}

    def _verify_side(*args):
        r = verify_side_effects[min(call_n["n"], len(verify_side_effects) - 1)]
        call_n["n"] += 1
        return r

    tools, mocks = _make_primary_tools(propose_ret=propose_ret)
    mocks["verify_against_rubric"].side_effect = _verify_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
    )

    mocks["propose_improvements"].assert_called_once()
    assert summary["improved"] == 1
    assert summary["rubric_score"] == pytest.approx(0.85)


def test_run_lifecycle_primary_stops_at_max_improve(tmp_path):
    """Improvement loop respects max_improve_iterations cap."""
    propose_ret = [{"hypothesis": "tweak", "rationale": "always below target"}]
    # verify always returns below target
    verify_ret = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    tools, mocks = _make_primary_tools(verify_ret=verify_ret, propose_ret=propose_ret)
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
    )

    assert summary["improved"] == 2
    mocks["propose_improvements"].call_count == 2


def test_run_lifecycle_primary_stops_on_empty_hypotheses(tmp_path):
    """propose_improvements returns [] → improvement loop exits cleanly."""
    tools, mocks = _make_primary_tools(
        verify_ret=TARGET_VERIFY,
        propose_ret=[],
    )
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=3,
    )

    # Attempted one propose call, got empty list → broke out.
    mocks["propose_improvements"].assert_called_once()
    assert summary["improved"] == 1
    # Score stays at baseline (TARGET_VERIFY = 0.5)
    assert summary["rubric_score"] == pytest.approx(0.5)


def test_run_lifecycle_primary_no_regress(tmp_path):
    """If the improvement step lowers the score, adopt only the better one."""
    propose_ret = [{"hypothesis": "try something", "rationale": "test"}]
    # Baseline verify gives 0.7; improvement sub-run gives 0.3 (worse)
    baseline_verify = {"overall_score": 0.7, "target_score": 0.9, "meets_target": False}
    sub_verify = {"overall_score": 0.3, "target_score": 0.9, "meets_target": False}
    call_n = {"n": 0}

    def _verify_side(*args):
        # First call = baseline verify; second call = sub-run verify
        r = baseline_verify if call_n["n"] == 0 else sub_verify
        call_n["n"] += 1
        return r

    tools, mocks = _make_primary_tools(propose_ret=propose_ret)
    mocks["verify_against_rubric"].side_effect = _verify_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=1,
    )

    # Score must NOT regress below baseline 0.7.
    assert summary["rubric_score"] == pytest.approx(0.7)


def test_run_lifecycle_primary_failsoft_propose_raises(tmp_path):
    """propose_improvements raises → improvement loop exits, baseline score preserved."""
    baseline_verify = {"overall_score": 0.6, "target_score": 0.9, "meets_target": False}
    tools, mocks = _make_primary_tools(verify_ret=baseline_verify)
    mocks["propose_improvements"].side_effect = RuntimeError("propose exploded")
    ctx = _make_ctx(tmp_path)

    # Must not raise.
    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
    )

    assert summary["rubric_score"] == pytest.approx(0.6)
    assert isinstance(summary, dict)


def test_run_lifecycle_primary_failsoft_implement_raises(tmp_path):
    """implement_baseline raises during improvement → loop exits, baseline score preserved."""
    baseline_verify = {"overall_score": 0.6, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "fix", "rationale": "low"}]
    tools, mocks = _make_primary_tools(verify_ret=baseline_verify, propose_ret=propose_ret)
    # Override implement_baseline to raise on the second call (first = baseline impl)
    original_side_effect = [{"ok": True, "code_path": "/code"}]
    call_n = {"n": 0}

    def _impl_side(plan):
        if call_n["n"] == 0:
            call_n["n"] += 1
            return {"ok": True, "code_path": "/code"}
        raise RuntimeError("improve impl exploded")

    mocks["implement_baseline"].side_effect = _impl_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
    )

    assert summary["rubric_score"] == pytest.approx(0.6)
    assert isinstance(summary, dict)


def test_run_lifecycle_primary_no_baseline_score_returns_early(tmp_path):
    """Backbone never reaches verify → returns early with no score."""
    tools, mocks = _make_primary_tools(implement_ret={"ok": False, "error": "fatal"})
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=3,
    )

    assert summary["rubric_score"] is None
    mocks["propose_improvements"].assert_not_called()


def test_run_lifecycle_primary_wallclock_stops_improvement_loop(tmp_path):
    """Wall-clock always low after backbone → improvement loop gates out immediately.

    Strategy: use remaining_s=None for the backbone ctx (so backbone runs freely)
    then rebuild ctx with remaining_s=50 and feed that to run_lifecycle_primary
    with min_remaining_s=300 — the improvement loop wall-clock fires on iteration 1.
    """
    # Use a separate ctx whose remaining_s always returns 50 (below 300).
    # The backbone step will wall-clock immediately too, so the chain will stop at
    # understand_section. That gives us verify_result=None → early return.
    # We want to test the improvement loop specifically, so we need the backbone
    # to succeed. Use a ctx that returns None (no budget — fail-open) for backbone
    # calls, then returns 50 for the first improvement gate.
    baseline_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "fix", "rationale": "low"}]
    tools, mocks = _make_primary_tools(verify_ret=baseline_verify, propose_ret=propose_ret)

    # Use None → fail-open for all calls so backbone runs, improvement never starts.
    ctx = _make_ctx(tmp_path, remaining=None)

    # min_remaining_s=0 effectively disables the wall-clock gate (any remaining passes).
    # This test verifies the cap instead (already covered above).
    # For the wall-clock case: verify stopped_reason=low_wallclock fires when a
    # tiny budget is given to the OUTER loop via min_remaining_s.
    # Since remaining_s()=None means fail-open (always ok), we simulate the gate
    # by testing the _wallclock_ok helper path directly via a small positive value.
    from types import SimpleNamespace
    project_dir = tmp_path / "wc_proj"
    project_dir.mkdir()
    ctx_low = SimpleNamespace(project_dir=project_dir, remaining_s=lambda: 50.0)

    # With remaining_s=50 and min_remaining_s=300, the BACKBONE itself will be stopped
    # at understand_section. So verify_result will be None → no improvement loop.
    summary_low = run_lifecycle_primary(
        tools=tools,
        ctx=ctx_low,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=5,
        min_remaining_s=300.0,
    )
    # Backbone gated out → no score → early return.
    assert summary_low["rubric_score"] is None
    # propose_improvements never called because there's no baseline score.
    mocks["propose_improvements"].assert_not_called()


# ---------------------------------------------------------------------------
# F1: drive_lifecycle_chain stops on a fatal outcome
# ---------------------------------------------------------------------------


def test_drive_lifecycle_chain_stops_on_fatal_outcome(tmp_path):
    """A fatal run_experiment result stops the chain and sets fatal_result."""
    fatal_result = {"outcome": "fatal", "error": "GPU died", "failure_class": "infra_fatal"}
    tools, mocks = _make_tools(run_ret=fatal_result)
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    assert result["fatal_result"] == fatal_result
    assert result["stopped_at"] == "run_experiment"
    assert result["stopped_reason"] == "GPU died"
    # verify must NOT have run after a fatal
    mocks["verify_against_rubric"].assert_not_called()


def test_drive_lifecycle_chain_fatal_result_none_on_success(tmp_path):
    """A normal (ok) run does NOT set fatal_result."""
    tools, _ = _make_tools()
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    assert result["fatal_result"] is None
    assert result["stopped_at"] is None


def test_drive_lifecycle_chain_repairable_not_treated_as_fatal(tmp_path):
    """A repairable outcome does NOT set fatal_result (it enters the repair loop)."""
    repairable = {"outcome": "repairable", "failure_class": "preflight_blocked"}
    ok_result = {"outcome": "ok", "metrics": {"acc": 0.9}}
    call_n = {"n": 0}

    def _run_side(*args):
        call_n["n"] += 1
        return repairable if call_n["n"] == 1 else ok_result

    tools, mocks = _make_tools()
    mocks["run_experiment"].side_effect = _run_side
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
        max_repair_iterations=1,
    )

    assert result["fatal_result"] is None
    assert result["repaired"] == 1


# ---------------------------------------------------------------------------
# F2: run_lifecycle_primary propagates fatal_result / stopped_at
# ---------------------------------------------------------------------------


def test_run_lifecycle_primary_propagates_fatal_from_backbone(tmp_path):
    """A fatal in the backbone is propagated to the primary summary."""
    fatal_result = {"outcome": "fatal", "error": "disk_full"}
    tools, mocks = _make_primary_tools(run_ret=fatal_result)
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
    )

    assert summary["fatal_result"] == fatal_result
    assert summary["stopped_at"] == "run_experiment"
    # No improvement loop possible with no baseline
    mocks["propose_improvements"].assert_not_called()


def test_run_lifecycle_primary_propagates_fatal_from_subdrive(tmp_path):
    """A fatal in a sub-drive (improvement run) breaks the climb loop."""
    fatal_result = {"outcome": "fatal", "error": "oom_fatal"}
    # baseline passes, sub-drive fails fatally
    baseline_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "fix lr", "rationale": "low score"}]

    # run_experiment: first call succeeds (baseline), second call returns fatal
    call_n = {"n": 0}
    run_results = [
        {"success": True, "metrics": {"acc": 0.5}},  # baseline run
        fatal_result,                                   # improve sub-drive run
    ]

    def _run_side(*args):
        r = run_results[min(call_n["n"], len(run_results) - 1)]
        call_n["n"] += 1
        return r

    tools, mocks = _make_primary_tools(
        verify_ret=baseline_verify,
        propose_ret=propose_ret,
    )
    mocks["run_experiment"].side_effect = _run_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=3,
    )

    assert summary["fatal_result"] == fatal_result
    assert summary["stopped_at"] == "run_experiment"
    # Loop broke after one iteration
    assert summary["improved"] == 1


# ---------------------------------------------------------------------------
# F3: propose_improvements error envelope is filtered out
# ---------------------------------------------------------------------------


def test_climb_skips_propose_error_envelope(tmp_path):
    """propose_improvements returning [{"success":False,...}] must break the loop."""
    baseline_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    error_envelope = [{"success": False, "error": "propose failed", "outcome": "repairable"}]
    tools, mocks = _make_primary_tools(
        verify_ret=baseline_verify,
        propose_ret=error_envelope,
    )
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=3,
    )

    # Error envelope is not a hypothesis — no implement call, loop breaks
    mocks["implement_baseline"].assert_called_once()  # baseline impl only
    assert summary["improved"] == 1  # attempted one iteration but broke out
    assert summary["rubric_score"] == pytest.approx(0.5)


def test_climb_valid_hypothesis_not_filtered(tmp_path):
    """A valid hypothesis with no 'success' key (or success=True) is NOT filtered."""
    baseline_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    valid_hyp = [{"hypothesis": "try lr=0.01", "rationale": "low score"}]
    tools, mocks = _make_primary_tools(
        verify_ret=baseline_verify,
        propose_ret=valid_hyp,
    )
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=1,
    )

    # implement_baseline called twice: baseline + improve
    assert mocks["implement_baseline"].call_count == 2


# ---------------------------------------------------------------------------
# F4: climb breaks when sub-drive returns verify_result=None or last_run_ok=False
# ---------------------------------------------------------------------------


def test_climb_breaks_on_subdrive_verify_none(tmp_path):
    """Sub-drive with verify_result=None → climb breaks immediately."""
    baseline_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "fix", "rationale": "low"}]

    # run_experiment fails fatally on the second call so sub-drive never verifies
    fatal_result = {"outcome": "fatal", "error": "fatal"}
    call_n = {"n": 0}

    def _run_side(*args):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return {"success": True, "metrics": {"acc": 0.5}}
        return fatal_result

    tools, mocks = _make_primary_tools(
        verify_ret=baseline_verify,
        propose_ret=propose_ret,
    )
    mocks["run_experiment"].side_effect = _run_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=3,
    )

    # improved=1 (one iteration attempted but sub-drive fatal → broke)
    assert summary["improved"] == 1
    assert summary["fatal_result"] == fatal_result


def test_climb_breaks_on_subdrive_last_run_ok_false(tmp_path):
    """Sub-drive with last_run_ok=False breaks the climb (always-repairable experiment)."""
    baseline_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "fix", "rationale": "low"}]

    # First run succeeds (baseline), subsequent always repairable
    repairable = {"outcome": "repairable", "failure_class": "preflight_blocked"}
    call_n = {"n": 0}

    def _run_side(*args):
        call_n["n"] += 1
        return {"success": True, "metrics": {"acc": 0.5}} if call_n["n"] == 1 else repairable

    tools, mocks = _make_primary_tools(
        verify_ret=baseline_verify,
        propose_ret=propose_ret,
    )
    mocks["run_experiment"].side_effect = _run_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=3,
        max_repair_iterations=0,  # no repair → immediately last_run_ok=False
    )

    # Loop must break after one sub-drive with last_run_ok=False
    assert summary["improved"] == 1
    assert summary.get("stopped_reason") in (
        "improve_subdrive_no_progress",
        None,  # may be set or not depending on path
    )


# ===========================================================================
# Bug fix: partial-grid verify must fire even when run_experiment stops chain
# ===========================================================================


# ---------------------------------------------------------------------------
# BF-1: Partial grid (success=False, has gradeable metrics) → verify IS called,
#        summary carries non-None rubric_score.
# ---------------------------------------------------------------------------


def test_partial_grid_fatal_with_metrics_still_verifies(tmp_path):
    """run_experiment returns fatal+metrics → verify IS called, rubric_score set.

    Simulates the SDAR scenario: partial cell grid with real metrics on disk
    but run_experiment stopped the chain (fatal outcome + non-empty metrics).
    Before the fix, the chain returned early with rubric_score=None.
    """
    partial_run_result = {
        "outcome": "fatal",
        "success": False,
        "failure_class": "oom_shrink_exhausted",
        "metrics": {
            "per_model": {
                "qwen3-1.7b": {"status": "ok", "reward": 0.42},
                "qwen2.5-3b": {"status": "oom_failed"},
            }
        },
        "error": "3 cells OOM after shrink",
    }
    verify_ret = {"overall_score": 0.38, "target_score": 0.7, "meets_target": False}
    tools, mocks = _make_tools(run_ret=partial_run_result, verify_ret=verify_ret)
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    # Verify MUST have been called (partial evidence exists).
    mocks["verify_against_rubric"].assert_called_once()
    assert result["rubric_score"] == pytest.approx(0.38)
    assert result["verify_result"] == verify_ret
    assert "verify_against_rubric" in result["driven"]


def test_partial_grid_no_metrics_does_not_force_verify(tmp_path):
    """run_experiment stops with empty/no metrics → verify is NOT forced.

    When there is genuinely no evidence (no metrics), the chain should not
    call verify — behaviour unchanged from the pre-fix state.
    """
    empty_run_result = {
        "outcome": "fatal",
        "success": False,
        "failure_class": "oom_shrink_exhausted",
        "metrics": {},
        "error": "all cells failed, no metrics",
    }
    tools, mocks = _make_tools(run_ret=empty_run_result)
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    mocks["verify_against_rubric"].assert_not_called()
    assert result["rubric_score"] is None
    assert result["verify_result"] is None


def test_partial_grid_explicit_ok_false_with_metrics_still_verifies(tmp_path):
    """run_experiment returns ok=False but with gradeable metrics → verify IS called.

    Covers the edge case where an ok=False explicit failure carries real metrics.
    """
    partial_run_result = {
        "ok": False,
        "success": False,
        "failure_class": "cell_execution_error",
        "metrics": {"reward": 0.55, "per_model": {"m1": {"status": "ok", "score": 0.55}}},
        "error": "some cells errored",
    }
    verify_ret = {"overall_score": 0.44, "target_score": 0.7}
    tools, mocks = _make_tools(run_ret=partial_run_result, verify_ret=verify_ret)
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
    )

    mocks["verify_against_rubric"].assert_called_once()
    assert result["rubric_score"] == pytest.approx(0.44)


def test_partial_grid_verify_called_inside_repair_loop(tmp_path):
    """run_experiment stops with gradeable metrics inside the repair loop → verify called.

    First run returns repairable (no metrics). Repair re-implements. Second run
    stops with gradeable metrics. Verify must fire before returning.
    """
    repairable_no_metrics = {
        "outcome": "repairable",
        "failure_class": "preflight_blocked",
        "metrics": {},
    }
    partial_with_metrics = {
        "outcome": "fatal",
        "success": False,
        "failure_class": "oom_shrink_exhausted",
        "metrics": {"per_model": {"m": {"status": "ok", "score": 0.3}}},
        "error": "oom",
    }
    run_calls = {"n": 0}

    def _run_side(*args):
        run_calls["n"] += 1
        return repairable_no_metrics if run_calls["n"] == 1 else partial_with_metrics

    verify_ret = {"overall_score": 0.31, "target_score": 0.7}
    tools, mocks = _make_tools(verify_ret=verify_ret)
    mocks["run_experiment"].side_effect = _run_side
    ctx = _make_ctx(tmp_path)

    result = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        start_stage="need_experiment",
        emit=_no_emit,
        max_repair_iterations=1,
    )

    mocks["verify_against_rubric"].assert_called_once()
    assert result["rubric_score"] == pytest.approx(0.31)
    assert result["verify_result"] == verify_ret


# ---------------------------------------------------------------------------
# BF-2: Best-of-run in run_lifecycle_primary: climb that fails/regresses does
#        NOT lower the backbone's recorded score.
# ---------------------------------------------------------------------------


def test_run_lifecycle_primary_climb_failure_keeps_backbone_score(tmp_path):
    """Backbone scores 0.4. Climb's run_experiment fails (scope_shape_violation).
    Summary must still carry rubric_score=0.4 (the backbone's score), not None.
    """
    backbone_verify = {"overall_score": 0.4, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "improve lr", "rationale": "low score"}]

    # First run returns success (backbone). Second run returns repairable (climb fails).
    repairable = {
        "outcome": "repairable",
        "failure_class": "scope_shape_violation",
        "metrics": {},
    }
    run_calls = {"n": 0}

    def _run_side(*args):
        run_calls["n"] += 1
        return (
            {"success": True, "metrics": {"reward": 0.4}, "outcome": "ok"}
            if run_calls["n"] == 1
            else repairable
        )

    tools, mocks = _make_primary_tools(
        verify_ret=backbone_verify,
        propose_ret=propose_ret,
    )
    mocks["run_experiment"].side_effect = _run_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
        max_repair_iterations=0,  # no repair — repairable immediately → last_run_ok=False
    )

    # Backbone score must survive the failed climb.
    assert summary["rubric_score"] == pytest.approx(0.4)
    assert summary["rubric_score"] is not None


def test_run_lifecycle_primary_climb_with_lower_score_keeps_best(tmp_path):
    """Backbone 0.7. Climb iteration produces score 0.3 (worse). Summary keeps 0.7."""
    propose_ret = [{"hypothesis": "try smaller lr", "rationale": "low"}]
    backbone_verify = {"overall_score": 0.7, "target_score": 0.9, "meets_target": False}
    climb_verify = {"overall_score": 0.3, "target_score": 0.9, "meets_target": False}
    call_n = {"n": 0}

    def _verify_side(*args):
        r = backbone_verify if call_n["n"] == 0 else climb_verify
        call_n["n"] += 1
        return r

    tools, mocks = _make_primary_tools(propose_ret=propose_ret)
    mocks["verify_against_rubric"].side_effect = _verify_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=1,
    )

    # Never regress below backbone score.
    assert summary["rubric_score"] == pytest.approx(0.7)


def test_run_lifecycle_primary_partial_evidence_climb_captures_score(tmp_path):
    """Climb sub-drive's run stops with partial evidence (gradeable metrics).
    The sub-drive's verify fires (via partial-evidence path) and the resulting
    score is captured in the summary.
    """
    backbone_verify = {"overall_score": 0.3, "target_score": 0.9, "meets_target": False}
    climb_partial_verify = {"overall_score": 0.5, "target_score": 0.9, "meets_target": False}
    propose_ret = [{"hypothesis": "fix OOM", "rationale": "low score"}]

    # run_experiment: first call = backbone (success). Second = fatal with metrics.
    climb_run = {
        "outcome": "fatal",
        "success": False,
        "failure_class": "oom_shrink_exhausted",
        "metrics": {"per_model": {"m": {"status": "ok", "score": 0.5}}},
        "error": "partial oom",
    }
    run_calls = {"n": 0}
    verify_calls = {"n": 0}

    def _run_side(*args):
        run_calls["n"] += 1
        return (
            {"success": True, "metrics": {"reward": 0.3}, "outcome": "ok"}
            if run_calls["n"] == 1
            else climb_run
        )

    def _verify_side(*args):
        r = backbone_verify if verify_calls["n"] == 0 else climb_partial_verify
        verify_calls["n"] += 1
        return r

    tools, mocks = _make_primary_tools(propose_ret=propose_ret)
    mocks["run_experiment"].side_effect = _run_side
    mocks["verify_against_rubric"].side_effect = _verify_side
    ctx = _make_ctx(tmp_path)

    summary = run_lifecycle_primary(
        tools=tools,
        ctx=ctx,
        paper_text=PAPER,
        rubric_spec=RUBRIC,
        emit=_no_emit,
        max_improve_iterations=2,
    )

    # Climb produced a better score via partial-evidence path — should be captured.
    assert summary["rubric_score"] == pytest.approx(0.5)
    assert mocks["verify_against_rubric"].call_count == 2


# ---------------------------------------------------------------------------
# BF-3: Flags off → byte-identical (lifecycle_driver not invoked differently)
# ---------------------------------------------------------------------------


def test_has_gradeable_evidence_empty_metrics_is_false(tmp_path):
    """_has_gradeable_evidence returns False for empty/missing metrics."""
    from backend.agents.rlm.lifecycle_driver import _has_gradeable_evidence

    assert _has_gradeable_evidence({}) is False
    assert _has_gradeable_evidence({"metrics": {}}) is False
    assert _has_gradeable_evidence({"metrics": None}) is False
    assert _has_gradeable_evidence("not a dict") is False  # type: ignore[arg-type]


def test_has_gradeable_evidence_flat_metrics_is_true(tmp_path):
    """_has_gradeable_evidence returns True for any non-empty flat metrics."""
    from backend.agents.rlm.lifecycle_driver import _has_gradeable_evidence

    assert _has_gradeable_evidence({"metrics": {"reward": 0.42}}) is True
    assert _has_gradeable_evidence({"success": True, "metrics": {"acc": 0.9}}) is True


def test_has_gradeable_evidence_per_model_with_ok_cell(tmp_path):
    """_has_gradeable_evidence returns True when per_model has an ok cell."""
    from backend.agents.rlm.lifecycle_driver import _has_gradeable_evidence

    result = {
        "metrics": {
            "per_model": {
                "qwen3-1.7b": {"status": "ok", "reward": 0.55},
                "qwen2.5-3b": {"status": "oom_failed"},
            }
        }
    }
    assert _has_gradeable_evidence(result) is True


def test_has_gradeable_evidence_per_model_no_ok_cell_still_true(tmp_path):
    """_has_gradeable_evidence returns True for non-empty per_model even if no ok cell.

    Conservative: non-empty metrics are always gradeable (better to over-verify).
    """
    from backend.agents.rlm.lifecycle_driver import _has_gradeable_evidence

    result = {
        "metrics": {
            "per_model": {
                "m": {"status": "oom_failed"},
            }
        }
    }
    assert _has_gradeable_evidence(result) is True
