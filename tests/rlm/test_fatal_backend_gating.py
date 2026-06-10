import json
import os
from unittest.mock import MagicMock, patch

import pytest
from rlm.core.types import CodeBlock, REPLResult, RLMIteration
from rlm.utils.parsing import format_iteration

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)
from backend.agents.rlm.run import (
    _FatalBackendGateLogger,
    _FatalPrimitiveAbort,
    _finalize_fatal_primitive_abort,
    _outcome_value,
    _record_last_primitive_result_tools,
)

apply_forced_iteration_patch()


def _fatal_result() -> dict:
    return {
        "success": False,
        "metrics": {},
        "error": "RUNPOD_BALANCE_TOO_LOW: add funds",
        "failure_class": "balance_too_low",
        "outcome": "fatal",
    }


def _iteration_with_fatal_result(result: dict) -> RLMIteration:
    return RLMIteration(
        prompt=[],
        response="I will run the experiment.",
        code_blocks=[
            CodeBlock(
                code="result = run_experiment(code_path, env_id)",
                result=REPLResult(stdout="", stderr="", locals={"result": result}),
            )
        ],
    )


def test_fatal_outcome_terminates_before_history_append(make_context, tmp_path):
    ctx = make_context(tmp_path)
    ctx._last_primitive_name = "run_experiment"
    ctx._last_primitive_result = _fatal_result()
    emitted: list[dict] = []
    checkpointer = MagicMock()
    logger = _FatalBackendGateLogger(
        emit=emitted.append,
        checkpointer=checkpointer,
        ctx=ctx,
    )
    iteration = _iteration_with_fatal_result(ctx._last_primitive_result)
    message_history: list[dict] = []

    with pytest.raises(_FatalPrimitiveAbort):
        logger.log(iteration)
        message_history.extend(format_iteration(iteration))

    assert message_history == []
    assert logger.iteration_count == 1
    checkpointer.record.assert_called_once()


def test_fatal_finalize_writes_failed_status_and_failed_report(make_context, tmp_path):
    # The only experiment on disk is a tri-state "partial evidence" row:
    # success=False with metrics-like numbers from a balance_too_low abort (the
    # run hit a credit wall, it never really executed). Per the evidence gate
    # (FM-004, ported from feat/rlm-wedge-hardening 2026-06-09), success==False
    # does NOT license a partial/reproduced VERDICT — the report verdict is
    # "failed", though the partial numbers stay in baseline_metrics for the record.
    ctx = make_context(tmp_path)
    result = _fatal_result()
    abort = _FatalPrimitiveAbort(primitive_name="run_experiment", result=result)
    (ctx.project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({
            "timestamp": "2026-05-26T00:00:00+00:00",
            "success": False,
            "metrics": {"accuracy": 0.42},
            "failure_class": "balance_too_low",
        })
        + "\n",
        encoding="utf-8",
    )
    emitted: list[dict] = []

    run_result = _finalize_fatal_primitive_abort(
        abort=abort,
        ctx=ctx,
        iterations=1,
        project_dir=ctx.project_dir,
        emit=emitted.append,
        tools_label="real",
    )

    status = json.loads((ctx.project_dir / "demo_status.json").read_text(encoding="utf-8"))
    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))

    assert run_result.status == "failed"
    assert status["status"] == "failed"
    assert status["error"]["outcome"] == "fatal"
    assert status["error"]["primitive"] == "run_experiment"
    assert report["verdict"] == "failed"  # tri-state partial-evidence no longer earns a verdict
    assert "evidence_gap" in report["reproduction_summary"]
    assert report["baseline_metrics"] == {"accuracy": 0.42}  # numbers kept for the record
    assert any(event.get("event") == "run_fatal" for event in emitted)


def test_fatal_finalize_with_timeout_partial_evidence_caps_at_partial(make_context, tmp_path):
    # Audit 2026-06-09: a HARNESS-finalized partial (exec_timeout/exec_stalled →
    # primitives._finalize_timeout_result loads the on-disk metrics.json and
    # stamps failure_class=partial_timeout) is real completed work — the verdict
    # caps at "partial" with an evidence_cap note, instead of the forced "failed"
    # the balance_too_low case above correctly gets. This is the wall-clock-expiry
    # scenario the 2026-06-08 finalize-on-timeout redesign was built for.
    ctx = make_context(tmp_path)
    result = {
        "success": False,
        "metrics": {"accuracy": 0.61},
        "error": "exec_timeout: wall clock exhausted after 4/5 families",
        "failure_class": "partial_timeout",
        "outcome": "fatal",
    }
    abort = _FatalPrimitiveAbort(primitive_name="run_experiment", result=result)
    (ctx.project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({
            "timestamp": "2026-06-09T00:00:00+00:00",
            "success": False,
            "metrics": {"accuracy": 0.61},
            "failure_class": "partial_timeout",
            "partial_timeout": True,
        })
        + "\n",
        encoding="utf-8",
    )
    # The timeout row was persisted by a REAL run_experiment call — record it in
    # the in-process ledger exactly as binding.wrap_primitive would. Without this
    # the gate's forge cross-check (session count == 0) rightly forces "failed".
    from datetime import datetime, timezone

    from backend.agents.resilience.cost import CostLedgerEntry

    ctx.cost_ledger.append(
        CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id="run_experiment",
            attempt_index=0,
            provider="openai",
            model="gpt-5",
        )
    )

    run_result = _finalize_fatal_primitive_abort(
        abort=abort,
        ctx=ctx,
        iterations=1,
        project_dir=ctx.project_dir,
        emit=lambda _e: None,
        tools_label="real",
    )

    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))

    assert run_result.status == "failed"  # the RUN still ended fatally
    assert report["verdict"] == "partial"  # ...but the verdict reflects the real partial work
    assert "evidence_cap" in report["reproduction_summary"]
    assert "evidence_gap" not in report["reproduction_summary"]
    assert report["baseline_metrics"] == {"accuracy": 0.61}


# ---------------------------------------------------------------------------
# PR-α followup integration: repairable outcome → record_repair_attempt →
# forced_iteration_policy refuses FINAL_VAR
# ---------------------------------------------------------------------------

def _repairable_result(failure_class: str = "preflight_blocked") -> dict:
    return {
        "success": False,
        "metrics": {},
        "error": "preflight checks failed: 5 AST violations",
        "failure_class": failure_class,
        "outcome": "repairable",
    }


def test_repairable_outcome_records_repair_attempt_via_tool_wrapper(make_context, tmp_path):
    """_record_last_primitive_result_tools calls policy.record_repair_attempt when
    run_experiment returns outcome='repairable' and the policy holder is populated.

    This is the integration path: tool wrapper → repair_policy_holder → policy.
    """
    ctx = make_context(tmp_path)
    repair_policy_holder: list = []

    # Build a minimal fake tool that returns a repairable result.
    fake_tool_result = _repairable_result("preflight_blocked")
    fake_tools = {
        "run_experiment": {
            "tool": lambda *a, **kw: fake_tool_result,
            "description": "fake",
        }
    }

    wrapped = _record_last_primitive_result_tools(fake_tools, ctx, repair_policy_holder)

    # Policy doesn't exist yet — tool call before policy creation should be safe.
    wrapped["run_experiment"]["tool"]()
    assert ctx._last_primitive_result == fake_tool_result

    # Now create and register the policy (mirrors run.py's late-binding pattern).
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.0, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
    )
    repair_policy_holder.append(policy)

    # Second tool call — now the holder is populated, so record_repair_attempt fires.
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        wrapped["run_experiment"]["tool"]()

    assert policy._repair_iter_count == 1
    assert policy._last_repair_failure_class == "preflight_blocked"


def test_repairable_outcome_forces_final_var_refusal_end_to_end(make_context, tmp_path):
    """Synthetic repairable outcome → orchestrator records repair attempt →
    forced_iteration_policy refuses FINAL_VAR.

    Full integration scenario from the PR description: iter 1 returns
    outcome='repairable' with failure_class='preflight_blocked'; the root
    tries to FINAL_VAR; the policy blocks it and emits forced_repair_iteration.
    """
    from rlm.environments.local_repl import LocalREPL

    ctx = make_context(tmp_path)
    repair_policy_holder: list = []
    repair_warnings: list[str] = []

    fake_tools = {
        "run_experiment": {
            "tool": lambda *a, **kw: _repairable_result("preflight_blocked"),
            "description": "fake",
        }
    }
    wrapped = _record_last_primitive_result_tools(fake_tools, ctx, repair_policy_holder)

    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.0, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        on_repair_refusal=lambda m: repair_warnings.append(m),
    )
    repair_policy_holder.append(policy)

    repl = LocalREPL()
    repl.locals["report"] = "{'score': 0.0}"

    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        # Simulate run_experiment call (records the repair attempt).
        wrapped["run_experiment"]["tool"]()
        # In production the real run_experiment primitive also feeds the
        # policy's experiment tracker (primitives.py: _fip.record_run_experiment)
        # — the fake tool bypasses primitives, so mirror that here; without it
        # the BUG-NEW-046 zero-experiments refusal (ported 2026-06-09) fires
        # first and masks the repair refusal under test.
        policy.record_run_experiment("repairable")

        # Simulate root calling FINAL_VAR immediately after the repairable outcome.
        with forced_iteration_policy(policy):
            out = repl._final_var("report")

    # The FINAL_VAR must be blocked.
    assert "Variable '" in out
    assert "' not found" in out
    assert "FINAL_VAR" in out
    assert "repairable outcome" in out
    assert "preflight_blocked" in out

    # on_repair_refusal callback must have been invoked.
    assert len(repair_warnings) == 1
    assert "preflight_blocked" in repair_warnings[0]

    # repair_iter_count must reflect the one attempt.
    assert policy._repair_iter_count == 1
