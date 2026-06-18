"""Tests for stage-specific + escalating FINAL_VAR refusal text (Task 3).

The no-experiment refusal must NAME the real missing lifecycle step inferred
by ``required_stage`` (``need_baseline`` → ``implement_baseline``, not just
``run_experiment``), and at the degenerate threshold the message must
explicitly name the loop and — for ``claude-oauth`` roots — append a one-line
recovery skeleton.

When no ``required_stage`` callable is injected (the production default until
Task 4, and most existing tests), the no-experiment message stays byte-for-byte
the legacy text.

Tested at two levels (mirroring ``test_forced_iteration.py``):
1. direct ``should_refuse()`` calls — message construction,
2. end-to-end through the patched ``LocalREPL._final_var`` — escalation wiring.
"""

from __future__ import annotations

from typing import Any

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)


apply_forced_iteration_patch()


def _make_policy(
    *,
    score: float | None = None,
    target: float | None = None,
    iteration: int = 0,
    min_iterations: int = 2,
    remaining_s: float | None = 3600.0,
    total_run_experiments: int = 0,
    required_stage: Any | None = None,
    oauth_root: bool = False,
    on_degenerate_refusal_loop: Any | None = None,
) -> ForcedIterationPolicy:
    captured: list[str] = []
    policy = ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (score, target, iteration),
        current_iteration=lambda: iteration,
        remaining_s=lambda: remaining_s,
        on_refusal=lambda msg: captured.append(msg),
        required_stage=required_stage,
        oauth_root=oauth_root,
        on_degenerate_refusal_loop=on_degenerate_refusal_loop,
    )
    policy._total_run_experiments = total_run_experiments
    return policy


def _make_local_repl() -> Any:
    from rlm.environments.local_repl import LocalREPL

    return LocalREPL()


# ---------------------------------------------------------------------------
# Stage-specific no-experiment message
# ---------------------------------------------------------------------------


def test_no_code_refusal_mentions_implement_baseline() -> None:
    """need_baseline stage → message names implement_baseline."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_baseline",
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "implement_baseline" in msg


def test_existing_code_refusal_does_not_reimplement() -> None:
    """need_experiment stage → message names run_experiment, NOT implement_baseline."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_experiment",
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "run_experiment" in msg
    assert "implement_baseline" not in msg


def test_need_environment_mentions_build_environment() -> None:
    """need_environment stage → message names build_environment."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_environment",
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "build_environment" in msg


def test_need_verification_mentions_verify() -> None:
    """need_verification stage → message names verify_against_rubric."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_verification",
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "verify_against_rubric" in msg


def test_none_stage_is_byte_for_byte_legacy() -> None:
    """required_stage=None → exact legacy no-experiment message."""
    policy = _make_policy(total_run_experiments=0, required_stage=None)
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg == (
        "FINAL_VAR refused at iteration 0: run_experiment has never "
        "been called. You must execute the baseline code at least once "
        "before terminating. Next steps: call build_environment (if the "
        "image is not built yet), then run_experiment(code_path, env_id) "
        "to execute the code, then verify_against_rubric to score it, "
        "then FINAL_VAR."
    )


def test_unknown_stage_falls_back_to_legacy() -> None:
    """can_finalize / unmapped stage → legacy message (clean fallback)."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "can_finalize",
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    # Falls back to the legacy text (mapping miss).
    assert "run_experiment(code_path, env_id)" in msg


def test_required_stage_raise_falls_back_to_legacy() -> None:
    """A required_stage callable that raises → legacy message, no crash."""

    def _bad() -> str:
        raise RuntimeError("disk read failed")

    policy = _make_policy(total_run_experiments=0, required_stage=_bad)
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "run_experiment(code_path, env_id)" in msg
    assert policy._pending_refusal_signature == "no_experiment"


def test_stage_signature_unchanged() -> None:
    """The no_experiment signature is preserved regardless of stage."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_baseline",
    )
    policy.should_refuse()
    assert policy._pending_refusal_signature == "no_experiment"


# ---------------------------------------------------------------------------
# Escalation at the degenerate threshold (end-to-end)
# ---------------------------------------------------------------------------


def test_escalation_names_loop_and_baseline(monkeypatch) -> None:
    """3 no_experiment refusals → last block message names the loop + baseline."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.0}"

    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_baseline",
        oauth_root=False,
    )
    assert policy.degenerate_threshold == 3

    with forced_iteration_policy(policy):
        out1 = repl._final_var("report")
        out2 = repl._final_var("report")
        out3 = repl._final_var("report")

    # First two are normal refusals (no escalation yet).
    assert "zero progress" not in out1
    assert "zero progress" not in out2
    # The third trips the degenerate threshold → escalated text.
    assert "zero progress" in out3
    assert "STOP" in out3
    assert "implement_baseline" in out3


def test_escalation_oauth_includes_skeleton(monkeypatch) -> None:
    """oauth_root=True → escalated message appends the recovery skeleton."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.0}"

    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_baseline",
        oauth_root=True,
    )

    with forced_iteration_policy(policy):
        repl._final_var("report")
        repl._final_var("report")
        out3 = repl._final_var("report")

    assert "zero progress" in out3
    assert "Recovery:" in out3
    assert "plan_reproduction" in out3


def test_escalation_non_oauth_omits_skeleton(monkeypatch) -> None:
    """oauth_root=False → escalated message has NO recovery skeleton."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.0}"

    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_baseline",
        oauth_root=False,
    )

    with forced_iteration_policy(policy):
        repl._final_var("report")
        repl._final_var("report")
        out3 = repl._final_var("report")

    assert "zero progress" in out3
    assert "Recovery:" not in out3


def test_escalate_refusal_message_direct() -> None:
    """escalate_refusal_message() builds loop-naming text from the stage."""
    policy = _make_policy(
        total_run_experiments=0,
        required_stage=lambda: "need_experiment",
    )
    policy._noprogress_refusals = 4
    msg = policy.escalate_refusal_message("no_experiment")
    assert "zero progress" in msg
    assert "4" in msg
    assert "run_experiment" in msg
    assert "Recovery:" not in msg  # oauth_root defaults False
