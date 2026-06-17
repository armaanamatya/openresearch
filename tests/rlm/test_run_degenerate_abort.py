"""Tests for the degenerate-refusal-loop early-abort wiring (Task 4).

Task 4 wires the no-progress detector (Tasks 1-3) into the live RLM run:

  * ``run.py`` builds an ``on_degenerate_refusal_loop`` callback (via the
    module-level factory ``_make_degenerate_loop_callback``) that emits a
    ``root_degenerate_refusal_loop`` run_warning and — with AUTODRIVE OFF (the
    default) — marks a terminal stop so the run finalizes fast with
    ``failure_class="root_degenerate_loop"`` instead of churning to the
    16-refusal cap / wall clock.
  * Wall-clock floor and pre-existing terminal stops take precedence.
  * ``required_stage`` / ``oauth_root`` injection must not change any healthy
    decision.

Tested at two levels:
1. the factory callback in isolation (event emission, terminal-stop marking,
   wall-clock precedence, autodrive-ON no-op);
2. end-to-end through the patched ``LocalREPL._final_var`` (stops at threshold,
   not 16; healthy sequence unaffected).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    _MAX_REFUSALS_PER_RUN,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)
from backend.agents.rlm.run import (
    _make_degenerate_loop_callback,
    _record_last_primitive_result_tools,
)


# Ensure the FINAL_VAR interceptor is installed once for the e2e checks.
apply_forced_iteration_patch()


def _make_local_repl() -> Any:
    from rlm.environments.local_repl import LocalREPL

    return LocalREPL()


def _fake_ctx(*, remaining_s: float | None = 3600.0) -> Any:
    ctx = MagicMock()
    ctx.remaining_s.return_value = remaining_s
    # _terminal_stop_reason is intentionally absent until the callback sets it;
    # MagicMock would otherwise auto-create the attribute, so reset it to a
    # sentinel we can detect.
    ctx._terminal_stop_reason = None
    return ctx


def _degenerate_payload() -> dict:
    return {"signature": "no_experiment", "count": 3, "required_stage": "need_baseline"}


# ---------------------------------------------------------------------------
# Factory callback — isolated behaviour
# ---------------------------------------------------------------------------


def test_callback_emits_event_and_marks_terminal_stop() -> None:
    """AUTODRIVE OFF: emit root_degenerate_refusal_loop + mark terminal stop."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append, ctx=ctx, policy=policy, autodrive_enabled=False
    )
    cb(_degenerate_payload())

    # Exactly one run_warning with the right code, carrying the stage.
    warnings = [e for e in emitted if e.get("event") == "run_warning"]
    assert len(warnings) == 1
    ev = warnings[0]
    assert ev["code"] == "root_degenerate_refusal_loop"
    assert ev.get("required_stage") == "need_baseline"
    assert ev.get("stage") == "need_baseline"
    assert ev.get("signature") == "no_experiment"
    assert ev.get("count") == 3

    # Terminal stop marked on ctx with the degenerate failure class.
    assert ctx._terminal_stop_reason is not None
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"
    assert ctx._terminal_stop_reason["kind"] == "root_degenerate_loop"

    # The policy now ACCEPTS the next FINAL_VAR (terminal class recognized).
    refuse, msg = policy.should_refuse()
    assert refuse is False
    assert msg is None


def test_callback_wall_clock_precedence() -> None:
    """remaining_s <= 60 → emit the warning but do NOT mark a terminal stop."""
    emitted: list[dict] = []
    ctx = _fake_ctx(remaining_s=30.0)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append, ctx=ctx, policy=policy, autodrive_enabled=False
    )
    cb(_degenerate_payload())

    assert any(e.get("code") == "root_degenerate_refusal_loop" for e in emitted)
    assert ctx._terminal_stop_reason is None
    # Policy did not get a terminal class (no note_terminal_failure called).
    assert policy._terminal_failure_class is None


def test_callback_autodrive_on_does_not_abort() -> None:
    """AUTODRIVE ON → emit the warning but do NOT early-abort (Task 6 fills drive)."""
    emitted: list[dict] = []
    ctx = _fake_ctx()
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append, ctx=ctx, policy=policy, autodrive_enabled=True
    )
    cb(_degenerate_payload())

    assert any(e.get("code") == "root_degenerate_refusal_loop" for e in emitted)
    assert ctx._terminal_stop_reason is None
    assert policy._terminal_failure_class is None


def test_callback_emit_failure_is_failsoft() -> None:
    """A raising emit must not propagate, and the terminal stop is still set."""
    def _bad_emit(_event: dict) -> None:
        raise RuntimeError("emit broke")

    ctx = _fake_ctx()
    policy = ForcedIterationPolicy(min_iterations=2)
    cb = _make_degenerate_loop_callback(
        emit=_bad_emit, ctx=ctx, policy=policy, autodrive_enabled=False
    )
    cb(_degenerate_payload())  # must not raise

    assert ctx._terminal_stop_reason is not None
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"


# ---------------------------------------------------------------------------
# End-to-end through the patched LocalREPL._final_var
# ---------------------------------------------------------------------------


def test_e2e_always_final_var_stops_at_threshold_not_16(monkeypatch) -> None:
    """Simulated always-FINAL_VAR root stops at ~threshold, far below 16.

    With the degenerate callback wired (autodrive OFF), the threshold trip
    marks a terminal stop so the SUBSEQUENT FINAL_VAR is ACCEPTED — the run
    would stop at ~threshold with failure_class=root_degenerate_loop.
    """
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "the-final-answer"

    emitted: list[dict] = []
    ctx = _fake_ctx(remaining_s=3600.0)

    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (None, None, 0),
        current_iteration=lambda: 0,
        remaining_s=lambda: 3600.0,
        required_stage=lambda: "need_baseline",
    )
    policy.degenerate_threshold = 3
    policy._total_run_experiments = 0  # no_experiment shape → consistent signature
    policy.on_degenerate_refusal_loop = _make_degenerate_loop_callback(
        emit=emitted.append, ctx=ctx, policy=policy, autodrive_enabled=False
    )

    with forced_iteration_policy(policy):
        outs = []
        accepted_index = None
        for i in range(_MAX_REFUSALS_PER_RUN):
            out = repl._final_var("report")
            outs.append(out)
            if "Variable '" not in out:
                accepted_index = i
                break

    # The degenerate event fired exactly once.
    degen = [e for e in emitted if e.get("code") == "root_degenerate_refusal_loop"]
    assert len(degen) == 1

    # The run was ACCEPTED far below the 16 cap (threshold 3 → accept at the
    # 4th call, index 3 — refusals 1,2,3 then the terminal class accepts).
    assert accepted_index is not None
    assert accepted_index <= 4
    assert policy.refusal_count <= 4
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"


def test_e2e_healthy_sequence_no_degenerate_event(monkeypatch) -> None:
    """Healthy run (experiment between refusals, then satisfied) → no degenerate event."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "healthy-answer"

    emitted: list[dict] = []
    ctx = _fake_ctx(remaining_s=3600.0)

    # Score reaches target → should_refuse accepts immediately. State changes
    # (run_experiment) reset the counter, so the degenerate trip never fires.
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.95, 0.7, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        required_stage=lambda: "can_finalize",
    )
    policy.degenerate_threshold = 3
    policy._total_run_experiments = 1
    policy.on_degenerate_refusal_loop = _make_degenerate_loop_callback(
        emit=emitted.append, ctx=ctx, policy=policy, autodrive_enabled=False
    )

    with forced_iteration_policy(policy):
        out = repl._final_var("report")

    # Accepted (score >= target), zero degenerate events, no terminal stop.
    assert out == "healthy-answer"
    assert [e for e in emitted if e.get("code") == "root_degenerate_refusal_loop"] == []
    assert ctx._terminal_stop_reason is None


# ---------------------------------------------------------------------------
# State-changing primitives reset the no-progress counter (Task 4 / plan Task 2:
# implement_baseline / build_environment / run_experiment ALL count as progress).
# ---------------------------------------------------------------------------


def _make_noprogress_policy(*, signature: str = "no_experiment", count: int = 2) -> ForcedIterationPolicy:
    policy = ForcedIterationPolicy(min_iterations=0)
    policy._noprogress_refusals = count
    policy._last_refusal_signature = signature
    return policy


def test_implement_baseline_resets_noprogress_counter() -> None:
    """Calling implement_baseline re-arms the degenerate detector (no false trip)."""
    policy = _make_noprogress_policy()
    holder = [policy]
    tools = {"implement_baseline": {"tool": lambda **_: {"code_path": "code/"}}}
    wrapped = _record_last_primitive_result_tools(tools, MagicMock(), holder)

    wrapped["implement_baseline"]["tool"]()

    assert policy._noprogress_refusals == 0
    assert policy._last_refusal_signature is None
    assert policy._degenerate_fired is False


def test_build_environment_resets_noprogress_counter() -> None:
    """build_environment (returns {'ok': ...}, no 'outcome' key) also resets the counter."""
    policy = _make_noprogress_policy()
    holder = [policy]
    tools = {"build_environment": {"tool": lambda **_: {"ok": True, "image_tag": ""}}}
    wrapped = _record_last_primitive_result_tools(tools, MagicMock(), holder)

    wrapped["build_environment"]["tool"]()

    assert policy._noprogress_refusals == 0
    assert policy._last_refusal_signature is None


def test_non_state_changing_primitive_does_not_reset() -> None:
    """A read-only primitive (understand_section) must NOT reset the counter."""
    policy = _make_noprogress_policy(count=2)
    holder = [policy]
    tools = {"understand_section": {"tool": lambda **_: {"summary": "x"}}}
    wrapped = _record_last_primitive_result_tools(tools, MagicMock(), holder)

    wrapped["understand_section"]["tool"]()

    assert policy._noprogress_refusals == 2  # unchanged


def test_implement_baseline_between_refusals_prevents_false_trip() -> None:
    """2 no_experiment refusals, implement_baseline, 2 more → never reaches threshold 3."""
    fired: list[dict] = []
    policy = ForcedIterationPolicy(
        min_iterations=0,
        degenerate_threshold=3,
        on_degenerate_refusal_loop=lambda p: fired.append(p),
        required_stage=lambda: "need_baseline",
    )
    holder = [policy]
    tools = {"implement_baseline": {"tool": lambda **_: {"code_path": "code/"}}}
    wrapped = _record_last_primitive_result_tools(tools, MagicMock(), holder)

    policy.register_refusal("no_experiment")
    policy.register_refusal("no_experiment")  # count == 2, no trip yet
    wrapped["implement_baseline"]["tool"]()  # reset
    policy.register_refusal("no_experiment")
    policy.register_refusal("no_experiment")  # count == 2 again

    assert policy._noprogress_refusals == 2
    assert fired == []  # never reached the degenerate threshold
