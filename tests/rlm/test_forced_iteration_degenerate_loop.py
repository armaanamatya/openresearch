"""Tests for the no-progress FINAL_VAR refusal-loop detector (Task 2).

A weak root can call ``FINAL_VAR`` repeatedly without ever doing real work
(``implement_baseline`` / ``run_experiment``).  ``should_refuse()`` correctly
refuses each one, but only returns text; a degenerate root ignores it and
loops to the ``_MAX_REFUSALS_PER_RUN`` cap.

Task 2 adds a *no-progress refusal counter* to ``ForcedIterationPolicy``:

  * it increments only across CONSECUTIVE refusals that share a signature with
    NO intervening state-changing primitive,
  * a new signature resets the run to 1,
  * a state-changing primitive (``record_state_change`` /
    ``record_run_experiment``) resets it to 0,
  * when it first reaches ``degenerate_threshold`` it fires
    ``on_degenerate_refusal_loop`` exactly once.

It is purely additive: when no callback is registered, nothing observable
changes and accept-after-16 remains the last-ditch escape.

Tested at two levels (mirroring ``test_forced_iteration.py``):
1. direct method calls on ``register_refusal`` / ``record_state_change`` — pure
   counter logic,
2. end-to-end through the patched ``LocalREPL._final_var`` — proves the
   interceptor wiring fires the callback.
"""

from __future__ import annotations

from typing import Any

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    _MAX_REFUSALS_PER_RUN,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)


# Ensure the patch is installed once for end-to-end checks.
apply_forced_iteration_patch()


def _make_policy(
    *,
    score: float | None = None,
    target: float | None = None,
    iteration: int = 0,
    min_iterations: int = 2,
    remaining_s: float | None = 3600.0,
    refusals: list[str] | None = None,
    total_run_experiments: int = 0,
    on_degenerate_refusal_loop: Any | None = None,
    required_stage: Any | None = None,
) -> ForcedIterationPolicy:
    """Factory mirroring ``test_forced_iteration._make_policy``.

    Defaults to the no-experiment shape (``total_run_experiments=0``) so the
    no-experiment refusal branch fires and stamps the ``"no_experiment"``
    signature.
    """
    captured: list[str] = refusals if refusals is not None else []
    policy = ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (score, target, iteration),
        current_iteration=lambda: iteration,
        remaining_s=lambda: remaining_s,
        on_refusal=lambda msg: captured.append(msg),
        on_degenerate_refusal_loop=on_degenerate_refusal_loop,
        required_stage=required_stage,
    )
    policy._total_run_experiments = total_run_experiments
    return policy


def _make_local_repl() -> Any:
    from rlm.environments.local_repl import LocalREPL

    return LocalREPL()


# ---------------------------------------------------------------------------
# Direct-method unit tests — counter logic
# ---------------------------------------------------------------------------


def test_register_refusal_three_same_signature_fires_callback_once() -> None:
    """3 consecutive same-signature refusals → callback fires exactly once."""
    captured: list[dict] = []
    policy = ForcedIterationPolicy(
        min_iterations=2,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
    )
    policy.degenerate_threshold = 3

    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 1
    assert captured == []

    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 2
    assert captured == []

    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 3
    assert len(captured) == 1
    assert captured[0]["signature"] == "no_experiment"
    assert captured[0]["count"] == 3
    assert captured[0]["required_stage"] is None

    # A 4th refusal must NOT fire the callback a second time.
    policy.register_refusal("no_experiment")
    assert len(captured) == 1


def test_state_change_between_refusals_resets_counter() -> None:
    """A state-changing primitive between refusals resets the counter to 0."""
    captured: list[dict] = []
    policy = ForcedIterationPolicy(
        min_iterations=2,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
    )
    policy.degenerate_threshold = 3

    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 1

    policy.record_run_experiment("ok")  # state change → reset
    assert policy._noprogress_refusals == 0

    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 1  # not 2
    assert captured == []  # threshold never reached


def test_record_state_change_directly_resets_counter() -> None:
    """record_state_change() resets the no-progress counter and signature."""
    policy = ForcedIterationPolicy(min_iterations=2)
    policy.degenerate_threshold = 3
    policy.register_refusal("no_experiment")
    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 2

    policy.record_state_change()
    assert policy._noprogress_refusals == 0
    assert policy._last_refusal_signature is None

    policy.register_refusal("no_experiment")
    assert policy._noprogress_refusals == 1


def test_mixed_signatures_do_not_accumulate() -> None:
    """Alternating signatures never accumulate into a false degenerate trip."""
    captured: list[dict] = []
    policy = ForcedIterationPolicy(
        min_iterations=2,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
    )
    policy.degenerate_threshold = 3

    policy.register_refusal("no_experiment")
    policy.register_refusal("below_target")
    policy.register_refusal("no_experiment")

    assert captured == []
    assert policy._noprogress_refusals == 1


def test_required_stage_threaded_into_payload() -> None:
    """When required_stage is set, its value flows into the callback payload."""
    captured: list[dict] = []
    policy = ForcedIterationPolicy(
        min_iterations=2,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
        required_stage=lambda: "need_experiment",
    )
    policy.degenerate_threshold = 2

    policy.register_refusal("no_experiment")
    policy.register_refusal("no_experiment")

    assert len(captured) == 1
    assert captured[0]["required_stage"] == "need_experiment"


def test_required_stage_raise_is_failsoft() -> None:
    """A required_stage callable that raises must not crash the policy."""
    captured: list[dict] = []

    def _bad_stage() -> str:
        raise RuntimeError("disk read failed")

    policy = ForcedIterationPolicy(
        min_iterations=2,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
        required_stage=_bad_stage,
    )
    policy.degenerate_threshold = 2

    policy.register_refusal("no_experiment")
    policy.register_refusal("no_experiment")  # must not raise

    assert len(captured) == 1
    assert captured[0]["required_stage"] is None


def test_degenerate_callback_raise_is_failsoft() -> None:
    """A raising on_degenerate_refusal_loop callback must not propagate."""
    def _bad_cb(payload: dict) -> None:
        raise RuntimeError("emit broke")

    policy = ForcedIterationPolicy(
        min_iterations=2,
        on_degenerate_refusal_loop=_bad_cb,
    )
    policy.degenerate_threshold = 2

    policy.register_refusal("no_experiment")
    policy.register_refusal("no_experiment")  # must not raise


def test_no_callback_registered_counter_inert() -> None:
    """on_degenerate_refusal_loop=None → counter updates but nothing fires."""
    policy = ForcedIterationPolicy(min_iterations=2)
    policy.degenerate_threshold = 3
    for _ in range(10):
        policy.register_refusal("no_experiment")
    # Counter advanced, but with no callback nothing observable happened and
    # the policy did not raise.
    assert policy._noprogress_refusals == 10


def test_threshold_default_from_env(monkeypatch) -> None:
    """Default degenerate_threshold reads OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "5")
    policy = ForcedIterationPolicy(min_iterations=2)
    assert policy.degenerate_threshold == 5


def test_threshold_default_falls_back_to_three(monkeypatch) -> None:
    """Unset / invalid env → default threshold 3."""
    monkeypatch.delenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", raising=False)
    policy = ForcedIterationPolicy(min_iterations=2)
    assert policy.degenerate_threshold == 3

    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "not-a-number")
    policy2 = ForcedIterationPolicy(min_iterations=2)
    assert policy2.degenerate_threshold == 3

    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "0")
    policy3 = ForcedIterationPolicy(min_iterations=2)
    assert policy3.degenerate_threshold == 3


# ---------------------------------------------------------------------------
# End-to-end via the patched LocalREPL._final_var
# ---------------------------------------------------------------------------


def test_e2e_three_no_experiment_refusals_fires_callback_once(monkeypatch) -> None:
    """3 FINAL_VAR calls with zero experiments → callback fires once, count==3.

    Proves the interceptor wiring: each blocked _final_var must call
    register_refusal("no_experiment").
    """
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.0}"

    captured: list[dict] = []
    policy = _make_policy(
        score=None,
        target=None,
        iteration=0,
        min_iterations=2,
        total_run_experiments=0,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
    )
    assert policy.degenerate_threshold == 3

    with forced_iteration_policy(policy):
        out1 = repl._final_var("report")
        out2 = repl._final_var("report")
        out3 = repl._final_var("report")

    # All three were blocked (no_experiment refusal).
    for out in (out1, out2, out3):
        assert "Variable '" in out
        assert "FINAL_VAR" in out

    assert len(captured) == 1
    assert captured[0]["signature"] == "no_experiment"
    assert captured[0]["count"] == 3


def test_e2e_state_change_between_refusals_prevents_trip(monkeypatch) -> None:
    """An experiment recorded between refusals resets the counter; no trip."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.0}"

    captured: list[dict] = []
    policy = _make_policy(
        score=None,
        target=None,
        iteration=0,
        min_iterations=2,
        total_run_experiments=0,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
    )

    with forced_iteration_policy(policy):
        repl._final_var("report")  # refusal 1 (no_experiment)
        assert policy._noprogress_refusals == 1
        policy.record_run_experiment("ok")  # state change → reset
        # Now total_run_experiments == 1, so the no_experiment branch no longer
        # fires; force the no-experiment shape back to drive another refusal of
        # the same signature without a state change in between would require 0
        # experiments. Reset to the degenerate shape explicitly.
        policy._total_run_experiments = 0
        repl._final_var("report")  # refusal 2, but counter reset to 1

    assert policy._noprogress_refusals == 1
    assert captured == []


def test_e2e_no_callback_accept_after_16_unchanged() -> None:
    """on_degenerate_refusal_loop=None → accept-after-16 still governs.

    Drives _MAX_REFUSALS_PER_RUN refusals then asserts the next _final_var is
    ACCEPTED by the existing accept-after-16 branch — the new counter changed
    nothing.
    """
    repl = _make_local_repl()
    repl.locals["report"] = "blocked-or-accepted"

    policy = _make_policy(
        score=0.1,
        target=0.9,
        iteration=0,
        min_iterations=2,
        total_run_experiments=1,  # below-target shape (not no_experiment)
        on_degenerate_refusal_loop=None,
    )

    with forced_iteration_policy(policy):
        # First _MAX_REFUSALS_PER_RUN calls are refused.
        for _ in range(_MAX_REFUSALS_PER_RUN):
            out = repl._final_var("report")
            assert "Variable '" in out

        assert policy.refusal_count == _MAX_REFUSALS_PER_RUN

        # The next call hits the accept-after-16 branch and is ACCEPTED.
        out_accepted = repl._final_var("report")

    assert out_accepted == "blocked-or-accepted"
    assert repl._last_final_answer == "blocked-or-accepted"
