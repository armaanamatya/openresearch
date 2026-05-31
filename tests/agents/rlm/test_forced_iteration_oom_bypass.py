"""Forced-iteration terminal-OOM bypass (spec 2026-05-31, component 6).

A shrink-exhausted OOM (or explicit capacity exhaustion) must NOT be
force-iterated — refusing FINAL_VAR only re-runs the same OOM-prone config (the
2026-05-31 death spiral). These tests pin that ``should_refuse`` ACCEPTS
FINAL_VAR for terminal classes while still refusing ordinary repairable ones.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm import forced_iteration as fi


def _would_refuse_policy(**overrides):
    """A policy whose default decision (sans bypass) is REFUSE.

    score 0.0 < target 0.6 and iteration 0 < min_iterations 2 → check #4 refuses.
    """
    kw = dict(
        min_iterations=2,
        rubric_snapshot=lambda: (0.0, 0.6, 0),
        current_iteration=lambda: 0,
        remaining_s=lambda: 600.0,
    )
    kw.update(overrides)
    return fi.ForcedIterationPolicy(**kw)


def test_sanity_without_bypass_it_refuses():
    refuse, msg = _would_refuse_policy().should_refuse()
    assert refuse is True
    assert msg is not None


@pytest.mark.parametrize("klass", ["oom_shrink_exhausted", "capacity_exhausted"])
def test_note_terminal_failure_accepts_final_var(klass):
    policy = _would_refuse_policy()
    policy.note_terminal_failure(klass)
    refuse, msg = policy.should_refuse()
    assert refuse is False
    assert msg is None


def test_terminal_class_via_record_repair_attempt_also_bypasses():
    # Robustness: if component 4 routes the terminal class through the existing
    # repairable wiring instead of note_terminal_failure(), the bypass still fires.
    policy = _would_refuse_policy()
    policy.record_repair_attempt("oom_shrink_exhausted")
    refuse, _ = policy.should_refuse()
    assert refuse is False


def test_ordinary_repairable_class_still_refuses(monkeypatch):
    # A normal repairable failure (not terminal) must NOT be bypassed.
    monkeypatch.setenv("REPROLAB_MIN_REPAIR_ITERATIONS", "2")
    policy = fi.ForcedIterationPolicy(min_iterations=0, remaining_s=lambda: 600.0)
    policy.record_repair_attempt("preflight_blocked")
    refuse, msg = policy.should_refuse()
    assert refuse is True  # repair floor not met → still refuses
    assert msg is not None


def test_note_nonterminal_class_does_not_bypass():
    policy = _would_refuse_policy()
    policy.note_terminal_failure("preflight_blocked")  # not a terminal class
    refuse, _ = policy.should_refuse()
    assert refuse is True  # falls through to the normal refusal


def test_terminal_bypass_survives_no_rubric_snapshot():
    # No rubric data + below iteration floor would normally refuse (BUG-LR-013);
    # a terminal failure still accepts.
    policy = fi.ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=None,
        current_iteration=lambda: 0,
        remaining_s=lambda: 600.0,
    )
    policy.note_terminal_failure("oom_shrink_exhausted")
    refuse, _ = policy.should_refuse()
    assert refuse is False
