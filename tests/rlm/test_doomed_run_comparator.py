"""tests/rlm/test_doomed_run_comparator.py — doomed-attempt early kill
comparator (spec §10.3, ``OPENRESEARCH_DOOMED_KILL``, default OFF).

Pure logic only: ``DoomedComparator.observe`` is exercised directly with
constructed ``CurvePoint`` sequences. NOT wired into any live path (WATCH
integration is a later, operator-validated step) -- these tests never touch
a real run, socket, GPU, or LLM.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.doomed_run_comparator import (
    DOOMED_KILL_ENV,
    DOOMED_MARGIN_ENV,
    DOOMED_MIN_PROGRESS_ENV,
    DOOMED_POLLS_ENV,
    CurvePoint,
    DoomedComparator,
    enabled,
    from_env,
)

# ---------------------------------------------------------------------------
# enabled() / from_env()
# ---------------------------------------------------------------------------


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv(DOOMED_KILL_ENV, raising=False)
    assert enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_enabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv(DOOMED_KILL_ENV, value)
    assert enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_enabled_falsy_values(monkeypatch, value):
    monkeypatch.setenv(DOOMED_KILL_ENV, value)
    assert enabled() is False


def test_from_env_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(DOOMED_MARGIN_ENV, raising=False)
    monkeypatch.delenv(DOOMED_POLLS_ENV, raising=False)
    monkeypatch.delenv(DOOMED_MIN_PROGRESS_ENV, raising=False)

    comp = from_env([CurvePoint(0, 1.0)], True, 100)

    assert comp.margin == pytest.approx(0.30)
    assert comp.polls_required == 6
    assert comp.min_progress == pytest.approx(0.2)
    assert comp.best_curve == [CurvePoint(0, 1.0)]
    assert comp.higher_is_better is True
    assert comp.planned_steps == 100


def test_from_env_reads_all_four_knobs(monkeypatch):
    monkeypatch.setenv(DOOMED_MARGIN_ENV, "0.5")
    monkeypatch.setenv(DOOMED_POLLS_ENV, "3")
    monkeypatch.setenv(DOOMED_MIN_PROGRESS_ENV, "0.1")

    comp = from_env(None, False, 50)

    assert comp.margin == pytest.approx(0.5)
    assert comp.polls_required == 3
    assert comp.min_progress == pytest.approx(0.1)


def test_from_env_garbage_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv(DOOMED_MARGIN_ENV, "not-a-float")
    monkeypatch.setenv(DOOMED_POLLS_ENV, "not-an-int")
    monkeypatch.setenv(DOOMED_MIN_PROGRESS_ENV, "garbage")

    comp = from_env(None, None, None)

    assert comp.margin == pytest.approx(0.30)
    assert comp.polls_required == 6
    assert comp.min_progress == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# observe() — conservative-discipline guard rails
# ---------------------------------------------------------------------------


def test_never_fires_without_best_curve():
    """No baseline (attempt 1) -- best_curve is None -- never fires no
    matter how bad the live value is."""
    comp = DoomedComparator(best_curve=None, higher_is_better=True, planned_steps=100, polls_required=1)
    for step in range(20, 26):
        assert comp.observe(CurvePoint(step, 0.0)) is False


def test_never_fires_with_empty_best_curve():
    comp = DoomedComparator(best_curve=[], higher_is_better=True, planned_steps=100, polls_required=1)
    assert comp.observe(CurvePoint(50, 0.0)) is False


def test_never_fires_with_unknown_direction():
    """higher_is_better is None (metric contract doesn't know) -- never
    fires regardless of the gap."""
    best = [CurvePoint(0, 1.0), CurvePoint(50, 1.0)]
    comp = DoomedComparator(best_curve=best, higher_is_better=None, planned_steps=100, polls_required=1)
    assert comp.observe(CurvePoint(50, 0.0)) is False


def test_never_fires_below_progress_floor():
    """point.step below min_progress * planned_steps never counts as
    worse, even with an enormous gap."""
    best = [CurvePoint(0, 1.0), CurvePoint(50, 1.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=100, min_progress=0.2, polls_required=1,
    )
    assert comp.observe(CurvePoint(19, 0.0)) is False  # 19 < 0.2*100=20
    assert comp.observe(CurvePoint(20, 0.0)) is True  # 20 >= floor, gap=1.0 >= margin


def test_never_fires_without_planned_steps():
    best = [CurvePoint(0, 1.0)]
    comp = DoomedComparator(best_curve=best, higher_is_better=True, planned_steps=None, polls_required=1)
    assert comp.observe(CurvePoint(50, 0.0)) is False


def test_no_comparable_point_never_fires():
    """best_curve has data only AFTER the live point's step -- last-value-
    carried-forward finds nothing at-or-before -- never fires."""
    best = [CurvePoint(60, 1.0), CurvePoint(80, 1.0)]
    comp = DoomedComparator(best_curve=best, higher_is_better=True, planned_steps=100, polls_required=1)
    assert comp.observe(CurvePoint(50, 0.0)) is False


def test_interpolation_last_value_carried_forward():
    """A live point BETWEEN two best-curve points uses the earlier
    (at-or-before) value, not the later one."""
    best = [CurvePoint(0, 1.0), CurvePoint(100, 0.0)]  # best degrades a lot, but LATER
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=100, margin=0.3, min_progress=0.0, polls_required=1,
    )
    # At step 5, last-value-carried-forward uses best=1.0 (from step 0), not
    # the step-100 value of 0.0 -- a live value of 0.9 is NOT a >=30% gap
    # against 1.0 (gap=0.1) even though it WOULD be against 0.0.
    assert comp.observe(CurvePoint(5, 0.9)) is False


# ---------------------------------------------------------------------------
# observe() — relative margin math, both directions
# ---------------------------------------------------------------------------


def test_relative_margin_higher_is_better_exact_boundary():
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10, margin=0.30, min_progress=0.0, polls_required=1,
    )
    # gap = (10 - 7) / 10 = 0.30 -- exactly at the margin -- fires (>=).
    assert comp.observe(CurvePoint(5, 7.0)) is True


def test_relative_margin_higher_is_better_just_under_boundary():
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10, margin=0.30, min_progress=0.0, polls_required=1,
    )
    # gap = (10 - 7.01) / 10 = 0.299 -- just under the margin -- never fires.
    assert comp.observe(CurvePoint(5, 7.01)) is False


def test_relative_margin_higher_is_better_live_ahead_never_fires():
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10, margin=0.30, min_progress=0.0, polls_required=1,
    )
    assert comp.observe(CurvePoint(5, 15.0)) is False  # live AHEAD of best


def test_relative_margin_lower_is_better_exact_boundary():
    """higher_is_better=False (e.g. loss): live is worse when it is HIGHER
    than best by the relative margin."""
    best = [CurvePoint(0, 1.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=False, planned_steps=10, margin=0.30, min_progress=0.0, polls_required=1,
    )
    # gap = (1.3 - 1.0) / 1.0 = 0.30 -- exactly at margin -- fires.
    assert comp.observe(CurvePoint(5, 1.3)) is True


def test_relative_margin_lower_is_better_live_better_never_fires():
    best = [CurvePoint(0, 1.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=False, planned_steps=10, margin=0.30, min_progress=0.0, polls_required=1,
    )
    assert comp.observe(CurvePoint(5, 0.1)) is False  # lower loss than best -- better, not worse


def test_zero_best_value_never_computes_a_gap():
    """A zero baseline has no computable RELATIVE gap -- treated as no
    gap, never fires (avoids a division by zero)."""
    best = [CurvePoint(0, 0.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10, margin=0.01, min_progress=0.0, polls_required=1,
    )
    assert comp.observe(CurvePoint(5, -100.0)) is False


# ---------------------------------------------------------------------------
# observe() — consecutive-streak discipline
# ---------------------------------------------------------------------------


def test_fires_on_sustained_gap_after_polls_required_consecutive():
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10,
        margin=0.30, min_progress=0.0, polls_required=3,
    )
    assert comp.observe(CurvePoint(1, 5.0)) is False  # 1st worse observation
    assert comp.observe(CurvePoint(2, 5.0)) is False  # 2nd
    assert comp.observe(CurvePoint(3, 5.0)) is True  # 3rd -- polls_required reached


def test_streak_resets_on_non_worse_observation():
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10,
        margin=0.30, min_progress=0.0, polls_required=3,
    )
    assert comp.observe(CurvePoint(1, 5.0)) is False  # worse (1/3)
    assert comp.observe(CurvePoint(2, 5.0)) is False  # worse (2/3)
    assert comp.observe(CurvePoint(3, 10.0)) is False  # NOT worse -- resets streak to 0
    assert comp.observe(CurvePoint(4, 5.0)) is False  # worse (1/3 again, not 3/3)
    assert comp.observe(CurvePoint(5, 5.0)) is False  # worse (2/3)
    assert comp.observe(CurvePoint(6, 5.0)) is True  # worse (3/3) -- fires now


def test_streak_continues_to_fire_while_still_worse():
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10,
        margin=0.30, min_progress=0.0, polls_required=2,
    )
    assert comp.observe(CurvePoint(1, 5.0)) is False
    assert comp.observe(CurvePoint(2, 5.0)) is True
    assert comp.observe(CurvePoint(3, 5.0)) is True  # still worse -- keeps firing


def test_below_progress_floor_observation_does_not_build_a_streak():
    """An observation below the progress floor is treated as non-worse (it
    resets/never advances the streak), so genuine worse-observations that
    follow must still accumulate from zero."""
    best = [CurvePoint(0, 10.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=100,
        margin=0.30, min_progress=0.5, polls_required=2,
    )
    assert comp.observe(CurvePoint(10, 0.0)) is False  # below floor (10 < 50)
    assert comp.observe(CurvePoint(60, 0.0)) is False  # 1st real worse observation
    assert comp.observe(CurvePoint(70, 0.0)) is True  # 2nd -- fires
