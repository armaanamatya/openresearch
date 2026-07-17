"""tests/rlm/test_doomed_run_comparator.py — doomed-attempt early kill
comparator (spec §10.3, ``OPENRESEARCH_DOOMED_KILL``, default OFF).

Pure logic + the deterministic on-disk evidence reader. ``DoomedComparator``
is exercised directly with constructed ``CurvePoint`` sequences;
``read_cell_curves``/``select_baseline`` against real (tmp_path) artifacts;
``DoomedWatch`` with constructed curve sets. Never touches a socket, GPU or
LLM. The campaign-side wiring (thread, process kill, ledger record) is
covered by ``tests/rlm/test_doomed_kill_campaign.py``.

The bulk of what is asserted here is FALSE-POSITIVE SAFETY: killing a
healthy-but-slow run is a false negative for a needle-in-haystack triage
product, and this repo has a painful history of guards false-blocking
faithful work (``learn.md`` 2026-07-07).
"""

from __future__ import annotations

import json
import math

import pytest

from backend.agents.rlm.doomed_run_comparator import (
    DOOMED_KILL_ENV,
    DOOMED_KILLED_CLASS,
    DOOMED_MARGIN_ENV,
    DOOMED_MIN_PROGRESS_ENV,
    DOOMED_POLLS_ENV,
    CellCurve,
    CurvePoint,
    DoomedComparator,
    DoomedWatch,
    enabled,
    from_env,
    read_cell_curves,
    select_baseline,
    watch_from_env,
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


# ===========================================================================
# read_cell_curves — the deterministic evidence reader (disk -> curves)
# ===========================================================================


def _curve_leaf(steps, values, *, metric="reward", status="ok"):
    return {"status": status, "training_curves": {"step": list(steps), metric: list(values)}}


def _write_metrics(code_dir, per_model):
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "metrics.json").write_text(json.dumps({"per_model": per_model}), encoding="utf-8")


def _rising(n=11, start=0.1, stop=1.0, step_size=10):
    steps = [i * step_size for i in range(n)]
    values = [start + (stop - start) * i / (n - 1) for i in range(n)]
    return steps, values


def test_read_cell_curves_missing_dir_is_empty(tmp_path):
    assert read_cell_curves(tmp_path / "nope") == {}


def test_read_cell_curves_nested_per_model_keys_by_cell_coordinate(tmp_path):
    code = tmp_path / "code"
    steps, values = _rising()
    _write_metrics(code, {"qwen-1.7b": {"alfworld": {"grpo": _curve_leaf(steps, values)}}})

    curves = read_cell_curves(code)

    assert set(curves) == {"qwen-1.7b/alfworld/grpo"}
    cell = curves["qwen-1.7b/alfworld/grpo"]
    assert cell.metric == "reward"
    assert cell.higher_is_better is True
    assert cell.total_steps == 100
    assert cell.latest.value == pytest.approx(1.0)


def test_read_cell_curves_loss_is_lower_is_better(tmp_path):
    code = tmp_path / "code"
    _write_metrics(code, {"m": {"e": {"b": _curve_leaf([0, 1, 2, 3], [2.3, 1.9, 1.4, 0.9], metric="loss")}}})

    cell = read_cell_curves(code)["m/e/b"]

    assert cell.metric == "loss"
    assert cell.higher_is_better is False


def test_read_cell_curves_refuses_a_curve_with_no_step_axis(tmp_path):
    """No explicit step series -> NO comparison. Synthesising steps from list
    indices would compare a run logging every step against one logging every
    tenth, off by a factor of ten -- a first-class false-positive vector."""
    code = tmp_path / "code"
    _write_metrics(code, {"m": {"e": {"b": {"status": "ok", "reward_history": [0.1, 0.2, 0.3, 0.4]}}}})

    assert read_cell_curves(code) == {}


def test_read_cell_curves_refuses_a_length_mismatched_step_axis(tmp_path):
    code = tmp_path / "code"
    _write_metrics(code, {"m": {"e": {"b": _curve_leaf([0, 1, 2], [0.1, 0.2, 0.3, 0.4])}}})

    assert read_cell_curves(code) == {}


def test_read_cell_curves_refuses_a_curve_below_the_minimum_point_count(tmp_path):
    code = tmp_path / "code"
    _write_metrics(code, {"m": {"e": {"b": _curve_leaf([0, 1], [0.1, 0.2])}}})

    assert read_cell_curves(code) == {}


def test_read_cell_curves_reads_per_cell_outputs_mid_flight(tmp_path):
    """The aggregated code/metrics.json only exists once run_experiment has
    FINISHED. Mid-flight, the per-cell outputs/ artifacts are the only curve
    evidence there is -- and a mid-flight kill is the entire point."""
    cell = tmp_path / "code" / "outputs" / "qwen" / "alfworld"
    cell.mkdir(parents=True)
    steps, values = _rising()
    (cell / "training_curves.json").write_text(
        json.dumps({"step": steps, "reward": values}), encoding="utf-8"
    )

    curves = read_cell_curves(tmp_path / "code")

    assert set(curves) == {"qwen/alfworld"}
    assert curves["qwen/alfworld"].total_steps == 100


def test_read_cell_curves_preserves_non_finite_readings(tmp_path):
    """A diverged curve is EVIDENCE, not missing data (the 2026-07-13
    IEEE-754 rule)."""
    code = tmp_path / "code"
    _write_metrics(code, {"m": {"e": {"b": _curve_leaf([0, 1, 2, 3], [1.0, 2.0, float("nan"), float("nan")])}}})

    cell = read_cell_curves(code)["m/e/b"]

    assert math.isnan(cell.latest.value)


def test_read_cell_curves_is_deterministic_and_fail_soft_on_garbage(tmp_path):
    code = tmp_path / "code"
    code.mkdir(parents=True)
    (code / "metrics.json").write_text("{not json", encoding="utf-8")

    assert read_cell_curves(code) == {}


# ===========================================================================
# _relative_gap — a diverged (non-finite) live reading is infinitely worse
# ===========================================================================


def test_non_finite_live_reading_is_worse_by_any_margin():
    best = [CurvePoint(0, 1.0), CurvePoint(100, 1.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=100, margin=0.30,
        min_progress=0.0, polls_required=3,
    )
    assert comp.observe(CurvePoint(10, float("nan"))) is False  # 1/3
    assert comp.observe(CurvePoint(20, float("inf"))) is False  # 2/3
    assert comp.observe(CurvePoint(30, float("-inf"))) is True  # 3/3 -- diverged, killed


def test_transient_non_finite_spike_that_recovers_resets_the_streak():
    """A grad-scaler-recovered fp16 overflow is NOT a diverged run (mirrors
    dead_training_guard.NONFINITE_WINDOW)."""
    best = [CurvePoint(0, 1.0), CurvePoint(100, 1.0)]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=100, margin=0.30,
        min_progress=0.0, polls_required=3,
    )
    assert comp.observe(CurvePoint(10, float("nan"))) is False
    assert comp.observe(CurvePoint(20, float("nan"))) is False
    assert comp.observe(CurvePoint(30, 1.0)) is False  # recovered -- streak reset
    assert comp.observe(CurvePoint(40, float("nan"))) is False  # back to 1/3


def test_non_finite_baseline_never_computes_a_gap():
    best = [CurvePoint(0, float("nan"))]
    comp = DoomedComparator(
        best_curve=best, higher_is_better=True, planned_steps=10, margin=0.01,
        min_progress=0.0, polls_required=1,
    )
    assert comp.observe(CurvePoint(5, -100.0)) is False


# ===========================================================================
# DoomedWatch — staleness, consensus, and the global poll floor
# ===========================================================================


def _cell(key, steps, values, *, metric="reward", higher_is_better=True):
    return CellCurve(
        key=key,
        metric=metric,
        higher_is_better=higher_is_better,
        points=tuple(CurvePoint(int(s), float(v)) for s, v in zip(steps, values)),
    )


def _baseline_cells(keys=("m/e/b",)):
    steps, values = _rising()
    return {k: _cell(k, steps, values) for k in keys}


def _live_flat(keys=("m/e/b",), *, upto_index, value=0.02):
    """A diverging live curve: pinned near zero while the reference rose."""
    steps, _ = _rising()
    steps = steps[: upto_index + 1]
    return {k: _cell(k, steps, [value] * len(steps)) for k in keys}


def _watch(baseline, **kw):
    kw.setdefault("margin", 0.30)
    kw.setdefault("polls_required", 3)
    kw.setdefault("min_progress", 0.2)
    return DoomedWatch(baseline, **kw)


def test_watch_kills_a_diverging_run_after_the_configured_polls():
    watch = _watch(_baseline_cells())
    verdicts = [watch.observe(_live_flat(upto_index=i)) for i in (2, 3, 4, 5, 6)]

    # steps 20/30/40 are the first three ABOVE the 0.2*100 progress floor.
    assert verdicts[0] is None  # step 20 -- worse 1/3
    assert verdicts[1] is None  # step 30 -- worse 2/3
    assert verdicts[2] is not None  # step 40 -- worse 3/3 -> kill
    kill = verdicts[2]
    assert kill.reason == "curve_below_best_completed_attempt"
    assert kill.cells == ("m/e/b",)
    assert kill.polls_required == 3
    assert kill.detail and "relative gap" in kill.detail[0]


def test_watch_never_kills_on_a_single_bad_poll():
    watch = _watch(_baseline_cells(), polls_required=6)
    assert watch.observe(_live_flat(upto_index=6)) is None
    assert watch.observations == 1


def test_watch_never_kills_without_a_baseline():
    """Attempt 1 is structurally unkillable: no completed attempt to compare
    against means no comparator is ever built."""
    watch = _watch({})
    for i in range(3, 11):
        assert watch.observe(_live_flat(upto_index=i)) is None
    assert watch.matched_cells == ()


def test_watch_never_kills_an_unmatched_cell():
    """The live run is training a DIFFERENT cell than the reference ever ran.
    Comparing them would be apples-to-oranges, so it is simply not compared."""
    watch = _watch(_baseline_cells(("m/e/b",)))
    for i in range(3, 11):
        assert watch.observe(_live_flat(("other/cell/x",), upto_index=i)) is None
    assert watch.matched_cells == ()


def test_watch_does_not_kill_during_a_long_warmup():
    """The whole early curve sits below the progress floor. Nothing there is
    ever judged -- and the reference run was warming up at those same steps
    anyway, so a shared warmup cancels out by construction."""
    watch = _watch(_baseline_cells(), polls_required=1, min_progress=0.5)
    for i in range(2, 5):  # steps 20, 30, 40 -- all below 0.5*100
        assert watch.observe(_live_flat(upto_index=i)) is None


def test_watch_does_not_kill_a_healthy_but_slow_starting_curve():
    """Behind early, catches up later. The margin is relative to the reference
    AT THE SAME STEP, and any single non-worse reading resets the streak."""
    steps, values = _rising()
    watch = _watch(_baseline_cells())
    for i in range(2, len(steps)):
        # 25% behind the reference: real, but inside the 30% margin.
        live = {"m/e/b": _cell("m/e/b", steps[: i + 1], [v * 0.75 for v in values[: i + 1]])}
        assert watch.observe(live) is None


def test_watch_does_not_kill_a_run_that_recovers_before_the_streak_completes():
    steps, values = _rising()
    watch = _watch(_baseline_cells(), polls_required=3)
    bad = {"m/e/b": _cell("m/e/b", steps[:3], [0.02] * 3)}  # step 20 -- worse 1/3
    assert watch.observe(bad) is None
    bad2 = {"m/e/b": _cell("m/e/b", steps[:4], [0.02] * 4)}  # step 30 -- worse 2/3
    assert watch.observe(bad2) is None
    recovered = {"m/e/b": _cell("m/e/b", steps[:5], [*([0.02] * 4), values[4]])}  # step 40 -- healthy
    assert watch.observe(recovered) is None
    # ...and the streak restarted from zero, so one more bad reading is 1/3.
    assert watch.observe({"m/e/b": _cell("m/e/b", steps[:6], [0.02] * 6)}) is None


def test_watch_ignores_stale_readings_during_a_quiet_eval_phase():
    """A quiet eval phase writes NO new curve points. Re-reading the same last
    point six times must not manufacture six observations -- otherwise a run
    that merely paused to evaluate gets killed for standing still."""
    watch = _watch(_baseline_cells(), polls_required=3)
    stale = _live_flat(upto_index=2)  # one genuinely worse reading, at step 20
    assert watch.observe(stale) is None
    for _ in range(20):  # the curve never advances: pure re-reads
        assert watch.observe(stale) is None
    assert watch.observations == 1  # exactly ONE real observation was ever taken


def test_watch_consensus_one_healthy_matched_cell_vetoes_the_kill():
    """Cell A is diverging; cell B is tracking the reference. The attempt as a
    whole is not doomed, so it is not killed."""
    steps, values = _rising()
    baseline = _baseline_cells(("a/e/b", "b/e/b"))
    watch = _watch(baseline, polls_required=2)
    for i in range(2, 8):
        live = {
            "a/e/b": _cell("a/e/b", steps[: i + 1], [0.02] * (i + 1)),  # diverging
            "b/e/b": _cell("b/e/b", steps[: i + 1], values[: i + 1]),  # healthy
        }
        assert watch.observe(live) is None
    assert set(watch.matched_cells) == {"a/e/b", "b/e/b"}


def test_watch_kills_when_every_matched_cell_is_diverging():
    steps, _ = _rising()
    watch = _watch(_baseline_cells(("a/e/b", "b/e/b")), polls_required=2)
    verdict = None
    for i in range(2, 8):
        live = {k: _cell(k, steps[: i + 1], [0.02] * (i + 1)) for k in ("a/e/b", "b/e/b")}
        verdict = watch.observe(live)
        if verdict is not None:
            break
    assert verdict is not None
    assert verdict.cells == ("a/e/b", "b/e/b")


def test_watch_global_poll_floor_holds_even_when_a_cell_fires_early():
    """Belt-and-braces: no kill before polls_required ADVANCING polls, however
    the per-cell streaks line up."""
    watch = _watch(_baseline_cells(), polls_required=4)
    assert watch.observe(_live_flat(upto_index=2)) is None
    assert watch.observe(_live_flat(upto_index=3)) is None
    assert watch.observe(_live_flat(upto_index=4)) is None
    assert watch.observations == 3
    assert watch.observe(_live_flat(upto_index=5)) is not None
    assert watch.observations == 4


def test_watch_from_env_reads_the_three_documented_knobs(monkeypatch):
    monkeypatch.setenv(DOOMED_MARGIN_ENV, "0.6")
    monkeypatch.setenv(DOOMED_POLLS_ENV, "9")
    monkeypatch.setenv(DOOMED_MIN_PROGRESS_ENV, "0.4")

    watch = watch_from_env(_baseline_cells())

    assert watch.margin == pytest.approx(0.6)
    assert watch.polls_required == 9
    assert watch.min_progress == pytest.approx(0.4)


# ===========================================================================
# select_baseline — trust filter + evidence ranking (never the LLM grade)
# ===========================================================================


def _assessed_row(attempt_n, *, predicates=3, hard=False, failure_class=None):
    return {
        "attempt_n": attempt_n,
        "status": "assessed",
        "assessment": {
            "attempt_n": attempt_n,
            "hard_quarantined": hard,
            "failure_class": failure_class,
            "evidence_predicates": {
                "backed_by_ledger": predicates > 0,
                "provenance_present": predicates > 1,
                "metrics_non_degenerate": predicates > 2,
                "metric_keys_real": predicates > 3,
                "run_level_clean": True,  # never counted
            },
            # A deliberately HIGH score on an otherwise weak attempt: if the
            # ranking ever keyed on the LLM grade, this would win.
            "final_report": {"score": 0.99, "target": 0.8},
        },
    }


def _code_with_curve(tmp_path, name):
    code = tmp_path / name / "code"
    steps, values = _rising()
    _write_metrics(code, {"m": {"e": {"b": _curve_leaf(steps, values)}}})
    return str(code)


def test_select_baseline_none_when_no_prior_attempt(tmp_path):
    assert select_baseline([], exclude_attempt_n=1, resolve_code_dir=lambda n: None) is None


def test_select_baseline_excludes_the_in_flight_attempt(tmp_path):
    rows = [_assessed_row(1)]
    got = select_baseline(
        rows, exclude_attempt_n=1, resolve_code_dir=lambda n: _code_with_curve(tmp_path, f"a{n}")
    )
    assert got is None


def test_select_baseline_never_uses_a_hard_quarantined_attempt(tmp_path):
    """A fabrication-flagged attempt must never become the yardstick a real
    run is killed against."""
    rows = [_assessed_row(1, hard=True)]
    got = select_baseline(
        rows, exclude_attempt_n=2, resolve_code_dir=lambda n: _code_with_curve(tmp_path, f"a{n}")
    )
    assert got is None


def test_select_baseline_never_uses_an_itself_doomed_killed_attempt(tmp_path):
    rows = [_assessed_row(1, failure_class=DOOMED_KILLED_CLASS)]
    got = select_baseline(
        rows, exclude_attempt_n=2, resolve_code_dir=lambda n: _code_with_curve(tmp_path, f"a{n}")
    )
    assert got is None


def test_select_baseline_ranks_by_evidence_not_by_score(tmp_path):
    """Attempt 1 has the higher LLM grade baked into its report; attempt 2 has
    more MEASURED evidence. Evidence wins -- the red line."""
    rows = [_assessed_row(1, predicates=1), _assessed_row(2, predicates=4)]
    got = select_baseline(
        rows, exclude_attempt_n=3, resolve_code_dir=lambda n: _code_with_curve(tmp_path, f"a{n}")
    )
    assert got is not None
    assert got[0] == 2
    assert set(got[1]) == {"m/e/b"}


def test_select_baseline_skips_an_attempt_with_no_readable_curves(tmp_path):
    rows = [_assessed_row(1, predicates=4), _assessed_row(2, predicates=1)]

    def _resolve(n):
        return None if n == 1 else _code_with_curve(tmp_path, f"a{n}")

    got = select_baseline(rows, exclude_attempt_n=3, resolve_code_dir=_resolve)
    assert got is not None
    assert got[0] == 2
