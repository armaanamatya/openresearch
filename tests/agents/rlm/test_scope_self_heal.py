"""Scope self-healing: a repeated-identical scope-shape violation becomes a tolerated
SCOPE REDUCTION (converge) instead of an infinite repair loop (the 2026-05-30 WebShop loop).
"""
from __future__ import annotations

from backend.agents.rlm.primitives import (
    _decline_advisory_note,
    _gap_in_load_failures,
    _rubric_declining,
    _rubric_plateaued,
    _scope_reduce_or_fail,
    _scope_violation_key,
)

HINT = "per_dataset_incomplete: model 'qwen3_1_7b' missing datasets ['WebShop'] in per_dataset."
HINT_OTHER_MODEL = "per_dataset_incomplete: model 'qwen2_5_3b' missing datasets ['WebShop'] in per_dataset."
HINT_ALFWORLD = "per_dataset_incomplete: model 'x' missing datasets ['ALFWorld'] in per_dataset."


def test_violation_key_stable_across_model_names():
    # The missing element (WebShop) is the stable signature; the model named varies.
    assert _scope_violation_key(HINT) == _scope_violation_key(HINT_OTHER_MODEL)
    assert "webshop" in _scope_violation_key(HINT)
    assert _scope_violation_key(HINT) != _scope_violation_key(HINT_ALFWORLD)


def test_first_violation_is_repairable_failure():
    counts: dict = {}
    res, tol = _scope_reduce_or_fail({"success": True, "metrics": {"a": 1}}, HINT, counts, 2)
    assert tol is False
    assert res["success"] is False
    assert res["scope_shape_violation"] is True


def test_kth_identical_violation_tolerated_as_reduction():
    counts: dict = {}
    r1, tol1 = _scope_reduce_or_fail({"success": True, "metrics": {"a": 1}}, HINT, counts, 2)
    assert tol1 is False and r1["success"] is False  # 1st → repairable

    r2, tol2 = _scope_reduce_or_fail({"success": True, "metrics": {"a": 1}}, HINT, counts, 2)
    assert tol2 is True                # 2nd identical → tolerated
    assert r2["success"] is True       # success preserved → root converges, no loop
    assert r2.get("scope_reduced") is True
    assert any("webshop" in g for g in r2["metrics"]["scope_gaps"])


def test_disabled_when_max_repeats_zero():
    counts: dict = {}
    for _ in range(5):
        _r, tol = _scope_reduce_or_fail({"success": True, "metrics": {}}, HINT, counts, 0)
        assert tol is False  # 0 disables the reduction — always repairable


def test_different_missing_elements_counted_independently():
    counts: dict = {}
    _scope_reduce_or_fail({"success": True, "metrics": {}}, HINT, counts, 2)  # webshop #1
    _r, tol = _scope_reduce_or_fail({"success": True, "metrics": {}}, HINT_ALFWORLD, counts, 2)
    assert tol is False  # alfworld's 1st miss is independent of webshop's


# --- rubric plateau (no-progress) detector -------------------------------------


def test_plateau_needs_full_window():
    assert _rubric_plateaued([0.2, 0.2], window=3, epsilon=0.005) is False  # only 2 samples


def test_plateau_flags_flatlined_score():
    assert _rubric_plateaued([0.0, 0.23, 0.23, 0.23], window=3, epsilon=0.005) is True


def test_plateau_ignores_improving_run():
    # Score still climbing across the window → not plateaued, keep iterating.
    assert _rubric_plateaued([0.10, 0.18, 0.27], window=3, epsilon=0.005) is False


def test_plateau_tolerates_sub_epsilon_noise():
    # Tiny churn below epsilon counts as flat (stuck), not progress.
    assert _rubric_plateaued([0.230, 0.231, 0.2305], window=3, epsilon=0.005) is True


def test_plateau_disabled_when_window_le_1():
    assert _rubric_plateaued([0.2, 0.2, 0.2], window=1, epsilon=0.005) is False
    assert _rubric_plateaued([0.2, 0.2, 0.2], window=0, epsilon=0.005) is False


# --- rubric decline (regression) detector --------------------------------------
# §3.5 (grader-noise companion): the overthinking / inverse-scaling signal the
# flatline detector above MISSES — the score peaked and recent changes made it
# worse. Drives a regression convergence_note (flag OPENRESEARCH_RUBRIC_DECLINE_ADVISORY).


def test_decline_needs_full_window():
    assert _rubric_declining([0.5, 0.4], window=3, epsilon=0.005) is False  # only 2 samples


def test_decline_flags_regression_off_peak():
    # Peaked at 0.62; latest 0.55 is below the peak AND the last step regressed.
    assert _rubric_declining([0.30, 0.62, 0.55], window=3, epsilon=0.005) is True


def test_decline_flags_monotone_drop():
    assert _rubric_declining([0.62, 0.58, 0.55], window=3, epsilon=0.005) is True


def test_decline_ignores_improving_run():
    # Latest score is the window peak → real progress, never flagged.
    assert _rubric_declining([0.10, 0.18, 0.27], window=3, epsilon=0.005) is False


def test_decline_ignores_recovering_run():
    # Dipped then climbed back: the LAST step was up, so don't nag it to restore.
    assert _rubric_declining([0.62, 0.40, 0.61], window=3, epsilon=0.005) is False


def test_decline_ignores_flatline():
    # A flatline is _rubric_plateaued's job; the decline detector stays quiet.
    assert _rubric_declining([0.23, 0.23, 0.23], window=3, epsilon=0.005) is False


def test_decline_tolerates_sub_epsilon_noise():
    # Sub-epsilon churn off the peak is not a regression.
    assert _rubric_declining([0.231, 0.2305, 0.2308], window=3, epsilon=0.005) is False


def test_decline_disabled_when_window_le_1():
    assert _rubric_declining([0.5, 0.4, 0.3], window=1, epsilon=0.005) is False
    assert _rubric_declining([0.5, 0.4, 0.3], window=0, epsilon=0.005) is False


def test_decline_and_plateau_mutually_exclusive():
    declining = [0.62, 0.58, 0.55]
    flat = [0.23, 0.23, 0.23]
    assert _rubric_declining(declining, window=3, epsilon=0.005) is True
    assert _rubric_plateaued(declining, window=3, epsilon=0.005) is False
    assert _rubric_declining(flat, window=3, epsilon=0.005) is False
    assert _rubric_plateaued(flat, window=3, epsilon=0.005) is True


# --- decline advisory note (the full firing decision) --------------------------
# "Validate the advisory fires": the helper exercises the exact branch condition
# (below target AND past the floor AND declining) plus the note text, with no LLM
# and no env — so the firing decision is deterministically tested.

_DECLINING = [0.62, 0.58, 0.55]


def test_decline_note_fires_on_regression():
    note = _decline_advisory_note(
        _DECLINING, 3, 0.005,
        meets_target=False, current_iteration=3, min_iterations=2,
        overall_score=0.55, target=0.70,
    )
    assert note is not None
    assert "REGRESSING" in note
    assert "0.620" in note  # the recent high it tells the root to restore
    assert "RESTORE" in note


def test_decline_note_silent_when_target_met():
    assert _decline_advisory_note(
        _DECLINING, 3, 0.005,
        meets_target=True, current_iteration=3, min_iterations=2,
        overall_score=0.55, target=0.50,
    ) is None


def test_decline_note_silent_below_iteration_floor():
    assert _decline_advisory_note(
        _DECLINING, 3, 0.005,
        meets_target=False, current_iteration=1, min_iterations=2,
        overall_score=0.55, target=0.70,
    ) is None


def test_decline_note_silent_when_improving():
    assert _decline_advisory_note(
        [0.10, 0.18, 0.27], 3, 0.005,
        meets_target=False, current_iteration=3, min_iterations=2,
        overall_score=0.27, target=0.70,
    ) is None


def test_decline_note_silent_when_recovering():
    assert _decline_advisory_note(
        [0.62, 0.40, 0.61], 3, 0.005,
        meets_target=False, current_iteration=3, min_iterations=2,
        overall_score=0.61, target=0.70,
    ) is None


def test_decline_note_carries_no_budget():
    # "no budget for now": the note must not surface wall-clock / seconds.
    note = _decline_advisory_note(
        _DECLINING, 3, 0.005,
        meets_target=False, current_iteration=3, min_iterations=2,
        overall_score=0.55, target=0.70,
    )
    assert note is not None
    assert "wall-clock" not in note and "remains" not in note and "seconds" not in note


# --- data_load_failures bridge → tolerate-on-first-sight --------------------------


def test_gap_matches_recorded_load_failure():
    metrics = {"data_load_failures": [{"dataset": "WebShop", "error": "HTTP 404"}]}
    assert _gap_in_load_failures(HINT, metrics) is True


def test_gap_not_matched_without_failure_record():
    assert _gap_in_load_failures(HINT, {"data_load_failures": []}) is False
    assert _gap_in_load_failures(HINT, {}) is False


def test_gap_string_entry_form_matches():
    assert _gap_in_load_failures(HINT, {"data_load_failures": ["webshop"]}) is True


def test_two_element_gap_needs_all_recorded():
    two = "per_dataset_incomplete: model 'x' missing datasets ['WebShop', 'ALFWorld']."
    # only webshop recorded → not all covered → no force-reduce
    assert _gap_in_load_failures(two, {"data_load_failures": ["webshop"]}) is False
    # both recorded → covered
    assert _gap_in_load_failures(two, {"data_load_failures": ["webshop", "alfworld"]}) is True


def test_force_reduce_tolerates_on_first_sight():
    counts: dict = {}
    res, tol = _scope_reduce_or_fail(
        {"success": True, "metrics": {"a": 1}}, HINT, counts, 2, force_reduce=True
    )
    assert tol is True          # 1st miss, but provably uncontrollable → tolerated now
    assert res["success"] is True
    assert res.get("scope_reduced") is True
    assert any("webshop" in g for g in res["metrics"]["scope_gaps"])
