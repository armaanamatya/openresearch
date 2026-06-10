"""Time-budget gate — the cell matrix must fit the run's remaining wall clock.

Pins the 2026-06-10 Adam v6 lesson: a 100-epoch re-grid got per_cell × waves
of budget with ~4h of run left, sailed into the 14h watchdog, and was
hard-killed mid-cell at a salvaged 0.151 — when the runner's own deadline
machinery (trim un-launched cells to honest `timeout` results) would have
returned a scoreable partial if anyone had told it the real budget.
"""

from __future__ import annotations

from backend.agents.rlm.cell_scheduler import cap_overall_budget


def test_caps_to_remaining_minus_reserve():
    # 4h left, default 45min reserve -> 3h15m budget
    assert cap_overall_budget(100_000.0, 14_400.0) == 14_400.0 - 2_700.0


def test_no_remaining_time_means_no_change():
    assert cap_overall_budget(100_000.0, None) == 100_000.0
    assert cap_overall_budget(100_000.0, 0.0) == 100_000.0
    assert cap_overall_budget(None, None) is None


def test_generous_budget_untouched_when_it_fits():
    assert cap_overall_budget(3_600.0, 50_400.0) == 3_600.0


def test_floor_protects_late_run_matrix():
    # 20 min left: remaining - reserve is negative; floor gives min(rem/2, 900)
    assert cap_overall_budget(100_000.0, 1_200.0) == 600.0
    # 1h left: floor 900 beats 3600-2700=900 (equal)
    assert cap_overall_budget(100_000.0, 3_600.0) == 900.0


def test_none_overall_adopts_capped_budget():
    assert cap_overall_budget(None, 14_400.0) == 11_700.0


def test_custom_reserve():
    assert cap_overall_budget(100_000.0, 10_000.0, reserve_s=1_000.0) == 9_000.0
