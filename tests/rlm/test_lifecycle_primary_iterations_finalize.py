"""Under lifecycle-primary, the finalize iteration count must come from the
driver summary, not rlm_logger.iteration_count (which is 0 in this mode)."""

from __future__ import annotations

from backend.agents.rlm import run as run_mod


def test_primary_iterations_prefers_summary_over_logger():
    assert run_mod._resolve_run_iterations(
        primary_active=True, summary={"iterations": 6}, logger_iterations=0
    ) == 6
    assert run_mod._resolve_run_iterations(
        primary_active=False, summary=None, logger_iterations=4
    ) == 4
    assert run_mod._resolve_run_iterations(
        primary_active=True, summary={}, logger_iterations=3
    ) == 3
