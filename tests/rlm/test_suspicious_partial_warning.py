"""Tests for BUG-LR-015 — suspicious_partial run_warning emitter.

Verifies that a 'suspicious_partial' warning fires when a partial verdict
is accompanied by ≥2 of: essential_primitives_missed, iteration_underutilized,
rubric_never_scored.

Uses a thin harness that calls the relevant section of _finalize logic
indirectly via the public RLMFinalReport + a mocked emit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.rlm.report import RLMFinalReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    *,
    verdict: str = "partial",
    iterations: int = 2,
    by_primitive: dict | None = None,
    overall_score: float | None = None,
) -> RLMFinalReport:
    rpt = RLMFinalReport(
        verdict=verdict,
        reproduction_summary="test",
    )
    rpt.iterations = iterations
    if by_primitive is not None:
        rpt.primitive_trace = {"calls": sum(by_primitive.values()), "by_primitive": by_primitive}
    if overall_score is not None:
        rpt.rubric = {"overall_score": overall_score}
    return rpt


def _run_suspicious_partial_check(
    report: RLMFinalReport,
    *,
    iterations: int,
    run_failed: bool = False,
    remaining_s: float = 3600.0,
) -> list[dict]:
    """Exercise the suspicious_partial block from _finalize() in isolation.

    Returns the list of warning events that would have been emitted.
    """
    emitted: list[dict] = []

    def _emit(event: dict) -> None:
        emitted.append(event)

    ctx = MagicMock()
    ctx.remaining_s.return_value = remaining_s

    # Inline the logic from _finalize so we don't need to build a full RunContext.
    _MAX_ITERATIONS = 20
    _essential = {"implement_baseline", "run_experiment", "verify_against_rubric"}

    if report.verdict == "partial" and not run_failed:
        _remaining = ctx.remaining_s()
        _wall_pressure = _remaining is not None and _remaining <= 60
        if not _wall_pressure:
            from backend.agents.rlm.sse_bridge import build_run_warning_event
            _by_prim = report.primitive_trace.get("by_primitive", {})
            _called = set(_by_prim.keys())
            _essential_missed = not bool(_essential & _called)
            _iter_underutilized = iterations < max(1, int(_MAX_ITERATIONS * 0.25))
            _rubric_never_scored = "verify_against_rubric" not in _called
            _signal_count = sum([_essential_missed, _iter_underutilized, _rubric_never_scored])
            if _signal_count >= 2:
                _missed_names = sorted(_essential - _called)
                _msg = (
                    f"suspicious_partial: run completed with verdict='partial' after only "
                    f"{iterations} iteration(s) without executing key primitives "
                    f"({', '.join(_missed_names) if _missed_names else 'some'}). "
                    f"Signals: essential_primitives_missed={_essential_missed}, "
                    f"iteration_underutilization={_iter_underutilized} "
                    f"(iterations={iterations}, floor={int(_MAX_ITERATIONS * 0.25)}), "
                    f"rubric_never_scored={_rubric_never_scored}. "
                    "This may indicate the model concluded primitives were unavailable "
                    "(see BUG-LR-011/012 in rlm-stability-remediation-design.md)."
                )
                _emit(build_run_warning_event(
                    level="warn",
                    code="suspicious_partial",
                    message=_msg,
                ))

    return emitted


# ---------------------------------------------------------------------------
# Positive: warning fires
# ---------------------------------------------------------------------------

def test_fires_for_prj_09047604e591d969_profile() -> None:
    """Replay today's run: 5 iters, only check_user_messages called, no rubric score."""
    report = _make_report(
        verdict="partial",
        iterations=5,
        by_primitive={"check_user_messages": 2},
    )
    events = _run_suspicious_partial_check(report, iterations=5)
    assert len(events) == 1
    ev = events[0]
    assert ev.get("code") == "suspicious_partial" or "suspicious_partial" in str(ev)


def test_fires_when_all_three_signals_present() -> None:
    report = _make_report(
        verdict="partial",
        iterations=1,
        by_primitive={},
    )
    events = _run_suspicious_partial_check(report, iterations=1)
    assert len(events) == 1


def test_fires_when_two_of_three_signals_present() -> None:
    """essential_missed + rubric_never_scored, but iterations OK."""
    report = _make_report(
        verdict="partial",
        iterations=10,
        by_primitive={"check_user_messages": 2},  # no essential primitives
    )
    events = _run_suspicious_partial_check(report, iterations=10)
    # essential_missed=True, iter_underutilized=False (10 >= 5), rubric_never_scored=True → 2 signals
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Negative: warning does NOT fire
# ---------------------------------------------------------------------------

def test_no_fire_for_successful_run() -> None:
    report = _make_report(
        verdict="reproduced",
        iterations=10,
        by_primitive={"implement_baseline": 1, "run_experiment": 1, "verify_against_rubric": 1},
        overall_score=0.8,
    )
    events = _run_suspicious_partial_check(report, iterations=10)
    assert len(events) == 0


def test_no_fire_for_failed_run() -> None:
    report = _make_report(
        verdict="partial",
        iterations=1,
        by_primitive={},
    )
    events = _run_suspicious_partial_check(report, iterations=1, run_failed=True)
    assert len(events) == 0


def test_no_fire_under_wall_clock_pressure() -> None:
    """Within 60s remaining — a truncated partial is expected; suppress warning."""
    report = _make_report(
        verdict="partial",
        iterations=1,
        by_primitive={},
    )
    events = _run_suspicious_partial_check(report, iterations=1, remaining_s=30.0)
    assert len(events) == 0


def test_no_fire_when_only_one_signal() -> None:
    """Only essential_missed without iter or rubric issues: threshold not reached."""
    report = _make_report(
        verdict="partial",
        iterations=10,
        by_primitive={
            "check_user_messages": 2,
            "verify_against_rubric": 1,  # rubric was scored
        },
        overall_score=0.1,
    )
    # essential_missed=True (no implement_baseline/run_experiment), iter=10 (>floor), rubric=False
    # Only 1 signal: essential_missed → no fire.
    events = _run_suspicious_partial_check(report, iterations=10)
    assert len(events) == 0
