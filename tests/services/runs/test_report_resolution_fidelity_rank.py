"""A5 — fidelity-first best-attempt selection in resolve_best_report.

The Codex blocker: best attempt was chosen purely by rubric score, so a broken-
but-high-score attempt would be surfaced ahead of a faithful attempt that
refutes the paper.  These tests pin the fix and confirm legacy runs are
unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.services.runs.report_resolution import (
    resolve_best_report,
    two_axis_fidelity_key,
)


def _attempt(run_dir: Path, name: str, report: dict) -> None:
    d = run_dir / "attempts" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_report.json").write_text(json.dumps(report), encoding="utf-8")


def _two_axis(impl: str, fidelity: float, overall: float, replication: str = "inconclusive") -> dict:
    return {
        "schema_version": 2,
        "implementation_verdict": impl,
        "replication_verdict": replication,
        "reproducibility": {"fidelity_score": fidelity},
        "rubric": {"overall_score": overall, "compute_adjusted_score": overall},
        "verdict": {"faithful": "reproduced", "partial": "partial", "broken": "failed"}[impl],
    }


def _legacy(overall: float, verdict: str = "partial") -> dict:
    return {"verdict": verdict, "rubric": {"overall_score": overall, "compute_adjusted_score": overall}}


def test_faithful_contradicted_beats_broken_high_score(tmp_path):
    """The whole point of A5: a faithful attempt that refutes the paper (lower
    rubric score) is selected over a broken attempt with a HIGHER rubric score."""
    _attempt(tmp_path, "a1_broken_highscore", _two_axis("broken", fidelity=0.85, overall=0.80))
    _attempt(tmp_path, "a2_faithful_contradicted",
             _two_axis("faithful", fidelity=0.90, overall=0.50, replication="contradicted"))

    best = resolve_best_report(tmp_path)
    assert best.report["implementation_verdict"] == "faithful"
    assert best.report["replication_verdict"] == "contradicted"


def test_faithful_outranks_partial_among_two_axis(tmp_path):
    _attempt(tmp_path, "p", _two_axis("partial", fidelity=0.95, overall=0.95))
    _attempt(tmp_path, "f", _two_axis("faithful", fidelity=0.70, overall=0.40))
    best = resolve_best_report(tmp_path)
    assert best.report["implementation_verdict"] == "faithful"


def test_legacy_ranking_unchanged(tmp_path):
    """Legacy (no two-axis fields) runs still pick the highest rubric score."""
    _attempt(tmp_path, "lo", _legacy(0.40))
    _attempt(tmp_path, "hi", _legacy(0.80))
    best = resolve_best_report(tmp_path)
    assert best.report["rubric"]["overall_score"] == 0.80


def test_two_axis_fidelity_key_none_for_legacy():
    assert two_axis_fidelity_key(_legacy(0.8)) is None
    assert two_axis_fidelity_key({"rubric": {"overall_score": 0.5}}) is None


def test_two_axis_fidelity_key_ranks_by_impl_then_fidelity():
    assert two_axis_fidelity_key(_two_axis("faithful", 0.9, 0.5)) == (2, 0.9)
    assert two_axis_fidelity_key(_two_axis("partial", 0.8, 0.8)) == (1, 0.8)
    assert two_axis_fidelity_key(_two_axis("broken", 0.7, 0.95)) == (0, 0.7)
