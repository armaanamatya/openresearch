"""P0 honesty fix: verdict must never claim more than the rubric leaf score supports.

Symptom: the `ftrl` RLM run self-reported verdict="reproduced" while its
post-run leaf score was 0.000.  reconcile_verdict_with_score() and the call
site in amend_final_report() together close that gap.

Tests:
- Unit: reconcile_verdict_with_score() threshold matrix.
- Integration: amend_final_report() rewrites verdict in the JSON and the
  re-rendered markdown reflects it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


from backend.agents.rlm.report import reconcile_verdict_with_score
from backend.evals.paperbench.leaf_scorer import amend_final_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rlm_report_dict(verdict: str = "reproduced") -> dict:
    """Return a dict carrying every RLMFinalReport field.

    _rerender_report_markdown checks fields.issubset(report.keys()), so all
    declared model fields must be present for the markdown re-render path to
    be exercised.
    """
    return {
        "paper": {"id": "ftrl-test", "title": "FTRL Test Paper"},
        "verdict": verdict,
        "reproduction_summary": "Baseline reproduced.",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.0, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {"calls": 0, "by_primitive": {}},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 1,
    }


def _score_dict(overall_score: float) -> dict:
    """Return a minimal score dict as produced by score_reproduction()."""
    return {
        "overall_score": overall_score,
        "leaf_count": 10,
        "graded": 10,
        "rubric_source": "paperbench_bundle",
        "leaf_scores": [],
    }


# ---------------------------------------------------------------------------
# Unit tests: reconcile_verdict_with_score
# ---------------------------------------------------------------------------


class TestReconcileVerdictWithScore:
    """reconcile_verdict_with_score() must never allow a verdict to claim more
    than the authoritative rubric score supports."""

    def test_reproduced_at_zero_score_becomes_failed(self):
        """Symptom: `ftrl` run — verdict 'reproduced', leaf score 0.000.

        A zero score provides no evidence of reproduction; verdict must be
        downgraded all the way to 'failed'.
        """
        assert reconcile_verdict_with_score("reproduced", 0.0) == "failed"

    def test_reproduced_at_partial_score_becomes_partial(self):
        """A score of 0.3 is above the partial floor but below the reproduced
        threshold — 'reproduced' must be capped at 'partial'."""
        assert reconcile_verdict_with_score("reproduced", 0.3) == "partial"

    def test_reproduced_at_sufficient_score_unchanged(self):
        """A score of 0.7 satisfies the reproduced threshold — the verdict is
        left untouched."""
        assert reconcile_verdict_with_score("reproduced", 0.7) == "reproduced"

    def test_partial_at_below_floor_score_becomes_failed(self):
        """A score of 0.05 is below the partial floor — 'partial' must be
        downgraded to 'failed'."""
        assert reconcile_verdict_with_score("partial", 0.05) == "failed"

    def test_failed_at_high_score_never_upgraded(self):
        """reconcile_verdict_with_score() NEVER upgrades a verdict.  Even a
        score of 0.95 cannot turn 'failed' into 'partial' or 'reproduced'."""
        assert reconcile_verdict_with_score("failed", 0.95) == "failed"

    def test_partial_at_high_score_never_upgraded(self):
        """A score of 0.99 cannot upgrade 'partial' to 'reproduced' — only
        downgrading is allowed."""
        assert reconcile_verdict_with_score("partial", 0.99) == "partial"

    def test_unknown_verdict_treated_as_partial(self):
        """An unrecognised verdict string is treated as rank 'partial'
        (mirrors _reconcile_verdict, which also downgrades unknowns)."""
        # score 0.0 is below partial floor → ceiling is 'failed'
        # unknown rank == partial(1) > failed(0) → downgrade to 'failed'
        assert reconcile_verdict_with_score("something_weird", 0.0) == "failed"
        # score 0.99 → ceiling 'reproduced'; unknown rank == partial(1) < reproduced(2) → unchanged
        assert reconcile_verdict_with_score("something_weird", 0.99) == "something_weird"

    def test_exact_reproduced_threshold(self):
        """Score exactly at _VERDICT_REPRODUCED_MIN_SCORE (0.60) permits
        'reproduced'."""
        assert reconcile_verdict_with_score("reproduced", 0.60) == "reproduced"

    def test_exact_partial_threshold(self):
        """Score exactly at _VERDICT_PARTIAL_MIN_SCORE (0.15) permits
        'partial' but not 'reproduced'."""
        assert reconcile_verdict_with_score("partial", 0.15) == "partial"
        assert reconcile_verdict_with_score("reproduced", 0.15) == "partial"

    def test_just_below_partial_threshold(self):
        """A score just below 0.15 (e.g. 0.149) forces ceiling to 'failed'."""
        assert reconcile_verdict_with_score("partial", 0.149) == "failed"
        assert reconcile_verdict_with_score("reproduced", 0.149) == "failed"


# ---------------------------------------------------------------------------
# Integration test: amend_final_report reconciles the verdict on disk
# ---------------------------------------------------------------------------


def test_amend_final_report_reconciles_reproduced_verdict_at_zero_score():
    """amend_final_report() must rewrite a dishonest 'reproduced' verdict to
    'failed' when the authoritative leaf score is 0.0.

    Symptom: the `ftrl` run wrote verdict='reproduced' to final_report.json,
    then amend_final_report() updated only the rubric block — the dishonest
    verdict survived on disk and in the served markdown.

    After this fix the JSON verdict must be 'failed', the rubric block must be
    populated, and the re-rendered markdown must reflect the corrected verdict.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)

        # Write a final_report.json that mimics the ftrl run: claim reproduced,
        # score 0.0.  Include ALL RLMFinalReport fields so the markdown re-render
        # path is fully exercised (the guard checks fields.issubset(report.keys())).
        initial = _rlm_report_dict(verdict="reproduced")
        (run_dir / "final_report.json").write_text(
            json.dumps(initial), encoding="utf-8"
        )

        # Provide a minimal final_report.md so the markdown re-render path runs.
        (run_dir / "final_report.md").write_text(
            "# REPRODUCED\n\n## Rubric Score\n\n**Overall score:** 0.000\n",
            encoding="utf-8",
        )

        amend_final_report(run_dir, _score_dict(overall_score=0.0))

        report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
        md = (run_dir / "final_report.md").read_text(encoding="utf-8")

    # The verdict must have been downgraded
    assert report["verdict"] == "failed", (
        f"Expected 'failed' but got {report['verdict']!r} — "
        "the ftrl symptom (reproduced @ score 0.0) was not fixed"
    )

    # The rubric block must be populated with the authoritative score
    assert report["rubric"]["overall_score"] == 0.0
    assert report["rubric"]["leaf_count"] == 10
    assert report["rubric"]["graded"] == 10

    # The re-rendered markdown must not claim a successful reproduction
    assert "REPRODUCTION FAILED" in md, (
        "Markdown re-render did not reflect the corrected 'failed' verdict"
    )
    # And must not still show the original dishonest header
    assert "# REPRODUCED\n" not in md
