"""Tests for backend.evals.paperbench.leaf_scorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.evals.paperbench.leaf_scorer import (
    amend_final_report,
    flatten_leaves,
    roll_up,
    score_reproduction,
)

# ---------------------------------------------------------------------------
# Synthetic rubric fixture
# ---------------------------------------------------------------------------

TINY_TREE = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {
            "id": "branch-a",
            "requirements": "branch a",
            "weight": 3,
            "sub_tasks": [
                {
                    "id": "leaf-a1",
                    "requirements": "leaf a1",
                    "weight": 1,
                    "sub_tasks": [],
                },
                {
                    "id": "leaf-a2",
                    "requirements": "leaf a2",
                    "weight": 1,
                    "sub_tasks": [],
                },
            ],
        },
        {
            "id": "leaf-b",
            "requirements": "leaf b",
            "weight": 1,
            "sub_tasks": [],
        },
    ],
}


# ---------------------------------------------------------------------------
# Test 1: flatten_leaves count
# ---------------------------------------------------------------------------


def test_flatten_leaves_count():
    leaves = flatten_leaves(TINY_TREE)
    ids = {leaf["id"] for leaf in leaves}
    assert len(leaves) == 3
    assert ids == {"leaf-a1", "leaf-a2", "leaf-b"}


# ---------------------------------------------------------------------------
# Test 2: roll_up weighted math
# ---------------------------------------------------------------------------


def test_roll_up_weighted_math():
    # branch-a has weight=3, leaf-b has weight=1 at root level.
    # branch-a score = (score(a1)*1 + score(a2)*1) / 2
    # root score = (branch_a_score * 3 + leaf_b_score * 1) / 4
    #
    # With: leaf-a1=1.0, leaf-a2=0.0, leaf-b=0.5
    # branch-a = (1.0 + 0.0) / 2 = 0.5
    # root = (0.5*3 + 0.5*1) / 4 = (1.5 + 0.5) / 4 = 2.0/4 = 0.5
    leaf_scores = {"leaf-a1": 1.0, "leaf-a2": 0.0, "leaf-b": 0.5}
    result = roll_up(TINY_TREE, leaf_scores)
    assert abs(result - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Test 3: score_reproduction with mock client
# ---------------------------------------------------------------------------


class MockLlmClient:
    """Returns canned scores for the synthetic tree leaves."""

    def complete(self, *, system: str, user: str) -> str:
        return json.dumps([
            {"leaf_id": "leaf-a1", "score": 1.0, "justification": "fully satisfied"},
            {"leaf_id": "leaf-a2", "score": 0.0, "justification": "not found"},
            {"leaf_id": "leaf-b", "score": 0.5, "justification": "partial"},
        ])


def test_score_reproduction_overall():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        # Provide a minimal final_report.json so evidence gathering has something
        (run_dir / "final_report.json").write_text(
            json.dumps({"reproduction_summary": "test run", "metrics": {}}),
            encoding="utf-8",
        )

        result = score_reproduction(TINY_TREE, run_dir, MockLlmClient(), batch_size=15)

    assert abs(result["overall_score"] - 0.5) < 1e-9
    assert result["leaf_count"] == 3
    assert result["graded"] == 3
    assert result["rubric_source"] == "paperbench_bundle"
    assert len(result["leaf_scores"]) == 3


# ---------------------------------------------------------------------------
# Test 4: rubric_source="generated" is propagated to the result dict
# ---------------------------------------------------------------------------


def test_score_reproduction_generated_rubric_source():
    """Passing rubric_source='generated' puts 'generated' in the result dict.

    Locks in: the new rubric_source keyword param of score_reproduction is
    forwarded verbatim — the arXiv self-generated-rubric path sets it so the
    report is honest about its rubric origin.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "final_report.json").write_text(
            json.dumps({"reproduction_summary": "arxiv run", "metrics": {}}),
            encoding="utf-8",
        )

        result = score_reproduction(
            TINY_TREE, run_dir, MockLlmClient(), rubric_source="generated"
        )

    assert result["rubric_source"] == "generated"


# ---------------------------------------------------------------------------
# Test 5: amend_final_report keeps final_report.md consistent with the JSON
# ---------------------------------------------------------------------------


def _rlm_report_dict() -> dict:
    """A dict carrying every RLMFinalReport field (RLM-mode report shape)."""
    return {
        "paper": {"id": "2510.25013", "title": "Test Paper"},
        "verdict": "partial",
        "reproduction_summary": "ran the baseline",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.999, "meets_target": True, "areas": []},
        "improvements": [],
        "primitive_trace": {"calls": 0, "by_primitive": {}},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 3,
    }


def test_amend_final_report_rerenders_markdown():
    """amend_final_report rewrites final_report.md with the leaf score.

    Regression: score_run.py updated only final_report.json, so the markdown
    GET /runs/{id}/final-report serves kept showing the stale in-loop
    verify_against_rubric score. The markdown must track the JSON.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "final_report.json").write_text(
            json.dumps(_rlm_report_dict()), encoding="utf-8"
        )
        (run_dir / "final_report.md").write_text(
            "# PARTIAL REPRODUCTION\n\n## Rubric Score\n\n"
            "**Overall score:** 0.999  (meets target)\n\n## Reproduction Summary\n\nx\n",
            encoding="utf-8",
        )
        amend_final_report(run_dir, {
            "overall_score": 0.42, "rubric_source": "generated",
            "leaf_count": 10, "graded": 10,
        })
        md = (run_dir / "final_report.md").read_text(encoding="utf-8")
        report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))

    assert report["rubric"]["overall_score"] == 0.42
    assert "0.420" in md
    assert "self-generated rubric" in md
    assert "10/10 rubric leaves graded" in md
    assert "0.999" not in md  # the stale score is gone


def test_amend_final_report_leaves_non_rlm_markdown_untouched():
    """A non-RLM final_report.json must not have its markdown clobbered — the
    RLM markdown renderer only applies to RLM-shaped reports."""
    stale_md = "# Some SDK report\n\nOverall: 0.7\n"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "final_report.json").write_text(
            json.dumps({"sdk_field": 1, "another": 2}), encoding="utf-8"
        )
        (run_dir / "final_report.md").write_text(stale_md, encoding="utf-8")
        amend_final_report(run_dir, {
            "overall_score": 0.3, "rubric_source": "paperbench_bundle",
            "leaf_count": 5, "graded": 5,
        })
        md = (run_dir / "final_report.md").read_text(encoding="utf-8")

    assert md == stale_md  # untouched — the RLM-shape guard prevented a re-render
