"""Tests for backend.evals.paperbench.leaf_scorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.evals.paperbench.leaf_scorer import (
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
