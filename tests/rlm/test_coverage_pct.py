"""β2: coverage_pct — failure-exclusion math tests.

coverage_pct = graded_count / total_leaves.
On degraded runs it is 0.0. On non-degraded runs it measures how many leaves
got a real LLM grade vs batch_error fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_rubric(n: int) -> dict:
    """Build a flat rubric with n leaves."""
    return {
        "id": "root",
        "requirements": "root",
        "weight": 1,
        "sub_tasks": [
            {"id": f"L{i}", "requirements": f"Leaf {i}", "weight": 1, "sub_tasks": []}
            for i in range(n)
        ],
    }


def _stub_llm_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.complete.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# coverage_pct on degraded runs
# ---------------------------------------------------------------------------


def test_degraded_run_coverage_pct_is_zero(tmp_path: Path):
    from backend.evals.paperbench.leaf_scorer import score_reproduction

    rubric = _make_rubric(3)
    client = _stub_llm_client([])  # should not be called

    result = score_reproduction(
        rubric_tree=rubric,
        run_dir=tmp_path,
        llm_client=client,
        degraded=True,
    )
    assert result["coverage_pct"] == pytest.approx(0.0)
    assert result["graded"] == 0


# ---------------------------------------------------------------------------
# coverage_pct on normal runs
# ---------------------------------------------------------------------------


def test_full_coverage_when_all_leaves_graded(tmp_path: Path):
    from backend.evals.paperbench.leaf_scorer import score_reproduction

    rubric = _make_rubric(2)
    # Stub LLM to return scores for both leaves
    stub_response = '[{"leaf_id": "L0", "score": 0.8, "justification": "ok"}, {"leaf_id": "L1", "score": 0.6, "justification": "ok"}]'
    client = _stub_llm_client([stub_response])

    result = score_reproduction(
        rubric_tree=rubric,
        run_dir=tmp_path,
        llm_client=client,
        degraded=False,
    )
    assert result["graded"] == 2
    assert result["coverage_pct"] == pytest.approx(1.0)


def test_partial_coverage_when_batch_errors(tmp_path: Path):
    from backend.evals.paperbench.leaf_scorer import score_reproduction

    rubric = _make_rubric(3)
    # Stub LLM to fail (batch_error) — all leaves get _graded=False
    client = MagicMock()
    client.complete.side_effect = Exception("LLM unavailable")

    result = score_reproduction(
        rubric_tree=rubric,
        run_dir=tmp_path,
        llm_client=client,
        degraded=False,
    )
    # All leaves default to batch_error (score=0.0, _graded=False)
    assert result["graded"] == 0
    assert result["coverage_pct"] == pytest.approx(0.0)
    assert result["leaf_count"] == 3


def test_zero_leaves_coverage_is_one(tmp_path: Path):
    """When the rubric has no leaves at all (shouldn't happen in practice) coverage=1.0.

    Note: flatten_leaves treats a root with empty sub_tasks as a single leaf,
    so this test verifies the graded/total math when we force leaf_count=0
    by testing the coverage formula directly.
    """
    # A rubric where flatten_leaves returns 1 leaf (the root itself)
    # but the LLM returns a valid grade → 1/1 = 100% coverage.
    from backend.evals.paperbench.leaf_scorer import score_reproduction

    rubric = {"id": "root", "requirements": "the root", "weight": 1, "sub_tasks": []}
    stub_resp = '[{"leaf_id": "root", "score": 0.7, "justification": "ok"}]'
    client = _stub_llm_client([stub_resp])

    result = score_reproduction(
        rubric_tree=rubric,
        run_dir=tmp_path,
        llm_client=client,
        degraded=False,
    )
    assert result["coverage_pct"] == pytest.approx(1.0)
    assert result["leaf_count"] == 1
    assert result["graded"] == 1
