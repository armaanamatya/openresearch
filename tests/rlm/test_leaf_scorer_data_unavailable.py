"""Tests for PR-κ: data-unavailable leaf pre-filter in score_reproduction.

Covers:
1. Skip via data_load_failures (fuzzy match)
2. Skip via scope.gaps alone (no data_load_failures entry)
3. Skip via metrics_shape (exact json_path match — PR-θ integration)
4. Fuzzy match path without metrics_shape
5. No skip when NEITHER signal mentions the dataset
6. roll_up excludes skipped leaves from both numerator AND denominator
7. coverage_pct uses eligible_count (not raw total) as denominator
8. Backward compat: no signals present → behavior identical to pre-κ
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.evals.paperbench.leaf_scorer import (
    _detect_data_unavailable_leaves,
    roll_up,
    score_reproduction,
)

# ---------------------------------------------------------------------------
# Synthetic rubric with 4 leaves.
# Leaf 3 mentions "Frey Face" — the unavailable dataset in our fixtures.
# ---------------------------------------------------------------------------

FOUR_LEAF_TREE = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {
            "id": "area-a",
            "requirements": "area a",
            "weight": 2,
            "sub_tasks": [
                {
                    "id": "leaf-1",
                    "requirements": "MNIST Nz=5 test ELBO",
                    "weight": 1,
                    "sub_tasks": [],
                },
                {
                    "id": "leaf-2",
                    "requirements": "MNIST Nz=10 test ELBO",
                    "weight": 1,
                    "sub_tasks": [],
                },
            ],
        },
        {
            "id": "area-b",
            "requirements": "area b",
            "weight": 2,
            "sub_tasks": [
                {
                    "id": "leaf-3",
                    "requirements": "Frey Face Nz=5 test ELBO (HTTP-gated dataset)",
                    "weight": 1,
                    "sub_tasks": [],
                },
                {
                    "id": "leaf-4",
                    "requirements": "Log-likelihood: AEVB vs Wake-Sleep on binarised MNIST",
                    "weight": 1,
                    "sub_tasks": [],
                },
            ],
        },
    ],
}

LEAVES = [
    FOUR_LEAF_TREE["sub_tasks"][0]["sub_tasks"][0],  # leaf-1
    FOUR_LEAF_TREE["sub_tasks"][0]["sub_tasks"][1],  # leaf-2
    FOUR_LEAF_TREE["sub_tasks"][1]["sub_tasks"][0],  # leaf-3 (frey face)
    FOUR_LEAF_TREE["sub_tasks"][1]["sub_tasks"][1],  # leaf-4
]


def _write_metrics(run_dir: Path, data: dict) -> None:
    outputs = run_dir / "code" / "outputs" / "run-abc"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "metrics.json").write_text(json.dumps(data), encoding="utf-8")


def _write_report(run_dir: Path, data: dict) -> None:
    (run_dir / "final_report.json").write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_dir_with_frey_failure(tmp_path):
    """metrics.json has data_load_failures for frey_face; final_report has scope.gaps."""
    _write_metrics(tmp_path, {
        "mnist_nz5_test_elbo": -128.8,
        "mnist_nz10_test_elbo": -111.5,
        "frey_face_status": "data_unavailable",
        "data_load_failures": [
            {"dataset": "frey_face", "loader": "scipy_mat", "error": "HTTP 403"},
        ],
    })
    _write_report(tmp_path, {
        "baseline_metrics": {"mnist_nz5_test_elbo": -128.8},
        "verdict": "partial",
        "scope": {
            "gaps": ["Frey Face: HTTP 403 — dataset requires institutional access"],
        },
    })
    return tmp_path


@pytest.fixture()
def run_dir_gaps_only(tmp_path):
    """scope.gaps mentions Frey Face but data_load_failures is absent."""
    _write_metrics(tmp_path, {
        "mnist_nz5_test_elbo": -128.8,
    })
    _write_report(tmp_path, {
        "baseline_metrics": {"mnist_nz5_test_elbo": -128.8},
        "verdict": "partial",
        "scope": {
            "gaps": ["Frey Face dataset not downloaded — licence gated"],
        },
    })
    return tmp_path


@pytest.fixture()
def run_dir_no_unavailable(tmp_path):
    """Neither data_load_failures NOR scope.gaps mentions any dataset."""
    _write_metrics(tmp_path, {
        "mnist_nz5_test_elbo": -128.8,
        "mnist_nz10_test_elbo": -111.5,
    })
    _write_report(tmp_path, {
        "baseline_metrics": {"mnist_nz5_test_elbo": -128.8},
        "verdict": "partial",
    })
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1 — skip via data_load_failures (fuzzy match)
# ---------------------------------------------------------------------------


def test_detect_skips_via_data_load_failures(run_dir_with_frey_failure):
    result = _detect_data_unavailable_leaves(LEAVES, run_dir_with_frey_failure)
    assert "leaf-3" in result, "leaf-3 (Frey Face) must be detected as unavailable"
    assert "leaf-1" not in result
    assert "leaf-2" not in result
    assert "leaf-4" not in result


# ---------------------------------------------------------------------------
# Test 2 — skip via scope.gaps alone
# ---------------------------------------------------------------------------


def test_detect_skips_via_scope_gaps_only(run_dir_gaps_only):
    result = _detect_data_unavailable_leaves(LEAVES, run_dir_gaps_only)
    assert "leaf-3" in result, "leaf-3 must be skipped when scope.gaps mentions Frey Face"
    assert "leaf-1" not in result
    assert "leaf-4" not in result


# ---------------------------------------------------------------------------
# Test 3 — metrics_shape exact json_path match (PR-θ integration)
# ---------------------------------------------------------------------------


def test_detect_skips_via_metrics_shape_exact(run_dir_with_frey_failure):
    """When metrics_shape declares a json_path absent from metrics.json AND the
    metric_id contains a failed-dataset token, all declared rubric_leaf_ids are
    marked unavailable — no fuzzy description match needed."""
    metrics_shape = [
        {
            "metric_id": "frey_face_nz5_test_elbo",
            "json_path": "frey_face.nz5.test_elbo",
            "rubric_leaf_ids": ["leaf-3"],
        },
        {
            "metric_id": "mnist_nz5_test_elbo",
            "json_path": "mnist_nz5_test_elbo",
            "rubric_leaf_ids": ["leaf-1"],
        },
    ]
    result = _detect_data_unavailable_leaves(
        LEAVES, run_dir_with_frey_failure, metrics_shape=metrics_shape
    )
    # frey_face.nz5.test_elbo is absent from metrics.json AND metric_id contains "frey_face"
    assert "leaf-3" in result
    # mnist_nz5_test_elbo IS present in metrics.json — must NOT be skipped
    assert "leaf-1" not in result


# ---------------------------------------------------------------------------
# Test 4 — fuzzy match without metrics_shape
# ---------------------------------------------------------------------------


def test_detect_fuzzy_without_metrics_shape(run_dir_with_frey_failure):
    """Without metrics_shape, the fuzzy token match on leaf description works."""
    # Remove metrics_shape — fall back to fuzzy path
    result = _detect_data_unavailable_leaves(LEAVES, run_dir_with_frey_failure, metrics_shape=None)
    assert "leaf-3" in result
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test 5 — no skip when NEITHER signal mentions the dataset
# ---------------------------------------------------------------------------


def test_detect_no_skip_when_no_signal(run_dir_no_unavailable):
    """Agent forgot to compute Frey Face but didn't record a failure — not skipped."""
    result = _detect_data_unavailable_leaves(LEAVES, run_dir_no_unavailable)
    assert result == set(), f"Expected empty skip set, got {result}"


# ---------------------------------------------------------------------------
# Test 6 — roll_up excludes skipped leaves from numerator AND denominator
# ---------------------------------------------------------------------------


def test_roll_up_with_skip_set():
    """4 leaves total; leaf-3 is skipped; leaves 1, 2, 4 each score 1.0.

    Without skip: area-b = (1.0 + 1.0) / 2 = 1.0; root = (1.0*2 + 1.0*2)/4 = 1.0
    With leaf-3 skipped: area-b has only leaf-4 (weight 1) → area-b = 1.0/1 = 1.0
    root = (area-a*2 + area-b*2) / 4 = (1.0*2 + 1.0*2)/4 = 1.0

    Now test the critical case: leaves 1,2,4 score 1.0, leaf-3 would score 0 →
    without skip: area-b = (0+1)/2 = 0.5; root = (1.0*2 + 0.5*2)/4 = 0.75
    with skip: area-b = 1.0; root = 1.0
    """
    # All three eligible leaves score 1.0; leaf-3 is skipped (not in leaf_scores).
    leaf_scores = {"leaf-1": 1.0, "leaf-2": 1.0, "leaf-4": 1.0}
    skip_set = frozenset({"leaf-3"})

    result = roll_up(FOUR_LEAF_TREE, leaf_scores, skip_set)
    assert abs(result - 1.0) < 1e-9, (
        f"roll_up with skip_set should score 1.0 when all eligible leaves score 1.0, "
        f"got {result}"
    )

    # Confirm WITHOUT skip, the same scores still produce 1.0 (leaf-3 defaults to 0.0
    # but that's not our test — we tested the WITH-skip case above).
    # Now test the "drag-down" scenario: with skip leaf-3 defaults 0 without skip.
    leaf_scores_drag = {"leaf-1": 1.0, "leaf-2": 1.0, "leaf-4": 1.0}
    # Without skip: leaf-3 is absent from leaf_scores → roll_up returns 0.0 for it.
    result_no_skip = roll_up(FOUR_LEAF_TREE, leaf_scores_drag)
    # area-b = (0.0*1 + 1.0*1)/2 = 0.5; root = (1.0*2 + 0.5*2)/4 = 0.75
    assert abs(result_no_skip - 0.75) < 1e-9, (
        f"Without skip, missing leaf-3 should drag root to 0.75, got {result_no_skip}"
    )

    # With skip: root should be 1.0
    assert abs(result - result_no_skip) > 0.1, (
        "skip_set should materially change the roll_up result"
    )


# ---------------------------------------------------------------------------
# Test 7 — coverage_pct uses eligible_count as denominator
# ---------------------------------------------------------------------------


class _FullScoreLlmClient:
    """Grades every leaf at 1.0 (for eligible leaves only — skipped not sent).

    Parses leaf_ids from the tasks JSON block embedded in the user message by
    the _USER_TEMPLATE format string.  The block is a raw JSON array after the
    '## Rubric leaf tasks' heading — no code fence.
    """

    def complete(self, *, system: str, user: str) -> str:
        import json as _json
        # The tasks block is the first [...] array in the user message that
        # contains leaf_id keys.  Find the first '[' after the heading marker.
        marker = "## Rubric leaf tasks"
        start_idx = user.find(marker)
        if start_idx == -1:
            return "[]"
        bracket_start = user.find("[", start_idx)
        if bracket_start == -1:
            return "[]"
        # Walk forward to find the matching closing bracket
        depth = 0
        bracket_end = bracket_start
        for i, ch in enumerate(user[bracket_start:], bracket_start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break
        tasks_raw = user[bracket_start : bracket_end + 1]
        try:
            tasks = _json.loads(tasks_raw)
        except Exception:
            tasks = []
        return _json.dumps([
            {"leaf_id": t["leaf_id"], "score": 1.0, "justification": "fully satisfied"}
            for t in tasks
            if isinstance(t, dict) and "leaf_id" in t
        ])


def test_coverage_pct_uses_eligible_count(run_dir_with_frey_failure):
    """3 eligible leaves graded, 1 skipped → coverage_pct = 3/3 = 1.0, not 3/4 = 0.75."""
    result = score_reproduction(
        FOUR_LEAF_TREE,
        run_dir_with_frey_failure,
        _FullScoreLlmClient(),
        batch_size=10,
    )

    assert result["leaf_count"] == 4, "total leaf count unchanged"
    assert result["unavailable_count"] == 1, "1 leaf skipped (leaf-3)"
    assert result["eligible_count"] == 3, "3 leaves eligible"
    assert result["graded"] == 3, "3 leaves graded by LLM"
    assert abs(result["coverage_pct"] - 1.0) < 1e-9, (
        f"coverage_pct should be 1.0 (3/3), got {result['coverage_pct']}"
    )

    # The overall_score should reflect only the 3 eligible leaves at 1.0
    # (leaf-3 excluded from roll_up).
    assert abs(result["overall_score"] - 1.0) < 1e-9, (
        f"overall_score should be 1.0 when all eligible leaves score 1.0, "
        f"got {result['overall_score']}"
    )

    # The skipped record must be present with score=None and state=skipped_data_unavailable
    skipped = [r for r in result["leaf_scores"] if r.get("state") == "skipped_data_unavailable"]
    assert len(skipped) == 1
    assert skipped[0]["id"] == "leaf-3"
    assert skipped[0]["score"] is None


# ---------------------------------------------------------------------------
# Test 8 — backward compat: no signals → behavior identical to pre-κ
# ---------------------------------------------------------------------------


class _CannedLlmClient:
    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def complete(self, *, system: str, user: str) -> str:
        import json as _json
        # Same parsing as _FullScoreLlmClient
        marker = "## Rubric leaf tasks"
        start_idx = user.find(marker)
        if start_idx == -1:
            return "[]"
        bracket_start = user.find("[", start_idx)
        if bracket_start == -1:
            return "[]"
        depth, bracket_end = 0, bracket_start
        for i, ch in enumerate(user[bracket_start:], bracket_start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break
        try:
            tasks = _json.loads(user[bracket_start : bracket_end + 1])
        except Exception:
            tasks = []
        return _json.dumps([
            {
                "leaf_id": t["leaf_id"],
                "score": self._scores.get(t["leaf_id"], 0.0),
                "justification": "test",
            }
            for t in tasks
            if isinstance(t, dict) and "leaf_id" in t
        ])


def test_backward_compat_no_signals(run_dir_no_unavailable):
    """When no signals present, all leaves are graded — unavailable_count=0,
    eligible_count=leaf_count, and roll_up is identical to pre-κ behaviour."""
    scores = {"leaf-1": 1.0, "leaf-2": 1.0, "leaf-3": 0.0, "leaf-4": 1.0}
    result = score_reproduction(
        FOUR_LEAF_TREE,
        run_dir_no_unavailable,
        _CannedLlmClient(scores),
        batch_size=10,
    )

    assert result["unavailable_count"] == 0
    assert result["eligible_count"] == 4
    assert result["leaf_count"] == 4
    assert result["graded"] == 4

    # area-a = (1.0 + 1.0)/2 = 1.0; area-b = (0.0 + 1.0)/2 = 0.5
    # root = (1.0*2 + 0.5*2)/4 = 0.75
    assert abs(result["overall_score"] - 0.75) < 1e-9, (
        f"Expected 0.75 with no skips, got {result['overall_score']}"
    )
    assert abs(result["coverage_pct"] - 1.0) < 1e-9
