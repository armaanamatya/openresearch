import json

import pytest

from backend.agents.rlm.primitives import verify_against_rubric

# Minimal 2-level PaperBench tree rubric (the new contract).
RUBRIC = {
    "id": "root",
    "requirements": "reproduce the paper",
    "weight": 1.0,
    "source": "generated",
    "target_score": 0.7,
    "sub_tasks": [
        {
            "id": "code",
            "requirements": "code is implemented",
            "weight": 0.6,
            "sub_tasks": [],
        },
        {
            "id": "results",
            "requirements": "results are reported",
            "weight": 0.4,
            "sub_tasks": [],
        },
    ],
}

# The leaf scorer calls llm_client.complete once per batch and expects a JSON
# array of {"leaf_id", "score", "justification"}.  Both leaves are in one batch.
_LLM_BATCH_RESPONSE = json.dumps([
    {"leaf_id": "code", "score": 0.9, "justification": "code implemented"},
    {"leaf_id": "results", "score": 0.8, "justification": "results reported"},
])


def test_verify_returns_overall_score(make_context, tmp_path):
    """verify_against_rubric returns a non-degenerate overall_score on a tree rubric."""
    ctx = make_context(tmp_path, llm_responses=[_LLM_BATCH_RESPONSE])
    result = verify_against_rubric({"success": True, "metrics": {"r": 1}}, RUBRIC, ctx=ctx)
    # weighted rollup: 0.9*0.6 + 0.8*0.4 = 0.54+0.32 = 0.86
    assert result["overall_score"] == pytest.approx(0.86)


def test_verify_meets_target(make_context, tmp_path):
    """meets_target reflects overall_score >= target_score."""
    ctx = make_context(tmp_path, llm_responses=[_LLM_BATCH_RESPONSE])
    result = verify_against_rubric({"success": True, "metrics": {"r": 1}}, RUBRIC, ctx=ctx)
    assert result["meets_target"] is True
    assert result["target_score"] == pytest.approx(0.7)


def test_verify_has_weak_leaves(make_context, tmp_path):
    """Result includes weak_leaves sorted ascending by score."""
    ctx = make_context(tmp_path, llm_responses=[_LLM_BATCH_RESPONSE])
    result = verify_against_rubric({"success": True, "metrics": {"r": 1}}, RUBRIC, ctx=ctx)
    assert "weak_leaves" in result
    scores = [e["score"] for e in result["weak_leaves"]]
    assert scores == sorted(scores)


def test_verify_has_leaf_scores(make_context, tmp_path):
    """Result includes leaf_scores with id and score fields."""
    ctx = make_context(tmp_path, llm_responses=[_LLM_BATCH_RESPONSE])
    result = verify_against_rubric({"success": True, "metrics": {"r": 1}}, RUBRIC, ctx=ctx)
    assert "leaf_scores" in result
    ids = {e["id"] for e in result["leaf_scores"]}
    assert ids == {"code", "results"}


def test_verify_fail_soft_on_empty_rubric(make_context, tmp_path):
    """verify_against_rubric returns error dict on empty/None rubric (fail-soft)."""
    ctx = make_context(tmp_path)
    result = verify_against_rubric({}, {}, ctx=ctx)
    assert result.get("success") is False
    assert "error" in result

    result2 = verify_against_rubric({}, None, ctx=ctx)  # type: ignore[arg-type]
    assert result2.get("success") is False
    assert "error" in result2
