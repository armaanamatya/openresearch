import json

import pytest

from backend.agents.rlm.primitives import verify_against_rubric

RUBRIC = {
    "areas": [{"area": "code", "weight": 0.6}, {"area": "results", "weight": 0.4}],
    "source": "generated",
    "target_score": 0.7,
}
LLM_SCORES = json.dumps({"areas": [
    {"area": "code", "score": 0.9, "weak_points": []},
    {"area": "results", "score": 0.8, "weak_points": []},
], "confidence": 0.8})


def test_verify_caps_a_failed_run(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[LLM_SCORES])
    result = verify_against_rubric({"success": False, "metrics": {"r": 1}}, RUBRIC, ctx=ctx)
    assert all(a["score"] <= 0.35 for a in result["areas"])


def test_verify_caps_a_metric_less_run(make_context, tmp_path):
    # success=True but no metrics — run_experiment returns metrics={} in Phase 2,
    # so a metric-less run must not score high (extends the honesty backstop).
    ctx = make_context(tmp_path, llm_responses=[LLM_SCORES])
    result = verify_against_rubric({"success": True, "metrics": {}}, RUBRIC, ctx=ctx)
    assert all(a["score"] <= 0.35 for a in result["areas"])


def test_verify_uses_llm_scores_for_a_real_run(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[LLM_SCORES])
    result = verify_against_rubric(
        {"success": True, "metrics": {"mean_reward": 200.0}}, RUBRIC, ctx=ctx)
    assert any(a["score"] > 0.35 for a in result["areas"])
    assert result["overall_score"] == pytest.approx(0.86)
