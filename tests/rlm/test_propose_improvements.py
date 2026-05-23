import json

from backend.agents.rlm.primitives import propose_improvements

HYP_JSON = json.dumps({"hypotheses": [
    {"path_id": "p1", "hypothesis": "Tune the learning rate.",
     "rationale": "Learning rate is the highest-leverage PPO hyperparameter.",
     "expected_outcome": "Higher mean reward at the same step budget.",
     "category": "optimizer"},
    {"path_id": "p2", "hypothesis": "Swap the backbone.",
     "rationale": "A wider value network may fit the return better.",
     "expected_outcome": "Lower value loss and faster convergence.",
     "category": "architecture"},
]})


def test_propose_improvements_returns_variable_length_tagged_list(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[HYP_JSON])
    result = propose_improvements({"success": True}, {"areas": []}, ctx=ctx)
    assert isinstance(result, list)
    assert len(result) == 2
    assert {h["category"] for h in result} == {"optimizer", "architecture"}


def test_propose_improvements_drops_malformed_items(make_context, tmp_path):
    bad = json.dumps({"hypotheses": [
        {"path_id": "p1", "hypothesis": "A good hypothesis.",
         "rationale": "sound reasoning", "expected_outcome": "better metric",
         "category": "optimizer"},
        {"path_id": "p2"},  # missing required fields -> dropped fail-soft
    ]})
    ctx = make_context(tmp_path, llm_responses=[bad])
    result = propose_improvements({"success": True}, {"areas": []}, ctx=ctx)
    assert len(result) == 1
    assert result[0]["category"] == "optimizer"


def test_propose_improvements_caps_result_at_k(make_context, tmp_path):
    three = json.dumps({"hypotheses": [
        {"path_id": f"p{i}", "hypothesis": "h", "rationale": "r",
         "expected_outcome": "o", "category": "c"} for i in range(3)]})
    ctx = make_context(tmp_path, llm_responses=[three])
    result = propose_improvements({"success": True}, {"areas": []}, k=1, ctx=ctx)
    assert len(result) == 1


def test_propose_improvements_coerces_string_k(make_context, tmp_path):
    # LLM-generated REPL code may pass k as a string like "2"; it must be
    # coerced to an int and applied as the cap, not ignored.
    three = json.dumps({"hypotheses": [
        {"path_id": f"p{i}", "hypothesis": "h", "rationale": "r",
         "expected_outcome": "o", "category": "c"} for i in range(3)]})
    ctx = make_context(tmp_path, llm_responses=[three])
    result = propose_improvements({"success": True}, {"areas": []}, k="2", ctx=ctx)
    assert len(result) == 2


def test_propose_improvements_clamps_nonpositive_k(make_context, tmp_path):
    # k <= 0 from the root model must not empty the result via out[:k]; clamp to 1.
    three = json.dumps({"hypotheses": [
        {"path_id": f"p{i}", "hypothesis": "h", "rationale": "r",
         "expected_outcome": "o", "category": "c"} for i in range(3)]})
    ctx = make_context(tmp_path, llm_responses=[three])
    result = propose_improvements({"success": True}, {"areas": []}, k=0, ctx=ctx)
    assert len(result) == 1


def test_propose_improvements_failsoft_on_truncated_json(make_context, tmp_path):
    """Symptom: a truncated LLM response crashes the primitive with ValueError.

    propose_improvements diverged from plan_reproduction / verify_against_rubric
    (review I3 / T11) — both of those wrap _extract_json in a fail-soft except,
    propose_improvements did not. Verify: a truncated JSON response yields a
    single-item error list, never a raised ValueError.
    """
    # A truncated JSON object — _extract_json raises ValueError on this exact input.
    truncated = '{"hypotheses": [{"path_id":"p1","hypothesis":"H'
    ctx = make_context(tmp_path, llm_responses=[truncated])

    # Must NOT raise.
    result = propose_improvements({"success": True}, {}, ctx=ctx)

    assert isinstance(result, list)
    assert result, "fail-soft contract violated: empty list returned instead of error item"
    assert result[0].get("success") is False
    # The error string should surface the underlying exception type/class.
    err = result[0].get("error", "")
    assert "propose_improvements:" in err
    assert ("ValueError" in err) or ("truncated" in err.lower())
