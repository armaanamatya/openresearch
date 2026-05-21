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
