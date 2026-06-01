from unittest.mock import MagicMock
from backend.agents.gepa.adapters.propose_improvements import ProposeImprovementsEvaluator


def test_evaluate_valid_hypotheses():
    llm = MagicMock()
    llm.complete = MagicMock(return_value='''{
        "hypotheses": [
            {"path_id":"p1","title":"LR","category":"hyperparameter",
             "hypothesis":"reduce","rationale":"plateau","expected_outcome":"better"},
            {"path_id":"p2","title":"Reg","category":"regularization",
             "hypothesis":"dropout","rationale":"overfit","expected_outcome":"better"}
        ]
    }''')
    ev = ProposeImprovementsEvaluator(llm_client=llm)
    score, info = ev(
        candidate={"propose_improvements_system": "You are an improver."},
        example={"weak_leaves": ["hyperparameter"]},
    )
    assert 0.0 < score <= 1.0


def test_evaluate_bad_json_returns_zero():
    llm = MagicMock()
    llm.complete = MagicMock(return_value="not json")
    ev = ProposeImprovementsEvaluator(llm_client=llm)
    score, info = ev(
        candidate={"propose_improvements_system": "broken"},
        example={"weak_leaves": ["hyperparameter"]},
    )
    assert score == 0.0
    assert "error" in info
