from backend.agents.gepa.adapters.implement_baseline import ImplementBaselineProxyEvaluator


def test_proxy_score_valid_output():
    ev = ImplementBaselineProxyEvaluator()
    score, info = ev(
        candidate={"implement_baseline_system": "You are an implementer."},
        example={
            "mode": "adapt", "code_path": "runs/x/code/",
            "dockerfile_path": "runs/x/Dockerfile",
            "diff_summary": "applied changes",
            "commands_to_run": ["python train.py"],
            "assumptions_applied": ["A001"],
        },
    )
    assert score == 1.0


def test_proxy_score_missing_fields():
    ev = ImplementBaselineProxyEvaluator()
    score, info = ev(
        candidate={"implement_baseline_system": "You are an implementer."},
        example={"mode": "adapt"},  # missing many fields
    )
    assert score < 1.0
