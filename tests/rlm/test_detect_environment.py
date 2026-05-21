from backend.agents.rlm.primitives import detect_environment


def test_detect_environment_produces_env_spec(make_context, tmp_path):
    ctx = make_context(tmp_path)
    method_spec = {"core_contribution": "A PyTorch RL agent.", "claims": [],
                   "datasets": [], "metrics": []}
    result = detect_environment(method_spec, ctx=ctx)
    assert result["python_version"]
    assert result["framework"]
    assert isinstance(result["pip_packages"], dict)
    assert result["dockerfile"].startswith("FROM")
