import json

from backend.agents.rlm.primitives import plan_reproduction, _PLAN_REPRODUCTION_SYSTEM

CONTRACT_JSON = json.dumps({
    "reproduction_definition": "Same algorithm, same dataset.",
    "smoke_test_plan": "1000 timesteps.",
    "full_run_plan": "500k timesteps.",
    "expected_outputs": ["metrics.json"],
    "evaluation_plan": "Mean reward over 100 episodes.",
})


def test_plan_reproduction_parses_llm_contract(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[CONTRACT_JSON])
    result = plan_reproduction({"core_contribution": "X"}, {"framework": "pytorch"}, ctx=ctx)
    assert result["reproduction_definition"] == "Same algorithm, same dataset."
    assert result["expected_outputs"] == ["metrics.json"]
    assert len(ctx.llm_client.calls) == 1
    assert ctx.llm_client.calls[0]["system"] == _PLAN_REPRODUCTION_SYSTEM


def test_plan_reproduction_prompt_names_the_exact_contract_keys(make_context, tmp_path):
    # ReproductionContract is extra="ignore": prose-guessed keys would be
    # silently dropped to a near-empty contract. The prompt must name the
    # real JSON keys.
    from backend.agents.schemas import ReproductionContract

    ctx = make_context(tmp_path, llm_responses=[CONTRACT_JSON])
    plan_reproduction({"core_contribution": "X"}, {"framework": "pytorch"}, ctx=ctx)
    user_prompt = ctx.llm_client.calls[0]["user"]
    for key in ReproductionContract.model_fields:
        assert key in user_prompt
