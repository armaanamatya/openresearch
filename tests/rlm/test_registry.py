from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_DESCRIPTIONS
from backend.agents.rlm.binding import build_custom_tools

EXPECTED = {
    "understand_section", "extract_hyperparameters", "detect_environment",
    "build_environment", "plan_reproduction", "implement_baseline",
    "run_experiment", "verify_against_rubric", "propose_improvements",
    "record_candidate_outcome", "check_user_messages", "respond_to_user",
}


def test_registry_and_descriptions_cover_all_primitives():
    assert set(PRIMITIVE_REGISTRY) == EXPECTED
    assert set(PRIMITIVE_DESCRIPTIONS) == EXPECTED


def test_build_custom_tools_produces_rlm_tool_dict(make_context, tmp_path):
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    assert set(tools) == EXPECTED
    for entry in tools.values():
        assert callable(entry["tool"])
        assert isinstance(entry["description"], str) and entry["description"]
