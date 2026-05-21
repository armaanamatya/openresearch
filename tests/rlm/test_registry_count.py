from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY

NINE = {
    "understand_section", "extract_hyperparameters", "detect_environment",
    "build_environment", "plan_reproduction", "implement_baseline",
    "run_experiment", "verify_against_rubric", "propose_improvements",
}


def test_registry_has_exactly_the_nine_brief_primitives():
    assert set(PRIMITIVE_REGISTRY) == NINE
    assert "set_final" not in PRIMITIVE_REGISTRY
