from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_DESCRIPTIONS
from backend.agents.rlm.binding import build_custom_tools

EXPECTED = {
    "understand_section", "extract_hyperparameters", "detect_environment",
    "build_environment", "plan_reproduction", "implement_baseline",
    "run_experiment", "verify_against_rubric", "propose_improvements",
    "record_candidate_outcome", "check_user_messages", "respond_to_user",
    "heartbeat", "recommend_next_tool",
    "resolve_gpu_requirements",  # dynamic-GPU spec 2026-05-23
    "codex_repair",  # optional Codex CLI repo-editing subagent, default off
    "read_context_map",  # E1: intra-run context map; advertised only when OPENRESEARCH_CONTEXT_MAP on
}


def test_registry_and_descriptions_cover_all_primitives():
    assert set(PRIMITIVE_REGISTRY) == EXPECTED
    assert set(PRIMITIVE_DESCRIPTIONS) == EXPECTED


def test_build_custom_tools_produces_rlm_tool_dict(make_context, tmp_path, monkeypatch):
    # E1: read_context_map is in the registry unconditionally but advertised to the
    # root only when OPENRESEARCH_CONTEXT_MAP is on. Off (default) → the advertised tool
    # list is the 16 always-on primitives (byte-for-byte today).
    monkeypatch.delenv("OPENRESEARCH_CONTEXT_MAP", raising=False)
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    assert set(tools) == EXPECTED - {"read_context_map"}
    for entry in tools.values():
        assert callable(entry["tool"])
        assert isinstance(entry["description"], str) and entry["description"]


def test_context_map_tool_advertised_when_flag_on(make_context, tmp_path, monkeypatch):
    # E1: with OPENRESEARCH_CONTEXT_MAP on, read_context_map IS advertised → all 17.
    monkeypatch.setenv("OPENRESEARCH_CONTEXT_MAP", "1")
    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)
    assert set(tools) == EXPECTED
