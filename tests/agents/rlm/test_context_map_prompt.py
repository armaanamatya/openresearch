from backend.agents.rlm import system_prompt as sp


def test_prompt_mentions_read_context_map():
    assert "read_context_map" in sp.SYSTEM_PROMPT
