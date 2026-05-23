"""Tests for the recommend_next_tool() primitive (W4 — Reflexion-lite)."""

from __future__ import annotations

import json


def test_returns_dict_with_required_keys(make_context, tmp_path):
    """Happy path: llm_query returns clean JSON → dict with tool/reason/alternatives."""
    response = json.dumps({
        "tool": "rlm_query",
        "reason": "The passage is >10K chars and a focused question is known.",
        "alternatives": ["understand_section"],
    })
    ctx = make_context(tmp_path, llm_responses=[response])

    from backend.agents.rlm.primitives import recommend_next_tool

    result = recommend_next_tool("I have a 15K-char methods section and need metrics.", ctx=ctx)

    assert isinstance(result, dict)
    assert result["tool"] == "rlm_query"
    assert isinstance(result["reason"], str) and result["reason"]
    assert isinstance(result["alternatives"], list)
    assert "understand_section" in result["alternatives"]


def test_handles_json_wrapped_in_markdown_fences(make_context, tmp_path):
    """llm_query wraps JSON in ```json fences → still parsed correctly via _extract_json."""
    inner = {"tool": "understand_section", "reason": "Short slice.", "alternatives": []}
    fenced = f"```json\n{json.dumps(inner)}\n```"
    ctx = make_context(tmp_path, llm_responses=[fenced])

    from backend.agents.rlm.primitives import recommend_next_tool

    result = recommend_next_tool("Methods section is 3K chars.", ctx=ctx)

    assert result["tool"] == "understand_section"
    assert result["reason"] == "Short slice."
    assert result["alternatives"] == []


def test_graceful_on_llm_exception(make_context, tmp_path, monkeypatch):
    """If the LLM call raises, returns error dict with empty tool — never raises."""
    ctx = make_context(tmp_path)

    # Monkeypatch the llm_client to raise
    def _raise(**_kw):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(ctx.llm_client, "complete", _raise)

    from backend.agents.rlm.primitives import recommend_next_tool

    result = recommend_next_tool("Should I call rlm_query or understand_section?", ctx=ctx)

    assert isinstance(result, dict)
    assert result["tool"] == ""
    assert "recommend_next_tool failed" in result["reason"]
    assert "RuntimeError" in result["reason"]
    assert result["alternatives"] == []


def test_emits_primitive_call_event(make_context, tmp_path):
    """wrap_primitive emits a primitive_call event for recommend_next_tool."""
    response = json.dumps({"tool": "llm_query", "reason": "Simple task.", "alternatives": []})
    ctx = make_context(tmp_path, llm_responses=[response])

    from backend.agents.rlm.binding import build_custom_tools

    tools = build_custom_tools(ctx)
    assert "recommend_next_tool" in tools

    tools["recommend_next_tool"]["tool"]("Which tool next?")

    events = [
        json.loads(line)
        for line in (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    primitive_events = [
        e for e in events
        if e.get("event") == "primitive_call" and e.get("primitive") == "recommend_next_tool"
    ]
    assert len(primitive_events) >= 1
    # At least one completed event (status ok or error — not just start)
    completed = [e for e in primitive_events if e.get("status") in ("ok", "error")]
    assert len(completed) >= 1
