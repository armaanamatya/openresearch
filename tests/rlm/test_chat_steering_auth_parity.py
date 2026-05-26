"""Parametrized parity tests: chat-steering primitives work identically under
both --model claude (ANTHROPIC_API_KEY) and --model claude-oauth (subscription).

Goal: lock in API/OAuth wiring parity before prod cutover.

The primitives themselves are file-I/O only (no LLM), so the tests assert that:
  1. build_system_prompt includes both chat-steering primitives for either model.
  2. build_custom_tools exposes both primitives for either model.
  3. POST /runs/<id>/messages is agnostic to root-model auth (regression guard).
  4. check_user_messages and respond_to_user work with no credentials at all.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _reset_settings_cache() -> None:
    import backend.config as _cfg
    _cfg._settings_cache = None


# ---------------------------------------------------------------------------
# Test 1 — system prompt includes chat-steering for both model auth modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_alias", ["claude", "claude-oauth"])
def test_system_prompt_includes_chat_steering_for_both_models(model_alias, monkeypatch):
    """build_system_prompt must mention both chat-steering primitives regardless of
    which root-model auth path is configured.

    build_system_prompt takes an already-resolved RootModel dataclass — no auth
    check is performed at prompt-build time. The parametrization documents that
    both registry entries produce a prompt with identical chat-steering coverage,
    confirming neither model suppresses or omits the steering section.
    """
    # build_system_prompt never touches credentials; set env vars only to
    # document the intended auth context for readers of this test.
    if model_alias == "claude":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from backend.agents.rlm.models import ROOT_MODELS
    from backend.agents.rlm.system_prompt import build_system_prompt

    # Pull the RootModel directly from the registry — no resolve_root_model call,
    # so no credential-check code path is exercised here.
    root_model = ROOT_MODELS[model_alias]
    prompt = build_system_prompt(
        context_metadata={
            "paper_text": {"type": "str", "length": 80_000},
        },
        root_model=root_model,
    )

    assert "check_user_messages" in prompt, (
        f"[{model_alias}] 'check_user_messages' missing from system prompt"
    )
    assert "respond_to_user" in prompt, (
        f"[{model_alias}] 'respond_to_user' missing from system prompt"
    )


# ---------------------------------------------------------------------------
# Test 2 — primitive registry exposes both chat primitives for either model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_alias", ["claude", "claude-oauth"])
def test_primitive_registry_binds_chat_primitives_for_both_models(
    model_alias, monkeypatch, make_context, tmp_path
):
    """build_custom_tools must include check_user_messages and respond_to_user
    regardless of which root-model auth is configured."""
    if model_alias == "claude":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        monkeypatch.setattr(
            "backend.agents.runtime.factory._has_claude_subscription_oauth",
            lambda: False,
        )
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(
            "backend.agents.runtime.factory._has_claude_subscription_oauth",
            lambda: True,
        )

    from backend.agents.rlm.binding import build_custom_tools

    ctx = make_context(tmp_path)
    tools = build_custom_tools(ctx)

    assert "check_user_messages" in tools, (
        f"[{model_alias}] 'check_user_messages' missing from custom_tools"
    )
    assert "respond_to_user" in tools, (
        f"[{model_alias}] 'respond_to_user' missing from custom_tools"
    )
    # Sanity: each entry is properly formed for rlm consumption.
    for name in ("check_user_messages", "respond_to_user"):
        assert callable(tools[name]["tool"]), f"[{model_alias}] {name}['tool'] is not callable"
        assert isinstance(tools[name]["description"], str) and tools[name]["description"], (
            f"[{model_alias}] {name}['description'] is empty or not a str"
        )


# ---------------------------------------------------------------------------
# Test 3 — POST /runs/<id>/messages is agnostic to root-model config
# ---------------------------------------------------------------------------

def test_post_message_endpoint_works_regardless_of_root_model(monkeypatch, tmp_path):
    """Regression guard: the messages endpoint must never gate on root-model auth.

    It is pure file I/O — no LLM credential is consulted. This test runs
    without any API key or OAuth credential set, proving the endpoint is
    fully credential-free.
    """
    # Wipe all LLM credentials so any accidental auth check would fail loudly.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory._has_claude_subscription_oauth",
        lambda: False,
    )

    # Isolate runs root and settings cache.
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("REPROLAB_RUNS_ROOT", str(runs_root))
    monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)
    _reset_settings_cache()

    try:
        from fastapi.testclient import TestClient
        from backend.app import create_app

        client = TestClient(create_app())

        # Prepare a minimal run directory the endpoint can find.
        run_dir = runs_root / "prj_parity"
        run_dir.mkdir(parents=True)
        (run_dir / "demo_status.json").write_text("{}")

        r = client.post(
            "/runs/prj_parity/messages",
            json={"role": "user", "content": "redirect to training section"},
        )
        assert r.status_code == 202, f"Expected 202, got {r.status_code}: {r.text}"
        assert r.json() == {"ok": True}

        # Confirm the message was written — endpoint is pure file I/O.
        jsonl_path = run_dir / "user_messages.jsonl"
        assert jsonl_path.exists(), "user_messages.jsonl was not created"
        entry = json.loads(jsonl_path.read_text().strip())
        assert entry["content"] == "redirect to training section"
    finally:
        _reset_settings_cache()


# ---------------------------------------------------------------------------
# Test 4 — primitives work with zero LLM credentials (pure file I/O)
# ---------------------------------------------------------------------------

def test_check_user_messages_works_without_any_llm_credentials(monkeypatch, make_context, tmp_path):
    """check_user_messages and respond_to_user must work with no credentials at all.

    Both primitives are file-I/O only — they never touch an LLM. This test
    wipes all credential env vars and OAuth detection to prove neither
    primitive requires them.
    """
    # Strip every LLM credential.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory._has_claude_subscription_oauth",
        lambda: False,
    )

    from backend.agents.rlm.primitives import check_user_messages, respond_to_user

    ctx = make_context(tmp_path)

    # --- check_user_messages: no file → empty list ---
    result = check_user_messages(ctx=ctx)
    assert result == [], f"Expected [], got {result!r}"

    # Seed a user message then read it back.
    msgs_path = ctx.project_dir / "user_messages.jsonl"
    msgs_path.write_text(
        json.dumps({"role": "user", "content": "focus on data section", "ts": "t"}) + "\n"
    )
    result = check_user_messages(ctx=ctx)
    assert len(result) == 1
    assert result[0]["content"] == "focus on data section"

    # Cursor advanced — second call returns empty.
    result2 = check_user_messages(ctx=ctx)
    assert result2 == []

    # --- respond_to_user: file write, no LLM ---
    reply = respond_to_user("Understood, pivoting to data section.", ctx=ctx)
    assert reply["sent"] is True, f"Expected sent=True, got {reply!r}"
    assert reply["outcome"] == "ok"

    lines = msgs_path.read_text().splitlines()
    # The original user message + the assistant reply.
    assert len(lines) == 2
    entry = json.loads(lines[-1])
    assert entry["role"] == "assistant"
    assert entry["content"] == "Understood, pivoting to data section."

    # Also confirm dashboard event was written.
    events_path = ctx.project_dir / "dashboard_events.jsonl"
    assert events_path.exists()
    event = json.loads(events_path.read_text().strip())
    assert event["event"] == "user_message_response"
