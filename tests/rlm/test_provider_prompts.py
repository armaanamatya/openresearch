"""Tests for the per-provider reliability tail (OPENRESEARCH_PROVIDER_PROMPTS).

Default-OFF flag: build_system_prompt() appends a short model-family-specific
reliability tail (keyed off root_model.key) right after the MODEL-SPECIFIC
ADDENDUM section. Off => byte-identical (no tail; the single
{custom_tools_section} brace-escape placeholder invariant still holds).
"""

from __future__ import annotations

from backend.agents.rlm.models import resolve_root_model
from backend.agents.rlm.system_prompt import (
    _provider_prompt_tail,
    build_system_prompt,
)

_CONTEXT_METADATA = {"paper_text": {"type": "str", "length": 1000}}

_SONNET_PHRASE = "STOPPING TOO EARLY"
_REASONING_CHAT_PHRASE = "CHURN"


def _patch_claude_oauth_credentials(monkeypatch) -> None:
    """claude-oauth resolution needs has_provider_credentials() -> True."""
    monkeypatch.setattr(
        "backend.agents.runtime.factory.has_provider_credentials",
        lambda provider=None: True,
    )


# ---------------------------------------------------------------------------
# Flag OFF (unset) — byte-identical: no tail, brace-escape invariant intact.
# ---------------------------------------------------------------------------


def test_flag_off_claude_oauth_has_no_tail(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_PROVIDER_PROMPTS", raising=False)
    _patch_claude_oauth_credentials(monkeypatch)

    root_model = resolve_root_model("claude-oauth")
    prompt = build_system_prompt(context_metadata=_CONTEXT_METADATA, root_model=root_model)

    assert _SONNET_PHRASE not in prompt
    assert prompt.count("{custom_tools_section}") == 1


# ---------------------------------------------------------------------------
# Flag ON — Sonnet family (claude-oauth, claude, sonnet-foundry).
# ---------------------------------------------------------------------------


def test_flag_on_claude_oauth_gets_sonnet_tail(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_PROVIDER_PROMPTS", "1")
    _patch_claude_oauth_credentials(monkeypatch)

    root_model = resolve_root_model("claude-oauth")
    prompt = build_system_prompt(context_metadata=_CONTEXT_METADATA, root_model=root_model)

    assert _SONNET_PHRASE in prompt
    assert _REASONING_CHAT_PHRASE not in prompt
    assert prompt.count("{custom_tools_section}") == 1


def test_flag_on_sonnet_foundry_gets_sonnet_tail(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_PROVIDER_PROMPTS", "1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "dummy")

    root_model = resolve_root_model("sonnet-foundry")
    prompt = build_system_prompt(context_metadata=_CONTEXT_METADATA, root_model=root_model)

    assert _SONNET_PHRASE in prompt
    assert _REASONING_CHAT_PHRASE not in prompt
    assert prompt.count("{custom_tools_section}") == 1


def test_flag_on_claude_api_gets_sonnet_tail(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_PROVIDER_PROMPTS", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    root_model = resolve_root_model("claude")
    prompt = build_system_prompt(context_metadata=_CONTEXT_METADATA, root_model=root_model)

    assert _SONNET_PHRASE in prompt
    assert _REASONING_CHAT_PHRASE not in prompt
    assert prompt.count("{custom_tools_section}") == 1


# ---------------------------------------------------------------------------
# Flag ON — reasoning-chat family (azure-foundry).
# ---------------------------------------------------------------------------


def test_flag_on_azure_foundry_gets_reasoning_chat_tail(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_PROVIDER_PROMPTS", "1")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", "grok-4.3")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "dummy")

    root_model = resolve_root_model("azure-foundry")
    prompt = build_system_prompt(context_metadata=_CONTEXT_METADATA, root_model=root_model)

    assert _REASONING_CHAT_PHRASE in prompt
    assert _SONNET_PHRASE not in prompt
    assert prompt.count("{custom_tools_section}") == 1


# ---------------------------------------------------------------------------
# Flag ON — everything else (gpt-5) gets no tail.
# ---------------------------------------------------------------------------


def test_flag_on_gpt5_gets_no_tail(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_PROVIDER_PROMPTS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    root_model = resolve_root_model("gpt-5")
    prompt = build_system_prompt(context_metadata=_CONTEXT_METADATA, root_model=root_model)

    assert _SONNET_PHRASE not in prompt
    assert _REASONING_CHAT_PHRASE not in prompt
    assert prompt.count("{custom_tools_section}") == 1


# ---------------------------------------------------------------------------
# Pure helper, directly.
# ---------------------------------------------------------------------------


def test_provider_prompt_tail_helper_claude_oauth(monkeypatch):
    _patch_claude_oauth_credentials(monkeypatch)
    root_model = resolve_root_model("claude-oauth")

    from backend.agents.rlm.system_prompt import _SONNET_RELIABILITY_TAIL

    assert _provider_prompt_tail(root_model) == _SONNET_RELIABILITY_TAIL


def test_provider_prompt_tail_helper_gpt5(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    root_model = resolve_root_model("gpt-5")

    assert _provider_prompt_tail(root_model) == ""
