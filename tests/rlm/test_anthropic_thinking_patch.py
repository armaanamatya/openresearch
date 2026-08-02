"""Regression: the rlm AnthropicClient must tolerate extended-THINKING responses.

Claude with extended thinking (e.g. Foundry ``claude-sonnet-5``) returns a ThinkingBlock
as ``content[0]``; the stock client's ``content[0].text`` crashed the root at iteration 0
(GCP smoke 2026-08-01: ``AttributeError: 'ThinkingBlock' object has no attribute 'text'``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.rlm._anthropic_thinking_patch import (
    apply_anthropic_thinking_safe_patch,
    extract_text,
)


def _thinking_then_text_response():
    thinking = SimpleNamespace(type="thinking", thinking="let me reason...")
    text = SimpleNamespace(type="text", text="the real answer")
    return SimpleNamespace(
        content=[thinking, text],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


class TestExtractText:
    def test_skips_thinking_returns_text(self):
        assert extract_text(_thinking_then_text_response()) == "the real answer"

    def test_plain_text_block_unchanged(self):
        r = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
        assert extract_text(r) == "hi"

    def test_only_thinking_returns_empty_not_crash(self):
        r = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="...")])
        assert extract_text(r) == ""

    def test_multiple_text_blocks_joined(self):
        r = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="a"),
            SimpleNamespace(type="text", text="b"),
        ])
        assert extract_text(r) == "ab"

    def test_plain_string_content(self):
        assert extract_text(SimpleNamespace(content="raw")) == "raw"


def test_patched_completion_survives_thinking_block():
    """The actual bug: AnthropicClient.completion crashed on content[0].text."""
    pytest.importorskip("rlm.clients.anthropic")
    from rlm.clients.anthropic import AnthropicClient

    apply_anthropic_thinking_safe_patch()
    client = AnthropicClient(api_key="test-key", model_name="claude-sonnet-5")
    # stub the SDK client so no network call happens; return a thinking-first response
    client.client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: _thinking_then_text_response())
    )
    assert client.completion("hi") == "the real answer"


class TestDisableThinking:
    """Foundry claude-sonnet-5/opus-4-8 default to extended thinking, which eats the
    max_tokens budget and truncates structured output (rubric JSON, root code). Disable
    it on those models; leave every other model byte-identical."""

    def test_disable_for_thinking_default_models(self):
        from backend.agents.rlm._anthropic_thinking_patch import disable_thinking_for
        assert disable_thinking_for("claude-sonnet-5") is True
        assert disable_thinking_for("claude-opus-4-8") is True

    def test_not_disabled_for_other_models(self):
        from backend.agents.rlm._anthropic_thinking_patch import disable_thinking_for
        assert disable_thinking_for("claude-sonnet-4-6") is False
        assert disable_thinking_for("gpt-5") is False
        assert disable_thinking_for(None) is False

    def test_root_completion_passes_thinking_disabled_for_sonnet5(self):
        pytest.importorskip("rlm.clients.anthropic")
        from rlm.clients.anthropic import AnthropicClient

        apply_anthropic_thinking_safe_patch()
        captured = {}
        client = AnthropicClient(api_key="k", model_name="claude-sonnet-5")

        def _create(**kw):
            captured.update(kw)
            return _thinking_then_text_response()

        client.client = SimpleNamespace(messages=SimpleNamespace(create=_create))
        client.completion("hi")
        assert captured.get("thinking") == {"type": "disabled"}

    def test_grader_client_disables_thinking_for_sonnet5(self):
        pytest.importorskip("anthropic")
        from backend.services.context.workspace.tools.anthropic_messages_client import (
            AnthropicMessagesClient,
        )

        captured = {}
        c = AnthropicMessagesClient(model="claude-sonnet-5", api_key="k")
        c._client = SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kw: captured.update(kw) or SimpleNamespace(
                content=[SimpleNamespace(type="text", text="{}")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1))))
        c.complete(system="s", user="u")
        assert captured.get("thinking") == {"type": "disabled"}

    def test_grader_client_unchanged_for_default_model(self):
        pytest.importorskip("anthropic")
        from backend.services.context.workspace.tools.anthropic_messages_client import (
            AnthropicMessagesClient,
        )

        captured = {}
        c = AnthropicMessagesClient(model="claude-sonnet-4-6", api_key="k")
        c._client = SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kw: captured.update(kw) or SimpleNamespace(
                content=[SimpleNamespace(type="text", text="{}")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1))))
        c.complete(system="s", user="u")
        assert "thinking" not in captured
