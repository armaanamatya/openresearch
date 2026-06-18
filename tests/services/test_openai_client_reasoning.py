"""Tests for reasoning-model-aware param selection in OpenAILlmClient.

Covers _is_reasoning_model() detector and _token_temp_kwargs() routing so
gpt-chat-latest / o1 / gpt-5 get max_completion_tokens (no temperature) and
chat models (gpt-4o, grok-*) keep the current max_tokens + temperature kwargs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services.context.workspace.tools.openai_client import (
    OpenAILlmClient,
    _is_reasoning_model,
)


# ---------------------------------------------------------------------------
# _is_reasoning_model() unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-chat-latest", True),
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("o1-preview", True),
        ("o3-mini", True),
        ("o4-mini", True),
        ("reasoning-v1", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("grok-4.3", False),
        ("qwen3-coder", False),
        ("", False),
        (None, False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert _is_reasoning_model(model) is expected


# ---------------------------------------------------------------------------
# Helpers — build a fake OpenAI client whose create() records its kwargs
# ---------------------------------------------------------------------------


def _make_fake_openai(recorded: list):
    """Return a fake ``openai`` module whose OpenAI() gives a mock client.

    ``recorded`` is mutated in-place with the kwargs of every create() call.
    """
    msg = MagicMock()
    msg.content = "x"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None

    def _create(**kwargs):
        recorded.append(kwargs)
        return resp

    completions = MagicMock()
    completions.create = _create
    chat = MagicMock()
    chat.completions = completions
    client_instance = MagicMock()
    client_instance.chat = chat

    fake_openai_cls = MagicMock(return_value=client_instance)
    fake_module = MagicMock()
    fake_module.OpenAI = fake_openai_cls
    return fake_module


# ---------------------------------------------------------------------------
# Reasoning client — complete()
# ---------------------------------------------------------------------------


def test_reasoning_client_complete_uses_max_completion_tokens():
    recorded = []
    fake = _make_fake_openai(recorded)
    with patch.dict("sys.modules", {"openai": fake}):
        client = OpenAILlmClient(model="gpt-chat-latest", api_key="test")
        out = client.complete(system="s", user="u")

    assert out == "x"
    assert len(recorded) == 1
    kwargs = recorded[0]
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


# ---------------------------------------------------------------------------
# Chat client — complete()
# ---------------------------------------------------------------------------


def test_chat_client_complete_uses_max_tokens_and_temperature():
    recorded = []
    fake = _make_fake_openai(recorded)
    with patch.dict("sys.modules", {"openai": fake}):
        client = OpenAILlmClient(model="gpt-4o-mini", api_key="test")
        out = client.complete(system="s", user="u")

    assert out == "x"
    assert len(recorded) == 1
    kwargs = recorded[0]
    assert "max_tokens" in kwargs
    assert "temperature" in kwargs
    assert "max_completion_tokens" not in kwargs


# ---------------------------------------------------------------------------
# Reasoning client — complete_samples()
# ---------------------------------------------------------------------------


def test_reasoning_client_complete_samples_no_temperature():
    recorded = []
    fake = _make_fake_openai(recorded)

    # complete_samples with n=2 on a reasoning model
    msg = MagicMock()
    msg.content = "x"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice, choice]
    resp.usage = None

    def _create_n(**kwargs):
        recorded.append(kwargs)
        return resp

    fake._make_fake_openai = None  # just a marker
    # Override the completions.create to return 2 choices
    with patch.dict("sys.modules", {"openai": fake}):
        client = OpenAILlmClient(model="gpt-chat-latest", api_key="test")
        # Patch directly on the constructed _client
        client._client.chat.completions.create = lambda **kw: (
            recorded.append(kw) or resp
        )
        results = client.complete_samples(system="s", user="u", n=2)

    assert len(results) == 2
    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("n") == 2
    assert "max_completion_tokens" in kwargs
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
