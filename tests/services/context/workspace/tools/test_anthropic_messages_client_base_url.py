from unittest.mock import MagicMock, patch

import pytest

from backend.services.context.workspace.tools.anthropic_messages_client import (
    AnthropicMessagesClient,
)


def _fake_resp(text: str = "ok"):
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = None
    return resp


def test_base_url_is_forwarded_to_sdk():
    with patch("anthropic.Anthropic") as mock_anthropic:
        AnthropicMessagesClient(model="claude-sonnet-5", api_key="k",
                                base_url="https://x/anthropic")
    _, kwargs = mock_anthropic.call_args
    assert kwargs["base_url"] == "https://x/anthropic"
    assert kwargs["api_key"] == "k"


def test_base_url_none_is_omitted_byte_identical():
    with patch("anthropic.Anthropic") as mock_anthropic:
        AnthropicMessagesClient(model="claude-sonnet-5")
    _, kwargs = mock_anthropic.call_args
    # None must NOT be passed (or must be None) so the SDK resolves its default.
    assert kwargs.get("base_url", None) is None


# --------------------------------------------------------------------------- #
# temperature-deprecation resilience (Claude Opus 4.8 / Sonnet 5 reject the
# `temperature` param — "temperature is deprecated for this model"). Regression
# for the 2026-07-07 fix: probe-then-drop + latch, byte-identical otherwise.
# --------------------------------------------------------------------------- #


def test_temperature_passed_for_accepting_model_byte_identical():
    """A model that accepts `temperature` still gets `temperature=0` (no change)."""
    with patch("anthropic.Anthropic") as mock_anthropic:
        inst = mock_anthropic.return_value
        inst.messages.create.return_value = _fake_resp("hi")
        client = AnthropicMessagesClient(model="claude-sonnet-4-6", api_key="k")
        out = client.complete(system="s", user="u")
    assert out == "hi"
    _, kwargs = inst.messages.create.call_args
    assert kwargs.get("temperature") == 0
    assert client._omit_temperature is False


def test_temperature_dropped_and_latched_when_deprecated():
    """On the deprecation 400, retry without `temperature` and latch it off."""
    with patch("anthropic.Anthropic") as mock_anthropic:
        inst = mock_anthropic.return_value
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError(
                    "Error code: 400 - {'message': '`temperature` is deprecated for this model.'}"
                )
            return _fake_resp("ok")

        inst.messages.create.side_effect = _create
        client = AnthropicMessagesClient(model="claude-opus-4-8", api_key="k")
        out1 = client.complete(system="s", user="u")
        out2 = client.complete(system="s", user="u")

    assert out1 == "ok" and out2 == "ok"
    # complete #1: probe-with-temp (raises) → retry-without = 2 calls;
    # complete #2: latched, skips the probe = 1 call. Total 3.
    assert len(calls) == 3
    assert "temperature" in calls[0]      # probe
    assert "temperature" not in calls[1]  # retry drops it
    assert "temperature" not in calls[2]  # latched — no probe
    assert client._omit_temperature is True


def test_non_temperature_error_is_reraised():
    """A 400 that is NOT the temperature deprecation must propagate, not be swallowed."""
    with patch("anthropic.Anthropic") as mock_anthropic:
        inst = mock_anthropic.return_value
        inst.messages.create.side_effect = ValueError("Error code: 400 - unrelated bad request")
        client = AnthropicMessagesClient(model="claude-opus-4-8", api_key="k")
        with pytest.raises(ValueError, match="unrelated bad request"):
            client.complete(system="s", user="u")
    assert client._omit_temperature is False
