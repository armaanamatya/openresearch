from unittest.mock import patch
from backend.services.context.workspace.tools.anthropic_messages_client import (
    AnthropicMessagesClient,
)


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
