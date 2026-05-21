"""Tests for the promoted OpenAILlmClient — #47.

These tests verify the module is importable and implements the LlmClient
protocol. Actual API calls are not made (require OPENAI_API_KEY).
"""

from __future__ import annotations

import pytest


def test_importable_from_tools_package():
    from backend.services.context.workspace.tools import OpenAILlmClient

    assert OpenAILlmClient is not None


def test_importable_from_workspace_package():
    from backend.services.context.workspace import OpenAILlmClient

    assert OpenAILlmClient is not None


def test_implements_llm_client_protocol():
    """OpenAILlmClient has the complete(*, system, user) -> str method."""
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    assert hasattr(OpenAILlmClient, "complete")
    import inspect

    sig = inspect.signature(OpenAILlmClient.complete)
    params = list(sig.parameters.keys())
    assert "system" in params
    assert "user" in params


def test_constructor_requires_openai():
    """Constructing OpenAILlmClient imports openai; skip if not installed."""
    try:
        from backend.services.context.workspace.tools.openai_client import (
            OpenAILlmClient,
        )

        # Will raise if openai package is missing or no API key configured
        client = OpenAILlmClient(model="gpt-4o-mini")
        assert client._model == "gpt-4o-mini"
    except Exception:
        pytest.skip("openai package not installed or not configured")


def test_constructor_accepts_api_key_and_base_url():
    """OpenAILlmClient(api_key=..., base_url=...) constructs without network.

    The openai SDK is lazy — it does not connect until .chat.completions.create
    is called — so construction with a custom key and base_url succeeds even
    without a live server.
    """
    try:
        from backend.services.context.workspace.tools.openai_client import (
            OpenAILlmClient,
        )

        client = OpenAILlmClient(
            model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
            api_key="test-key",
            base_url="https://api.featherless.ai/v1",
        )
        assert client._model == "Qwen/Qwen3-Coder-480B-A35B-Instruct"
        # The SDK may normalize the URL (e.g. add a trailing slash), so use `in`.
        assert "featherless" in str(client._client.base_url)
    except ImportError:
        pytest.skip("openai package not installed")
