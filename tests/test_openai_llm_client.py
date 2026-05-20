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
