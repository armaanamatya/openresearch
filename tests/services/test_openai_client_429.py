"""Tests for the 429/503 retry/backoff decorator (handoff P1-I6 / T14)."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Unit tests for the decorator itself
# ---------------------------------------------------------------------------

def test_with_429_backoff_retries_on_429(monkeypatch):
    """Decorator retries on 429-like exceptions and eventually succeeds."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    from backend.services.context.workspace.tools._retry import with_429_backoff

    calls: list[int] = []

    @with_429_backoff
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert len(calls) == 3


def test_with_429_backoff_does_not_retry_on_auth_error(monkeypatch):
    """Non-retryable errors (401) are raised immediately without backoff."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    from backend.services.context.workspace.tools._retry import with_429_backoff

    calls: list[int] = []

    @with_429_backoff
    def auth_fail() -> str:
        calls.append(1)
        raise RuntimeError("HTTP 401 Unauthorized — invalid API key")

    with pytest.raises(RuntimeError, match="401"):
        auth_fail()
    assert len(calls) == 1


def test_with_429_backoff_raises_after_max_retries(monkeypatch):
    """After _MAX_RETRIES exhausted, the final exception is re-raised."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    from backend.services.context.workspace.tools._retry import (
        _MAX_RETRIES,
        with_429_backoff,
    )

    calls: list[int] = []

    @with_429_backoff
    def always_429() -> str:
        calls.append(1)
        raise RuntimeError("rate limit exceeded — 429")

    with pytest.raises(RuntimeError, match="429"):
        always_429()
    assert len(calls) == _MAX_RETRIES + 1


def test_with_429_backoff_retries_on_503(monkeypatch):
    """503 service unavailable also triggers backoff."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    from backend.services.context.workspace.tools._retry import with_429_backoff

    calls: list[int] = []

    @with_429_backoff
    def flaky_503() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("503 Service Unavailable")
        return "recovered"

    result = flaky_503()
    assert result == "recovered"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Integration: decorator is applied to OpenAILlmClient.complete
# ---------------------------------------------------------------------------

def test_openai_client_complete_has_retry_decorator():
    """OpenAILlmClient.complete is decorated with with_429_backoff."""
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    # functools.wraps sets __wrapped__ on the wrapper pointing to the original.
    assert hasattr(OpenAILlmClient.complete, "__wrapped__"), (
        "OpenAILlmClient.complete must be decorated with @with_429_backoff "
        "(no __wrapped__ attribute found — decorator not applied)"
    )


def test_openai_client_sdk_max_retries():
    """OpenAILlmClient passes max_retries=6 to the OpenAI SDK (belt+suspenders)."""
    # The OpenAI SDK natively retries 429/503 with exponential backoff and
    # honours Retry-After. max_retries=6 is a first-line defence; the
    # with_429_backoff decorator is a second-line defence if the SDK
    # exhausts its own retries.
    import inspect
    import textwrap

    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    src = inspect.getsource(OpenAILlmClient.__init__)
    assert "max_retries=6" in src, (
        "OpenAILlmClient.__init__ must pass max_retries=6 to the OpenAI SDK"
    )


# ---------------------------------------------------------------------------
# Integration: decorator is applied to ClaudeLlmClient.complete
# ---------------------------------------------------------------------------

def test_claude_llm_client_complete_has_retry_decorator():
    """ClaudeLlmClient.complete is decorated with with_429_backoff."""
    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

    assert hasattr(ClaudeLlmClient.complete, "__wrapped__"), (
        "ClaudeLlmClient.complete must be decorated with @with_429_backoff "
        "(no __wrapped__ attribute found — decorator not applied)"
    )


def test_openai_client_retries_on_429_via_api_call(monkeypatch):
    """Symptom: concurrent RLM run 429s instead of waiting for a slot (P1-I6).

    Patch the inner SDK call (chat.completions.create) to 429 twice then
    succeed. The decorator should absorb the 429s and return the final result.

    Note: OpenAILlmClient also has max_retries=6 in the SDK which handles
    429 natively. This test exercises the outer decorator as an extra layer.
    """
    monkeypatch.setattr("time.sleep", lambda s: None)

    from unittest.mock import MagicMock, patch

    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    calls: list[int] = []

    def fake_create(**kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("HTTP 429 Too Many Requests")
        msg = MagicMock()
        msg.content = "ok"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create
        mock_openai_cls.return_value = mock_client

        client = OpenAILlmClient()
        result = client.complete(system="sys", user="usr")

    assert result == "ok"
    assert len(calls) == 3
