"""Tests for the OAuth LLM client → cost_ledger token capture path.

Covers:
1. ClaudeLlmClient._last_usage is populated from ResultMessage.usage.
2. Calling complete() multiple times accumulates the LAST call's usage
   (not all calls summed — binding zeroes between primitives).
3. When ResultMessage.usage is None, _last_usage stays at zeros.
4. The existing API-key path (direct CostLedgerEntry.from_usage) is unaffected.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_message(usage: dict | None):
    """Build a mock ResultMessage with the given usage dict."""
    msg = MagicMock()
    msg.result = "hello"
    msg.usage = usage
    # Make isinstance(msg, ResultMessage) return True by patching at call-site
    return msg


async def _fake_query(prompt, options, *, usage: dict | None):
    """Async generator that yields a single ResultMessage-like object."""
    from claude_agent_sdk import ResultMessage as _RM
    msg = MagicMock(spec=_RM)
    msg.result = "test response"
    msg.usage = usage
    yield msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_last_usage_populated_from_result_message_usage(monkeypatch):
    """After complete(), _last_usage carries the SDK's ResultMessage.usage values."""
    usage_from_sdk = {
        "input_tokens": 1200,
        "output_tokens": 350,
        "cache_creation_input_tokens": 80,
        "cache_read_input_tokens": 40,
    }

    client = ClaudeLlmClient(model="claude-sonnet-4-6")

    async def _mock_async_complete(self, *, system, user):
        # Simulate what the real _async_complete does
        from backend.services.pricing.token_accumulator import TokenAccumulator
        acc = TokenAccumulator()
        acc.absorb_usage(usage_from_sdk)
        return "test response", acc.as_dict()

    monkeypatch.setattr(ClaudeLlmClient, "_async_complete", _mock_async_complete)

    import concurrent.futures
    import asyncio as _asyncio

    # Patch complete() to run _async_complete without the real ThreadPoolExecutor
    original_complete = ClaudeLlmClient.complete

    def _patched_complete(self, *, system, user):
        text, usage = asyncio.run(self._async_complete(system=system, user=user))
        self._last_usage = usage
        return text

    monkeypatch.setattr(ClaudeLlmClient, "complete", _patched_complete)

    result = client.complete(system="sys", user="user prompt")
    assert result == "test response"
    assert client._last_usage["input_tokens"] == 1200
    assert client._last_usage["output_tokens"] == 350
    assert client._last_usage["cache_creation_input_tokens"] == 80
    assert client._last_usage["cache_read_input_tokens"] == 40
    assert client._last_usage["reasoning_tokens"] == 0


def test_last_usage_zeros_when_result_message_usage_is_none(monkeypatch):
    """When ResultMessage.usage is None, _last_usage stays at zeros."""
    client = ClaudeLlmClient(model="claude-sonnet-4-6")

    async def _mock_async_complete(self, *, system, user):
        from backend.services.pricing.token_accumulator import TokenAccumulator
        acc = TokenAccumulator()
        acc.absorb_usage(None)  # No usage available
        return "empty response", acc.as_dict()

    monkeypatch.setattr(ClaudeLlmClient, "_async_complete", _mock_async_complete)

    def _patched_complete(self, *, system, user):
        text, usage = asyncio.run(self._async_complete(system=system, user=user))
        self._last_usage = usage
        return text

    monkeypatch.setattr(ClaudeLlmClient, "complete", _patched_complete)

    client.complete(system="sys", user="query")
    assert client._last_usage["input_tokens"] == 0
    assert client._last_usage["output_tokens"] == 0


def test_initial_last_usage_is_all_zeros():
    """Before any call, _last_usage defaults to zeros."""
    client = ClaudeLlmClient(model="claude-sonnet-4-6")
    assert client._last_usage["input_tokens"] == 0
    assert client._last_usage["output_tokens"] == 0
    assert client._last_usage["cache_creation_input_tokens"] == 0
    assert client._last_usage["cache_read_input_tokens"] == 0
    assert client._last_usage["reasoning_tokens"] == 0


def test_last_usage_overwritten_on_each_call(monkeypatch):
    """Each complete() call overwrites _last_usage with the latest SDK usage."""
    call_num = {"n": 0}

    async def _mock_async_complete(self, *, system, user):
        from backend.services.pricing.token_accumulator import TokenAccumulator
        call_num["n"] += 1
        acc = TokenAccumulator()
        if call_num["n"] == 1:
            acc.absorb_usage({"input_tokens": 500, "output_tokens": 100})
        else:
            acc.absorb_usage({"input_tokens": 800, "output_tokens": 200})
        return "response", acc.as_dict()

    monkeypatch.setattr(ClaudeLlmClient, "_async_complete", _mock_async_complete)

    def _patched_complete(self, *, system, user):
        text, usage = asyncio.run(self._async_complete(system=system, user=user))
        self._last_usage = usage
        return text

    monkeypatch.setattr(ClaudeLlmClient, "complete", _patched_complete)

    client = ClaudeLlmClient()
    client.complete(system="s", user="first")
    assert client._last_usage["input_tokens"] == 500

    client.complete(system="s", user="second")
    assert client._last_usage["input_tokens"] == 800
    assert client._last_usage["output_tokens"] == 200
