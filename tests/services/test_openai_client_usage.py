"""OpenAI/Azure cheap-call clients must capture API usage into ``_last_usage`` so the
cost ledger records accelerator spend (was logging 0 tokens for Qwen calls — 2026-05-29).
"""
from __future__ import annotations

from backend.services.context.workspace.tools.openai_client import (
    OpenAILlmClient,
    _usage_from_response,
    _zero_usage,
)


class _Details:
    def __init__(self, cached: int = 0, reasoning: int = 0) -> None:
        self.cached_tokens = cached
        self.reasoning_tokens = reasoning


class _Usage:
    def __init__(self, p: int, c: int, cached: int = 0, reasoning: int = 0) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.prompt_tokens_details = _Details(cached=cached)
        self.completion_tokens_details = _Details(reasoning=reasoning)


def test_usage_from_response_extracts_tokens():
    u = _usage_from_response(_Usage(120, 30, cached=50, reasoning=7))
    assert u["input_tokens"] == 120
    assert u["output_tokens"] == 30
    assert u["cache_read_input_tokens"] == 50
    assert u["reasoning_tokens"] == 7


def test_usage_from_response_none_is_zero():
    assert _usage_from_response(None) == _zero_usage()


def test_usage_from_response_tolerates_missing_details():
    class _Bare:
        prompt_tokens = 5
        completion_tokens = 2

    u = _usage_from_response(_Bare())
    assert u["input_tokens"] == 5
    assert u["output_tokens"] == 2
    assert u["cache_read_input_tokens"] == 0
    assert u["reasoning_tokens"] == 0


def test_openai_client_complete_captures_usage(monkeypatch):
    c = OpenAILlmClient(api_key="test", model="x")
    assert c._last_usage == _zero_usage()  # zero before any call

    class _Msg:
        content = "hi"

    class _Choice:
        message = _Msg()

    class _Resp:
        usage = _Usage(200, 40, cached=10)
        choices = [_Choice()]

    monkeypatch.setattr(c._client.chat.completions, "create", lambda **kw: _Resp())
    out = c.complete(system="s", user="u")
    assert out == "hi"
    assert c._last_usage["input_tokens"] == 200
    assert c._last_usage["output_tokens"] == 40
    assert c._last_usage["cache_read_input_tokens"] == 10
