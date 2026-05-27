"""Tests for Phase 2 reflection-LM factories.

The api-key path requires ``gepa`` installed (provides ``gepa.lm.LM``).
The claude-oauth path is mocked — we test the wrapping logic, not the
live SDK call (the live call is exercised in the e2e Docker smoke).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_make_api_key_reflection_lm_returns_callable() -> None:
    """Returns a (prompt, **kwargs) -> str callable matching gepa's LanguageModel protocol."""
    from backend.agents.optimization.reflection_lms import make_api_key_reflection_lm

    fn = make_api_key_reflection_lm("openai/gpt-4.1")
    assert callable(fn)
    # protocol: __call__(prompt, **kwargs) -> str
    import inspect

    sig = inspect.signature(fn)
    # accepts at least one positional + **kwargs
    assert len(sig.parameters) >= 1


def test_make_api_key_reflection_lm_tracks_cost(tmp_path) -> None:
    """When a tracker is passed, each call is recorded with estimated cost.

    Mocks the litellm module entirely — keeps the test runnable in environments
    without litellm installed (e.g. our .venv-gepa).
    """
    from backend.agents.optimization.cost_tracker import LMCostTracker
    from backend.agents.optimization.reflection_lms import make_api_key_reflection_lm

    tracker = LMCostTracker(run_dir=tmp_path / "r")

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="x" * 800))]
    fake_litellm = MagicMock()
    fake_litellm.completion.return_value = fake_resp

    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        fn = make_api_key_reflection_lm("openai/gpt-4.1", tracker=tracker)
        out = fn("test prompt")
        assert "x" in out
        fake_litellm.completion.assert_called_once()

    snap = tracker.snapshot()
    assert snap.call_count == 1
    # 800 chars / 4 = 200 tokens out at $0.012/1K = 0.0024; prompt is small
    assert snap.total_cost_usd > 0
    assert snap.per_model["openai/gpt-4.1"]["calls"] == 1


def test_make_claude_oauth_reflection_lm_returns_callable() -> None:
    """The OAuth factory returns a (messages, **kwargs) -> str callable."""
    fake_client = MagicMock()
    fake_client.completion.return_value = "reflection text from oauth"

    with patch(
        "backend.agents.rlm.claude_oauth_client.ClaudeOauthClient",
        return_value=fake_client,
    ):
        from backend.agents.optimization.reflection_lms import (
            make_claude_oauth_reflection_lm,
        )

        fn = make_claude_oauth_reflection_lm(model_name="claude-sonnet-4-6")
        assert callable(fn)
        result = fn([{"role": "user", "content": "diagnose this trace"}])
        assert result == "reflection text from oauth"
        fake_client.completion.assert_called_once()


def test_claude_oauth_reflection_lm_swallows_exceptions() -> None:
    """A failing completion returns '' instead of crashing the optimize run."""
    fake_client = MagicMock()
    fake_client.completion.side_effect = RuntimeError("SDK timed out")

    with patch(
        "backend.agents.rlm.claude_oauth_client.ClaudeOauthClient",
        return_value=fake_client,
    ):
        from backend.agents.optimization.reflection_lms import (
            make_claude_oauth_reflection_lm,
        )

        fn = make_claude_oauth_reflection_lm()
        result = fn("ignored")
        assert result == ""  # graceful degradation


def test_claude_oauth_reflection_lm_passes_model_kwarg() -> None:
    """``model`` kwarg from gepa flows through to ClaudeOauthClient.completion."""
    fake_client = MagicMock()
    fake_client.completion.return_value = "out"

    with patch(
        "backend.agents.rlm.claude_oauth_client.ClaudeOauthClient",
        return_value=fake_client,
    ):
        from backend.agents.optimization.reflection_lms import (
            make_claude_oauth_reflection_lm,
        )

        fn = make_claude_oauth_reflection_lm()
        fn([{"role": "user", "content": "x"}], model="claude-haiku-4-5-20251001")
        kwargs = fake_client.completion.call_args.kwargs
        assert kwargs.get("model") == "claude-haiku-4-5-20251001"
