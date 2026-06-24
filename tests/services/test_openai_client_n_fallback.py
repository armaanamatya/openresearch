"""complete_samples must survive an OpenAI-compatible endpoint that rejects the
native n>1 (multi-completion) request.

Observed 2026-06-18: the Azure AI Foundry / grok endpoint 422s on n>1 with an
internal ``bootstrap_host`` routing error, while single completions succeed. The
grok grader (median-of-3) therefore failed every batch → all leaves 0.0. The fix
falls back to n SEQUENTIAL single-completion calls for n>1; for n<=1 a genuine
API error is re-raised rather than masked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services.context.workspace.tools.openai_client import OpenAILlmClient


def _one_choice(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def test_complete_samples_falls_back_to_sequential_on_n_gt_1_error() -> None:
    """A provider error on the native n>1 request → n sequential single calls."""
    calls: list[int] = []

    def fake_create(**kwargs):
        n = kwargs.get("n", 1)
        calls.append(n)
        if n > 1:
            raise RuntimeError(
                "422 Invalid input — bootstrap_host should be a valid string"
            )
        return _one_choice("graded")

    with patch("openai.OpenAI") as mock_cls:
        mc = MagicMock()
        mc.chat.completions.create.side_effect = fake_create
        mock_cls.return_value = mc
        client = OpenAILlmClient()
        out = client.complete_samples(system="grade", user="leaves", n=3, temperature=0)

    assert out == ["graded", "graded", "graded"]
    # One native n=3 attempt (raises), then 3 sequential single-completion calls.
    assert calls == [3, 1, 1, 1]


def test_complete_samples_native_n_success_path_unchanged() -> None:
    """An endpoint that supports native n>1 still returns all choices in one call."""

    def fake_create(**kwargs):
        n = kwargs.get("n", 1)
        choices = []
        for i in range(n):
            m = MagicMock()
            m.content = f"s{i}"
            c = MagicMock()
            c.message = m
            choices.append(c)
        resp = MagicMock()
        resp.choices = choices
        resp.usage = None
        return resp

    with patch("openai.OpenAI") as mock_cls:
        mc = MagicMock()
        mc.chat.completions.create.side_effect = fake_create
        mock_cls.return_value = mc
        client = OpenAILlmClient()
        out = client.complete_samples(system="s", user="u", n=3, temperature=0)

    assert out == ["s0", "s1", "s2"]
    # Exactly one native multi-choice call — no sequential fallback.
    assert mc.chat.completions.create.call_count == 1


def test_complete_samples_n1_genuine_error_reraises() -> None:
    """For n<=1 a real API error must propagate, not be masked by a retry."""

    def always_raise(**kwargs):
        raise RuntimeError("boom-genuine-error")  # no 429/503 → decorator won't retry

    with patch("openai.OpenAI") as mock_cls:
        mc = MagicMock()
        mc.chat.completions.create.side_effect = always_raise
        mock_cls.return_value = mc
        client = OpenAILlmClient()
        with pytest.raises(RuntimeError, match="boom-genuine-error"):
            client.complete_samples(system="s", user="u", n=1, temperature=0)
