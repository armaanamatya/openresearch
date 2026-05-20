"""OpenAI-backed LlmClient for RLM queries.

Promoted from tools/test-rlm-on-paper.py for production use.
Pins temperature=0 for deterministic recursion replay.
"""

from __future__ import annotations


class OpenAILlmClient:
    """LlmClient backed by OpenAI's Chat Completions.

    Default model is gpt-4o-mini — cheap enough for the routing and
    leaf calls that RLM generates, strong enough for focused Q&A.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=600,
        )
        return resp.choices[0].message.content or ""


__all__ = ["OpenAILlmClient"]
