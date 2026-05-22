"""OpenAI-backed LlmClient for RLM queries.

Promoted from tools/test-rlm-on-paper.py for production use.
Pins temperature=0 for deterministic recursion replay.
"""

from __future__ import annotations


class OpenAILlmClient:
    """LlmClient backed by OpenAI's Chat Completions.

    Default model is gpt-4o-mini — cheap enough for the routing and
    leaf calls that RLM generates, strong enough for focused Q&A.

    ``max_tokens`` defaults to 4096: primitive callers (``verify_against_rubric``,
    ``propose_improvements``) and the PaperBench leaf scorer return structured
    JSON that does not fit in a few hundred tokens — too small a ceiling
    truncates the response mid-object and the caller fails to parse it.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 300.0,
    ) -> None:
        from openai import OpenAI

        # Bound every request: the OpenAI SDK default is 600 s, so one hung
        # primitive call stalls the whole run for ten minutes. 300 s matches
        # rlm's own client default and still clears any real response.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""


__all__ = ["OpenAILlmClient"]
