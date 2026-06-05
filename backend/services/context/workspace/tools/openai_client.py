"""OpenAI-backed LlmClient for RLM queries.

Promoted from tools/test-rlm-on-paper.py for production use.
Pins temperature=0 for deterministic recursion replay.
"""

from __future__ import annotations

from backend.services.context.workspace.tools._retry import with_429_backoff


def _zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def _usage_from_response(usage: object) -> dict[str, int]:
    """Extract token counts from a Chat Completions ``usage`` object.

    Robust to missing fields / providers that omit cache or reasoning details
    (e.g. vLLM-served Qwen). prompt_tokens→input, completion_tokens→output,
    prompt_tokens_details.cached_tokens→cache_read.
    """
    if usage is None:
        return _zero_usage()

    def _int(name: str, src: object = usage) -> int:
        return int(getattr(src, name, 0) or 0)

    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = _int("cached_tokens", details)
    reasoning = 0
    cdetails = getattr(usage, "completion_tokens_details", None)
    if cdetails is not None:
        reasoning = _int("reasoning_tokens", cdetails)
    return {
        "input_tokens": _int("prompt_tokens"),
        "output_tokens": _int("completion_tokens"),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": reasoning,
    }


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
        # max_retries=6: the SDK natively retries 429s with exponential
        # backoff and honours the Retry-After header, handling Featherless
        # concurrency caps without any hand-rolled retry logic.
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=6
        )
        self._model = model
        self._max_tokens = max_tokens
        # Per-call token usage, mirrored from the API ``usage`` object so the cost
        # ledger (binding._ledger reads ``ctx.llm_client._last_usage``) records
        # accelerator / cheap-call spend instead of zeros. Mirrors ClaudeLlmClient.
        self._last_usage: dict[str, int] = _zero_usage()

    @with_429_backoff
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
        self._last_usage = _usage_from_response(getattr(resp, "usage", None))
        return resp.choices[0].message.content or ""


__all__ = ["OpenAILlmClient", "_usage_from_response", "_zero_usage"]
