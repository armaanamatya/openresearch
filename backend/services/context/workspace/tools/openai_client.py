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


def _is_reasoning_model(model: str | None) -> bool:
    """True for OpenAI reasoning-class models that reject `max_tokens` and a
    non-default `temperature` (they require `max_completion_tokens` and only the
    default temperature). Covers o1/o3/o4, gpt-5*, and Azure Foundry's
    `gpt-chat-latest`. Chat models (gpt-4o, grok-*, qwen-*) return False."""
    n = (model or "").lower()
    return n.startswith(("o1", "o3", "o4", "gpt-5", "gpt-chat")) or "reasoning" in n


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

    def _token_temp_kwargs(self, temperature: float) -> dict:
        """Completion kwargs appropriate for this client's model.

        Reasoning-class models: ``max_completion_tokens`` + NO ``temperature``
        (they only accept the default). Chat models: ``max_tokens`` +
        ``temperature`` (unchanged behavior). Selected dynamically by model
        name so a ``.env`` deployment swap (grok ⇄ gpt-chat-latest) needs no
        code change.
        """
        if _is_reasoning_model(self._model):
            return {"max_completion_tokens": self._max_tokens}
        return {"max_tokens": self._max_tokens, "temperature": temperature}

    @with_429_backoff
    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._token_temp_kwargs(0),
        )
        self._last_usage = _usage_from_response(getattr(resp, "usage", None))
        return resp.choices[0].message.content or ""

    @with_429_backoff
    def complete_samples(
        self,
        *,
        system: str,
        user: str,
        n: int = 1,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Return ``n`` completions in ONE round-trip (Chat Completions ``n``).

        Optional grader-fidelity sampling path (spec 2026-06-16 §A5). Pins
        ``temperature=0`` by default (deterministic) but honours an explicit
        ``temperature``; passes ``seed`` for near-determinism on backends that
        support it. If the SDK rejects ``n``/``seed`` (older SDK, or a provider
        that doesn't accept them), falls back to ``n`` SEQUENTIAL
        single-completion calls so the caller always gets ``n`` strings.

        ``_last_usage`` mirrors the API usage: on the native ``n`` path it is
        the single multi-choice response's usage (it covers all ``n`` choices),
        and on the fallback path it is the LAST call's usage (mirrors
        ``complete``).
        """
        eff_temp = 0 if temperature is None else temperature
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                n=n,
                seed=seed,
                **self._token_temp_kwargs(eff_temp),
            )
            self._last_usage = _usage_from_response(getattr(resp, "usage", None))
            return [(c.message.content or "") for c in resp.choices]
        except TypeError:
            # SDK signature rejected n/seed (older SDK) — fall through to sequential.
            pass
        except Exception:
            # Some OpenAI-COMPATIBLE endpoints (Azure AI Foundry / grok) reject the
            # multi-completion (n>1) request itself — e.g. an Azure ML 422
            # "bootstrap_host" routing error — even though single completions
            # succeed. Fall back to n sequential single-completion calls for n>1.
            # For n<=1 the failure is a genuine API/transport error, so re-raise it
            # rather than masking it behind an identical retry.
            if n <= 1:
                raise
        # Fallback: n SEQUENTIAL single-completion calls — universally supported,
        # including on endpoints without native multi-completion (n>1).
        return [
            self._complete_once(system=system, user=user, temperature=eff_temp)
            for _ in range(n)
        ]

    def _complete_once(self, *, system: str, user: str, temperature: float) -> str:
        """One single-choice completion at an explicit temperature (fallback path)."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._token_temp_kwargs(temperature),
        )
        self._last_usage = _usage_from_response(getattr(resp, "usage", None))
        return resp.choices[0].message.content or ""


__all__ = ["OpenAILlmClient", "_is_reasoning_model", "_usage_from_response", "_zero_usage"]
