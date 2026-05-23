"""Retry-on-429/503 decorator for synchronous LLM clients (T14 / P1-I6).

Featherless caps concurrent units; a second RLM run 429s. Hard-failing on
429 means concurrent runs cannot share a cap — they must back off and wait
for a slot. This decorator adds exponential backoff with jitter on 429 / 503.
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_MAX_RETRIES = 6
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0

T = TypeVar("T")


def _is_retryable(exc: Exception) -> bool:
    """True if `exc` is a 429 / 503 / rate-limit / service-unavailable error.

    We pattern-match on the exception message because the LLM clients raise
    a mix of provider-SDK exception types (anthropic.RateLimitError,
    openai.RateLimitError, HTTPError) — string match is the least fragile
    coverage across providers.
    """
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "rate limit", "503", "service unavailable"))


def with_429_backoff(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator: retry `fn` on 429/503 with exponential backoff + jitter.

    Up to _MAX_RETRIES attempts. Non-retryable exceptions are re-raised
    immediately. Auth/schema errors (401/403/400) MUST not be retried —
    `_is_retryable` only matches rate-limit / service-unavailable patterns.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — pattern-match retryable below
                if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                    raise
                wait = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** attempt))
                wait += random.uniform(0, wait * 0.3)  # 30% jitter
                logger.warning(
                    "%s: 429/503 (attempt %d/%d) — backing off %.1fs",
                    fn.__qualname__, attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = exc
        # Unreachable, but satisfies type-checker.
        raise last_exc  # type: ignore[misc]

    return wrapper


__all__ = ["with_429_backoff"]
