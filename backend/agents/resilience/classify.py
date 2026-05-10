"""Provider exception classification.

String matching is intentionally concentrated in this module. The rest of
the system consumes typed ``RuntimeFailure`` values.
"""

from __future__ import annotations

import re
from email.utils import parsedate_to_datetime
from typing import Any

from backend.agents.resilience.failures import (
    AuthenticationError,
    GuardViolation,
    QuotaExhausted,
    RateLimited,
    RuntimeFailure,
    TransientError,
    TurnBudgetExhausted,
    from_agent_limit,
    from_guard_violation,
)
from backend.agents.runtime.base import (
    AgentLimitExceeded,
    ProviderName,
    RuntimeGuardViolation,
)


_TURN_LIMIT_RE = re.compile(r"maximum number of turns\s*\((\d+)\)", re.IGNORECASE)
_STEP_LIMIT_RE = re.compile(r"max(?:imum)?(?: number of)? steps?\s*\(?(\d+)?\)?", re.IGNORECASE)
_CLAUDE_QUOTA_PHRASES = (
    "you've hit your limit",
    "you have hit your limit",
    "claude code returned an error result: success",
)
_QUOTA_PHRASES = (
    "insufficient_quota",
    "quota exceeded",
    "current quota",
    "billing",
    "credit balance",
    "usage limit",
)
_TRANSIENT_PHRASES = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "temporary failure",
    "temporarily unavailable",
    "service unavailable",
)


def classify_failure(
    provider: ProviderName,
    exc: BaseException,
    *,
    agent_id: str,
    elapsed_seconds: float,
    partial_output: str = "",
) -> RuntimeFailure:
    if isinstance(exc, RuntimeFailure):
        return exc
    if isinstance(exc, AgentLimitExceeded):
        return from_agent_limit(exc, provider=provider)
    if isinstance(exc, RuntimeGuardViolation):
        return from_guard_violation(
            exc,
            provider=provider,
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
        )
    if provider == "anthropic":
        return classify_anthropic(
            exc,
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
        )
    return classify_openai(
        exc,
        agent_id=agent_id,
        elapsed_seconds=elapsed_seconds,
        partial_output=partial_output,
    )


def classify_anthropic(
    exc: BaseException,
    *,
    agent_id: str,
    elapsed_seconds: float,
    partial_output: str = "",
) -> RuntimeFailure:
    text = _failure_text(exc)
    status_code = _status_code(exc)
    class_name = type(exc).__name__.lower()
    turn_limit = _turn_limit(text)
    if turn_limit is not None:
        return TurnBudgetExhausted(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            limit_value=turn_limit,
            cause=exc,
        )
    if "authentication" in class_name or status_code in {401, 403}:
        return AuthenticationError(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    if any(phrase in text for phrase in _QUOTA_PHRASES):
        return QuotaExhausted(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            resets_at=_reset_time(exc),
            cause=exc,
        )
    if any(phrase in text for phrase in _CLAUDE_QUOTA_PHRASES):
        return QuotaExhausted(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            resets_at=_reset_time(exc),
            cause=exc,
        )
    if "ratelimit" in class_name or "rate_limit" in text or status_code == 429:
        if any(phrase in text for phrase in _QUOTA_PHRASES):
            return QuotaExhausted(
                str(exc),
                provider="anthropic",
                agent_id=agent_id,
                elapsed_seconds=elapsed_seconds,
                partial_output=partial_output,
                resets_at=_reset_time(exc),
                cause=exc,
            )
        return RateLimited(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            retry_after_seconds=_retry_after_seconds(exc),
            cause=exc,
        )
    if status_code is not None and status_code >= 500:
        return TransientError(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    if any(phrase in text for phrase in _TRANSIENT_PHRASES):
        return TransientError(
            str(exc),
            provider="anthropic",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    return TransientError(
        str(exc),
        provider="anthropic",
        agent_id=agent_id,
        elapsed_seconds=elapsed_seconds,
        partial_output=partial_output,
        cause=exc,
    )


def classify_openai(
    exc: BaseException,
    *,
    agent_id: str,
    elapsed_seconds: float,
    partial_output: str = "",
) -> RuntimeFailure:
    text = _failure_text(exc)
    status_code = _status_code(exc)
    class_name = type(exc).__name__.lower()
    turn_limit = _turn_limit(text)
    if turn_limit is not None or _STEP_LIMIT_RE.search(text):
        return TurnBudgetExhausted(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            limit_value=turn_limit or 0,
            cause=exc,
        )
    if "authentication" in class_name or status_code in {401, 403}:
        return AuthenticationError(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    if any(phrase in text for phrase in _QUOTA_PHRASES):
        return QuotaExhausted(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            resets_at=_reset_time(exc),
            cause=exc,
        )
    if "ratelimit" in class_name or "rate_limit" in text or status_code == 429:
        if any(phrase in text for phrase in _QUOTA_PHRASES):
            return QuotaExhausted(
                str(exc),
                provider="openai",
                agent_id=agent_id,
                elapsed_seconds=elapsed_seconds,
                partial_output=partial_output,
                resets_at=_reset_time(exc),
                cause=exc,
            )
        return RateLimited(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            retry_after_seconds=_retry_after_seconds(exc),
            cause=exc,
        )
    if "apiconnection" in class_name or "timeout" in class_name:
        return TransientError(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    if status_code is not None and status_code >= 500:
        return TransientError(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    if any(phrase in text for phrase in _TRANSIENT_PHRASES):
        return TransientError(
            str(exc),
            provider="openai",
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=exc,
        )
    return TransientError(
        str(exc),
        provider="openai",
        agent_id=agent_id,
        elapsed_seconds=elapsed_seconds,
        partial_output=partial_output,
        cause=exc,
    )


def _failure_text(exc: BaseException) -> str:
    pieces = [type(exc).__name__, str(exc)]
    for attr in ("body", "response", "message", "error"):
        value = getattr(exc, attr, None)
        if value:
            pieces.append(str(value))
    return "\n".join(pieces).lower()


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _headers(exc: BaseException) -> dict[str, Any]:
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
    if headers is None:
        return {}
    if isinstance(headers, dict):
        return headers
    try:
        return dict(headers)
    except Exception:
        return {}


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = {str(k).lower(): v for k, v in _headers(exc).items()}
    raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-seconds")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            dt = parsedate_to_datetime(str(raw))
            from datetime import datetime, timezone

            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


def _reset_time(exc: BaseException):
    headers = {str(k).lower(): v for k, v in _headers(exc).items()}
    raw = headers.get("x-ratelimit-reset") or headers.get("anthropic-ratelimit-reset")
    if raw is None:
        return None
    try:
        return parsedate_to_datetime(str(raw))
    except Exception:
        return None


def _turn_limit(text: str) -> int | None:
    match = _TURN_LIMIT_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "classify_anthropic",
    "classify_failure",
    "classify_openai",
]
