"""Typed provider-independent runtime failures.

The runtime adapters and the resilience engine use these types as the
coordination contract. Provider SDKs remain free to raise their native
exceptions; classification into this vocabulary happens at the boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from backend.agents.runtime.base import (
    AgentLimitExceeded,
    ProviderName,
    RuntimeGuardViolation,
)


FailureProvider = ProviderName | None


class RuntimeFailure(Exception):
    """Base class for every runtime failure visible to recovery policy."""

    recoverable = True

    def __init__(
        self,
        message: str = "",
        *,
        provider: FailureProvider,
        agent_id: str,
        elapsed_seconds: float = 0.0,
        partial_output: str = "",
        cause: BaseException | None = None,
    ) -> None:
        self.provider = provider
        self.agent_id = agent_id
        self.elapsed_seconds = elapsed_seconds
        self.partial_output = partial_output
        self.cause = cause
        Exception.__init__(self, message or self.__class__.__name__)


class TransientError(RuntimeFailure):
    """Recoverable network, timeout, or provider 5xx failure."""


class RateLimited(RuntimeFailure):
    """Recoverable short-horizon rate limit."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: FailureProvider,
        agent_id: str,
        elapsed_seconds: float = 0.0,
        partial_output: str = "",
        retry_after_seconds: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            message,
            provider=provider,
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=cause,
        )


class QuotaExhausted(RuntimeFailure):
    """Provider quota/subscription budget is exhausted for this window."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: FailureProvider,
        agent_id: str,
        elapsed_seconds: float = 0.0,
        partial_output: str = "",
        resets_at: datetime | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.resets_at = resets_at
        super().__init__(
            message,
            provider=provider,
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=cause,
        )


class TurnBudgetExhausted(RuntimeFailure, AgentLimitExceeded):
    """Agent turn cap fired."""

    _limit_kind: LimitKind = "turns"

    def __init__(
        self,
        message: str = "",
        *,
        provider: FailureProvider,
        agent_id: str,
        elapsed_seconds: float = 0.0,
        partial_output: str = "",
        limit_value: int,
        cause: BaseException | None = None,
    ) -> None:
        self.limit_value = int(limit_value)
        self.kind = self._limit_kind
        super().__init__(
            message,
            provider=provider,
            agent_id=agent_id,
            elapsed_seconds=elapsed_seconds,
            partial_output=partial_output,
            cause=cause,
        )


class ToolBudgetExhausted(TurnBudgetExhausted):
    """Agent tool-call cap fired."""

    _limit_kind: LimitKind = "tool_calls"


class WallClockExceeded(TurnBudgetExhausted):
    """Agent wall-clock cap fired."""

    _limit_kind: LimitKind = "wall_clock"


class FatalRuntimeFailure(RuntimeFailure):
    recoverable = False


class AuthenticationError(FatalRuntimeFailure):
    """Credentials are missing, invalid, expired, or unauthorized."""


class GuardViolation(FatalRuntimeFailure):
    """Runtime guard blocked the attempted action."""


class SchemaValidationError(FatalRuntimeFailure):
    """Provider output cannot be made to match required schema."""


class BudgetExhausted(FatalRuntimeFailure):
    """Run-level cost/wall-clock/invocation budget blocks another attempt."""


def from_agent_limit(
    exc: AgentLimitExceeded,
    *,
    provider: FailureProvider,
) -> TurnBudgetExhausted:
    """Convert legacy ``AgentLimitExceeded`` into the new failure taxonomy."""

    common = {
        "provider": provider,
        "agent_id": exc.agent_id,
        "elapsed_seconds": exc.elapsed_seconds,
        "partial_output": exc.partial_output,
        "limit_value": exc.limit_value,
        "cause": exc,
    }
    if exc.kind == "tool_calls":
        return ToolBudgetExhausted(str(exc), **common)
    if exc.kind == "wall_clock":
        return WallClockExceeded(str(exc), **common)
    return TurnBudgetExhausted(str(exc), **common)


def from_guard_violation(
    exc: RuntimeGuardViolation,
    *,
    provider: FailureProvider,
    agent_id: str,
    elapsed_seconds: float,
    partial_output: str = "",
) -> GuardViolation:
    return GuardViolation(
        str(exc),
        provider=provider,
        agent_id=agent_id,
        elapsed_seconds=elapsed_seconds,
        partial_output=partial_output,
        cause=exc,
    )


LimitKind = Literal["turns", "tool_calls", "wall_clock"]


__all__ = [
    "AuthenticationError",
    "BudgetExhausted",
    "FailureProvider",
    "FatalRuntimeFailure",
    "GuardViolation",
    "LimitKind",
    "QuotaExhausted",
    "RateLimited",
    "RuntimeFailure",
    "SchemaValidationError",
    "ToolBudgetExhausted",
    "TransientError",
    "TurnBudgetExhausted",
    "WallClockExceeded",
    "from_agent_limit",
    "from_guard_violation",
]
