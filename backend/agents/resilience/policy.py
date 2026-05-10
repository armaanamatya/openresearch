"""Composable recovery policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from backend.agents.resilience.context import AttemptContext
from backend.agents.resilience.failures import (
    QuotaExhausted,
    RateLimited,
    RuntimeFailure,
    TransientError,
    TurnBudgetExhausted,
    WallClockExceeded,
)
from backend.agents.resilience.health import ProviderHealthMonitor
from backend.agents.runtime.base import ProviderName


@dataclass(frozen=True)
class RecoveryDecision:
    action: Literal["retry_same", "fallback_to", "salvage_partial", "fail"]
    target_provider: ProviderName | None = None
    delay_seconds: float = 0.0
    note: str = ""
    bumped_max_turns: int | None = None


class RecoveryPolicy(Protocol):
    def decide(
        self,
        *,
        failure: RuntimeFailure,
        context: AttemptContext,
    ) -> RecoveryDecision | None: ...


class BackoffOnTransient:
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, multiplier: float = 2.0) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.multiplier = multiplier

    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        if not isinstance(failure, TransientError):
            return None
        seen = _failure_count(context, TransientError)
        if seen >= self.max_retries:
            return None
        return RecoveryDecision(
            action="retry_same",
            delay_seconds=self.initial_delay * (self.multiplier ** max(0, seen - 1)),
            note="transient provider failure; retrying same provider",
        )


class BackoffOnRateLimit:
    def __init__(self, max_retries: int = 1, default_delay: float = 5.0) -> None:
        self.max_retries = max_retries
        self.default_delay = default_delay

    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        if not isinstance(failure, RateLimited):
            return None
        seen = _failure_count(context, RateLimited)
        if seen >= self.max_retries:
            return None
        return RecoveryDecision(
            action="retry_same",
            delay_seconds=failure.retry_after_seconds or self.default_delay,
            note="short rate limit; retrying same provider",
        )


class RolloverOnQuota:
    def __init__(
        self,
        chain: list[ProviderName],
        *,
        health: ProviderHealthMonitor | None = None,
    ) -> None:
        self.chain = chain
        self.health = health

    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        if not isinstance(failure, QuotaExhausted):
            return None
        current = failure.provider
        if current is None:
            return None
        try:
            start = self.chain.index(current) + 1
        except ValueError:
            start = 0
        for provider in self.chain[start:]:
            if self.health is not None and not self.health.is_healthy(provider):
                continue
            return RecoveryDecision(
                action="fallback_to",
                target_provider=provider,
                note=f"{current} quota exhausted; falling back to {provider}",
            )
        return RecoveryDecision(
            action="fail",
            note="quota exhausted and no healthy fallback provider remains",
        )


class BumpOnTurnBudget:
    def __init__(self, factor: float = 2.0, max_attempts: int = 2) -> None:
        self.factor = factor
        self.max_attempts = max_attempts

    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        if not isinstance(failure, TurnBudgetExhausted) or isinstance(failure, WallClockExceeded):
            return None
        seen = _failure_count(context, TurnBudgetExhausted)
        if seen >= self.max_attempts:
            return None
        bumped = max(failure.limit_value + 1, int(failure.limit_value * self.factor))
        return RecoveryDecision(
            action="retry_same",
            bumped_max_turns=bumped,
            note=f"turn cap hit; retrying with max_turns={bumped}",
        )


class SalvageOnTurnBudget:
    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        if isinstance(failure, TurnBudgetExhausted) and context.latest_partial_output():
            return RecoveryDecision(
                action="salvage_partial",
                note="turn/tool budget hit; partial output is available",
            )
        return None


class SalvageOnWallClock:
    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        if isinstance(failure, WallClockExceeded) and context.latest_partial_output():
            return RecoveryDecision(
                action="salvage_partial",
                note="wall-clock cap hit; partial output is available",
            )
        return None


class FailFast:
    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision:
        return RecoveryDecision(action="fail", note=failure.__class__.__name__)


class CompositePolicy:
    def __init__(self, *policies: RecoveryPolicy) -> None:
        self.policies = policies

    def iter_decisions(
        self,
        *,
        failure: RuntimeFailure,
        context: AttemptContext,
    ):
        for policy in self.policies:
            decision = policy.decide(failure=failure, context=context)
            if decision is not None:
                yield decision

    def decide(self, *, failure: RuntimeFailure, context: AttemptContext) -> RecoveryDecision | None:
        return next(self.iter_decisions(failure=failure, context=context), None)


def _failure_count(context: AttemptContext, cls: type[RuntimeFailure]) -> int:
    cls_name = cls.__name__
    return sum(1 for attempt in context.attempts if attempt.failure_kind == cls_name)


__all__ = [
    "BackoffOnRateLimit",
    "BackoffOnTransient",
    "BumpOnTurnBudget",
    "CompositePolicy",
    "FailFast",
    "RecoveryDecision",
    "RecoveryPolicy",
    "RolloverOnQuota",
    "SalvageOnTurnBudget",
    "SalvageOnWallClock",
]
