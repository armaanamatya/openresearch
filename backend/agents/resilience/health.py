"""Provider health and cooldown tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from backend.agents.resilience.failures import QuotaExhausted, RuntimeFailure, TransientError
from backend.agents.runtime.base import ProviderName


@dataclass
class ProviderHealth:
    provider: ProviderName
    consecutive_quota_failures: int = 0
    consecutive_transient_failures: int = 0
    last_quota_failure_at: datetime | None = None
    last_transient_failure_at: datetime | None = None
    cooldown_until: datetime | None = None


class ProviderHealthMonitor:
    def __init__(self) -> None:
        self._health: dict[ProviderName, ProviderHealth] = {}

    def record_success(self, provider: ProviderName) -> None:
        state = self._state(provider)
        state.consecutive_quota_failures = 0
        state.consecutive_transient_failures = 0
        state.cooldown_until = None

    def record_failure(self, provider: ProviderName, failure: RuntimeFailure) -> None:
        state = self._state(provider)
        now = datetime.now(timezone.utc)
        if isinstance(failure, QuotaExhausted):
            state.consecutive_quota_failures += 1
            state.consecutive_transient_failures = 0
            state.last_quota_failure_at = now
            if failure.resets_at is not None:
                state.cooldown_until = failure.resets_at
            elif state.consecutive_quota_failures >= 3:
                state.cooldown_until = now + timedelta(minutes=30)
            return
        if isinstance(failure, TransientError):
            if (
                state.last_transient_failure_at is not None
                and (now - state.last_transient_failure_at).total_seconds() > 60
            ):
                state.consecutive_transient_failures = 0
            state.consecutive_transient_failures += 1
            state.last_transient_failure_at = now
            if state.consecutive_transient_failures >= 5:
                state.cooldown_until = now + timedelta(minutes=5)

    def is_healthy(self, provider: ProviderName, *, now: datetime | None = None) -> bool:
        state = self._state(provider)
        current = now or datetime.now(timezone.utc)
        return state.cooldown_until is None or state.cooldown_until <= current

    def cooldown_remaining(
        self,
        provider: ProviderName,
        *,
        now: datetime | None = None,
    ) -> float:
        state = self._state(provider)
        if state.cooldown_until is None:
            return 0.0
        current = now or datetime.now(timezone.utc)
        return max(0.0, (state.cooldown_until - current).total_seconds())

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            provider: {
                "consecutive_quota_failures": state.consecutive_quota_failures,
                "consecutive_transient_failures": state.consecutive_transient_failures,
                "cooldown_until": (
                    state.cooldown_until.isoformat() if state.cooldown_until else None
                ),
            }
            for provider, state in self._health.items()
        }

    def _state(self, provider: ProviderName) -> ProviderHealth:
        if provider not in self._health:
            self._health[provider] = ProviderHealth(provider=provider)
        return self._health[provider]


__all__ = ["ProviderHealth", "ProviderHealthMonitor"]
