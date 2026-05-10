"""Run-level budget checks for resilient provider attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.agents.resilience.cost import RunCostLedger
from backend.agents.resilience.failures import BudgetExhausted


@dataclass(frozen=True)
class RunBudget:
    max_usd: float | None = None
    max_wall_clock_seconds: float | None = None
    max_invocations_per_agent: dict[str, int] = field(default_factory=dict)

    def check(
        self,
        *,
        ledger: RunCostLedger,
        started_at: datetime,
        agent_id: str,
        attempt_count: int,
    ) -> None:
        if self.max_usd is not None and ledger.total_usd() >= self.max_usd:
            raise BudgetExhausted(
                f"Run cost budget exhausted before invoking {agent_id}: "
                f"${ledger.total_usd():.4f} >= ${self.max_usd:.4f}",
                provider=None,
                agent_id=agent_id,
            )
        if self.max_wall_clock_seconds is not None:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            if elapsed >= self.max_wall_clock_seconds:
                raise BudgetExhausted(
                    f"Run wall-clock budget exhausted before invoking {agent_id}: "
                    f"{elapsed:.1f}s >= {self.max_wall_clock_seconds:.1f}s",
                    provider=None,
                    agent_id=agent_id,
                    elapsed_seconds=elapsed,
                )
        max_attempts = self.max_invocations_per_agent.get(agent_id)
        if max_attempts is not None and attempt_count >= max_attempts:
            raise BudgetExhausted(
                f"Invocation budget exhausted for {agent_id}: "
                f"{attempt_count} >= {max_attempts}",
                provider=None,
                agent_id=agent_id,
            )


__all__ = ["RunBudget"]
