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
    max_pod_seconds: float | None = None
    max_invocations_per_agent: dict[str, int] = field(default_factory=dict)
    rlm_calls_remaining: int = 120
    max_run_gpu_usd: float | None = None

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

    def check_pod_seconds(
        self,
        *,
        pod_started_at: datetime | None,
        agent_id: str,
        now: datetime | None = None,
    ) -> None:
        if self.max_pod_seconds is None or pod_started_at is None:
            return
        current = now if now is not None else datetime.now(timezone.utc)
        elapsed = (current - pod_started_at).total_seconds()
        if elapsed >= self.max_pod_seconds:
            raise BudgetExhausted(
                f"Run pod-time budget exhausted before invoking {agent_id}: "
                f"{elapsed:.1f}s >= {self.max_pod_seconds:.1f}s",
                provider=None,
                agent_id=agent_id,
                elapsed_seconds=elapsed,
            )

    def check_run_gpu_usd(
        self,
        *,
        cumulative_pod_usd: float,
        agent_id: str,
    ) -> None:
        """Raise BudgetExhausted when cumulative pod spend >= max_run_gpu_usd.

        `cumulative_pod_usd` is the total RunPod cost incurred by this run so
        far — caller is responsible for the running tally (typically
        wall_clock_seconds * sku_usd_per_hr / 3600 summed across pod
        lifetimes). Cap is honored only when set and > 0; None or 0 disables
        the check.
        """
        if self.max_run_gpu_usd is None or self.max_run_gpu_usd <= 0:
            return
        if cumulative_pod_usd >= self.max_run_gpu_usd:
            raise BudgetExhausted(
                f"Run pod-USD budget exhausted before invoking {agent_id}: "
                f"${cumulative_pod_usd:.4f} >= ${self.max_run_gpu_usd:.4f}",
                provider=None,
                agent_id=agent_id,
            )


__all__ = ["RunBudget"]
