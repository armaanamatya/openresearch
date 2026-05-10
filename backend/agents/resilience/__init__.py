"""Provider resilience primitives for agent runtime invocations."""

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.context import AgentRunResult, AttemptContext, AttemptRecord
from backend.agents.resilience.cost import CostLedgerEntry, RunCostLedger
from backend.agents.resilience.failures import (
    AuthenticationError,
    BudgetExhausted,
    GuardViolation,
    QuotaExhausted,
    RateLimited,
    RuntimeFailure,
    SchemaValidationError,
    ToolBudgetExhausted,
    TransientError,
    TurnBudgetExhausted,
    WallClockExceeded,
)
from backend.agents.resilience.health import ProviderHealthMonitor
from backend.agents.resilience.policy import (
    BackoffOnRateLimit,
    BackoffOnTransient,
    BumpOnTurnBudget,
    CompositePolicy,
    FailFast,
    RolloverOnQuota,
    SalvageOnTurnBudget,
    SalvageOnWallClock,
)

__all__ = [
    "AgentRunResult",
    "AttemptContext",
    "AttemptRecord",
    "AuthenticationError",
    "BackoffOnRateLimit",
    "BackoffOnTransient",
    "BudgetExhausted",
    "BumpOnTurnBudget",
    "CompositePolicy",
    "CostLedgerEntry",
    "FailFast",
    "GuardViolation",
    "ProviderHealthMonitor",
    "QuotaExhausted",
    "RateLimited",
    "RolloverOnQuota",
    "RunBudget",
    "RunCostLedger",
    "RuntimeFailure",
    "SalvageOnTurnBudget",
    "SalvageOnWallClock",
    "SchemaValidationError",
    "ToolBudgetExhausted",
    "TransientError",
    "TurnBudgetExhausted",
    "WallClockExceeded",
]
