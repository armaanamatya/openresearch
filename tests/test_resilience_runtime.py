from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from backend.agents.resilience import (
    BudgetExhausted,
    ProviderHealthMonitor,
    QuotaExhausted,
    RunBudget,
    RunCostLedger,
    TransientError,
)
from backend.agents.resilience.classify import classify_anthropic, classify_openai
from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.resilience.engine import (
    RuntimeKwargs,
    default_recovery_policy,
    run_agent_with_resilience,
)
from backend.agents.runtime.base import (
    AgentRuntimeSpec,
    ProviderName,
    RuntimeGuard,
    StreamEvent,
    StreamText,
    StreamUsage,
)


class FakeRuntime:
    def __init__(
        self,
        provider_name: ProviderName,
        *,
        output: str = '{"ok": true}',
        raises: Exception | None = None,
        partial: str = "",
    ) -> None:
        self.provider_name = provider_name
        self.output = output
        self.raises = raises
        self.partial = partial
        self.prompts: list[str] = []

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        self.prompts.append(user_input)
        if self.partial:
            yield StreamText(self.partial)
        if self.raises is not None:
            raise self.raises
        yield StreamText(self.output)
        yield StreamUsage(input_tokens=100, output_tokens=50, reasoning_tokens=0)


def _spec(runtime: FakeRuntime, max_turns: int | None) -> AgentRuntimeSpec:
    return AgentRuntimeSpec(
        name="paper-understanding",
        instructions="test",
        model="gpt-4o" if runtime.provider_name == "openai" else "claude-sonnet-4-6",
        max_turns=max_turns,
        guard=RuntimeGuard(max_tool_calls=4),
    )


def test_classifies_claude_subscription_quota_as_typed_failure() -> None:
    failure = classify_anthropic(
        Exception("You've hit your limit - resets 9:40pm"),
        agent_id="paper-understanding",
        elapsed_seconds=1.0,
        partial_output="partial",
    )

    assert isinstance(failure, QuotaExhausted)
    assert failure.provider == "anthropic"
    assert failure.partial_output == "partial"


def test_classifies_openai_insufficient_quota_as_typed_failure() -> None:
    class FakeOpenAIRateLimitError(Exception):
        status_code = 429

    failure = classify_openai(
        FakeOpenAIRateLimitError("insufficient_quota: current quota exceeded"),
        agent_id="paper-understanding",
        elapsed_seconds=1.0,
    )

    assert isinstance(failure, QuotaExhausted)
    assert failure.provider == "openai"


def test_cost_ledger_roundtrip_and_budget_guard(tmp_path: Path) -> None:
    path = tmp_path / "cost_ledger.jsonl"
    ledger = RunCostLedger(project_id="prj", path=path)
    ledger.append(
        CostLedgerEntry.from_usage(
            agent_id="paper-understanding",
            attempt_index=0,
            provider="openai",
            model="gpt-4o",
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
        )
    )
    # Lane G batches writes; force a flush before reading back from disk.
    ledger.flush()

    loaded = RunCostLedger.load_jsonl(path, project_id="prj")

    assert loaded.total_usd() == 2.5
    with pytest.raises(BudgetExhausted):
        RunBudget(max_usd=1.0).check(
            ledger=loaded,
            started_at=loaded.entries[0].timestamp,
            agent_id="paper-understanding",
            attempt_count=1,
        )


def test_health_monitor_cools_down_after_repeated_quota_failures() -> None:
    health = ProviderHealthMonitor()
    for _ in range(3):
        health.record_failure(
            "anthropic",
            QuotaExhausted("quota", provider="anthropic", agent_id="agent"),
        )

    assert health.is_healthy("anthropic") is False
    assert health.cooldown_remaining("anthropic") > 0
    health.record_success("anthropic")
    assert health.is_healthy("anthropic") is True


def test_engine_falls_back_anthropic_to_openai_with_partial_carryover(tmp_path: Path) -> None:
    anthropic = FakeRuntime(
        "anthropic",
        raises=Exception("You've hit your limit"),
        partial='{"partial": "claim map started"}',
    )
    openai = FakeRuntime("openai", output='{"core_contribution": "fallback ok"}')
    runtimes = {"anthropic": anthropic, "openai": openai}
    health = ProviderHealthMonitor()
    ledger = RunCostLedger(project_id="prj", path=tmp_path / "cost_ledger.jsonl")

    result = asyncio.run(
        run_agent_with_resilience(
            agent_id="paper-understanding",
            base_prompt="Analyze paper.",
            primary_provider="anthropic",
            runtime_for=lambda provider: runtimes[provider],
            chain=["anthropic", "openai"],
            policy=default_recovery_policy(chain=["anthropic", "openai"], health=health),
            health=health,
            ledger=ledger,
            budget=RunBudget(),
            runtime_kwargs=RuntimeKwargs(
                cwd=tmp_path,
                max_turns=3,
                wall_clock_seconds=10,
                build_runtime_spec=_spec,
                summary_path=tmp_path / "fallback_summary.json",
            ),
        )
    )

    assert result.output_text == '{"core_contribution": "fallback ok"}'
    assert "[FALLBACK CONTINUATION]" in openai.prompts[0]
    assert "claim map started" in openai.prompts[0]
    assert ledger.total_by_provider()["anthropic"].attempts == 1
    assert ledger.total_by_provider()["openai"].attempts == 1
    assert (tmp_path / "fallback_summary.json").exists()


def test_engine_falls_back_openai_to_anthropic_on_quota(tmp_path: Path) -> None:
    openai = FakeRuntime(
        "openai",
        raises=Exception("insufficient_quota: current quota exceeded"),
        partial="partial openai work",
    )
    anthropic = FakeRuntime("anthropic", output="anthropic finished")
    runtimes = {"anthropic": anthropic, "openai": openai}
    health = ProviderHealthMonitor()
    ledger = RunCostLedger(project_id="prj", path=tmp_path / "cost_ledger.jsonl")

    result = asyncio.run(
        run_agent_with_resilience(
            agent_id="paper-understanding",
            base_prompt="Analyze paper.",
            primary_provider="openai",
            runtime_for=lambda provider: runtimes[provider],
            chain=["openai", "anthropic"],
            policy=default_recovery_policy(chain=["openai", "anthropic"], health=health),
            health=health,
            ledger=ledger,
            budget=RunBudget(),
            runtime_kwargs=RuntimeKwargs(
                cwd=tmp_path,
                max_turns=3,
                wall_clock_seconds=10,
                build_runtime_spec=_spec,
            ),
        )
    )

    assert result.output_text == "anthropic finished"
    assert "[FALLBACK CONTINUATION]" in anthropic.prompts[0]
    assert "partial openai work" in anthropic.prompts[0]


def test_engine_retries_transient_without_cross_provider_fallback(tmp_path: Path) -> None:
    flaky = FakeRuntime("anthropic", raises=Exception("network timeout"))
    openai = FakeRuntime("openai", output="should not be used")
    health = ProviderHealthMonitor()
    ledger = RunCostLedger(project_id="prj")

    with pytest.raises(TransientError):
        asyncio.run(
            run_agent_with_resilience(
                agent_id="paper-understanding",
                base_prompt="Analyze paper.",
                primary_provider="anthropic",
                runtime_for=lambda provider: flaky if provider == "anthropic" else openai,
                chain=["anthropic", "openai"],
                policy=default_recovery_policy(
                    chain=["anthropic", "openai"],
                    health=health,
                ),
                health=health,
                ledger=ledger,
                budget=RunBudget(),
                runtime_kwargs=RuntimeKwargs(
                    cwd=tmp_path,
                    max_turns=3,
                    wall_clock_seconds=10,
                    build_runtime_spec=_spec,
                ),
            )
        )

    assert len(flaky.prompts) == 2
    assert openai.prompts == []


# ---------------------------------------------------------------------------
# Track C — runtime credential health: AuthenticationError on a non-primary
# fallback should mark that provider dead and let the primary retry. The
# previous behaviour surfaced the fallback's 401 as the run's terminal
# error even when the operator only configured the primary intentionally.
# ---------------------------------------------------------------------------

class _FakeAuthError(Exception):
    """Class name contains 'authentication' so classify_* tags it as AuthenticationError."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


# Ensure the class name actually contains "authentication" for classify_*.
_FakeAuthError.__name__ = "FakeAuthenticationError"


def test_health_monitor_mark_dead_excludes_from_healthy() -> None:
    health = ProviderHealthMonitor()
    assert health.is_healthy("openai") is True
    assert health.is_dead("openai") is False
    health.mark_dead("openai", reason="authentication")
    assert health.is_dead("openai") is True
    assert health.is_healthy("openai") is False
    assert health.snapshot()["openai"]["dead_reason"] == "authentication"


def test_health_monitor_mark_dead_rejects_empty_reason() -> None:
    health = ProviderHealthMonitor()
    with pytest.raises(ValueError):
        health.mark_dead("openai", reason="")


def test_engine_marks_fallback_provider_dead_on_authentication_error(tmp_path: Path) -> None:
    # Primary (anthropic) hits quota -> RolloverOnQuota policy triggers
    # fallback to openai. Openai returns AuthenticationError -> engine
    # marks openai dead and resets to primary. Primary's second attempt
    # succeeds. End state: dead-mark on openai, success result returned.
    # (A plain transient on primary would trigger BackoffOnTransient, not
    # cross-provider fallback — quota is the canonical trigger.)
    attempt = {"count": 0}

    class QuotaThenSuccessful:
        provider_name: ProviderName = "anthropic"
        prompts: list[str] = []

        async def run_agent(
            self, *, agent: AgentRuntimeSpec, user_input: str
        ) -> AsyncIterator[StreamEvent]:
            self.prompts.append(user_input)
            attempt["count"] += 1
            if attempt["count"] == 1:
                # First call: quota -> rollover policy falls back to openai
                raise Exception("insufficient_quota: current quota exceeded")
            # After openai-dead-mark and reset to primary: succeed
            yield StreamText('{"core_contribution": "primary retried after dead-mark"}')
            yield StreamUsage(input_tokens=10, output_tokens=10, reasoning_tokens=0)

    auth_failing_openai = FakeRuntime(
        "openai",
        raises=_FakeAuthError("Incorrect API key: sk-svcacct-***"),
    )
    primary = QuotaThenSuccessful()
    runtimes = {"anthropic": primary, "openai": auth_failing_openai}
    health = ProviderHealthMonitor()
    ledger = RunCostLedger(project_id="prj_dead", path=tmp_path / "cost_ledger.jsonl")

    result = asyncio.run(
        run_agent_with_resilience(
            agent_id="paper-understanding",
            base_prompt="Analyze paper.",
            primary_provider="anthropic",
            runtime_for=lambda provider: runtimes[provider],
            chain=["anthropic", "openai"],
            policy=default_recovery_policy(chain=["anthropic", "openai"], health=health),
            health=health,
            ledger=ledger,
            budget=RunBudget(),
            runtime_kwargs=RuntimeKwargs(
                cwd=tmp_path,
                max_turns=3,
                wall_clock_seconds=10,
                build_runtime_spec=_spec,
                summary_path=tmp_path / "fallback_summary.json",
                max_total_attempts=5,
            ),
        )
    )

    assert "primary retried after dead-mark" in result.output_text
    assert health.is_dead("openai") is True
    assert health.is_dead("anthropic") is False
    # Anthropic was tried twice (initial transient + retry-after-dead-mark).
    assert attempt["count"] == 2
    # Openai was tried exactly once (the auth error) then never again.
    assert len(auth_failing_openai.prompts) == 1


def test_engine_raises_when_authentication_error_hits_primary_provider(tmp_path: Path) -> None:
    # Auth failure on the PRIMARY provider IS fatal — no escape hatch.
    auth_failing_primary = FakeRuntime(
        "anthropic",
        raises=_FakeAuthError("Incorrect API key: sk-ant-bad"),
    )
    other = FakeRuntime("openai", output="unused")
    runtimes = {"anthropic": auth_failing_primary, "openai": other}
    health = ProviderHealthMonitor()
    ledger = RunCostLedger(project_id="prj_primary_dead", path=tmp_path / "cost_ledger.jsonl")

    with pytest.raises(Exception):
        asyncio.run(
            run_agent_with_resilience(
                agent_id="paper-understanding",
                base_prompt="Analyze paper.",
                primary_provider="anthropic",
                runtime_for=lambda provider: runtimes[provider],
                chain=["anthropic"],  # no fallback configured
                policy=default_recovery_policy(chain=["anthropic"], health=health),
                health=health,
                ledger=ledger,
                budget=RunBudget(),
                runtime_kwargs=RuntimeKwargs(
                    cwd=tmp_path,
                    max_turns=3,
                    wall_clock_seconds=10,
                    build_runtime_spec=_spec,
                ),
            )
        )
    # Primary not marked dead by engine — that semantic is reserved for
    # non-primary fallbacks (operator can fix primary, has no point trying
    # to "skip" it).
    assert health.is_dead("anthropic") is False
