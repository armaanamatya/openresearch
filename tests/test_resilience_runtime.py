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
