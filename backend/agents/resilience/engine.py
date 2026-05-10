"""Resilient provider invocation engine."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.classify import classify_failure
from backend.agents.resilience.context import (
    AgentRunResult,
    AttemptContext,
    AttemptRecord,
)
from backend.agents.resilience.cost import CostLedgerEntry, RunCostLedger
from backend.agents.resilience.failures import (
    BudgetExhausted,
    FatalRuntimeFailure,
    RuntimeFailure,
)
from backend.agents.resilience.health import ProviderHealthMonitor
from backend.agents.resilience.policy import (
    BackoffOnRateLimit,
    BackoffOnTransient,
    BumpOnTurnBudget,
    CompositePolicy,
    FailFast,
    RecoveryDecision,
    RecoveryPolicy,
    RolloverOnQuota,
    SalvageOnTurnBudget,
    SalvageOnWallClock,
)
from backend.agents.runtime.base import (
    AgentLimitExceeded,
    AgentRuntime,
    AgentRuntimeSpec,
    ProviderName,
    StreamText,
    StreamToolCall,
    StreamUsage,
)
from backend.agents.telemetry import (
    AgentInvocationRecord,
    AgentTelemetryRecorder,
    coerce_usage,
    utc_now_iso,
)


@dataclass(frozen=True)
class RuntimeKwargs:
    cwd: Path
    max_turns: int | None
    wall_clock_seconds: float | None
    build_runtime_spec: Callable[[AgentRuntime, int | None], AgentRuntimeSpec]
    telemetry: AgentTelemetryRecorder | None = None
    run_started_at: datetime | None = None
    salvage_validator: Callable[[str], bool] | None = None
    summary_path: Path | None = None
    max_total_attempts: int = 3


class _NullAsyncContext:
    async def __aenter__(self) -> "_NullAsyncContext":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def default_recovery_policy(
    *,
    chain: list[ProviderName],
    health: ProviderHealthMonitor,
) -> RecoveryPolicy:
    return CompositePolicy(
        BackoffOnTransient(max_retries=2, initial_delay=1.0, multiplier=2.0),
        BackoffOnRateLimit(max_retries=1, default_delay=5.0),
        SalvageOnTurnBudget(),
        SalvageOnWallClock(),
        BumpOnTurnBudget(factor=2.0, max_attempts=2),
        RolloverOnQuota(chain, health=health),
        FailFast(),
    )


async def run_agent_with_resilience(
    *,
    agent_id: str,
    base_prompt: str,
    primary_provider: ProviderName,
    runtime_for: Callable[[ProviderName], AgentRuntime],
    chain: list[ProviderName],
    policy: RecoveryPolicy,
    health: ProviderHealthMonitor,
    ledger: RunCostLedger,
    budget: RunBudget,
    runtime_kwargs: RuntimeKwargs,
) -> AgentRunResult:
    context = AttemptContext(
        agent_id=agent_id,
        base_prompt=base_prompt,
        cwd=runtime_kwargs.cwd,
        cost_ledger=ledger,
    )
    current_provider = _first_healthy(primary_provider, chain, health)
    current_max_turns = runtime_kwargs.max_turns
    run_started_at = runtime_kwargs.run_started_at or datetime.now(timezone.utc)

    while True:
        if len(context.attempts) >= runtime_kwargs.max_total_attempts:
            raise RuntimeFailure(
                f"Maximum provider attempts reached for {agent_id}",
                provider=current_provider,
                agent_id=agent_id,
            )
        try:
            budget.check(
                ledger=ledger,
                started_at=run_started_at,
                agent_id=agent_id,
                attempt_count=len(context.attempts),
            )
        except BudgetExhausted:
            _write_summary(runtime_kwargs.summary_path, context, ledger, health)
            raise
        runtime = runtime_for(current_provider)
        prompt = context.continuation_prompt(target_provider=current_provider)
        result, failure = await _run_single_attempt(
            agent_id=agent_id,
            runtime=runtime,
            prompt=prompt,
            max_turns=current_max_turns,
            context=context,
            health=health,
            ledger=ledger,
            runtime_kwargs=runtime_kwargs,
        )
        _write_summary(runtime_kwargs.summary_path, context, ledger, health)
        if result is not None:
            return result
        if failure is None:
            raise RuntimeFailure(
                f"Attempt failed without classified failure for {agent_id}",
                provider=current_provider,
                agent_id=agent_id,
            )
        if isinstance(failure, (FatalRuntimeFailure, BudgetExhausted)):
            raise failure

        handled = False
        for decision in _iter_policy_decisions(policy, failure=failure, context=context):
            if decision.action == "salvage_partial":
                partial = context.latest_partial_output()
                if partial and _salvage_is_valid(partial, runtime_kwargs.salvage_validator):
                    _mark_last_attempt(context, "salvaged", decision)
                    _write_summary(runtime_kwargs.summary_path, context, ledger, health)
                    return AgentRunResult(
                        output_text=partial,
                        trace_text=partial,
                        tool_calls=[],
                        elapsed_seconds=failure.elapsed_seconds,
                    )
                continue
            if decision.action == "retry_same":
                current_max_turns = decision.bumped_max_turns or current_max_turns
                if decision.delay_seconds > 0:
                    await asyncio.sleep(decision.delay_seconds)
                handled = True
                break
            if decision.action == "fallback_to":
                if decision.target_provider is None:
                    continue
                current_provider = decision.target_provider
                current_max_turns = runtime_kwargs.max_turns
                _mark_last_attempt(context, "fallback", decision)
                handled = True
                break
            if decision.action == "fail":
                _mark_last_attempt(context, "failed", decision)
                _write_summary(runtime_kwargs.summary_path, context, ledger, health)
                raise failure
        if not handled:
            raise failure


async def _run_single_attempt(
    *,
    agent_id: str,
    runtime: AgentRuntime,
    prompt: str,
    max_turns: int | None,
    context: AttemptContext,
    health: ProviderHealthMonitor,
    ledger: RunCostLedger,
    runtime_kwargs: RuntimeKwargs,
) -> tuple[AgentRunResult | None, RuntimeFailure | None]:
    started = datetime.now(timezone.utc)
    started_at = utc_now_iso()
    t0 = time.time()
    runtime_spec = runtime_kwargs.build_runtime_spec(runtime, max_turns)
    collected_text: list[str] = []
    trace_lines: list[str] = []
    tool_calls: list[str] = []
    msg_count = 0
    tool_call_count = 0
    usage: dict[str, object] = {
        "provider": runtime.provider_name,
        "model": runtime_spec.model,
    }
    success = False
    error_message = ""

    print(f"  [{agent_id}] starting ({runtime.provider_name})...", file=sys.stderr, flush=True)
    try:
        timeout_ctx = (
            asyncio.timeout(runtime_kwargs.wall_clock_seconds)
            if runtime_kwargs.wall_clock_seconds is not None
            else _NullAsyncContext()
        )
        async with timeout_ctx:
            async for event in runtime.run_agent(agent=runtime_spec, user_input=prompt):
                elapsed = time.time() - t0
                msg_count += 1
                if isinstance(event, StreamText):
                    collected_text.append(event.text)
                    trace_lines.append(event.text)
                    snippet = event.text[:120].replace("\n", " ").strip()
                    if snippet:
                        print(
                            f"  [{agent_id}] ({elapsed:.0f}s) {snippet}...",
                            file=sys.stderr,
                            flush=True,
                        )
                elif isinstance(event, StreamToolCall):
                    tool_call_count += 1
                    if (
                        runtime_spec.guard.max_tool_calls is not None
                        and tool_call_count > runtime_spec.guard.max_tool_calls
                    ):
                        raise AgentLimitExceeded(
                            agent_id=agent_id,
                            kind="tool_calls",
                            limit_value=runtime_spec.guard.max_tool_calls,
                            elapsed_seconds=elapsed,
                            partial_output="\n".join(collected_text),
                        )
                    tool_info = _tool_info(event)
                    tool_calls.append(tool_info)
                    trace_lines.append(f"tool: {tool_info}")
                    print(
                        f"  [{agent_id}] ({elapsed:.0f}s) tool: {tool_info}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif isinstance(event, StreamUsage):
                    usage.update(coerce_usage(event.as_dict()))
        success = True
        health.record_success(runtime.provider_name)
        output = "\n".join(collected_text)
        elapsed = time.time() - t0
        print(
            f"  [{agent_id}] completed in {elapsed:.0f}s ({msg_count} events, {sum(len(t) for t in collected_text)} chars)",
            file=sys.stderr,
            flush=True,
        )
        _append_attempt_records(
            context=context,
            ledger=ledger,
            runtime=runtime,
            runtime_spec=runtime_spec,
            started=started,
            outcome="success",
            usage=usage,
            decision_note="success",
        )
        if runtime_kwargs.telemetry is not None:
            runtime_kwargs.telemetry.append(
                AgentInvocationRecord(
                    agent_id=agent_id,
                    model=runtime_spec.model,
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                    duration_seconds=elapsed,
                    message_count=msg_count,
                    output_chars=sum(len(text) for text in collected_text),
                    success=True,
                    error_message="",
                    usage=dict(usage),
                    provider=runtime.provider_name,
                    attempt_index=len(context.attempts) - 1,
                    outcome="success",
                )
            )
        return (
            AgentRunResult(
                output_text=output,
                trace_text="\n".join(trace_lines),
                tool_calls=tool_calls,
                elapsed_seconds=elapsed,
            ),
            None,
        )
    except TimeoutError as exc:
        failure = classify_failure(
            runtime.provider_name,
            AgentLimitExceeded(
                agent_id=agent_id,
                kind="wall_clock",
                limit_value=int(runtime_kwargs.wall_clock_seconds or 0),
                elapsed_seconds=time.time() - t0,
                partial_output="\n".join(collected_text),
            ),
            agent_id=agent_id,
            elapsed_seconds=time.time() - t0,
            partial_output="\n".join(collected_text),
        )
        failure.cause = exc
    except Exception as exc:
        failure = classify_failure(
            runtime.provider_name,
            exc,
            agent_id=agent_id,
            elapsed_seconds=time.time() - t0,
            partial_output="\n".join(collected_text),
        )

    error_message = f"{failure.__class__.__name__}: {failure}"
    context.record_partial(runtime.provider_name, failure.partial_output or "\n".join(collected_text))
    health.record_failure(runtime.provider_name, failure)
    _append_attempt_records(
        context=context,
        ledger=ledger,
        runtime=runtime,
        runtime_spec=runtime_spec,
        started=started,
        outcome="failed",
        failure=failure,
        usage=usage,
        decision_note=error_message,
    )
    if runtime_kwargs.telemetry is not None:
        runtime_kwargs.telemetry.append(
            AgentInvocationRecord(
                agent_id=agent_id,
                model=runtime_spec.model,
                started_at=started_at,
                finished_at=utc_now_iso(),
                duration_seconds=time.time() - t0,
                message_count=msg_count,
                output_chars=sum(len(text) for text in collected_text),
                success=success,
                error_message=error_message,
                usage=dict(usage),
                provider=runtime.provider_name,
                attempt_index=len(context.attempts) - 1,
                outcome="failed",
                failure_kind=failure.__class__.__name__,
            )
        )
    return None, failure


def _append_attempt_records(
    *,
    context: AttemptContext,
    ledger: RunCostLedger,
    runtime: AgentRuntime,
    runtime_spec: AgentRuntimeSpec,
    started: datetime,
    outcome: str,
    usage: dict[str, object],
    decision_note: str,
    failure: RuntimeFailure | None = None,
) -> None:
    attempt_index = len(context.attempts)
    entry = CostLedgerEntry.from_usage(
        agent_id=context.agent_id,
        attempt_index=attempt_index,
        provider=runtime.provider_name,
        model=runtime_spec.model,
        usage=usage,
    )
    ledger.append(entry)
    context.attempts.append(
        AttemptRecord(
            attempt_index=attempt_index,
            provider=runtime.provider_name,
            model=runtime_spec.model,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            outcome=outcome,  # type: ignore[arg-type]
            failure_kind=failure.__class__.__name__ if failure else None,
            usage={
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "cache_read_input_tokens": entry.cache_read_input_tokens,
                "cache_creation_input_tokens": entry.cache_creation_input_tokens,
                "reasoning_tokens": entry.reasoning_tokens,
            },
            estimated_usd=entry.estimated_usd,
            decision_note=decision_note,
        )
    )

def _mark_last_attempt(
    context: AttemptContext,
    outcome: str,
    decision: RecoveryDecision,
) -> None:
    if not context.attempts:
        return
    last = context.attempts[-1]
    last.outcome = outcome  # type: ignore[assignment]
    last.decision_note = decision.note
    last.next_provider = decision.target_provider


def _iter_policy_decisions(
    policy: RecoveryPolicy,
    *,
    failure: RuntimeFailure,
    context: AttemptContext,
):
    iterator = getattr(policy, "iter_decisions", None)
    if callable(iterator):
        yield from iterator(failure=failure, context=context)
        return
    decision = policy.decide(failure=failure, context=context)
    if decision is not None:
        yield decision


def _salvage_is_valid(text: str, validator: Callable[[str], bool] | None) -> bool:
    if not text.strip():
        return False
    return validator(text) if validator is not None else True


def _first_healthy(
    primary_provider: ProviderName,
    chain: list[ProviderName],
    health: ProviderHealthMonitor,
) -> ProviderName:
    if health.is_healthy(primary_provider):
        return primary_provider
    for provider in chain:
        if health.is_healthy(provider):
            return provider
    return primary_provider


def _tool_info(event: StreamToolCall) -> str:
    tool_info = event.tool_name
    inp = event.tool_input or {}
    if "file_path" in inp:
        tool_info += f" {inp['file_path']}"
    elif "command" in inp:
        cmd = str(inp["command"])[:80]
        tool_info += f" `{cmd}`"
    elif "pattern" in inp:
        tool_info += f" {inp['pattern']}"
    return tool_info


def _write_summary(
    path: Path | None,
    context: AttemptContext,
    ledger: RunCostLedger,
    health: ProviderHealthMonitor,
) -> None:
    if path is None:
        return
    totals = {
        provider: values.__dict__
        for provider, values in ledger.total_by_provider().items()
    }
    payload = {
        "project_id": ledger.project_id,
        "agent_id": context.agent_id,
        "attempts": [attempt.to_json() for attempt in context.attempts],
        "total_usd": ledger.total_usd(),
        "total_by_provider": totals,
        "provider_health": health.snapshot(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["RuntimeKwargs", "default_recovery_policy", "run_agent_with_resilience"]
