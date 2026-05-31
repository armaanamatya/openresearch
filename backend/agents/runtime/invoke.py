"""Shared helpers for one-shot agent runtime invocations."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.resilience.cost import record_subagent_usage_to_path
from backend.agents.runtime.base import (
    AgentRuntime,
    ProviderName,
    StreamText,
    StreamToolCall,
    StreamUsage,
    blocked_terms_from_env,
)
from backend.agents.runtime.factory import make_runtime
from backend.agents.runtime.sdk_isolation import run_isolated
from backend.agents.worker_reports import (
    append_worker_report_instruction,
    build_worker_report,
    write_worker_report,
)

logger = logging.getLogger(__name__)

# FM-001 sub-agent transport retry (2026-05-30).
#
# The bundled-CLI (claude-agent-sdk) OAuth transport degrades under rapid
# call volume: after a burst of bundled-CLI round-trips the CLI starts
# returning ``API Error: Unable to connect to API (ConnectionRefused)``,
# which the SDK surfaces as a zero-text "success" result and re-raises as
# ``ProcessError("Claude Code returned an error result: success")``. This is
# NOT orphaned-child accumulation (confirmed 2026-05-30: reaping leaves 0
# children) — it is a rate/throughput limit on the shared transport. The
# durable, concurrency-safe fix is a bounded retry-with-backoff on the
# code-writing / grading sub-agent surface (where an empty result kills the
# whole run via implement_baseline -> no commands.json -> no experiment).
#
# Scope: this covers the sub-agent path only (implement_baseline,
# verify_against_rubric, propose_improvements via collect_agent_text). The
# high-concurrency root navigation path (ClaudeLlmClient) keeps its existing
# read-idle / stall-sentinel handling — siblings there make in-place reaping
# unsafe, and the root already retries on the sentinel.
_TRANSIENT_TRANSPORT_RE = re.compile(
    r"returned an error result:\s*success"   # zero-text success == transport wedge
    r"|ConnectionRefused"
    r"|Connection refused"
    r"|ECONNREFUSED"
    r"|Unable to connect to API"
    r"|error_during_execution",
    re.IGNORECASE,
)


def _is_transient_transport_error(message: str | None) -> bool:
    """True when *message* matches the FM-001 transient bundled-CLI transport signature."""
    return bool(message) and bool(_TRANSIENT_TRANSPORT_RE.search(message))


def _transport_attempts() -> int:
    """Total attempts (= retries + 1) for a transient sub-agent transport failure.

    ``REPROLAB_SUBAGENT_TRANSPORT_RETRIES`` (default 2 retries => 3 attempts);
    ``0`` disables retry (1 attempt).
    """
    raw = os.environ.get("REPROLAB_SUBAGENT_TRANSPORT_RETRIES", "").strip()
    try:
        retries = int(raw) if raw else 2
    except ValueError:
        logger.warning("invalid REPROLAB_SUBAGENT_TRANSPORT_RETRIES=%r; using 2", raw)
        retries = 2
    return max(1, retries + 1)


def _transport_backoff_s(attempt: int) -> float:
    """Exponential backoff before retrying attempt N (1-indexed): base * 2**(N-1).

    Default base 8s => 8, 16, 32 ... Bounded well under the implement_baseline
    pre-emit stall budget (``REPROLAB_PRE_EMIT_STALL_S``, default 900s) so a
    backoff never trips the code_dir watchdog. ``REPROLAB_SUBAGENT_TRANSPORT_BACKOFF_S``
    overrides the base.
    """
    try:
        base = float(os.environ.get("REPROLAB_SUBAGENT_TRANSPORT_BACKOFF_S", "8") or 8)
    except ValueError:
        base = 8.0
    return base * (2 ** max(0, attempt - 1))


async def _backoff_with_liveness(
    seconds: float, on_event: Callable[[], None] | None
) -> None:
    """Sleep ``seconds`` in small chunks, bumping ``on_event`` so the caller's
    stall watchdog (implement_baseline polls code_dir + SDK-event liveness)
    sees the backoff as intentional, not a hang."""
    slept = 0.0
    while slept < seconds:
        chunk = min(5.0, seconds - slept)
        await asyncio.sleep(chunk)
        slept += chunk
        if on_event is not None:
            try:
                on_event()
            except Exception:  # noqa: BLE001 — liveness ping must never break backoff
                pass


async def collect_agent_text(
    agent_id: str,
    prompt: str,
    *,
    project_dir: Path,
    ledger_dir: Path | None = None,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
    max_turns: int | None = None,
    on_event: Callable[[], None] | None = None,
    blocked_terms: tuple[str, ...] = (),
) -> str:
    """Run one agent and return concatenated text output.

    ``ledger_dir`` overrides where cost_ledger.jsonl and dashboard_events.jsonl
    are written.  Callers that set ``project_dir`` to a code subdirectory (e.g.
    ``runs/<id>/code/``) should pass ``ledger_dir=project_dir.parent`` so ledger
    entries land in the run root alongside demo_status.json.
    """
    selected_runtime = runtime or make_runtime(provider)
    started_at = datetime.now(timezone.utc).isoformat()
    # #7 benchmark integrity: when the caller didn't pass an explicit blocklist,
    # seed it from the curated env-var seam (OPENRESEARCH_BLOCKED_TERMS_JSON, set by
    # cli.py). collect_agent_text is the single chokepoint EVERY agent flows
    # through (baseline-implementation, rdr, patch-mode, future callers), so this
    # makes the RuntimeGuard uniform and un-forgettable — no per-caller threading
    # to forget. An explicit non-empty blocked_terms always wins.
    if not blocked_terms:
        blocked_terms = blocked_terms_from_env()
    spec = AGENT_REGISTRY[agent_id].to_runtime_spec(
        selected_runtime.provider_name,
        model_override=model,
        working_directory=project_dir,
        max_turns=max_turns,
        blocked_terms=blocked_terms,
    )
    collected: list[str] = []
    tool_calls: list[dict[str, object]] = []
    subagent_usage: dict[str, int] = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }

    async def _do_sdk_call() -> tuple[list[str], list[dict[str, object]], dict[str, int]]:
        # Inner coroutine so run_isolated can thread-isolate the SDK call and
        # contain its aclose race within the worker's event loop.
        _inner_collected: list[str] = []
        _inner_tool_calls: list[dict[str, object]] = []
        _inner_usage: dict[str, int] = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }
        async for event in selected_runtime.run_agent(
            agent=spec,
            user_input=append_worker_report_instruction(prompt),
        ):
            # Liveness signal: every streamed event (text token, tool-call, usage)
            # proves the sub-agent is actively producing output — even before it has
            # written any file. A polling watchdog uses this to distinguish a model
            # that is reasoning / generating a large file from a genuinely hung SDK,
            # so a working agent is never falsely cancelled. Never let it break the stream.
            if on_event is not None:
                try:
                    on_event()
                except Exception:  # noqa: BLE001
                    pass
            if isinstance(event, StreamText):
                _inner_collected.append(event.text)
            elif isinstance(event, StreamToolCall):
                _inner_tool_calls.append({
                    "tool_id": event.tool_id,
                    "tool_name": event.tool_name,
                    "tool_input": event.tool_input,
                })
            elif isinstance(event, StreamUsage):
                for k in _inner_usage:
                    _inner_usage[k] += getattr(event, k, 0) or 0
        return _inner_collected, _inner_tool_calls, _inner_usage

    attempts = _transport_attempts()
    for attempt in range(1, attempts + 1):
        try:
            collected, tool_calls, subagent_usage = await run_isolated(_do_sdk_call)
            break
        except Exception as exc:
            # FM-001: a transient bundled-CLI transport wedge surfaces as a
            # zero-text "success" / ConnectionRefused ProcessError. Back off and
            # retry — the transport recovers once the call cadence relaxes. The
            # failed attempt's SDK child has already errored out; a fresh query()
            # spawns its own child, and implement_baseline's finally-reaper sweeps
            # any straggler after this primitive returns, so no in-loop reap is
            # needed (and reaping here would risk killing a concurrent sibling).
            if attempt < attempts and _is_transient_transport_error(str(exc)):
                backoff = _transport_backoff_s(attempt)
                logger.warning(
                    "collect_agent_text[%s]: transient transport failure on "
                    "attempt %d/%d (%s) — backing off %.0fs then retrying",
                    agent_id, attempt, attempts, str(exc)[:200], backoff,
                )
                await _backoff_with_liveness(backoff, on_event)
                continue
            # Non-transient, or retries exhausted: record + propagate.
            if _is_transient_transport_error(str(exc)):
                logger.error(
                    "collect_agent_text[%s]: transient transport failure persisted "
                    "across %d attempts — giving up (%s)",
                    agent_id, attempts, str(exc)[:200],
                )
            raw_text = "\n".join(collected)
            report = build_worker_report(
                agent_id=agent_id,
                project_dir=project_dir,
                model=spec.model,
                provider=selected_runtime.provider_name,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                raw_text=raw_text,
                tool_calls=tool_calls,
                error=str(exc),
            )
            write_worker_report(project_dir, report)
            raise

    raw_text = "\n".join(collected)
    report = build_worker_report(
        agent_id=agent_id,
        project_dir=project_dir,
        model=spec.model,
        provider=selected_runtime.provider_name,
        status="completed",
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        raw_text=raw_text,
        tool_calls=tool_calls,
    )
    write_worker_report(project_dir, report)
    record_subagent_usage_to_path(
        ledger_dir or project_dir,
        agent_id,
        spec.model or "",
        selected_runtime.provider_name,
        subagent_usage,
    )
    return raw_text


__all__ = ["collect_agent_text"]
