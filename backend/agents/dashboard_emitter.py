"""Emits DashboardEvent-shaped JSONL for real-time frontend streaming.

Each event matches the frontend contract defined in
frontend/src/lib/events/contract.ts. Events are appended to
<runs_root>/<project_id>/dashboard_events.jsonl and picked up
by the SSE stream in live_runs.py.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

AgentType = Literal["orchestrator", "builder", "verifier", "improvement", "supervisor"]

_AGENT_TYPES: dict[str, AgentType] = {
    "root-orchestrator": "orchestrator",
    "paper-understanding": "builder",
    "artifact-discovery": "builder",
    "environment-detective": "builder",
    "reproduction-planner": "builder",
    "baseline-implementation": "builder",
    "experiment-runner": "builder",
    "method-fidelity-verifier": "verifier",
    "environment-verifier": "verifier",
    "data-metrics-verifier": "verifier",
    "artifact-diff-verifier": "verifier",
    "supervisor-verifier": "supervisor",
    "improvement-orchestrator": "orchestrator",
    "improvement-path": "improvement",
}

_AGENT_LABELS: dict[str, str] = {
    "root-orchestrator": "Root Orchestrator",
    "paper-understanding": "Paper Understanding",
    "artifact-discovery": "Artifact Discovery",
    "environment-detective": "Environment Detective",
    "reproduction-planner": "Reproduction Planner",
    "baseline-implementation": "Baseline Implementation",
    "experiment-runner": "Experiment Runner",
    "method-fidelity-verifier": "Method Fidelity Verifier",
    "environment-verifier": "Environment Verifier",
    "data-metrics-verifier": "Data & Metrics Verifier",
    "artifact-diff-verifier": "Artifact Diff Verifier",
    "supervisor-verifier": "Supervisor Verifier",
    "improvement-orchestrator": "Improvement Orchestrator",
    "improvement-path": "Improvement Path",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DashboardEmitter:
    """Appends dashboard events as JSONL for the live frontend stream."""

    def __init__(self, project_id: str, runs_root: Path) -> None:
        self._project_id = project_id
        self._dir = Path(runs_root) / project_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "dashboard_events.jsonl"
        self._lock = threading.Lock()

    def _emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, default=str) + "\n"
        # Serialize concurrent appends — parallel RDR cluster execution can
        # race here and Windows append-to-text-file is not atomic.
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def _agent_node(
        self,
        agent_id: str,
        status: str,
        current_task: str,
        *,
        parent_id: str | None = "root-orchestrator",
        output_target_ids: list[str] | None = None,
        context_variables: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": agent_id,
            "label": _AGENT_LABELS.get(agent_id, agent_id),
            "type": _AGENT_TYPES.get(agent_id, "builder"),
            "status": status,
            "parentId": parent_id,
            "currentTask": current_task,
            "lastUpdated": _now(),
            "outputTargetIds": output_target_ids or [],
            "contextVariables": context_variables or [],
        }

    def agent_started(
        self,
        agent_id: str,
        current_task: str,
        *,
        parent_id: str | None = "root-orchestrator",
        output_target_ids: list[str] | None = None,
        context_variables: list[str] | None = None,
    ) -> None:
        self._emit({
            "event": "agent_started",
            "timestamp": _now(),
            "agentId": agent_id,
            "agent": self._agent_node(
                agent_id, "running", current_task,
                parent_id=parent_id,
                output_target_ids=output_target_ids,
                context_variables=context_variables,
            ),
        })

    def agent_completed(
        self,
        agent_id: str,
        current_task: str,
        *,
        parent_id: str | None = "root-orchestrator",
        output_target_ids: list[str] | None = None,
        context_variables: list[str] | None = None,
    ) -> None:
        self._emit({
            "event": "agent_completed",
            "timestamp": _now(),
            "agentId": agent_id,
            "agent": self._agent_node(
                agent_id, "completed", current_task,
                parent_id=parent_id,
                output_target_ids=output_target_ids,
                context_variables=context_variables,
            ),
        })

    def agent_failed(
        self,
        agent_id: str,
        error: str,
        *,
        parent_id: str | None = "root-orchestrator",
    ) -> None:
        self._emit({
            "event": "agent_failed",
            "timestamp": _now(),
            "agentId": agent_id,
            "agent": self._agent_node(
                agent_id, "failed", error,
                parent_id=parent_id,
            ),
        })

    def reasoning_step(
        self,
        agent_id: str,
        title: str,
        detail: str,
        *,
        step_type: str = "analysis",
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._emit({
            "event": "agent_reasoning_step",
            "timestamp": _now(),
            "agentId": agent_id,
            "agentLabel": _AGENT_LABELS.get(agent_id, agent_id),
            "stepType": step_type,
            "title": title,
            "detail": detail,
            "citations": citations or [],
        })

    def verification_gate(
        self,
        stage: str,
        status: str,
        detail: str,
    ) -> None:
        """stage: 'plan' | 'baseline' | 'improvement'
        status: 'pending' | 'running' | 'passed' | 'failed' | 'caveat'
        """
        self._emit({
            "event": "verification_gate_result",
            "timestamp": _now(),
            "stage": stage,
            "status": status,
            "detail": detail,
        })

    def primitive_call(
        self,
        primitive: str,
        status: str,
        *,
        args_summary: dict | None = None,
        result_summary: str | None = None,
        iteration: int | None = None,
        rubric_delta: float | None = None,
        coerced: bool = False,
    ) -> None:
        """Emit a `primitive_call` event (RLM Phase 2 — issue #61 schema).

        `status` is "start" | "ok" | "error". `iteration` is the root-loop
        index — a bare primitive wrapper cannot know it, so it is None here.
        Phase 3 (`run.py`, #60) supplies it via a custom `RLMLogger` subclass
        passed to `rlm.RLM`: `rlm` calls `logger.log(iteration)` once per loop
        (the verified per-iteration hook — `on_iteration_*` never fire), and
        that subclass stashes the index for the wrapper to read.
        `rubric_delta` is not applicable to a primitive call (always None).
        `coerced` is True when `wrap_primitive` auto-coerced one or more args
        to fix an obvious type mismatch before calling the primitive; the UI
        can surface "(auto-coerced)" on these events.
        """
        self._emit({
            "event": "primitive_call",
            "timestamp": _now(),
            "primitive": primitive,
            "status": status,
            "args_summary": args_summary or {},
            "result_summary": result_summary,
            "iteration": iteration,
            "rubric_delta": rubric_delta,
            "coerced": coerced,
        })

    def shared_state_updated(
        self,
        agent_id: str,
        change_type: str,
        title: str,
        detail: str,
        *,
        from_agent_id: str | None = None,
        to_agent_id: str | None = None,
    ) -> None:
        self._emit({
            "event": "shared_state_updated",
            "timestamp": _now(),
            "agentId": agent_id,
            "changeType": change_type,
            "title": title,
            "detail": detail,
            "fromAgentId": from_agent_id,
            "toAgentId": to_agent_id,
        })

    def context_enrichment(
        self,
        agent_id: str,
        variable_name: str,
        summary: str,
    ) -> None:
        self._emit({
            "event": "context_enrichment",
            "timestamp": _now(),
            "agentId": agent_id,
            "variableName": variable_name,
            "summary": summary,
        })

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Generic event emitter for rdr lifecycle events.

        Appends a ``dashboard_event``-shaped JSONL line so the SSE stream can
        forward arbitrary typed events to the frontend without coupling the rdr
        controller to the specific structured methods above.

        ``payload`` must NOT contain raw paper corpus text or full file contents
        (corpus-leak invariant). Pass counts, IDs, and scores only.
        """
        self._emit({
            "event": "dashboard_event",
            "timestamp": _now(),
            "event_type": event_type,
            "payload": payload,
        })

    def hermes_check_updated(
        self,
        title: str,
        summary: str,
        overall_status: str,
        checks: list[dict[str, Any]],
    ) -> None:
        self._emit({
            "event": "hermes_check_updated",
            "timestamp": _now(),
            "panel": {
                "title": title,
                "summary": summary,
                "overallStatus": overall_status,
                "checks": checks,
            },
        })
