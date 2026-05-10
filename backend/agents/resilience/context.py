"""Attempt state and continuation prompt rendering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from backend.agents.resilience.cost import RunCostLedger
from backend.agents.runtime.base import ProviderName


@dataclass
class AttemptRecord:
    attempt_index: int
    provider: ProviderName
    model: str
    started_at: datetime
    finished_at: datetime
    outcome: Literal["success", "fallback", "salvaged", "failed"]
    failure_kind: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    estimated_usd: float | None = None
    decision_note: str = ""
    next_provider: ProviderName | None = None

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["finished_at"] = self.finished_at.isoformat()
        return data


@dataclass
class AttemptContext:
    agent_id: str
    base_prompt: str
    cwd: Path
    cost_ledger: RunCostLedger
    attempts: list[AttemptRecord] = field(default_factory=list)
    partial_output_by_provider: dict[ProviderName, str] = field(default_factory=dict)

    def continuation_prompt(self, *, target_provider: ProviderName) -> str:
        if not self.attempts:
            return self.base_prompt
        prior_output = self._last_partial_output()
        if not prior_output:
            return self.base_prompt
        prior = self.attempts[-1]
        partial = compact_partial_output(prior_output)
        return (
            "[FALLBACK CONTINUATION]\n"
            f"A previous attempt by {prior.provider} produced the following partial "
            f"output before failing with {prior.failure_kind or 'unknown_failure'}:\n"
            "---\n"
            f"{partial}\n"
            "---\n"
            f"The on-disk workspace at {self.cwd} reflects any files the prior "
            "attempt wrote. Use this as a starting point. Verify what is already "
            "on disk and complete the task without repeating finished work.\n\n"
            "[ORIGINAL TASK]\n"
            f"{self.base_prompt}"
        )

    def record_partial(self, provider: ProviderName, output: str) -> None:
        if output.strip():
            self.partial_output_by_provider[provider] = output

    def latest_partial_output(self) -> str:
        return self._last_partial_output()

    def _last_partial_output(self) -> str:
        for record in reversed(self.attempts):
            partial = self.partial_output_by_provider.get(record.provider, "")
            if partial.strip():
                return partial
        return ""


@dataclass(frozen=True)
class AgentRunResult:
    output_text: str
    trace_text: str
    tool_calls: list[str]
    elapsed_seconds: float


def compact_partial_output(text: str, *, limit_chars: int = 8192) -> str:
    raw = text.strip()
    if not raw:
        return ""
    json_block = _try_extract_json(raw)
    if json_block is not None:
        raw = json.dumps(json_block, sort_keys=True, separators=(",", ":"))
    if len(raw) <= limit_chars:
        return raw
    tail = raw[-limit_chars:]
    return f"[truncated to last {limit_chars} chars]\n{tail}"


def _try_extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace_start = text.find("{")
    if brace_start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


__all__ = [
    "AgentRunResult",
    "AttemptContext",
    "AttemptRecord",
    "compact_partial_output",
]
