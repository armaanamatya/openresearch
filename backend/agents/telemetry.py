"""Telemetry helpers for Claude Agent SDK invocations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentInvocationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    model: str = ""
    started_at: str
    finished_at: str
    duration_seconds: float
    message_count: int
    output_chars: int
    success: bool
    error_message: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)


class AgentTelemetryRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: AgentInvocationRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(), sort_keys=True) + "\n")


def coerce_usage(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if hasattr(raw, "__dict__"):
        return dict(raw.__dict__)
    return {"raw": str(raw)}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "AgentInvocationRecord",
    "AgentTelemetryRecorder",
    "coerce_usage",
    "utc_now_iso",
]
