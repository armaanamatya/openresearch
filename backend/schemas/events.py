"""Event stream payload schemas for frontend integration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    agent_started = "agent_started"
    agent_completed = "agent_completed"
    agent_failed = "agent_failed"
    agent_reasoning_step = "agent_reasoning_step"
    rlm_query_executed = "rlm_query_executed"
    semantic_search_executed = "semantic_search_executed"
    shared_state_updated = "shared_state_updated"
    verification_gate_result = "verification_gate_result"
    approval_requested = "approval_requested"
    approval_resolved = "approval_resolved"
    context_enrichment = "context_enrichment"
    task_status_changed = "task_status_changed"


class EventPayload(BaseModel):
    event: EventType
    agent_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
