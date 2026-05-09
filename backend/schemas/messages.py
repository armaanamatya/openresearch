"""Agent message schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    message_id: str
    agent_id: str
    content: str
    structured_outputs: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
