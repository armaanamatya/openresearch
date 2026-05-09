"""Run schemas for baseline and improvement runs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RunType(str, Enum):
    baseline = "baseline"
    improvement = "improvement"


class Run(BaseModel):
    run_id: str
    run_type: RunType
    status: str
    task_id: str
    parent_run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
