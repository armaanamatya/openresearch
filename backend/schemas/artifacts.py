"""Artifact schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    metrics = "metrics"
    logs = "logs"
    plots = "plots"
    dockerfile = "dockerfile"
    diff = "diff"
    report = "report"
    commands = "commands"
    assumptions = "assumptions"
    provenance = "provenance"
    environment_lock = "environment_lock"


class Artifact(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    run_id: str
    file_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
