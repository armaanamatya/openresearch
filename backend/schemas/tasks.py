"""Agent task schemas and lifecycle enums."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    created = "created"
    context_prepared = "context_prepared"
    running = "running"
    artifact_submitted = "artifact_submitted"
    verification_pending = "verification_pending"
    verified = "verified"
    failed = "failed"
    blocked_requires_human = "blocked_requires_human"


class FailureSubstatus(str, Enum):
    # Install / dependency
    failed_install = "failed_install"
    failed_dependency_resolution = "failed_dependency_resolution"
    # Data
    failed_data_download = "failed_data_download"
    failed_dataset_validation = "failed_dataset_validation"
    # Build
    failed_docker_build = "failed_docker_build"
    failed_smoke_test = "failed_smoke_test"
    # Training / evaluation
    failed_training = "failed_training"
    failed_evaluation = "failed_evaluation"
    failed_metric_validation = "failed_metric_validation"
    failed_plot_generation = "failed_plot_generation"
    # Remote
    failed_remote_sync = "failed_remote_sync"
    failed_remote_execution = "failed_remote_execution"
    # Resource
    timeout = "timeout"
    out_of_memory = "out_of_memory"
    out_of_disk = "out_of_disk"
    # Blocked
    blocked_approval = "blocked_approval"
    blocked_license = "blocked_license"
    blocked_credentials = "blocked_credentials"
    blocked_unavailable_dataset = "blocked_unavailable_dataset"
    inconclusive_budget = "inconclusive_budget"


class AgentTask(BaseModel):
    task_id: str
    agent_type: str
    status: TaskStatus
    parent_task_id: Optional[str] = None
    failure_substatus: Optional[FailureSubstatus] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
