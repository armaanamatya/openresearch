"""Failure diagnosis models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


FailureKind = Literal[
    "failed_install",
    "failed_dependency_resolution",
    "failed_data_download",
    "failed_dataset_validation",
    "failed_docker_build",
    "failed_smoke_test",
    "failed_training",
    "failed_evaluation",
    "failed_metric_validation",
    "failed_plot_generation",
    "failed_remote_sync",
    "failed_remote_execution",
    "timeout",
    "out_of_memory",
    "out_of_disk",
    "blocked_approval",
    "blocked_license",
    "blocked_credentials",
    "blocked_unavailable_dataset",
    "inconclusive_budget",
]


class FailureEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    failure_id: str
    project_id: str
    stage: str
    kind: FailureKind
    command: str = ""
    exit_code: int | None = None
    retryable: bool = False
    suspected_cause: str = ""
    recommended_next_step: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    artifact_refs: tuple[str, ...] = ()
    created_at: datetime


__all__ = ["FailureEvent", "FailureKind"]
