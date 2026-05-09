"""Runtime backend contracts for sandboxed command execution.

The interface is intentionally small and async. Concrete backends own IO
and process/container lifecycle; callers receive typed sandbox and command
results that can be persisted as provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeCauseKind(str, Enum):
    image_not_found = "image_not_found"
    network_unavailable = "network_unavailable"
    oom_killed = "oom_killed"
    exec_timeout = "exec_timeout"
    signal_terminated = "signal_terminated"
    build_failed = "build_failed"
    copy_failed = "copy_failed"
    backend_unavailable = "backend_unavailable"
    command_failed = "command_failed"


class SandboxConfig(BaseModel):
    """Configuration for one sandbox.

    `project_root` is mounted at `/work`. `artifact_root` is mounted at
    `/artifacts` and should be the only writable artifact surface for normal
    experiment runs.
    """

    model_config = ConfigDict(frozen=True)

    project_id: str
    run_id: str
    image: str = ""
    project_root: Path
    artifact_root: Path | None = None
    dockerfile_path: Path | None = None
    build_context: Path | None = None
    workdir: str = "/work"
    artifacts_dir: str = "/artifacts"
    readonly_project: bool = False
    network_disabled: bool = True
    environment: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    keepalive_command: tuple[str, ...] = ("sleep", "infinity")

    def resolved_artifact_root(self) -> Path:
        return self.artifact_root or self.project_root / "artifacts"


class Sandbox(BaseModel):
    """A live sandbox returned by a RuntimeBackend."""

    model_config = ConfigDict(frozen=True)

    sandbox_id: str
    name: str
    image: str
    config: SandboxConfig
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecResult(BaseModel):
    """Result of a command executed inside a sandbox."""

    model_config = ConfigDict(frozen=True)

    command: str
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    timed_out: bool = False
    cause_kind: RuntimeCauseKind | None = None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.cause_kind is None


class SandboxRuntimeError(RuntimeError):
    """Typed runtime failure suitable for metrics and retry policy."""

    def __init__(
        self,
        cause_kind: RuntimeCauseKind,
        message: str,
        *,
        retryable: bool = False,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.cause_kind = cause_kind
        self.retryable = retryable
        self.detail = detail or {}


class RuntimeBackend(ABC):
    """Abstract interface for sandboxed code execution."""

    @abstractmethod
    async def create_sandbox(self, config: SandboxConfig) -> Sandbox: ...

    @abstractmethod
    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult: ...

    @abstractmethod
    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes: ...

    @abstractmethod
    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None: ...

    @abstractmethod
    async def destroy(self, sandbox: Sandbox) -> None: ...


__all__ = [
    "ExecResult",
    "RuntimeBackend",
    "RuntimeCauseKind",
    "Sandbox",
    "SandboxConfig",
    "SandboxRuntimeError",
]
