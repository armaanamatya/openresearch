"""Local process runtime backend.

This backend executes commands on the host machine using the same artifact
directory contract as Docker. It is useful for fast iteration and environments
where Docker is unavailable, but it is not an isolation boundary.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


class LocalProcessBackend(RuntimeBackend):
    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        project_root = config.project_root.resolve()
        artifact_root = config.resolved_artifact_root().resolve()
        if not project_root.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Local project root does not exist: {project_root}",
            )
        artifact_root.mkdir(parents=True, exist_ok=True)
        return Sandbox(
            sandbox_id=f"local-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}",
            name=f"local-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}",
            image="local-process",
            config=config,
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        started_at = datetime.now(timezone.utc)
        env = {
            **os.environ,
            **sandbox.config.environment,
        }
        if sandbox.config.gpu_device_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(sandbox.config.gpu_device_ids)
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(sandbox.config.project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_raw, stderr_raw = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                finished_at = datetime.now(timezone.utc)
                return ExecResult(
                    command=command,
                    exit_code=None,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds.",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=(finished_at - started_at).total_seconds(),
                    timed_out=True,
                    cause_kind=RuntimeCauseKind.exec_timeout,
                )
        except Exception as exc:  # pragma: no cover - subprocess platform edge
            raise SandboxRuntimeError(
                RuntimeCauseKind.command_failed,
                str(exc),
                retryable=False,
            ) from exc

        finished_at = datetime.now(timezone.utc)
        exit_code = process.returncode
        return ExecResult(
            command=command,
            exit_code=exit_code,
            stdout=_decode(stdout_raw),
            stderr=_decode(stderr_raw),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            cause_kind=None if exit_code == 0 else RuntimeCauseKind.command_failed,
        )

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        host_path = _map_containerish_path(sandbox.config, path)
        try:
            return await asyncio.to_thread(host_path.read_bytes)
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not copy local file {host_path}: {exc}",
            ) from exc

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        host_path = _map_containerish_path(sandbox.config, path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(host_path.write_bytes, data)
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not write local file {host_path}: {exc}",
            ) from exc

    async def destroy(self, sandbox: Sandbox) -> None:
        return None


def _map_containerish_path(config: SandboxConfig, path: str) -> Path:
    posix = PurePosixPath(path)
    if not posix.is_absolute():
        return (config.project_root / path).resolve()

    workdir = PurePosixPath(config.workdir)
    artifacts_dir = PurePosixPath(config.artifacts_dir)
    if posix == workdir or workdir in posix.parents:
        return (config.project_root / posix.relative_to(workdir).as_posix()).resolve()
    if posix == artifacts_dir or artifacts_dir in posix.parents:
        return (
            config.resolved_artifact_root() / posix.relative_to(artifacts_dir).as_posix()
        ).resolve()
    raise SandboxRuntimeError(
        RuntimeCauseKind.copy_failed,
        f"Path {path!r} is outside local runtime mounts.",
    )


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:48]


__all__ = ["LocalProcessBackend"]
