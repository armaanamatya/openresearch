"""Local Docker runtime backend."""

from __future__ import annotations

import asyncio
import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


class LocalDockerBackend(RuntimeBackend):
    """RuntimeBackend implementation backed by the Docker SDK.

    A Docker client can be injected for unit tests. When no client is
    provided, the backend imports `docker` lazily and connects with
    `docker.from_env()`.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import docker  # type: ignore[import-untyped]
            except ImportError as exc:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    "Docker SDK is not installed. Install the 'docker' Python package.",
                    retryable=False,
                ) from exc
            self._client = docker.from_env()
        return self._client

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        project_root = config.project_root.resolve()
        artifact_root = config.resolved_artifact_root().resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)

        image = await self._ensure_image(config)
        name = _container_name(config)
        labels = {
            "reprolab.project_id": config.project_id,
            "reprolab.run_id": config.run_id,
            **config.labels,
        }
        volumes = {
            str(project_root): {
                "bind": config.workdir,
                "mode": "ro" if config.readonly_project else "rw",
            },
            str(artifact_root): {"bind": config.artifacts_dir, "mode": "rw"},
        }
        run_kwargs: dict[str, Any] = {
            "command": list(config.keepalive_command),
            "detach": True,
            "environment": config.environment,
            "labels": labels,
            "name": name,
            "volumes": volumes,
            "working_dir": config.workdir,
        }
        if config.network_disabled:
            run_kwargs["network_mode"] = "none"

        try:
            container = await asyncio.to_thread(
                self.client.containers.run,
                image,
                **run_kwargs,
            )
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.backend_unavailable) from exc

        sandbox_id = str(getattr(container, "id", name))
        return Sandbox(
            sandbox_id=sandbox_id,
            name=name,
            image=image,
            config=config,
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        started_at = datetime.now(timezone.utc)
        container = await self._get_container(sandbox)
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    container.exec_run,
                    command,
                    stdout=True,
                    stderr=True,
                    demux=True,
                    workdir=sandbox.config.workdir,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            finished_at = datetime.now(timezone.utc)
            return ExecResult(
                command=command,
                exit_code=None,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=(finished_at - started_at).total_seconds(),
                timed_out=True,
                cause_kind=RuntimeCauseKind.exec_timeout,
                stderr=f"Command timed out after {timeout} seconds.",
            )
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.command_failed) from exc

        finished_at = datetime.now(timezone.utc)
        exit_code, stdout, stderr = _decode_exec_result(raw)
        return ExecResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            cause_kind=None if exit_code == 0 else RuntimeCauseKind.command_failed,
        )

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        container = await self._get_container(sandbox)
        try:
            chunks, _stat = await asyncio.to_thread(container.get_archive, path)
            archive = b"".join(chunks)
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        extracted = tar.extractfile(member)
                        return extracted.read() if extracted is not None else b""
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.copy_failed) from exc
        raise SandboxRuntimeError(
            RuntimeCauseKind.copy_failed,
            f"No file payload found while copying {path!r} out of sandbox.",
        )

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        container = await self._get_container(sandbox)
        parent, name = _split_container_path(path)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            tar.addfile(info, io.BytesIO(data))
        try:
            ok = await asyncio.to_thread(container.put_archive, parent, archive.getvalue())
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.copy_failed) from exc
        if ok is False:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Docker refused copy into sandbox path {path!r}.",
            )

    async def destroy(self, sandbox: Sandbox) -> None:
        container = await self._get_container(sandbox)
        try:
            await asyncio.to_thread(container.stop, timeout=3)
        except Exception:
            pass
        try:
            await asyncio.to_thread(container.remove, force=True)
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.backend_unavailable) from exc

    async def _ensure_image(self, config: SandboxConfig) -> str:
        if config.dockerfile_path is None:
            if not config.image:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.image_not_found,
                    "SandboxConfig requires either image or dockerfile_path.",
                )
            return config.image

        dockerfile = config.dockerfile_path.resolve()
        context = (config.build_context or dockerfile.parent).resolve()
        try:
            dockerfile_arg = str(dockerfile.relative_to(context))
        except ValueError:
            context = dockerfile.parent
            dockerfile_arg = dockerfile.name
        tag = config.image or f"reprolab/{config.project_id}:{config.run_id}"
        try:
            await asyncio.to_thread(
                self.client.images.build,
                path=str(context),
                dockerfile=dockerfile_arg,
                tag=tag,
                rm=True,
            )
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.build_failed) from exc
        return tag

    async def _get_container(self, sandbox: Sandbox) -> Any:
        try:
            return await asyncio.to_thread(self.client.containers.get, sandbox.sandbox_id)
        except Exception as exc:  # pragma: no cover - docker-specific branches
            raise _map_docker_error(exc, RuntimeCauseKind.backend_unavailable) from exc


def _container_name(config: SandboxConfig) -> str:
    safe_project = _safe_name(config.project_id)
    safe_run = _safe_name(config.run_id)
    return f"reprolab-{safe_project}-{safe_run}"


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:48]


def _split_container_path(path: str) -> tuple[str, str]:
    posix = PurePosixPath(path)
    if not posix.is_absolute():
        raise SandboxRuntimeError(
            RuntimeCauseKind.copy_failed,
            f"Container path must be absolute, got {path!r}.",
        )
    name = posix.name
    if not name or name in {".", ".."}:
        raise SandboxRuntimeError(
            RuntimeCauseKind.copy_failed,
            f"Container path must point to a file, got {path!r}.",
        )
    parent = str(posix.parent)
    return parent, name


def _decode_exec_result(raw: Any) -> tuple[int | None, str, str]:
    exit_code = getattr(raw, "exit_code", None)
    output = getattr(raw, "output", None)
    if output is None and isinstance(raw, tuple) and len(raw) == 2:
        exit_code, output = raw

    stdout_bytes: bytes | None
    stderr_bytes: bytes | None
    if isinstance(output, tuple):
        stdout_bytes, stderr_bytes = output
    else:
        stdout_bytes, stderr_bytes = output, None

    return (
        exit_code,
        _decode_bytes(stdout_bytes),
        _decode_bytes(stderr_bytes),
    )


def _decode_bytes(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _map_docker_error(exc: Exception, default: RuntimeCauseKind) -> SandboxRuntimeError:
    text = str(exc)
    lower = text.lower()
    cause = default
    if "not found" in lower or "no such image" in lower:
        cause = RuntimeCauseKind.image_not_found
    elif "network" in lower:
        cause = RuntimeCauseKind.network_unavailable
    elif "oom" in lower or "out of memory" in lower:
        cause = RuntimeCauseKind.oom_killed
    return SandboxRuntimeError(cause, text, retryable=cause != RuntimeCauseKind.build_failed)


__all__ = ["LocalDockerBackend"]
