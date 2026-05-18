"""Local Docker runtime backend."""

from __future__ import annotations

import asyncio
import io
import sys
import tarfile
from datetime import datetime, timezone
from types import SimpleNamespace
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


DEFAULT_BUILD_TIMEOUT_SECONDS = 1800


def _resolve_build_context(dockerfile_path: Path, context_dir: Path | None) -> tuple[Path, str]:
    """Resolve (build_context, dockerfile_arg) for a docker build.

    dockerfile_arg is the Dockerfile path relative to the context; falls back
    to building from the Dockerfile's own parent when it lives outside the
    requested context.
    """
    dockerfile = dockerfile_path.resolve()
    context = (context_dir or dockerfile.parent).resolve()
    try:
        return context, str(dockerfile.relative_to(context))
    except ValueError:
        return dockerfile.parent, dockerfile.name


async def build_image(
    dockerfile_path: Path,
    context_dir: Path,
    tag: str,
    *,
    timeout: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
    client: Any | None = None,
) -> tuple[bool, str, str]:
    """Build a Dockerfile (build-only — no container is run).

    Returns ``(ok, image_tag, error_text)``:
      - success            -> ``(True, tag, "")``
      - docker BuildError  -> ``(False, tag, <tail of the build log>)``  (the
        Dockerfile is broken — the caller may repair and retry)
      - build timeout      -> ``(False, tag, "Build exceeded ...")``

    Raises ``SandboxRuntimeError`` for *infrastructure* failures (Docker SDK
    missing, daemon unreachable, non-build API errors) — those are NOT the
    Dockerfile's fault and must not trigger a repair.
    """
    docker_client = client if client is not None else _make_docker_client()
    context, dockerfile_arg = _resolve_build_context(dockerfile_path, context_dir)
    build_kwargs: dict[str, Any] = {
        "path": str(context),
        "dockerfile": dockerfile_arg,
        "tag": tag,
        "rm": True,
    }
    try:
        await asyncio.wait_for(
            asyncio.to_thread(docker_client.images.build, **build_kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return (
            False,
            tag,
            f"Build exceeded {timeout:.0f}s wall-clock — likely too-heavy dependencies; "
            "split pip install into per-package layers or lighten the image.",
        )
    except Exception as exc:
        build_exc = _as_build_error(exc)
        if build_exc is not None:
            return (False, tag, _extract_build_error(exc))
        raise _map_docker_error(exc, RuntimeCauseKind.build_failed) from exc
    return (True, tag, "")


def _as_build_error(exc: Exception) -> Exception | None:
    """Return exc if it is a docker BuildError, else None (BuildError means
    the Dockerfile is at fault and is repairable)."""
    try:
        from docker.errors import BuildError  # type: ignore[import-untyped]
    except Exception:
        return None
    return exc if isinstance(exc, BuildError) else None


def _extract_build_error(exc: Exception) -> str:
    """Pull a useful error string from a docker BuildError: the message plus
    the tail (~40 lines) of the streamed build log.

    Defensive: BuildError.msg is sometimes a dict (docker SDK has shipped
    variants where the structured error payload leaks through as `.msg`)
    and a `stream`/`error` entry in build_log can be a dict too. Coerce
    every candidate to str so the returned value is always a string — the
    orchestrator's repair loop does `error_text.strip()`, which would
    AttributeError on a dict.
    """
    lines: list[str] = []
    for entry in getattr(exc, "build_log", None) or []:
        if isinstance(entry, dict):
            raw = entry.get("stream") or entry.get("error") or ""
        else:
            raw = entry
        text = str(raw).rstrip() if raw else ""
        if text:
            lines.append(text)
    tail = lines[-40:]
    msg = getattr(exc, "msg", None) or str(exc)
    if not isinstance(msg, str):
        msg = str(msg)
    return msg + "\n" + "\n".join(tail) if tail else msg


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
            self._client = _make_docker_client()
        return self._client

    @classmethod
    def verify_available(cls) -> None:
        """Fail fast when Docker mode cannot create SDK-backed sandboxes."""
        client = _make_docker_client()
        try:
            _ping_docker(client)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

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
        if config.platform:
            run_kwargs["platform"] = config.platform
        if config.memory_limit:
            run_kwargs["mem_limit"] = config.memory_limit
        if config.cpus:
            run_kwargs["nano_cpus"] = int(config.cpus * 1_000_000_000)
        if config.network_disabled:
            run_kwargs["network_mode"] = "none"
        if config.gpu_mode in {"prefer", "max"}:
            run_kwargs["device_requests"] = [_gpu_device_request()]
            run_kwargs["environment"] = {
                **config.environment,
                "REPROLAB_GPU_MODE": config.gpu_mode,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }

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
                    ["/bin/sh", "-lc", command],
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

        context, dockerfile_arg = _resolve_build_context(config.dockerfile_path, config.build_context)
        tag = config.image or f"reprolab/{config.project_id}:{config.run_id}"
        try:
            build_kwargs: dict[str, Any] = {
                "path": str(context),
                "dockerfile": dockerfile_arg,
                "tag": tag,
                "rm": True,
            }
            if config.platform:
                build_kwargs["platform"] = config.platform
            await asyncio.to_thread(
                self.client.images.build,
                **build_kwargs,
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


def _gpu_device_request() -> Any:
    try:
        from docker.types import DeviceRequest  # type: ignore[import-untyped]

        return DeviceRequest(count=-1, capabilities=[["gpu"]])
    except Exception:  # pragma: no cover - used by tests without Docker SDK
        return SimpleNamespace(count=-1, capabilities=[["gpu"]])


def ensure_local_docker_available(client: Any | None = None) -> None:
    """Validate the Python Docker SDK and daemon before starting a run."""
    if client is not None:
        _ping_docker(client)
        return
    LocalDockerBackend.verify_available()


def _make_docker_client() -> Any:
    try:
        import docker  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            (
                "Python Docker SDK is not installed for this environment "
                f"({sys.executable}). Install backend dependencies with "
                "`python -m pip install -r backend/requirements.txt` or "
                "`python -m pip install -e .` before using Docker sandbox mode."
            ),
            retryable=False,
            detail={"python": sys.executable, "missing_package": "docker"},
        ) from exc
    try:
        return docker.from_env()
    except Exception as exc:  # pragma: no cover - docker-specific branch
        raise _docker_daemon_unavailable(exc) from exc


def _ping_docker(client: Any) -> None:
    ping = getattr(client, "ping", None)
    if not callable(ping):
        return
    try:
        ping()
    except Exception as exc:  # pragma: no cover - docker-specific branch
        raise _docker_daemon_unavailable(exc) from exc


def _docker_daemon_unavailable(exc: Exception) -> SandboxRuntimeError:
    return SandboxRuntimeError(
        RuntimeCauseKind.backend_unavailable,
        (
            "Docker daemon is not reachable from this Python environment. "
            "Start Docker Desktop or Docker Engine, then verify `docker run hello-world`. "
            f"Original error: {exc}"
        ),
        retryable=True,
    )


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


__all__ = [
    "DEFAULT_BUILD_TIMEOUT_SECONDS",
    "LocalDockerBackend",
    "build_image",
    "ensure_local_docker_available",
]
