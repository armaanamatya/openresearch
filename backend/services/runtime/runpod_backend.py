"""Runpod runtime backend for remote GPU experiment execution."""

from __future__ import annotations

import asyncio
import io
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


@dataclass(frozen=True)
class _RunpodConnection:
    pod_id: str
    public_ip: str
    ssh_port: int
    remote_base: str
    remote_workdir: str
    remote_artifacts_dir: str


class RunpodBackend(RuntimeBackend):
    """RuntimeBackend implementation backed by a Runpod GPU Pod.

    Runpod requires a registry-accessible image. This backend defaults to an
    official PyTorch image, uploads the generated project files over SFTP, runs
    commands through SSH, and syncs `/artifacts` back to the local run folder.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base_url: str = "https://rest.runpod.io/v1",
        image_name: str = DEFAULT_RUNPOD_IMAGE,
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        gpu_count: int = 1,
        cloud_type: str = "SECURE",
        container_disk_gb: int = 50,
        volume_gb: int = 20,
        volume_mount_path: str = "/workspace",
        network_volume_id: str = "",
        data_center_ids: list[str] | None = None,
        ssh_key_path: str | Path | None = None,
        ssh_public_key: str = "",
        ssh_user: str = "root",
        boot_timeout_seconds: int = 900,
        delete_on_destroy: bool = True,
        bootstrap_command: str = "",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("REPROLAB_RUNPOD_API_KEY")
            or os.environ.get("RUNPOD_API_KEY")
            or ""
        ).strip()
        self.api_base_url = api_base_url.rstrip("/")
        self.image_name = image_name or DEFAULT_RUNPOD_IMAGE
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.cloud_type = cloud_type
        self.container_disk_gb = container_disk_gb
        self.volume_gb = volume_gb
        self.volume_mount_path = volume_mount_path.rstrip("/") or "/workspace"
        self.network_volume_id = network_volume_id.strip()
        self.data_center_ids = data_center_ids or []
        self.ssh_key_path = Path(ssh_key_path or "~/.ssh/id_ed25519").expanduser()
        self.ssh_public_key = ssh_public_key.strip()
        self.ssh_user = ssh_user
        self.boot_timeout_seconds = boot_timeout_seconds
        self.delete_on_destroy = delete_on_destroy
        self.bootstrap_command = bootstrap_command.strip()
        self._connections: dict[str, _RunpodConnection] = {}
        self._ssh_clients: dict[str, Any] = {}

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        if not self.api_key:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "Runpod API key is missing. Set REPROLAB_RUNPOD_API_KEY or RUNPOD_API_KEY.",
            )
        if not self.ssh_key_path.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod SSH private key not found: {self.ssh_key_path}",
            )
        project_root = config.project_root.resolve()
        if not project_root.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod project root does not exist: {project_root}",
            )

        config.resolved_artifact_root().mkdir(parents=True, exist_ok=True)
        image = self.image_name or config.image
        pod = await self._create_pod(config, image)
        pod_id = str(pod["id"])
        try:
            ready = await self._wait_for_pod_ssh(pod_id)
            remote_base = _join_posix(
                self.volume_mount_path,
                "reprolab",
                _safe_name(config.project_id),
                _safe_name(config.run_id),
            )
            connection = _RunpodConnection(
                pod_id=pod_id,
                public_ip=ready["public_ip"],
                ssh_port=ready["ssh_port"],
                remote_base=remote_base,
                remote_workdir=_join_posix(remote_base, "work"),
                remote_artifacts_dir=_join_posix(remote_base, "artifacts"),
            )
            self._connections[pod_id] = connection
            await self._prepare_remote_workspace(config, connection)
            return Sandbox(
                sandbox_id=pod_id,
                name=str(pod.get("name") or _pod_name(config)),
                image=image,
                config=config,
            )
        except Exception:
            await self._delete_pod_quietly(pod_id)
            raise

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        started_at = datetime.now(timezone.utc)
        try:
            conn = await self._ssh(sandbox.sandbox_id)
            script = _remote_command(sandbox.config, command)
            result = await asyncio.wait_for(
                conn.run(f"/bin/bash -lc {_shell_quote(script)}", check=False),
                timeout=timeout,
            )
            await self._sync_artifacts_to_host(sandbox)
        except TimeoutError:
            finished_at = datetime.now(timezone.utc)
            await self._sync_artifacts_to_host_quietly(sandbox)
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
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.command_failed,
                f"Runpod SSH command failed: {exc}",
            ) from exc

        finished_at = datetime.now(timezone.utc)
        exit_code = int(getattr(result, "returncode", 1))
        return ExecResult(
            command=command,
            exit_code=exit_code,
            stdout=_coerce_text(getattr(result, "stdout", "")),
            stderr=_coerce_text(getattr(result, "stderr", "")),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            cause_kind=None if exit_code == 0 else RuntimeCauseKind.command_failed,
        )

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        remote_path = self._map_remote_path(sandbox, path)
        try:
            conn = await self._ssh(sandbox.sandbox_id)
            async with conn.start_sftp_client() as sftp:
                async with sftp.open(remote_path, "rb") as handle:
                    return await handle.read()
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not copy Runpod file {remote_path}: {exc}",
            ) from exc

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        remote_path = self._map_remote_path(sandbox, path)
        try:
            conn = await self._ssh(sandbox.sandbox_id)
            async with conn.start_sftp_client() as sftp:
                await sftp.makedirs(str(PurePosixPath(remote_path).parent), exist_ok=True)
                async with sftp.open(remote_path, "wb") as handle:
                    await handle.write(data)
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not write Runpod file {remote_path}: {exc}",
            ) from exc

    async def destroy(self, sandbox: Sandbox) -> None:
        await self._sync_artifacts_to_host_quietly(sandbox)
        conn = self._ssh_clients.pop(sandbox.sandbox_id, None)
        if conn is not None:
            conn.close()
            await conn.wait_closed()
        self._connections.pop(sandbox.sandbox_id, None)
        if self.delete_on_destroy:
            await self._delete_pod(sandbox.sandbox_id)

    async def _create_pod(self, config: SandboxConfig, image: str) -> dict[str, Any]:
        public_key = self._public_key()
        env = dict(config.environment)
        if public_key:
            env.setdefault("PUBLIC_KEY", public_key)
            env.setdefault("SSH_PUBLIC_KEY", public_key)
        payload: dict[str, Any] = {
            "name": _pod_name(config),
            "cloudType": self.cloud_type,
            "computeType": "GPU",
            "imageName": image,
            "gpuTypeIds": [self.gpu_type],
            "gpuCount": self.gpu_count,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": self.volume_mount_path,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": env,
        }
        # Official RunPod images (runpod/*) already handle SSH via PUBLIC_KEY
        # env var. Only inject a custom start command for third-party images.
        if not image.startswith("runpod/"):
            payload["dockerStartCmd"] = [
                "bash",
                "-lc",
                _runpod_start_command(),
            ]
        if self.network_volume_id:
            payload["networkVolumeId"] = self.network_volume_id
        if self.data_center_ids:
            payload["dataCenterIds"] = self.data_center_ids
        return await self._request_json("POST", "/pods", json=payload)

    async def _wait_for_pod_ssh(self, pod_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.boot_timeout_seconds
        last_error = ""
        while asyncio.get_running_loop().time() < deadline:
            pod = await self._request_json("GET", f"/pods/{pod_id}")
            public_ip = pod.get("publicIp") or pod.get("publicIP")
            port = _ssh_port(pod.get("portMappings") or {})
            if public_ip and port:
                try:
                    conn = await self._connect_ssh(str(public_ip), int(port))
                    self._ssh_clients[pod_id] = conn
                    return {"public_ip": str(public_ip), "ssh_port": int(port)}
                except Exception as exc:
                    last_error = str(exc)
            await asyncio.sleep(10)
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Runpod pod {pod_id} did not become SSH-ready before timeout. {last_error}",
            retryable=True,
        )

    async def _prepare_remote_workspace(
        self,
        config: SandboxConfig,
        connection: _RunpodConnection,
    ) -> None:
        conn = await self._ssh(connection.pod_id)
        setup = "\n".join(
            [
                f"mkdir -p {_shell_quote(connection.remote_workdir)}",
                f"mkdir -p {_shell_quote(connection.remote_artifacts_dir)}",
                _replace_path_with_symlink(config.workdir, connection.remote_workdir),
                _replace_path_with_symlink(config.artifacts_dir, connection.remote_artifacts_dir),
            ]
        )
        result = await conn.run(f"/bin/bash -lc {_shell_quote(setup)}", check=False)
        if result.returncode != 0:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Could not prepare Runpod workspace: {result.stderr}",
            )
        async with conn.start_sftp_client() as sftp:
            await self._upload_directory(sftp, config.project_root, connection.remote_workdir)
        if self.bootstrap_command:
            bootstrap = await conn.run(
                f"/bin/bash -lc {_shell_quote(_remote_command(config, self.bootstrap_command))}",
                check=False,
            )
            if bootstrap.returncode != 0:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.build_failed,
                    bootstrap.stderr or bootstrap.stdout or "Runpod bootstrap command failed.",
                )

    async def _upload_directory(
        self,
        sftp: Any,
        local_root: Path,
        remote_root: str,
    ) -> None:
        await sftp.makedirs(remote_root, exist_ok=True)
        for local_path in sorted(local_root.rglob("*")):
            rel = local_path.relative_to(local_root).as_posix()
            remote_path = _join_posix(remote_root, rel)
            if local_path.is_dir():
                await sftp.makedirs(remote_path, exist_ok=True)
            elif local_path.is_file():
                await sftp.makedirs(str(PurePosixPath(remote_path).parent), exist_ok=True)
                await sftp.put(str(local_path), remote_path)

    async def _sync_artifacts_to_host(self, sandbox: Sandbox) -> None:
        connection = self._connections[sandbox.sandbox_id]
        conn = await self._ssh(sandbox.sandbox_id)
        command = f"test -d {_shell_quote(connection.remote_artifacts_dir)} && tar -C {_shell_quote(connection.remote_artifacts_dir)} -cf - ."
        result = await conn.run(command, check=False, encoding=None)
        if result.returncode != 0 or not result.stdout:
            return
        if isinstance(result.stdout, bytes):
            data = result.stdout
        elif isinstance(result.stdout, bytearray):
            data = bytes(result.stdout)
        else:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                "Runpod artifact archive was not returned as bytes.",
            )
        _extract_artifact_tar(data, sandbox.config.resolved_artifact_root())

    async def _sync_artifacts_to_host_quietly(self, sandbox: Sandbox) -> None:
        try:
            await self._sync_artifacts_to_host(sandbox)
        except Exception:
            return

    async def _ssh(self, pod_id: str) -> Any:
        conn = self._ssh_clients.get(pod_id)
        if conn is not None and not conn.is_closed():
            return conn
        connection = self._connections[pod_id]
        conn = await self._connect_ssh(connection.public_ip, connection.ssh_port)
        self._ssh_clients[pod_id] = conn
        return conn

    async def _connect_ssh(self, host: str, port: int) -> Any:
        try:
            import asyncssh
        except ImportError as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "asyncssh is not installed. Install the 'asyncssh' Python package.",
            ) from exc
        return await asyncssh.connect(
            host,
            port=port,
            username=self.ssh_user,
            client_keys=[str(self.ssh_key_path)],
            known_hosts=None,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(
                base_url=self.api_base_url,
                headers=headers,
                timeout=60,
            ) as client:
                response = await client.request(method, path, json=json)
                response.raise_for_status()
                if not response.content:
                    return {}
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object response, got {type(payload).__name__}")
                return payload
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod API request failed: {exc}",
                retryable=True,
            ) from exc

    async def _delete_pod(self, pod_id: str) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(
                base_url=self.api_base_url,
                headers=headers,
                timeout=60,
            ) as client:
                response = await client.delete(f"/pods/{pod_id}")
                response.raise_for_status()
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod pod deletion failed for {pod_id}: {exc}",
                retryable=True,
            ) from exc

    async def _delete_pod_quietly(self, pod_id: str) -> None:
        try:
            await self._delete_pod(pod_id)
        except Exception:
            return

    def _public_key(self) -> str:
        if self.ssh_public_key:
            return self.ssh_public_key
        public_key_path = Path(f"{self.ssh_key_path}.pub")
        if public_key_path.exists():
            return public_key_path.read_text(encoding="utf-8").strip()
        return ""

    def _map_remote_path(self, sandbox: Sandbox, path: str) -> str:
        connection = self._connections[sandbox.sandbox_id]
        posix = PurePosixPath(path)
        if not posix.is_absolute():
            return _join_posix(connection.remote_workdir, path)

        workdir = PurePosixPath(sandbox.config.workdir)
        artifacts_dir = PurePosixPath(sandbox.config.artifacts_dir)
        if posix == workdir or workdir in posix.parents:
            return _join_posix(
                connection.remote_workdir,
                posix.relative_to(workdir).as_posix(),
            )
        if posix == artifacts_dir or artifacts_dir in posix.parents:
            return _join_posix(
                connection.remote_artifacts_dir,
                posix.relative_to(artifacts_dir).as_posix(),
            )
        raise SandboxRuntimeError(
            RuntimeCauseKind.copy_failed,
            f"Path {path!r} is outside Runpod runtime mounts.",
        )


def _runpod_start_command() -> str:
    return (
        "command -v sshd >/dev/null || "
        "(apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server); "
        "mkdir -p /root/.ssh /run/sshd; "
        "if [ -n \"$PUBLIC_KEY\" ]; then printf '%s\\n' \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; fi; "
        "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true; "
        "service ssh start 2>/dev/null || /usr/sbin/sshd; "
        "sleep infinity"
    )


def _remote_command(config: SandboxConfig, command: str) -> str:
    exports = "; ".join(
        f"export {key}={_shell_quote(value)}"
        for key, value in sorted(config.environment.items())
        if key.replace("_", "").isalnum()
    )
    prefix = f"{exports}; " if exports else ""
    return f"{prefix}cd {_shell_quote(config.workdir)} && {command}"


def _replace_path_with_symlink(path: str, target: str) -> str:
    posix = PurePosixPath(path)
    parent = str(posix.parent)
    return (
        f"mkdir -p {_shell_quote(parent)}; "
        f"rm -rf {_shell_quote(path)}; "
        f"ln -s {_shell_quote(target)} {_shell_quote(path)}"
    )


def _extract_artifact_tar(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                continue
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.copy_failed,
                    f"Runpod artifact archive contains unsafe path: {member.name}",
                )
            archive.extract(member, root)


def _ssh_port(port_mappings: dict[Any, Any]) -> int | None:
    for key, value in port_mappings.items():
        if str(key) == "22":
            return int(value)
    return None


def _pod_name(config: SandboxConfig) -> str:
    return f"reprolab-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}"


def _safe_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return safe[:48] or "run"


def _join_posix(*parts: str) -> str:
    result = PurePosixPath(parts[0])
    for part in parts[1:]:
        if part:
            result /= part
    return result.as_posix()


def _shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _coerce_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = ["RunpodBackend"]
