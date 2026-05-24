"""Runpod runtime backend for remote GPU experiment execution."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from backend.agents.schemas import GpuPlan

_log = logging.getLogger(__name__)

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.failures import BudgetExhausted
from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

# RunPod pod states from which recovery is impossible — raise immediately
# rather than spinning the full boot_timeout_seconds (A3-5).
_TERMINAL_POD_STATES: frozenset[str] = frozenset({"EXITED", "FAILED", "DEAD"})


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
        pod_id: str = "",
        run_budget: RunBudget | None = None,
        gpu_plan: "GpuPlan | None" = None,
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
        self.ssh_key_path = _normalize_ssh_key_path(ssh_key_path)
        self.ssh_public_key = ssh_public_key.strip()
        self.ssh_user = ssh_user
        self.boot_timeout_seconds = boot_timeout_seconds
        self.delete_on_destroy = delete_on_destroy
        self.bootstrap_command = bootstrap_command.strip()
        # Optional persistent pod. When set, create_sandbox attaches to
        # this pod instead of POSTing /pods. The pod is never added to
        # _owned_pod_ids, so the destroy() guard refuses to delete it.
        self.pod_id = pod_id.strip()
        self._connections: dict[str, _RunpodConnection] = {}
        self._ssh_clients: dict[str, Any] = {}
        # Session-scoped host-key pins keyed by (host, port). Populated on
        # first connect (TOFU), verified on reconnects within the same session
        # to defend against IP-recycling MITM attacks (A3-4).
        self._pinned_host_keys: dict[tuple[str, int], Any] = {}
        # Allowlist of pod IDs THIS backend instance created. Any delete
        # call against a pod ID NOT in this set is refused with a typed
        # error — defense in depth on top of ``delete_on_destroy=false``
        # so a logic bug or rogue caller can never delete a pod created
        # outside our process (e.g. a coworker's pod on the same account,
        # or a persistent pod attached via REPROLAB_RUNPOD_POD_ID).
        self._owned_pod_ids: set[str] = set()
        self._run_budget = run_budget

        # Dynamic GPU plan overrides explicit args ONLY when source != "informational"
        # (informational means dynamic_gpu_enabled=off; caller passes the plan for
        # telemetry/UI but expects the legacy gpu_type to provision the pod).
        if gpu_plan is not None and getattr(gpu_plan, "source", None) != "informational":
            self.gpu_type = gpu_plan.runpod_id
            self.gpu_count = gpu_plan.gpu_count
            self.cloud_type = gpu_plan.cloud_type
            self.container_disk_gb = max(self.container_disk_gb, gpu_plan.container_disk_gb)
            self.volume_gb = max(self.volume_gb, gpu_plan.volume_gb)
        self.gpu_plan = gpu_plan

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

        # Persistent-pod mode: REPROLAB_RUNPOD_POD_ID points at an existing pod.
        # If it's RUNNING, attach. If it's missing/stopped, fall through to
        # create a new pod (and log the new ID prominently so the user can
        # update .env). Either way the resulting pod is never deleted on
        # destroy because we do NOT add it to _owned_pod_ids.
        if self.pod_id:
            attached = await self._try_attach_existing_pod(config, image)
            if attached is not None:
                return attached
            _log.warning(
                "REPROLAB_RUNPOD_POD_ID=%r is unusable; creating a new persistent pod. "
                "Update .env with the new pod id printed below to reuse it next run.",
                self.pod_id,
            )
            pod = await self._create_pod(config, image)
            pod_id = str(pod["id"])
            _log.warning(
                "RUNPOD_PERSISTENT_POD_CREATED pod_id=%s name=%s — "
                "set REPROLAB_RUNPOD_POD_ID=%s in .env to reuse it.",
                pod_id,
                pod.get("name") or _pod_name(config),
                pod_id,
            )
            # Persistent semantics: never delete this pod from destroy().
            # Skip _owned_pod_ids on purpose.
            try:
                return await self._finish_create(config, image, pod_id, pod)
            except Exception:
                # We created it but it failed to come up; user will see it
                # in their dashboard with the warned id. We do NOT delete
                # because the user opted into persistent mode by setting
                # REPROLAB_RUNPOD_POD_ID.
                raise

        # Per-run create-and-(maybe-)destroy mode (existing behavior).
        pod = await self._create_pod(config, image)
        pod_id = str(pod["id"])
        # Record ownership BEFORE the SSH-wait try block so the cleanup
        # path's _delete_pod_quietly call is allowed to proceed.
        self._owned_pod_ids.add(pod_id)
        try:
            return await self._finish_create(config, image, pod_id, pod)
        except Exception:
            await self._delete_pod_quietly(pod_id)
            raise

    async def _try_attach_existing_pod(
        self,
        config: SandboxConfig,
        image: str,
    ) -> Sandbox | None:
        """Attach to ``self.pod_id`` if it exists and is RUNNING, else None."""
        try:
            pod = await self._request_json("GET", f"/pods/{self.pod_id}")
        except SandboxRuntimeError as exc:
            _log.warning(
                "Runpod GET /pods/%s failed (%s) — falling back to create.",
                self.pod_id,
                exc,
            )
            return None
        status = str(pod.get("desiredStatus") or "").upper()
        if status != "RUNNING":
            _log.warning(
                "Configured persistent pod %s is not RUNNING (status=%s).",
                self.pod_id,
                status or "unknown",
            )
            return None
        public_ip = pod.get("publicIp") or pod.get("publicIP")
        ssh_port = _ssh_port(pod.get("portMappings") or {})
        if not public_ip or not ssh_port:
            _log.warning(
                "Persistent pod %s has no SSH endpoint (publicIp=%s, port22=%s).",
                self.pod_id,
                public_ip,
                ssh_port,
            )
            return None
        try:
            conn = await self._connect_ssh(str(public_ip), int(ssh_port))
        except Exception as exc:
            _log.warning(
                "SSH connect to persistent pod %s@%s:%s failed: %s",
                self.ssh_user,
                public_ip,
                ssh_port,
                exc,
            )
            return None
        self._ssh_clients[self.pod_id] = conn
        _log.info(
            "Attached to persistent Runpod pod %s at %s:%s (will not delete on destroy).",
            self.pod_id,
            public_ip,
            ssh_port,
        )
        # Reuse _finish_create's workspace prep, but skip _wait_for_pod_ssh
        # since we already have a live SSH connection.
        return await self._finish_create(
            config,
            pod.get("imageName") or image,
            self.pod_id,
            pod,
            ssh_ready={"public_ip": str(public_ip), "ssh_port": int(ssh_port)},
        )

    async def _finish_create(
        self,
        config: SandboxConfig,
        image: str,
        pod_id: str,
        pod: dict[str, Any],
        *,
        ssh_ready: dict[str, Any] | None = None,
    ) -> Sandbox:
        ready = ssh_ready or await self._wait_for_pod_ssh(pod_id)
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

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        if self._run_budget is not None:
            try:
                self._run_budget.check_pod_seconds(
                    pod_started_at=sandbox.created_at,
                    agent_id="experiment-runner",
                )
                if self.gpu_plan is not None and sandbox.created_at is not None:
                    _elapsed_hr = (
                        (datetime.now(timezone.utc) - sandbox.created_at).total_seconds()
                        / 3600.0
                    )
                    self._run_budget.check_run_gpu_usd(
                        cumulative_pod_usd=_elapsed_hr * self.gpu_plan.total_usd_per_hr,
                        agent_id="experiment-runner",
                    )
            except BudgetExhausted:
                # Track whether destroy actually deleted the pod. For persistent
                # pods (REPROLAB_RUNPOD_POD_ID), the pod_id is intentionally not
                # in _owned_pod_ids so destroy() returns without deleting —
                # which means the pod keeps billing. The operator must see this
                # explicitly, not buried as an INFO log inside destroy().
                pod_was_owned = sandbox.sandbox_id in self._owned_pod_ids
                try:
                    await self.destroy(sandbox)
                except Exception as exc:
                    _log.error(
                        "RUNPOD_DESTROY_FAILED_AFTER_BUDGET_EXHAUSTION "
                        "sandbox_id=%s — pod may still be billing, manual cleanup required. cause=%s",
                        sandbox.sandbox_id,
                        exc,
                        exc_info=True,
                    )
                else:
                    if not pod_was_owned:
                        _log.error(
                            "RUNPOD_BUDGET_EXHAUSTED_PERSISTENT_POD_NOT_DELETED "
                            "sandbox_id=%s — persistent pod (REPROLAB_RUNPOD_POD_ID) "
                            "is unowned by this backend, destroy skipped; pod is STILL "
                            "RUNNING and billing. Stop it manually via the RunPod dashboard.",
                            sandbox.sandbox_id,
                        )
                raise
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
        # Persistent pods (attached via REPROLAB_RUNPOD_POD_ID, or any pod
        # not in our ownership allowlist) are never deleted. The
        # _delete_pod guard would also refuse, but short-circuiting here
        # avoids a noisy SandboxRuntimeError and keeps logs clean.
        if sandbox.sandbox_id not in self._owned_pod_ids:
            _log.info(
                "Runpod destroy() skipping delete for unowned pod %s (persistent).",
                sandbox.sandbox_id,
            )
            return
        if self.delete_on_destroy:
            # asyncio.shield ensures task cancellation cannot abort the
            # DELETE call — a paid RunPod pod must be terminated even if
            # the surrounding task is cancelled (e.g. wall-clock timeout).
            await asyncio.shield(self._delete_pod(sandbox.sandbox_id))

    async def _create_pod(self, config: SandboxConfig, image: str) -> dict[str, Any]:
        # network_disabled is not supported on RunPod: the pod communicates
        # exclusively over SSH (port 22), so disabling the network would make
        # the sandbox unreachable. Callers who set network_disabled=True on a
        # RunPod config should use the local-docker backend instead.
        if config.network_disabled:
            _log.warning(
                "SandboxConfig.network_disabled=True is ignored for RunPod pods: "
                "RunPod requires network access for SSH. "
                "Use the local-docker backend if true network isolation is required."
            )

        public_key = self._public_key()
        env = dict(config.environment)
        if public_key:
            env.setdefault("PUBLIC_KEY", public_key)
            env.setdefault("SSH_PUBLIC_KEY", public_key)

        # Map gpu_mode to RunPod compute type.
        # "off"          → CPU-only pod (no GPU allocation, lower cost).
        # "prefer"/"max" → GPU pod using the configured gpu_type/count.
        # "auto" (default) → GPU pod (preserves prior behavior).
        use_gpu = config.gpu_mode != "off"

        payload: dict[str, Any] = {
            "name": _pod_name(config),
            "cloudType": self.cloud_type,
            "computeType": "GPU" if use_gpu else "CPU",
            "imageName": image,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": self.volume_mount_path,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": env,
        }
        if use_gpu:
            payload["gpuTypeIds"] = [self.gpu_type]
            payload["gpuCount"] = self.gpu_count
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
            # Detect terminal pod states immediately — no point spinning the
            # full boot_timeout_seconds on a dead pod (A3-5).
            desired = str(pod.get("desiredStatus") or "").upper()
            current = str(pod.get("currentStatus") or "").upper()
            if desired in _TERMINAL_POD_STATES or current in _TERMINAL_POD_STATES:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"Runpod pod {pod_id} entered terminal state "
                    f"(desiredStatus={desired!r}, currentStatus={current!r}) during boot.",
                    retryable=False,
                )
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
            f"RUNPOD_SSH_TIMEOUT: pod {pod_id} did not become SSH-ready "
            f"after {self.boot_timeout_seconds}s. {last_error}",
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
        """Incrementally sync remote artifacts/ to the local artifact root.

        Walks the remote tree via SFTP and transfers only files whose
        (size, mtime) differs from the local copy.  Symlinks are skipped
        (same as the previous tar-based behaviour).  Hardlinks are NOT
        checked: SFTP presents hardlinks as ordinary regular files with
        nlink > 1, which is indistinguishable from regular files without
        per-file nlink inspection, and the original tar-format islnk()
        check was an archive-format concept that has no direct SFTP
        analogue.  The path-escape guard below is the real safety
        guarantee; hardlink-specific refusal has been intentionally
        dropped (see B.4 design notes).
        """
        connection = self._connections[sandbox.sandbox_id]
        conn = await self._ssh(sandbox.sandbox_id)
        local_root = sandbox.config.resolved_artifact_root().resolve()
        local_root.mkdir(parents=True, exist_ok=True)
        remote_root = connection.remote_artifacts_dir

        async with conn.start_sftp_client() as sftp:
            # Short-circuit: if the remote artifacts dir doesn't exist yet
            # (pod never wrote anything) this is a silent no-op, matching
            # the original `test -d ...` guard.
            try:
                await sftp.stat(remote_root)
            except (FileNotFoundError, OSError):
                return

            # Walk the remote tree recursively.  We use a manual stack
            # rather than glob('**') so the behaviour is explicit and
            # testable regardless of asyncssh version.
            stack: list[str] = [remote_root]
            while stack:
                current_dir = stack.pop()
                entries = await sftp.readdir(current_dir)
                for entry in entries:
                    name = entry.filename
                    if name in (".", ".."):
                        continue
                    entry_remote_path = _join_posix(current_dir, name)

                    # Re-stat via lstat (never follows symlinks) so we get
                    # the true type of the entry rather than the target.
                    attrs = await sftp.lstat(entry_remote_path)

                    # asyncssh exposes attrs.permissions as Optional[int]. A
                    # well-behaved SFTP server always populates it on lstat,
                    # but a misbehaving one would feed None into stat.S_ISLNK
                    # and crash the sync. Skip such entries defensively.
                    if attrs.permissions is None:
                        continue

                    # Refuse symlinks — skip silently (original tar behaviour).
                    if stat.S_ISLNK(attrs.permissions):
                        continue

                    # Compute the safe local path.
                    relative = _relative_posix(entry_remote_path, remote_root)
                    local_path = (local_root / relative).resolve()
                    # Safety: refuse paths that resolve outside local_root.
                    if local_root != local_path and local_root not in local_path.parents:
                        raise SandboxRuntimeError(
                            RuntimeCauseKind.copy_failed,
                            f"Runpod artifact sync refused unsafe relative path: {relative}",
                        )

                    if stat.S_ISDIR(attrs.permissions):
                        local_path.mkdir(parents=True, exist_ok=True)
                        stack.append(entry_remote_path)
                        continue

                    # The win: skip files that haven't changed.
                    if _file_unchanged(local_path, attrs):
                        continue

                    # Transfer the file in chunks.
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    async with sftp.open(entry_remote_path, "rb") as src:
                        with open(local_path, "wb") as dst:
                            while True:
                                chunk = await src.read(65536)
                                if not chunk:
                                    break
                                dst.write(chunk)

                    # Preserve remote mtime so subsequent syncs can skip
                    # this file when it hasn't changed on the remote side.
                    if attrs.atime is not None and attrs.mtime is not None:
                        os.utime(local_path, (attrs.atime, attrs.mtime))

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
        # Host-key pinning (A3-4): on the first connection to a (host, port)
        # pair we accept any key and record it; on subsequent reconnections
        # (e.g. after a dropped connection mid-run) we verify against the
        # pinned key. This defends against IP-recycling MITM within a session.
        #
        # Threat model: RunPod pods are freshly booted per run, so we cannot
        # pre-populate known_hosts from a trust store. We accept TOFU
        # (trust-on-first-use) for the first connection, then pin for the
        # session. A cold-start MITM on the very first connect would not be
        # detected — acceptable given the RunPod trust boundary (TLS-secured
        # API issues the IP, and the pod runs our own image).
        pin_key = (host, port)
        pinned = self._pinned_host_keys.get(pin_key)
        if pinned is None:
            # First connect: accept any key, then pin it.
            conn = await asyncssh.connect(
                host,
                port=port,
                username=self.ssh_user,
                client_keys=[str(self.ssh_key_path)],
                known_hosts=None,
            )
            host_key = conn.get_server_host_key()
            if host_key is not None:
                self._pinned_host_keys[pin_key] = host_key
            return conn
        # Subsequent connects: verify against the pinned host key.
        # asyncssh accepts a list of SSHKey objects as known_hosts — it will
        # reject any server presenting a different key, guarding against
        # IP-recycling MITM on reconnects within the same session.
        return await asyncssh.connect(
            host,
            port=port,
            username=self.ssh_user,
            client_keys=[str(self.ssh_key_path)],
            known_hosts=([pinned], [], []),
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
        except httpx.HTTPStatusError as exc:
            # Auth failures must not be retried — looping on a 401/403 wastes
            # time and quota (A3-6). All other HTTP errors are treated as
            # transient network/server faults and remain retryable.
            status = exc.response.status_code
            retryable = status not in (401, 403)
            # Capacity / quota errors come back as 500 with a specific body.
            # Surface a distinguishable message so the dynamic-GPU escalation
            # loop in run_experiment can advance to the next SKU on the ladder
            # instead of failing the whole run (spec 2026-05-23 §SSE event
            # types — gpu_escalated reason=runpod_capacity).
            body_text = ""
            try:
                body_text = exc.response.text or ""
            except Exception:  # noqa: BLE001 — body read must never crash this branch
                body_text = ""
            lower = body_text.lower()
            capacity_marker = (
                "no instances currently available" in lower
                or "no available" in lower
                or "out of capacity" in lower
            )
            balance_marker = "balance is too low" in lower or "add funds" in lower
            if status == 500 and capacity_marker:
                # Sentinel prefix so the run_experiment escalation loop can
                # match this on the exception message alone — avoids a new
                # exception type / RuntimeCauseKind churn for a single case.
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"RUNPOD_CAPACITY_EXHAUSTED: {exc}",
                    retryable=True,
                ) from exc
            if status == 500 and balance_marker:
                # Funding failures should NOT be retried — they need user action.
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"RUNPOD_BALANCE_TOO_LOW: {exc}",
                    retryable=False,
                ) from exc
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod API request failed (HTTP {status}): {exc}",
                retryable=retryable,
            ) from exc
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod API request failed: {exc}",
                retryable=True,
            ) from exc

    async def _delete_pod(self, pod_id: str) -> None:
        # Guardrail: refuse to delete any pod this backend instance did
        # not create. Defense in depth on top of ``delete_on_destroy=false``.
        # Catches logic bugs, rogue callers, and accidental cross-account
        # deletions (e.g. a coworker's pods on the same Runpod account).
        if pod_id not in self._owned_pod_ids:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Refusing to delete pod {pod_id!r}: not in owned-pod allowlist. "
                "This backend only deletes pods it created itself.",
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(
                base_url=self.api_base_url,
                headers=headers,
                timeout=60,
            ) as client:
                # Belt-and-suspenders: verify the pod's name still has our
                # prefix (`reprolab-…`) before issuing the DELETE. If the
                # name doesn't match, refuse — covers the case where a pod
                # ID was added to our allowlist via some future code path
                # but the pod was actually created by someone else.
                try:
                    info = await client.get(f"/pods/{pod_id}")
                    info.raise_for_status()
                    pod_name = str((info.json() or {}).get("name") or "")
                    if pod_name and not pod_name.startswith("reprolab-"):
                        raise SandboxRuntimeError(
                            RuntimeCauseKind.backend_unavailable,
                            f"Refusing to delete pod {pod_id!r} (name {pod_name!r}): "
                            "name does not start with 'reprolab-' — not ours.",
                        )
                except SandboxRuntimeError:
                    raise
                except Exception:
                    # GET failed (transient API error etc.). The allowlist
                    # check above already passed, so we proceed — the
                    # name-prefix check is best-effort hardening, not the
                    # primary guarantee.
                    pass

                response = await client.delete(f"/pods/{pod_id}")
                response.raise_for_status()
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Runpod pod deletion failed for {pod_id}: {exc}",
                retryable=True,
            ) from exc
        finally:
            # Drop ownership regardless of API outcome — the pod either
            # was deleted or the caller now knows our records are stale.
            self._owned_pod_ids.discard(pod_id)

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
        try:
            derived = subprocess.run(
                ["ssh-keygen", "-y", "-f", str(self.ssh_key_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""
        if derived.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
            return derived
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


def ensure_runpod_available() -> None:
    """Fail fast when RunPod is selected but local credentials are incomplete."""
    from backend.config import get_settings

    settings = get_settings(_force_reload=True)
    api_key = (
        settings.runpod_api_key
        or os.environ.get("REPROLAB_RUNPOD_API_KEY")
        or os.environ.get("RUNPOD_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "RunPod sandbox is selected but REPROLAB_RUNPOD_API_KEY or RUNPOD_API_KEY is not set.",
        )
    ssh_key_path = _normalize_ssh_key_path(settings.runpod_ssh_key_path or None)
    if not ssh_key_path.exists():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"RunPod sandbox is selected but SSH private key was not found: {ssh_key_path}. "
            "Set REPROLAB_RUNPOD_SSH_KEY_PATH or create ~/.ssh/id_ed25519.",
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


def _relative_posix(remote_path: str, remote_root: str) -> str:
    """Return the POSIX-relative path of *remote_path* under *remote_root*.

    Example: _relative_posix('/artifacts/a/b.txt', '/artifacts') -> 'a/b.txt'
    """
    root = remote_root.rstrip("/") + "/"
    if remote_path.startswith(root):
        return remote_path[len(root):]
    # Identical — the root itself (shouldn't normally be passed, but be safe).
    return ""


def _file_unchanged(local_path: Path, remote_attrs: Any) -> bool:
    """Return True when *local_path* exists and matches *remote_attrs* (size + mtime).

    The comparison is intentionally cheap: matching both size and mtime is
    sufficient for the artifact-sync use case where the remote is always the
    source of truth.
    """
    try:
        st = local_path.stat()
    except FileNotFoundError:
        return False
    remote_mtime = remote_attrs.mtime
    remote_size = remote_attrs.size
    if remote_mtime is None or remote_size is None:
        return False
    return st.st_size == remote_size and st.st_mtime >= remote_mtime


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


def _normalize_ssh_key_path(value: str | Path | None) -> Path:
    raw = str(value or "~/.ssh/id_ed25519").strip()
    expanded = Path(raw).expanduser()
    if expanded.exists():
        return expanded
    # Accept Windows-style absolute paths in .env when running from WSL/Linux.
    # Example: C:\Users\name\.ssh\id_ed25519 -> /mnt/c/Users/name/.ssh/id_ed25519
    if ":" in raw and "\\" in raw and len(raw) >= 3 and raw[1] == ":":
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        mapped = Path(f"/mnt/{drive}/{tail}")
        return mapped.expanduser()
    return expanded


__all__ = ["RunpodBackend", "ensure_runpod_available"]
