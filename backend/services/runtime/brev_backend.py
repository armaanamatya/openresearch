"""Brev (NVIDIA Brev) remote-GPU sandbox backend.

IMPORTANT — API surface note
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As of May 2026, NVIDIA Brev does NOT expose a documented, stable REST API
for instance lifecycle management.  Its only programmatic surface is the
``brev`` CLI (``brev start``, ``brev stop``, ``brev delete``, ``brev ls``,
``brev shell``), which is backed by an undocumented internal REST service.
The ``brev-mcp`` project (github.com/brevdev/brev-mcp) also shells out to
the CLI.

Authentication is CLI-first: ``brev login --token <token>`` caches credentials
in ``~/.brev/``; subsequent CLI calls read the cached token.  The short-lived
access-token (1 h) is refreshed automatically by running any ``brev`` command.

This backend therefore:

1. Shells out to the ``brev`` CLI for create / list / delete.
2. Connects to instances over SSH (standard asyncssh, same as the RunPod
   backend) using an ed25519 key uploaded at instance-creation time.
3. Transfers files over SFTP (same asyncssh path).

The Brev REST API base URL is included as a parameter to allow a future
native-HTTP switch without changing call sites once Brev publishes a stable
REST surface.

Hardening mirrors ``runpod_backend.py``:
- ``asyncio.shield`` on destroy so task-cancellation cannot leak a paid GPU.
- 401/403 → ``retryable=False`` on CLI auth errors.
- Terminal-instance-state detection (FAILED, STOPPED) during boot polling.
- SSH host-key TOFU pinning within a session.
- Owned-instance allowlist: only instances created by *this* backend object
  can be deleted by it.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

_log = logging.getLogger(__name__)

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BREV_IMAGE = "ubuntu:22.04"

# Brev internal REST base (undocumented; may change without notice).
# Included as a fallback for future native-HTTP upgrade; the CLI path is
# the primary supported route.
_BREV_API_BASE = "https://api.brev.dev"

# Instance states that can never recover — abort boot-wait immediately.
_TERMINAL_INSTANCE_STATES: frozenset[str] = frozenset(
    {"FAILED", "ERROR", "TERMINATED", "DELETED"}
)


# ---------------------------------------------------------------------------
# Internal connection record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BrevConnection:
    instance_id: str
    public_ip: str
    ssh_port: int
    remote_base: str
    remote_workdir: str
    remote_artifacts_dir: str


# ---------------------------------------------------------------------------
# BrevBackend
# ---------------------------------------------------------------------------


class BrevBackend(RuntimeBackend):
    """RuntimeBackend implementation backed by an NVIDIA Brev GPU instance.

    Instance lifecycle is driven through the ``brev`` CLI subprocess.  File
    I/O and command execution go over SSH / SFTP via asyncssh, matching the
    RunPod backend's approach.

    Parameters
    ----------
    api_key:
        Brev API token (``BREV_API_KEY`` / ``REPROLAB_BREV_API_KEY``).
        Used for ``brev login --token`` on the first call that needs the CLI.
    cli_path:
        Path to the ``brev`` binary.  Defaults to whatever ``PATH`` resolves.
    gpu_type:
        Brev GPU flavor string, e.g. ``"A10G"`` or ``"RTX4090"``.  Passed to
        ``brev start --gpu <gpu_type>:<count>``.
    gpu_count:
        Number of GPUs per instance.
    instance_type:
        CPU-only machine type name for ``gpu_mode="off"`` runs.
    image:
        Container / VM base image name.
    ssh_key_path:
        Local path to the ed25519 private key that was (or will be) uploaded
        to Brev.  If the key does not yet exist, generate one first:
        ``ssh-keygen -t ed25519 -f ~/.ssh/brev_ed25519``.
    ssh_user:
        Remote SSH user (typically ``ubuntu`` on Brev instances).
    boot_timeout_seconds:
        Maximum seconds to wait for an instance to become SSH-ready.
    delete_on_destroy:
        When False, ``destroy()`` closes the SSH connection but does not
        delete the instance (useful for persistent shared workers).
    bootstrap_command:
        Optional shell command to run once inside the instance after the
        workspace is prepared.  Failure raises ``SandboxRuntimeError``.
    instance_id:
        When set, ``create_sandbox`` attaches to this pre-existing instance
        rather than creating a new one.  The instance is never deleted on
        destroy (same semantics as ``REPROLAB_RUNPOD_POD_ID``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cli_path: str = "brev",
        api_base_url: str = _BREV_API_BASE,
        gpu_type: str = "A10G",
        gpu_count: int = 1,
        instance_type: str = "t3.medium",
        image: str = DEFAULT_BREV_IMAGE,
        ssh_key_path: str | Path | None = None,
        ssh_public_key: str = "",
        ssh_user: str = "ubuntu",
        boot_timeout_seconds: int = 900,
        delete_on_destroy: bool = True,
        bootstrap_command: str = "",
        instance_id: str = "",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("REPROLAB_BREV_API_KEY")
            or os.environ.get("BREV_API_KEY")
            or ""
        ).strip()
        self.cli_path = cli_path
        self.api_base_url = api_base_url.rstrip("/")
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.instance_type = instance_type
        self.image = image or DEFAULT_BREV_IMAGE
        self.ssh_key_path = _normalize_ssh_key_path(ssh_key_path)
        self.ssh_public_key = ssh_public_key.strip()
        self.ssh_user = ssh_user
        self.boot_timeout_seconds = boot_timeout_seconds
        self.delete_on_destroy = delete_on_destroy
        self.bootstrap_command = bootstrap_command.strip()
        self.instance_id = instance_id.strip()

        self._connections: dict[str, _BrevConnection] = {}
        self._ssh_clients: dict[str, Any] = {}
        # TOFU host-key pins: (host, port) → SSHKey
        self._pinned_host_keys: dict[tuple[str, int], Any] = {}
        # Only instances this backend object created are safe to delete.
        self._owned_instance_ids: set[str] = set()
        # CLI authentication is idempotent; track to avoid re-running it.
        self._cli_authenticated: bool = False

    # ------------------------------------------------------------------
    # Public RuntimeBackend interface
    # ------------------------------------------------------------------

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        self._check_api_key()
        self._check_ssh_key()

        project_root = config.project_root.resolve()
        if not project_root.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Brev project root does not exist: {project_root}",
            )
        config.resolved_artifact_root().mkdir(parents=True, exist_ok=True)

        await self._ensure_cli_authenticated()

        # Persistent-instance mode: attach to a pre-existing instance.
        if self.instance_id:
            attached = await self._try_attach_existing_instance(config)
            if attached is not None:
                return attached
            _log.warning(
                "REPROLAB_BREV_INSTANCE_ID=%r is unusable; creating a new instance. "
                "Update .env with the new instance ID printed below to reuse it.",
                self.instance_id,
            )

        instance = await self._create_instance(config)
        instance_id = instance["id"]
        # Record ownership BEFORE the SSH-wait so cleanup is allowed.
        self._owned_instance_ids.add(instance_id)
        try:
            return await self._finish_create(config, instance_id, instance)
        except Exception:
            await self._delete_instance_quietly(instance_id)
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
            await self._sync_artifacts_to_host_quietly(sandbox)
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
                f"Brev SSH command failed: {exc}",
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
                f"Could not copy Brev file {remote_path}: {exc}",
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
                f"Could not write Brev file {remote_path}: {exc}",
            ) from exc

    async def destroy(self, sandbox: Sandbox) -> None:
        await self._sync_artifacts_to_host_quietly(sandbox)
        conn = self._ssh_clients.pop(sandbox.sandbox_id, None)
        if conn is not None:
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:
                pass
        self._connections.pop(sandbox.sandbox_id, None)

        if sandbox.sandbox_id not in self._owned_instance_ids:
            _log.info(
                "Brev destroy() skipping delete for unowned instance %s (persistent).",
                sandbox.sandbox_id,
            )
            return
        if self.delete_on_destroy:
            # asyncio.shield: task-cancellation cannot abort the DELETE call —
            # a paid Brev instance must be terminated even if the task is
            # cancelled (e.g. wall-clock timeout).
            await asyncio.shield(self._delete_instance(sandbox.sandbox_id))

    # ------------------------------------------------------------------
    # Private helpers — instance lifecycle
    # ------------------------------------------------------------------

    async def _ensure_cli_authenticated(self) -> None:
        """Run ``brev login --token`` once per backend lifetime."""
        if self._cli_authenticated:
            return
        if not self.api_key:
            self._cli_authenticated = True  # rely on existing ~/.brev/ session
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _run_cli(
                    self.cli_path,
                    ["login", "--token", self.api_key],
                    timeout=30,
                ),
            )
            self._cli_authenticated = True
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Brev CLI login failed: {exc}",
                retryable=False,
            ) from exc

    async def _create_instance(self, config: SandboxConfig) -> dict[str, Any]:
        """Create a new Brev instance and return its parsed descriptor."""
        use_gpu = config.gpu_mode != "off"
        name = _instance_name(config)

        if use_gpu:
            gpu_spec = f"{self.gpu_type}:{self.gpu_count}"
            args = ["start", name, "--gpu", gpu_spec, "--output", "json"]
        else:
            args = ["start", name, "--instance-type", self.instance_type, "--output", "json"]

        if self.image and self.image != DEFAULT_BREV_IMAGE:
            args += ["--image", self.image]

        # Inject the SSH public key if Brev CLI supports --public-key.
        pub_key = self._public_key()
        if pub_key:
            args += ["--public-key", pub_key]

        raw = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _run_cli(self.cli_path, args, timeout=120),
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Brev CLI may output human-readable text; extract JSON block.
            data = _extract_json_block(raw)
        if not isinstance(data, dict) or "id" not in data:
            # Fall back: query the instance list to find the newly created one.
            data = await self._find_instance_by_name(name)
        if not data:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Brev instance creation succeeded but instance descriptor could not be parsed. "
                f"CLI output: {raw[:500]}",
                retryable=False,
            )
        return data

    async def _try_attach_existing_instance(
        self, config: SandboxConfig
    ) -> Sandbox | None:
        """Attach to ``self.instance_id`` if it is RUNNING, else None."""
        try:
            info = await self._describe_instance(self.instance_id)
        except SandboxRuntimeError as exc:
            _log.warning(
                "Brev describe instance %s failed (%s) — falling back to create.",
                self.instance_id,
                exc,
            )
            return None
        status = str(info.get("status") or info.get("state") or "").upper()
        if status not in {"RUNNING", "ACTIVE", "READY"}:
            _log.warning(
                "Configured persistent Brev instance %s is not RUNNING (status=%s).",
                self.instance_id,
                status or "unknown",
            )
            return None
        public_ip = _extract_ip(info)
        ssh_port = int(info.get("sshPort") or info.get("ssh_port") or 22)
        if not public_ip:
            _log.warning(
                "Persistent Brev instance %s has no public IP.", self.instance_id
            )
            return None
        try:
            conn = await self._connect_ssh(public_ip, ssh_port)
        except Exception as exc:
            _log.warning(
                "SSH connect to Brev instance %s@%s:%s failed: %s",
                self.ssh_user,
                public_ip,
                ssh_port,
                exc,
            )
            return None
        self._ssh_clients[self.instance_id] = conn
        _log.info(
            "Attached to persistent Brev instance %s at %s:%s (will not delete on destroy).",
            self.instance_id,
            public_ip,
            ssh_port,
        )
        return await self._finish_create(
            config,
            self.instance_id,
            info,
            ssh_ready={"public_ip": public_ip, "ssh_port": ssh_port},
        )

    async def _finish_create(
        self,
        config: SandboxConfig,
        instance_id: str,
        info: dict[str, Any],
        *,
        ssh_ready: dict[str, Any] | None = None,
    ) -> Sandbox:
        ready = ssh_ready or await self._wait_for_instance_ssh(instance_id)
        remote_base = _join_posix(
            "/home", self.ssh_user, "reprolab",
            _safe_name(config.project_id),
            _safe_name(config.run_id),
        )
        connection = _BrevConnection(
            instance_id=instance_id,
            public_ip=ready["public_ip"],
            ssh_port=ready["ssh_port"],
            remote_base=remote_base,
            remote_workdir=_join_posix(remote_base, "work"),
            remote_artifacts_dir=_join_posix(remote_base, "artifacts"),
        )
        self._connections[instance_id] = connection
        await self._prepare_remote_workspace(config, connection)
        name = str(info.get("name") or _instance_name(config))
        image = str(info.get("image") or self.image)
        return Sandbox(
            sandbox_id=instance_id,
            name=name,
            image=image,
            config=config,
        )

    async def _wait_for_instance_ssh(self, instance_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.boot_timeout_seconds
        last_error = ""
        while asyncio.get_running_loop().time() < deadline:
            try:
                info = await self._describe_instance(instance_id)
            except SandboxRuntimeError:
                await asyncio.sleep(10)
                continue
            status = str(info.get("status") or info.get("state") or "").upper()
            # Terminal states — no point waiting the full timeout.
            if status in _TERMINAL_INSTANCE_STATES:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"Brev instance {instance_id} entered terminal state "
                    f"(status={status!r}) during boot.",
                    retryable=False,
                )
            public_ip = _extract_ip(info)
            ssh_port = int(info.get("sshPort") or info.get("ssh_port") or 22)
            if public_ip and status in {"RUNNING", "ACTIVE", "READY"}:
                try:
                    conn = await self._connect_ssh(public_ip, ssh_port)
                    self._ssh_clients[instance_id] = conn
                    return {"public_ip": public_ip, "ssh_port": ssh_port}
                except Exception as exc:
                    last_error = str(exc)
            await asyncio.sleep(15)
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Brev instance {instance_id} did not become SSH-ready before timeout. {last_error}",
            retryable=True,
        )

    async def _describe_instance(self, instance_id: str) -> dict[str, Any]:
        """Return instance descriptor via ``brev ls --output json`` + filter."""
        raw = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _run_cli(self.cli_path, ["ls", "--output", "json"], timeout=30),
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = _extract_json_block(raw)
        instances: list[dict[str, Any]] = []
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict):
            instances = data.get("instances") or data.get("workspaces") or [data]
        for inst in instances:
            if isinstance(inst, dict) and str(inst.get("id") or "") == instance_id:
                return inst
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Brev instance {instance_id!r} not found in 'brev ls' output.",
            retryable=True,
        )

    async def _find_instance_by_name(self, name: str) -> dict[str, Any]:
        raw = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _run_cli(self.cli_path, ["ls", "--output", "json"], timeout=30),
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = _extract_json_block(raw)
        instances: list[dict[str, Any]] = []
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict):
            instances = data.get("instances") or data.get("workspaces") or []
        for inst in instances:
            if isinstance(inst, dict) and str(inst.get("name") or "") == name:
                return inst
        return {}

    async def _delete_instance(self, instance_id: str) -> None:
        if instance_id not in self._owned_instance_ids:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Refusing to delete Brev instance {instance_id!r}: not in owned-instance "
                "allowlist. This backend only deletes instances it created itself.",
            )
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _run_cli(
                    self.cli_path,
                    ["delete", instance_id, "--yes"],
                    timeout=120,
                ),
            )
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Brev instance deletion failed for {instance_id}: {exc}",
                retryable=True,
            ) from exc
        finally:
            self._owned_instance_ids.discard(instance_id)

    async def _delete_instance_quietly(self, instance_id: str) -> None:
        try:
            await self._delete_instance(instance_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers — SSH / SFTP
    # ------------------------------------------------------------------

    async def _ssh(self, instance_id: str) -> Any:
        conn = self._ssh_clients.get(instance_id)
        if conn is not None and not conn.is_closed():
            return conn
        connection = self._connections[instance_id]
        conn = await self._connect_ssh(connection.public_ip, connection.ssh_port)
        self._ssh_clients[instance_id] = conn
        return conn

    async def _connect_ssh(self, host: str, port: int) -> Any:
        try:
            import asyncssh
        except ImportError as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "asyncssh is not installed. Install the 'asyncssh' Python package.",
            ) from exc
        # Host-key pinning (TOFU): accept any key on first connect, then pin
        # for the session to defend against IP-recycling MITM on reconnects.
        pin_key = (host, port)
        pinned = self._pinned_host_keys.get(pin_key)
        if pinned is None:
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
        return await asyncssh.connect(
            host,
            port=port,
            username=self.ssh_user,
            client_keys=[str(self.ssh_key_path)],
            known_hosts=([pinned], [], []),
        )

    async def _prepare_remote_workspace(
        self, config: SandboxConfig, connection: _BrevConnection
    ) -> None:
        conn = await self._ssh(connection.instance_id)
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
                f"Could not prepare Brev workspace: {result.stderr}",
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
                    bootstrap.stderr or bootstrap.stdout or "Brev bootstrap command failed.",
                )

    async def _upload_directory(self, sftp: Any, local_root: Path, remote_root: str) -> None:
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
        command = (
            f"test -d {_shell_quote(connection.remote_artifacts_dir)} && "
            f"tar -C {_shell_quote(connection.remote_artifacts_dir)} -cf - ."
        )
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
                "Brev artifact archive was not returned as bytes.",
            )
        _extract_artifact_tar(data, sandbox.config.resolved_artifact_root())

    async def _sync_artifacts_to_host_quietly(self, sandbox: Sandbox) -> None:
        try:
            await self._sync_artifacts_to_host(sandbox)
        except Exception:
            pass

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
            f"Path {path!r} is outside Brev runtime mounts.",
        )

    # ------------------------------------------------------------------
    # Private helpers — key management / validation
    # ------------------------------------------------------------------

    def _check_api_key(self) -> None:
        if not self.api_key and not Path("~/.brev").expanduser().exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "Brev sandbox is selected but no API key is configured and no cached "
                "session was found in ~/.brev/. "
                "Set REPROLAB_BREV_API_KEY or BREV_API_KEY, or run 'brev login' first.",
                retryable=False,
            )

    def _check_ssh_key(self) -> None:
        if not self.ssh_key_path.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Brev SSH private key not found: {self.ssh_key_path}. "
                "Generate one with: ssh-keygen -t ed25519 -f ~/.ssh/brev_ed25519",
                retryable=False,
            )

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


# ---------------------------------------------------------------------------
# ensure_brev_available — called by ensure_sandbox_mode_available
# ---------------------------------------------------------------------------


def ensure_brev_available() -> None:
    """Fail fast when Brev is selected but local credentials or CLI are absent."""
    api_key = (
        os.environ.get("REPROLAB_BREV_API_KEY")
        or os.environ.get("BREV_API_KEY")
        or ""
    ).strip()
    cli_ok = shutil.which("brev") is not None
    session_ok = Path("~/.brev").expanduser().exists()

    if not api_key and not session_ok:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Brev sandbox is selected but REPROLAB_BREV_API_KEY or BREV_API_KEY is not set "
            "and no cached session was found in ~/.brev/. "
            "Run 'brev login' or set the env var.",
            retryable=False,
        )
    if not cli_ok:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Brev sandbox is selected but the 'brev' CLI binary was not found in PATH. "
            "Install it from https://docs.nvidia.com/brev/getting-started/quickstart",
            retryable=False,
        )
    ssh_key_path = _normalize_ssh_key_path(
        os.environ.get("REPROLAB_BREV_SSH_KEY_PATH") or None
    )
    if not ssh_key_path.exists():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Brev sandbox is selected but SSH private key was not found: {ssh_key_path}. "
            "Set REPROLAB_BREV_SSH_KEY_PATH or create ~/.ssh/id_ed25519.",
            retryable=False,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _run_cli(cli_path: str, args: list[str], *, timeout: int = 60) -> str:
    """Synchronously run the brev CLI and return stdout.

    Raises SandboxRuntimeError with retryable=False on auth errors (exit 1
    with '401' or '403' in output), retryable=True for other failures.
    """
    result = subprocess.run(
        [cli_path] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        auth_error = "401" in combined or "403" in combined or "unauthorized" in combined.lower()
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Brev CLI exited {result.returncode}: {combined[:500]}",
            retryable=not auth_error,
        )
    return result.stdout


def _extract_json_block(text: str) -> Any:
    """Extract the first JSON object or array from free-form CLI output."""
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return {}


def _extract_ip(info: dict[str, Any]) -> str:
    for key in ("publicIp", "public_ip", "ip", "host", "dnsName", "dns"):
        val = info.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


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
                    f"Brev artifact archive contains unsafe path: {member.name}",
                )
            archive.extract(member, root, filter="data")


def _instance_name(config: SandboxConfig) -> str:
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
    # Tolerate Windows-style paths from .env on WSL.
    if ":" in raw and "\\" in raw and len(raw) >= 3 and raw[1] == ":":
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        mapped = Path(f"/mnt/{drive}/{tail}")
        return mapped.expanduser()
    return expanded


__all__ = ["BrevBackend", "ensure_brev_available"]
