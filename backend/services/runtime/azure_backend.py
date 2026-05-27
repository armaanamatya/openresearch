"""Azure compute sandbox backend (--sandbox azure).

Provisions an Azure Virtual Machine with a GPU SKU, uploads the project tree
over SFTP, executes commands over SSH, and tears the VM down on destroy.
Mirrors the RuntimeBackend contract identically to runpod_backend.py and
brev_backend.py — callers do not need to know which backend they got.

Auth: ``azure.identity.DefaultAzureCredential`` resolves credentials from
``az login`` (local dev), service-principal env vars
(``AZURE_CLIENT_ID``/``_SECRET``/``_TENANT_ID``), or managed identity (prod).
No ``az`` CLI shellout — we use the management SDK directly.

Hardening (parallels RunpodBackend + BrevBackend):
- ``asyncio.shield`` on destroy so task cancellation cannot leak a paid VM.
- ``atexit`` handler registered at create time as a belt-and-braces safety net.
- Owned-instance allowlist: only VMs created by *this* backend object are
  deletable by it.  Refuses to delete VMs it didn't provision.
- TOFU host-key pinning on SSH within a session.
- Terminal-VM-state detection (Failed, Deallocated) during boot polling.
- Ephemeral resource-group cascade-delete is the cleanest tear-down path;
  the per-resource piecemeal path is the fallback for preexisting RGs.
- Every ARM call is logged to ``runs/<id>/azure_resource_log.jsonl`` with
  op, resource id, region, sku, status, and latency.
"""

from __future__ import annotations

import asyncio
import atexit
import io
import json
import logging
import os
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

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

DEFAULT_AZURE_IMAGE = "microsoft-dsvm:ubuntu-hpc:2204:latest"
DEFAULT_AZURE_REGION = "eastus"
DEFAULT_AZURE_VM_SIZE = "Standard_NC6s_v3"

# VM provisioning / power states that can never recover — abort wait.
_TERMINAL_VM_STATES: frozenset[str] = frozenset(
    {"failed", "canceled", "deleting", "deleted"}
)

# PowerState/* substrings indicating the VM is ready.
_READY_POWER_STATES: frozenset[str] = frozenset(
    {"running", "starting"}  # starting transitions to running; SSH may already work
)


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AzureConnection:
    vm_id: str
    vm_name: str
    resource_group: str
    public_ip: str
    ssh_port: int
    remote_base: str
    remote_workdir: str
    remote_artifacts_dir: str
    # Track ALL resources we created so destroy() can clean them up
    # individually if the RG is preexisting (not ephemeral).
    nic_id: str = ""
    public_ip_id: str = ""
    nsg_id: str = ""
    vnet_id: str = ""
    os_disk_id: str = ""
    ephemeral_rg: bool = False


@dataclass
class _ResourceLog:
    """Append-only JSONL writer for Azure ARM calls.

    Lives at ``runs/<project_id>/azure_resource_log.jsonl``.  Captures every
    SDK call shape (op, resource_type, resource_id, region, sku, status,
    latency_ms, error) so operators can reconstruct provisioning timelines
    and debug leaked resources.
    """

    log_path: Path | None
    enabled: bool = True

    def append(
        self,
        *,
        op: str,
        resource_type: str = "",
        resource_id: str = "",
        region: str = "",
        sku: str = "",
        status: str = "",
        latency_ms: int = 0,
        error: str = "",
    ) -> None:
        if not self.enabled or self.log_path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": op,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "region": region,
            "sku": sku,
            "status": status,
            "latency_ms": latency_ms,
        }
        if error:
            record["error"] = error[:500]
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001 — never let logging crash a run
            pass


# ---------------------------------------------------------------------------
# AzureBackend
# ---------------------------------------------------------------------------


class AzureBackend(RuntimeBackend):
    """RuntimeBackend implementation backed by an Azure VM.

    Parameters
    ----------
    subscription_id:
        Azure subscription id.  ``REPROLAB_AZURE_SUBSCRIPTION_ID`` or
        ``AZURE_SUBSCRIPTION_ID``.
    region:
        Azure region (e.g. ``"eastus"``).
    vm_size:
        Azure VM SKU (e.g. ``"Standard_NC6s_v3"``).  ``AzureBackend`` does
        NOT translate from a generic short_name; the caller (the RLM
        primitive factory) resolves the SKU via ``gpu_catalog`` before
        constructing the backend, or passes the env-configured default.
    image:
        VM image URN, e.g. ``"microsoft-dsvm:ubuntu-hpc:2204:latest"``.
    resource_group:
        When empty, ``create_sandbox`` provisions an ephemeral RG named
        ``reprolab-<run_id>`` and cascade-deletes it on destroy.  When set,
        the backend reuses the existing RG and only deletes the VM + its
        owned nested resources.
    ssh_key_path / ssh_public_key:
        Local path to the ed25519 private key whose public half is injected
        into ``osProfile.linuxConfiguration.ssh.publicKeys``.  If
        ``ssh_public_key`` is empty, derived from ``<ssh_key_path>.pub`` or
        ``ssh-keygen -y -f <key>``.
    ssh_user:
        Linux username on the VM (default ``"azureuser"``).
    os_disk_gb / data_disk_gb:
        OS disk size (Premium SSD); data_disk_gb=0 disables.
    delete_on_destroy:
        When False, the VM is left running for persistent attach (analogous
        to ``REPROLAB_RUNPOD_POD_ID``).  Default True.
    tags:
        Extra resource tags merged with ``{"reprolab": "true", "project_id": ..., "run_id": ...}``.
    max_boot_seconds:
        Maximum seconds to wait for the VM to become SSH-reachable.
    log_dir:
        Where to write ``azure_resource_log.jsonl`` and
        ``azure_provisioning.log`` (typically ``runs/<project_id>/``).
    sdk_overrides:
        Test-injection point.  Dict with keys ``compute_client``,
        ``network_client``, ``resource_client``, ``credential`` — when
        provided, ``_get_clients()`` returns these instead of building
        real SDK clients.  Production callers leave this ``None``.
    """

    def __init__(
        self,
        *,
        subscription_id: str = "",
        region: str = DEFAULT_AZURE_REGION,
        vm_size: str = DEFAULT_AZURE_VM_SIZE,
        image: str = DEFAULT_AZURE_IMAGE,
        resource_group: str = "",
        ssh_key_path: str | Path | None = None,
        ssh_public_key: str = "",
        ssh_user: str = "azureuser",
        os_disk_gb: int = 100,
        os_disk_tier: str = "StandardSSD_LRS",
        data_disk_gb: int = 0,
        boot_diagnostics: bool = True,
        delete_on_destroy: bool = True,
        tags: dict[str, str] | None = None,
        max_boot_seconds: int = 600,
        bootstrap_command: str = "",
        log_dir: Path | None = None,
        run_budget: Any = None,
        sdk_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.subscription_id = (
            subscription_id
            or os.environ.get("REPROLAB_AZURE_SUBSCRIPTION_ID")
            or os.environ.get("AZURE_SUBSCRIPTION_ID")
            or ""
        ).strip()
        self.region = (region or DEFAULT_AZURE_REGION).strip()
        self.vm_size = (vm_size or DEFAULT_AZURE_VM_SIZE).strip()
        self.image = (image or DEFAULT_AZURE_IMAGE).strip()
        self.resource_group = (resource_group or "").strip()
        self.ssh_key_path = _normalize_ssh_key_path(ssh_key_path)
        self.ssh_public_key = ssh_public_key.strip()
        self.ssh_user = (ssh_user or "azureuser").strip()
        self.os_disk_gb = max(30, int(os_disk_gb or 100))
        self.os_disk_tier = (os_disk_tier or "StandardSSD_LRS").strip()
        self.data_disk_gb = max(0, int(data_disk_gb or 0))
        self.boot_diagnostics = bool(boot_diagnostics)
        self.delete_on_destroy = bool(delete_on_destroy)
        self.tags = dict(tags or {})
        self.max_boot_seconds = max(60, int(max_boot_seconds or 600))
        self.bootstrap_command = (bootstrap_command or "").strip()
        self.run_budget = run_budget
        self._sdk_overrides = sdk_overrides or {}

        self._connections: dict[str, _AzureConnection] = {}
        self._ssh_clients: dict[str, Any] = {}
        self._pinned_host_keys: dict[tuple[str, int], Any] = {}
        # Only VMs we created — refuse to delete anything else.
        self._owned_vm_ids: set[str] = set()
        # atexit guards (synchronous fallback for cancelled destroy()).
        self._atexit_registered: set[str] = set()

        # Resource log is enabled when log_dir is provided.  It writes to
        # <log_dir>/azure_resource_log.jsonl and is created lazily on the
        # first append.
        log_path = (
            log_dir / "azure_resource_log.jsonl" if log_dir is not None else None
        )
        self.resource_log = _ResourceLog(log_path=log_path)
        # Provisioning console — line-buffered text log for the boot loop.
        self._provisioning_log_path = (
            log_dir / "azure_provisioning.log" if log_dir is not None else None
        )

    # ------------------------------------------------------------------
    # Public RuntimeBackend interface
    # ------------------------------------------------------------------

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        self._check_subscription()
        self._check_ssh_key()

        project_root = config.project_root.resolve()
        if not project_root.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Azure project root does not exist: {project_root}",
            )
        config.resolved_artifact_root().mkdir(parents=True, exist_ok=True)

        # Resolve clients lazily.  This either returns the test-injected stubs
        # or constructs real SDK clients.
        clients = self._get_clients()

        # Decide RG strategy.  Empty config => ephemeral.
        ephemeral_rg = not self.resource_group
        rg_name = self.resource_group or _ephemeral_rg_name(config)

        vm_name = _vm_name(config)
        try:
            await self._ensure_resource_group(clients, rg_name, ephemeral_rg, config)
        except Exception as exc:  # noqa: BLE001 — log + re-raise as typed
            self.resource_log.append(
                op="ResourceGroups.create_or_update",
                resource_type="resource_group",
                resource_id=rg_name,
                region=self.region,
                status="failed",
                error=str(exc),
            )
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Azure resource group setup failed for {rg_name!r}: {exc}",
                retryable=_is_retryable_arm_error(exc),
            ) from exc

        # Now provision the VM + networking.
        try:
            connection = await self._provision_vm(
                clients, rg_name, vm_name, config, ephemeral_rg=ephemeral_rg
            )
        except SandboxRuntimeError:
            # Best-effort cleanup of partial state.
            await self._delete_vm_quietly(rg_name, vm_name, ephemeral_rg)
            raise
        except Exception as exc:  # noqa: BLE001
            await self._delete_vm_quietly(rg_name, vm_name, ephemeral_rg)
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Azure VM provisioning failed: {exc}",
                retryable=_is_retryable_arm_error(exc),
            ) from exc

        self._connections[connection.vm_id] = connection
        self._owned_vm_ids.add(connection.vm_id)
        # Register atexit cleanup BEFORE we attempt remote work — if we crash
        # during workspace prep, the VM is still going to be cleaned up on
        # process exit.
        if self.delete_on_destroy:
            self._register_atexit_cleanup(rg_name, vm_name, ephemeral_rg)

        # Prepare remote workspace.
        try:
            await self._prepare_remote_workspace(config, connection)
        except Exception:
            await self._delete_vm_quietly(rg_name, vm_name, ephemeral_rg)
            self._connections.pop(connection.vm_id, None)
            self._owned_vm_ids.discard(connection.vm_id)
            raise

        return Sandbox(
            sandbox_id=connection.vm_id,
            name=connection.vm_name,
            image=self.image,
            config=config,
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        # RunBudget integration — mirrors RunpodBackend.exec contract.
        # ``check_pod_seconds`` bounds the time the VM has been running;
        # ``check_run_gpu_usd`` bounds the cumulative spend.  Both are no-ops
        # when the corresponding cap is unset on the budget object.
        if self.run_budget is not None:
            from backend.agents.resilience.failures import BudgetExhausted

            try:
                self.run_budget.check_pod_seconds(
                    pod_started_at=sandbox.created_at,
                    agent_id="experiment-runner",
                )
                if sandbox.created_at is not None and self._vm_usd_per_hr() > 0:
                    _elapsed_hr = (
                        (datetime.now(timezone.utc) - sandbox.created_at).total_seconds()
                        / 3600.0
                    )
                    self.run_budget.check_run_gpu_usd(
                        cumulative_pod_usd=_elapsed_hr * self._vm_usd_per_hr(),
                        agent_id="experiment-runner",
                    )
            except BudgetExhausted:
                # Tear the VM down so the caller doesn't leak it.
                try:
                    await self.destroy(sandbox)
                except Exception:  # noqa: BLE001
                    pass
                raise

        started_at = datetime.now(timezone.utc)
        try:
            conn = await self._ssh(sandbox.sandbox_id)
            script = _remote_command(sandbox.config, command)
            result = await asyncio.wait_for(
                conn.run(f"/bin/bash -lc {_shell_quote(script)}", check=False),
                timeout=timeout,
            )
            await self._sync_artifacts_to_host_quietly(sandbox)
        except (TimeoutError, asyncio.TimeoutError):
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
        except Exception as exc:  # noqa: BLE001
            raise SandboxRuntimeError(
                RuntimeCauseKind.command_failed,
                f"Azure SSH command failed: {exc}",
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
        except Exception as exc:  # noqa: BLE001
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not read Azure VM file {remote_path}: {exc}",
            ) from exc

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        remote_path = self._map_remote_path(sandbox, path)
        try:
            conn = await self._ssh(sandbox.sandbox_id)
            async with conn.start_sftp_client() as sftp:
                await sftp.makedirs(str(PurePosixPath(remote_path).parent), exist_ok=True)
                async with sftp.open(remote_path, "wb") as handle:
                    await handle.write(data)
        except Exception as exc:  # noqa: BLE001
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"Could not write Azure VM file {remote_path}: {exc}",
            ) from exc

    async def destroy(self, sandbox: Sandbox) -> None:
        # Pull artifacts first (best-effort).  If the VM is unreachable we
        # still proceed with delete — leaking a paid VM is worse than losing
        # artifacts.
        await self._sync_artifacts_to_host_quietly(sandbox)
        conn = self._ssh_clients.pop(sandbox.sandbox_id, None)
        if conn is not None:
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:  # noqa: BLE001
                pass

        if sandbox.sandbox_id not in self._owned_vm_ids:
            _log.info(
                "Azure destroy() skipping delete for unowned VM %s (persistent).",
                sandbox.sandbox_id,
            )
            self._connections.pop(sandbox.sandbox_id, None)
            return

        connection = self._connections.pop(sandbox.sandbox_id, None)
        if connection is None or not self.delete_on_destroy:
            self._owned_vm_ids.discard(sandbox.sandbox_id)
            return

        # asyncio.shield: task-cancellation cannot abort the DELETE call.
        # A paid Azure VM must be deleted even if the outer task is being
        # cancelled (wall-clock timeout, watchdog, etc.).
        await asyncio.shield(
            self._delete_vm(
                connection.resource_group,
                connection.vm_name,
                ephemeral_rg=connection.ephemeral_rg,
            )
        )
        self._owned_vm_ids.discard(sandbox.sandbox_id)

    # ------------------------------------------------------------------
    # Lane N: optional probe / soft-recovery
    # ------------------------------------------------------------------

    async def probe_alive(self, sandbox: Sandbox, *, timeout: float = 10.0) -> bool:
        connection = self._connections.get(sandbox.sandbox_id)
        if connection is None:
            return False
        try:
            # Open a FRESH SSH channel (not the cached one which may be
            # wedged).  Use a tight per-call timeout.
            conn = await asyncio.wait_for(
                self._connect_ssh(connection.public_ip, connection.ssh_port),
                timeout=timeout,
            )
            try:
                result = await asyncio.wait_for(
                    conn.run("true", check=False), timeout=timeout
                )
                return getattr(result, "returncode", 1) == 0
            finally:
                conn.close()
                try:
                    await conn.wait_closed()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            return False

    async def soft_recover(self, sandbox: Sandbox) -> bool:
        # Use the CACHED channel — that's the one whose process tree we
        # want to whack.  pkill -9 of common in-VM training processes.
        conn = self._ssh_clients.get(sandbox.sandbox_id)
        if conn is None:
            return False
        try:
            kill_cmd = (
                "pkill -9 -f 'python|train\\.py|run_experiment' || true; "
                "pkill -9 -f 'accelerate|deepspeed|torchrun' || true; "
                "echo recovered"
            )
            result = await asyncio.wait_for(
                conn.run(f"/bin/bash -lc {_shell_quote(kill_cmd)}", check=False),
                timeout=15,
            )
            return getattr(result, "returncode", 1) == 0
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Private helpers — VM lifecycle
    # ------------------------------------------------------------------

    def _get_clients(self) -> dict[str, Any]:
        """Return the dict of Azure SDK clients (or test-injected stubs)."""
        if self._sdk_overrides:
            return self._sdk_overrides
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.mgmt.network import NetworkManagementClient
            from azure.mgmt.resource import ResourceManagementClient
        except ImportError as exc:  # noqa: F841
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "Azure SDK is not installed.  Install azure-identity, "
                "azure-mgmt-compute, azure-mgmt-network, azure-mgmt-resource.",
                retryable=False,
            ) from exc

        credential = DefaultAzureCredential()
        return {
            "credential": credential,
            "compute_client": ComputeManagementClient(credential, self.subscription_id),
            "network_client": NetworkManagementClient(credential, self.subscription_id),
            "resource_client": ResourceManagementClient(credential, self.subscription_id),
        }

    async def _ensure_resource_group(
        self,
        clients: dict[str, Any],
        rg_name: str,
        ephemeral: bool,
        config: SandboxConfig,
    ) -> None:
        resource_client = clients["resource_client"]
        if not ephemeral:
            # Verify it exists; do not create.
            t0 = time.monotonic()
            try:
                rg = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: resource_client.resource_groups.get(rg_name)
                )
                self.resource_log.append(
                    op="ResourceGroups.get",
                    resource_type="resource_group",
                    resource_id=getattr(rg, "id", rg_name),
                    region=self.region,
                    status="ok",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
                return
            except Exception as exc:
                self.resource_log.append(
                    op="ResourceGroups.get",
                    resource_type="resource_group",
                    resource_id=rg_name,
                    region=self.region,
                    status="failed",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    error=str(exc),
                )
                raise

        # Ephemeral path — create or update.
        tags = self._effective_tags(config)
        params = {"location": self.region, "tags": tags}
        t0 = time.monotonic()
        rg = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: resource_client.resource_groups.create_or_update(rg_name, params),
        )
        self.resource_log.append(
            op="ResourceGroups.create_or_update",
            resource_type="resource_group",
            resource_id=getattr(rg, "id", rg_name),
            region=self.region,
            status="ok",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    async def _provision_vm(
        self,
        clients: dict[str, Any],
        rg_name: str,
        vm_name: str,
        config: SandboxConfig,
        *,
        ephemeral_rg: bool,
    ) -> _AzureConnection:
        network_client = clients["network_client"]
        compute_client = clients["compute_client"]
        tags = self._effective_tags(config)

        # 1) NSG with port 22 inbound.
        nsg_name = f"{vm_name}-nsg"
        nsg = await self._create_nsg(network_client, rg_name, nsg_name, tags)

        # 2) VNet + subnet.
        vnet_name = f"{vm_name}-vnet"
        vnet, subnet = await self._create_vnet(network_client, rg_name, vnet_name, tags)

        # 3) Public IP.
        pip_name = f"{vm_name}-ip"
        public_ip = await self._create_public_ip(network_client, rg_name, pip_name, tags)

        # 4) NIC.
        nic_name = f"{vm_name}-nic"
        nic = await self._create_nic(
            network_client,
            rg_name,
            nic_name,
            subnet_id=getattr(subnet, "id", ""),
            public_ip_id=getattr(public_ip, "id", ""),
            nsg_id=getattr(nsg, "id", ""),
            tags=tags,
        )

        # 5) VM itself.
        vm = await self._create_vm(
            compute_client,
            rg_name,
            vm_name,
            nic_id=getattr(nic, "id", ""),
            tags=tags,
        )
        vm_id = getattr(vm, "id", "") or f"/subscriptions/{self.subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.Compute/virtualMachines/{vm_name}"

        # 6) Wait for the VM to be reachable.
        public_ip_address = await self._wait_for_public_ip(
            network_client, rg_name, pip_name
        )
        if not public_ip_address:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Azure VM {vm_name} provisioned but no public IP was assigned.",
                retryable=True,
            )
        await self._wait_for_vm_ssh(public_ip_address, port=22)

        remote_base = _join_posix(
            "/home", self.ssh_user, "reprolab",
            _safe_name(config.project_id),
            _safe_name(config.run_id),
        )
        return _AzureConnection(
            vm_id=vm_id,
            vm_name=vm_name,
            resource_group=rg_name,
            public_ip=public_ip_address,
            ssh_port=22,
            remote_base=remote_base,
            remote_workdir=_join_posix(remote_base, "work"),
            remote_artifacts_dir=_join_posix(remote_base, "artifacts"),
            nic_id=getattr(nic, "id", ""),
            public_ip_id=getattr(public_ip, "id", ""),
            nsg_id=getattr(nsg, "id", ""),
            vnet_id=getattr(vnet, "id", ""),
            os_disk_id="",  # filled later if we need fine-grained deletion
            ephemeral_rg=ephemeral_rg,
        )

    async def _create_nsg(
        self,
        network_client: Any,
        rg_name: str,
        nsg_name: str,
        tags: dict[str, str],
    ) -> Any:
        params = {
            "location": self.region,
            "tags": tags,
            "security_rules": [
                {
                    "name": "AllowSSH",
                    "protocol": "Tcp",
                    "source_port_range": "*",
                    "destination_port_range": "22",
                    "source_address_prefix": "*",
                    "destination_address_prefix": "*",
                    "access": "Allow",
                    "priority": 100,
                    "direction": "Inbound",
                }
            ],
        }
        t0 = time.monotonic()
        poller = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: network_client.network_security_groups.begin_create_or_update(
                rg_name, nsg_name, params
            ),
        )
        nsg = await asyncio.get_running_loop().run_in_executor(None, poller.result)
        self.resource_log.append(
            op="NetworkSecurityGroups.begin_create_or_update",
            resource_type="nsg",
            resource_id=getattr(nsg, "id", nsg_name),
            region=self.region,
            status="ok",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return nsg

    async def _create_vnet(
        self,
        network_client: Any,
        rg_name: str,
        vnet_name: str,
        tags: dict[str, str],
    ) -> tuple[Any, Any]:
        vnet_params = {
            "location": self.region,
            "tags": tags,
            "address_space": {"address_prefixes": ["10.0.0.0/24"]},
        }
        t0 = time.monotonic()
        vnet_poller = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: network_client.virtual_networks.begin_create_or_update(
                rg_name, vnet_name, vnet_params
            ),
        )
        vnet = await asyncio.get_running_loop().run_in_executor(None, vnet_poller.result)
        self.resource_log.append(
            op="VirtualNetworks.begin_create_or_update",
            resource_type="vnet",
            resource_id=getattr(vnet, "id", vnet_name),
            region=self.region,
            status="ok",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        subnet_params = {"address_prefix": "10.0.0.0/29"}
        t0 = time.monotonic()
        subnet_poller = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: network_client.subnets.begin_create_or_update(
                rg_name, vnet_name, "default", subnet_params
            ),
        )
        subnet = await asyncio.get_running_loop().run_in_executor(None, subnet_poller.result)
        self.resource_log.append(
            op="Subnets.begin_create_or_update",
            resource_type="subnet",
            resource_id=getattr(subnet, "id", "default"),
            region=self.region,
            status="ok",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return vnet, subnet

    async def _create_public_ip(
        self,
        network_client: Any,
        rg_name: str,
        pip_name: str,
        tags: dict[str, str],
    ) -> Any:
        # Standard SKU + Static allocation:
        #   - Basic SKU is deprecated (retirement: 2025-09-30); new subscriptions
        #     can't create it.
        #   - Standard SKU requires Static allocation (Dynamic is invalid for it).
        #   - Standard SKU is zone-redundant by default and forward-compatible.
        params = {
            "location": self.region,
            "tags": tags,
            "public_ip_allocation_method": "Static",
            "sku": {"name": "Standard"},
        }
        t0 = time.monotonic()
        poller = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: network_client.public_ip_addresses.begin_create_or_update(
                rg_name, pip_name, params
            ),
        )
        pip = await asyncio.get_running_loop().run_in_executor(None, poller.result)
        self.resource_log.append(
            op="PublicIPAddresses.begin_create_or_update",
            resource_type="public_ip",
            resource_id=getattr(pip, "id", pip_name),
            region=self.region,
            status="ok",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return pip

    async def _create_nic(
        self,
        network_client: Any,
        rg_name: str,
        nic_name: str,
        *,
        subnet_id: str,
        public_ip_id: str,
        nsg_id: str,
        tags: dict[str, str],
    ) -> Any:
        params = {
            "location": self.region,
            "tags": tags,
            "ip_configurations": [
                {
                    "name": "ipconfig1",
                    "subnet": {"id": subnet_id},
                    "public_ip_address": {"id": public_ip_id} if public_ip_id else None,
                }
            ],
            "network_security_group": {"id": nsg_id} if nsg_id else None,
        }
        t0 = time.monotonic()
        poller = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: network_client.network_interfaces.begin_create_or_update(
                rg_name, nic_name, params
            ),
        )
        nic = await asyncio.get_running_loop().run_in_executor(None, poller.result)
        self.resource_log.append(
            op="NetworkInterfaces.begin_create_or_update",
            resource_type="nic",
            resource_id=getattr(nic, "id", nic_name),
            region=self.region,
            status="ok",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return nic

    async def _create_vm(
        self,
        compute_client: Any,
        rg_name: str,
        vm_name: str,
        *,
        nic_id: str,
        tags: dict[str, str],
    ) -> Any:
        publisher, offer, sku, version = _parse_image_urn(self.image)
        public_key = self._public_key()
        if not public_key:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "Azure VM creation requires an SSH public key.  Set "
                "REPROLAB_AZURE_SSH_PUBLIC_KEY or ensure "
                "<REPROLAB_AZURE_SSH_KEY_PATH>.pub exists.",
                retryable=False,
            )
        params = {
            "location": self.region,
            "tags": tags,
            "hardware_profile": {"vm_size": self.vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": publisher,
                    "offer": offer,
                    "sku": sku,
                    "version": version,
                },
                "os_disk": {
                    # Explicit name so piecemeal delete in non-ephemeral RGs
                    # finds the disk.  Without this, Azure auto-generates a
                    # name like `{vm}_OsDisk_1_<guid>` and the delete 404s,
                    # leaking the disk.
                    "name": f"{vm_name}-osdisk",
                    "create_option": "FromImage",
                    "disk_size_gb": self.os_disk_gb,
                    # StandardSSD_LRS works on every VM SKU (incl. B-series);
                    # Premium_LRS is rejected by B/Av2-series.  Override via
                    # REPROLAB_AZURE_OS_DISK_TIER for production GPU runs.
                    "managed_disk": {"storage_account_type": self.os_disk_tier},
                },
            },
            "os_profile": {
                "computer_name": vm_name[:15] or "reprolab",  # Linux hostname limit
                "admin_username": self.ssh_user,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{self.ssh_user}/.ssh/authorized_keys",
                                "key_data": public_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic_id}]},
            "diagnostics_profile": {"boot_diagnostics": {"enabled": self.boot_diagnostics}},
        }
        t0 = time.monotonic()
        poller = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: compute_client.virtual_machines.begin_create_or_update(
                rg_name, vm_name, params
            ),
        )
        try:
            vm = await asyncio.get_running_loop().run_in_executor(
                None, poller.result
            )
        except Exception as exc:  # noqa: BLE001
            self.resource_log.append(
                op="VirtualMachines.begin_create_or_update",
                resource_type="vm",
                resource_id=vm_name,
                region=self.region,
                sku=self.vm_size,
                status="failed",
                latency_ms=int((time.monotonic() - t0) * 1000),
                error=str(exc),
            )
            raise

        # Check terminal state — Failed/Canceled means we can never recover.
        prov_state = str(getattr(vm, "provisioning_state", "") or "").lower()
        if prov_state in _TERMINAL_VM_STATES:
            self.resource_log.append(
                op="VirtualMachines.begin_create_or_update",
                resource_type="vm",
                resource_id=getattr(vm, "id", vm_name),
                region=self.region,
                sku=self.vm_size,
                status=prov_state,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Azure VM {vm_name} entered terminal state {prov_state!r} during provisioning.",
                retryable=False,
            )

        self.resource_log.append(
            op="VirtualMachines.begin_create_or_update",
            resource_type="vm",
            resource_id=getattr(vm, "id", vm_name),
            region=self.region,
            sku=self.vm_size,
            status=prov_state or "succeeded",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return vm

    async def _wait_for_public_ip(
        self, network_client: Any, rg_name: str, pip_name: str
    ) -> str:
        deadline = asyncio.get_running_loop().time() + self.max_boot_seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                pip = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: network_client.public_ip_addresses.get(rg_name, pip_name),
                )
                ip = getattr(pip, "ip_address", "") or ""
                if ip:
                    return ip
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(5)
        return ""

    async def _wait_for_vm_ssh(self, host: str, *, port: int) -> None:
        deadline = asyncio.get_running_loop().time() + self.max_boot_seconds
        last_error = ""
        while asyncio.get_running_loop().time() < deadline:
            try:
                conn = await self._connect_ssh(host, port)
                # Cache the connection — _ssh() will reuse it.
                # We key by the eventual vm_id later; for now we just close it
                # so it's reopened via _ssh() once the connection record exists.
                try:
                    conn.close()
                    await conn.wait_closed()
                except Exception:  # noqa: BLE001
                    pass
                return
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._log_provisioning(f"ssh-wait: {host}:{port} not ready ({exc})")
            await asyncio.sleep(10)
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Azure VM at {host}:{port} did not become SSH-reachable before timeout. "
            f"Last error: {last_error}",
            retryable=True,
        )

    async def _delete_vm(
        self, rg_name: str, vm_name: str, *, ephemeral_rg: bool
    ) -> None:
        clients = self._get_clients()
        if ephemeral_rg:
            # Fire-and-forget the cascade.  Azure RG delete takes 5-15 min
            # server-side; blocking destroy() on it would stall the run.
            # Issue begin_delete, log the accepted request, and return.  The
            # periodic pod_sweep_scheduler picks up any leftover that fails
            # to cascade.
            t0 = time.monotonic()
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: clients["resource_client"].resource_groups.begin_delete(
                        rg_name
                    ),
                )
                self.resource_log.append(
                    op="ResourceGroups.begin_delete",
                    resource_type="resource_group",
                    resource_id=rg_name,
                    region=self.region,
                    status="accepted",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception as exc:  # noqa: BLE001
                self.resource_log.append(
                    op="ResourceGroups.begin_delete",
                    resource_type="resource_group",
                    resource_id=rg_name,
                    region=self.region,
                    status="failed",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    error=str(exc),
                )
                _log.warning(
                    "Azure ephemeral RG cascade-delete failed for %s: %s.  "
                    "Falling back to piecemeal delete.",
                    rg_name,
                    exc,
                )
                await self._delete_vm_piecemeal(clients, rg_name, vm_name)
            return
        # Preexisting RG — piecemeal delete only.
        await self._delete_vm_piecemeal(clients, rg_name, vm_name)

    async def _delete_vm_piecemeal(
        self, clients: dict[str, Any], rg_name: str, vm_name: str
    ) -> None:
        """Delete VM + its owned nested resources individually.

        Order: VM -> NIC -> public IP -> OS disk -> NSG -> vnet.
        Each delete swallows 404 (the resource may already be gone).
        """
        compute = clients["compute_client"]
        network = clients["network_client"]

        async def _delete(op_name: str, resource_type: str, fn: Any) -> None:
            t0 = time.monotonic()
            try:
                poller = await asyncio.get_running_loop().run_in_executor(None, fn)
                if hasattr(poller, "result"):
                    await asyncio.get_running_loop().run_in_executor(None, poller.result)
                self.resource_log.append(
                    op=op_name,
                    resource_type=resource_type,
                    resource_id=f"{rg_name}/{vm_name}",
                    region=self.region,
                    status="ok",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception as exc:  # noqa: BLE001
                status_code = getattr(exc, "status_code", None) or getattr(
                    exc, "response", None
                )
                # Swallow 404 silently — resource already gone.
                if "404" in str(exc) or "ResourceNotFound" in str(exc):
                    self.resource_log.append(
                        op=op_name,
                        resource_type=resource_type,
                        resource_id=f"{rg_name}/{vm_name}",
                        region=self.region,
                        status="not_found",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                    return
                self.resource_log.append(
                    op=op_name,
                    resource_type=resource_type,
                    resource_id=f"{rg_name}/{vm_name}",
                    region=self.region,
                    status="failed",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    error=str(exc),
                )

        await _delete(
            "VirtualMachines.begin_delete",
            "vm",
            lambda: compute.virtual_machines.begin_delete(rg_name, vm_name),
        )
        await _delete(
            "NetworkInterfaces.begin_delete",
            "nic",
            lambda: network.network_interfaces.begin_delete(rg_name, f"{vm_name}-nic"),
        )
        await _delete(
            "PublicIPAddresses.begin_delete",
            "public_ip",
            lambda: network.public_ip_addresses.begin_delete(rg_name, f"{vm_name}-ip"),
        )
        await _delete(
            "Disks.begin_delete",
            "os_disk",
            lambda: compute.disks.begin_delete(rg_name, f"{vm_name}-osdisk"),
        )
        await _delete(
            "NetworkSecurityGroups.begin_delete",
            "nsg",
            lambda: network.network_security_groups.begin_delete(rg_name, f"{vm_name}-nsg"),
        )
        await _delete(
            "VirtualNetworks.begin_delete",
            "vnet",
            lambda: network.virtual_networks.begin_delete(rg_name, f"{vm_name}-vnet"),
        )

    async def _delete_vm_quietly(
        self, rg_name: str, vm_name: str, ephemeral: bool
    ) -> None:
        try:
            await self._delete_vm(rg_name, vm_name, ephemeral_rg=ephemeral)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Private helpers — atexit cleanup
    # ------------------------------------------------------------------

    def _register_atexit_cleanup(
        self, rg_name: str, vm_name: str, ephemeral_rg: bool
    ) -> None:
        key = f"{rg_name}/{vm_name}"
        if key in self._atexit_registered:
            return
        self._atexit_registered.add(key)
        atexit.register(self._cleanup_atexit, rg_name, vm_name, ephemeral_rg)

    def _cleanup_atexit(self, rg_name: str, vm_name: str, ephemeral_rg: bool) -> None:
        """Synchronous fallback delete.  Called at process exit."""
        if vm_name not in {c.vm_name for c in self._connections.values()} and \
           not any(rg_name in v for v in self._owned_vm_ids):
            return  # already cleaned up
        try:
            self._delete_vm_sync(rg_name, vm_name, ephemeral_rg)
        except Exception:  # noqa: BLE001
            pass

    def _delete_vm_sync(
        self, rg_name: str, vm_name: str, ephemeral_rg: bool
    ) -> None:
        try:
            clients = self._get_clients()
        except SandboxRuntimeError:
            return
        if ephemeral_rg:
            try:
                poller = clients["resource_client"].resource_groups.begin_delete(rg_name)
                # Don't block process shutdown — fire-and-forget the polling.
                # The Azure platform will continue the delete server-side.
                _ = poller
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            clients["compute_client"].virtual_machines.begin_delete(rg_name, vm_name)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Private helpers — SSH / SFTP (lifted from brev_backend.py)
    # ------------------------------------------------------------------

    async def _ssh(self, sandbox_id: str) -> Any:
        conn = self._ssh_clients.get(sandbox_id)
        if conn is not None and not conn.is_closed():
            return conn
        connection = self._connections[sandbox_id]
        conn = await self._connect_ssh(connection.public_ip, connection.ssh_port)
        self._ssh_clients[sandbox_id] = conn
        return conn

    async def _connect_ssh(self, host: str, port: int) -> Any:
        try:
            import asyncssh
        except ImportError as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "asyncssh is not installed. Install the 'asyncssh' Python package.",
            ) from exc
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
        self, config: SandboxConfig, connection: _AzureConnection
    ) -> None:
        conn = await self._ssh(connection.vm_id)
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
                f"Could not prepare Azure VM workspace: {result.stderr}",
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
                    bootstrap.stderr or bootstrap.stdout or "Azure bootstrap command failed.",
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
                "Azure artifact archive was not returned as bytes.",
            )
        _extract_artifact_tar(data, sandbox.config.resolved_artifact_root())

    async def _sync_artifacts_to_host_quietly(self, sandbox: Sandbox) -> None:
        try:
            await self._sync_artifacts_to_host(sandbox)
        except Exception:  # noqa: BLE001
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
            f"Path {path!r} is outside Azure runtime mounts.",
        )

    # ------------------------------------------------------------------
    # Private helpers — validation / metadata
    # ------------------------------------------------------------------

    def _check_subscription(self) -> None:
        if not self.subscription_id:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                "Azure sandbox is selected but REPROLAB_AZURE_SUBSCRIPTION_ID "
                "(or AZURE_SUBSCRIPTION_ID) is not set.",
                retryable=False,
            )

    def _check_ssh_key(self) -> None:
        if not self.ssh_key_path.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"Azure SSH private key not found: {self.ssh_key_path}.  "
                "Generate one with: ssh-keygen -t ed25519 -f ~/.ssh/azure_ed25519",
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
        except Exception:  # noqa: BLE001
            return ""
        if derived.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
            return derived
        return ""

    def _vm_usd_per_hr(self) -> float:
        """Look up the catalog price for ``self.vm_size``.  Returns 0.0 when unknown.

        The RunBudget cost cap is enforced only when the rate is non-zero;
        an unknown SKU falls back to "no cap" which is safer than guessing.
        """
        try:
            from backend.services.runtime.gpu_catalog import AZURE_CATALOG

            for sku in AZURE_CATALOG:
                if sku.azure_id == self.vm_size:
                    return float(sku.approx_usd_per_hr)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _effective_tags(self, config: SandboxConfig) -> dict[str, str]:
        merged: dict[str, str] = {
            "reprolab": "true",
            "project_id": _safe_tag(config.project_id),
            "run_id": _safe_tag(config.run_id),
        }
        # Then env-configured tags.
        merged.update({k: str(v) for k, v in self.tags.items()})
        return merged

    def _log_provisioning(self, message: str) -> None:
        if self._provisioning_log_path is None:
            return
        try:
            self._provisioning_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._provisioning_log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"{datetime.now(timezone.utc).isoformat()} {message}\n"
                )
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# ensure_azure_available — called by ensure_sandbox_mode_available
# ---------------------------------------------------------------------------


def ensure_azure_available() -> None:
    """Fail fast when Azure is selected but credentials or SSH key are absent."""
    subscription_id = (
        os.environ.get("REPROLAB_AZURE_SUBSCRIPTION_ID")
        or os.environ.get("AZURE_SUBSCRIPTION_ID")
        or ""
    ).strip()
    if not subscription_id:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Azure sandbox is selected but REPROLAB_AZURE_SUBSCRIPTION_ID "
            "(or AZURE_SUBSCRIPTION_ID) is not set.",
            retryable=False,
        )
    try:
        from azure.identity import DefaultAzureCredential  # noqa: F401
        from azure.mgmt.compute import ComputeManagementClient  # noqa: F401
    except ImportError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Azure sandbox is selected but the Azure SDK is not installed.  "
            "Install azure-identity, azure-mgmt-compute, azure-mgmt-network, "
            "azure-mgmt-resource.",
            retryable=False,
        ) from exc
    ssh_key_path = _normalize_ssh_key_path(
        os.environ.get("REPROLAB_AZURE_SSH_KEY_PATH") or None
    )
    if not ssh_key_path.exists():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Azure sandbox is selected but SSH private key was not found: "
            f"{ssh_key_path}. Set REPROLAB_AZURE_SSH_KEY_PATH or create "
            "~/.ssh/id_ed25519.",
            retryable=False,
        )


# ---------------------------------------------------------------------------
# Azure VM SKU lookup — for translating short_name from GpuPlan
# ---------------------------------------------------------------------------


# Mapping from the generic short_name (gpu_catalog.GpuSku.short_name) to an
# Azure VM SKU that matches the GPU model + memory.  The values are sourced
# from Microsoft Learn `/azure/virtual-machines/sizes/gpu-accelerated`.
# Refresh quarterly.
AZURE_SKU_BY_SHORT_NAME: dict[str, str] = {
    # 16-24 GB tier
    "rtx4090": "Standard_NV6ads_A10_v5",      # closest: A10, 24 GB
    "a5000": "Standard_NV6ads_A10_v5",         # closest: A10, 24 GB
    # 40 GB tier
    "a100_40": "Standard_NC24ads_A100_v4",     # A100 40GB equiv path; Azure ships 80GB A100s
    # 48 GB tier
    "a6000": "Standard_NV6ads_A10_v5",         # Azure doesn't expose A6000; A10 is closest
    "l40s": "Standard_NV36ads_A10_v5",         # heavier A10 SKU
    # 80 GB tier
    "a100_80": "Standard_NC24ads_A100_v4",
    "h100_80": "Standard_NC40ads_H100_v5",
    # H200
    "h200": "Standard_ND96isr_H200_v5",
}


def resolve_azure_vm_size(short_name: str, fallback: str = DEFAULT_AZURE_VM_SIZE) -> str:
    """Translate a generic GpuPlan.short_name to an Azure VM size string."""
    return AZURE_SKU_BY_SHORT_NAME.get(short_name, fallback)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_retryable_arm_error(exc: BaseException) -> bool:
    """Best-effort retryability classification for Azure ARM exceptions."""
    text = str(exc).lower()
    if "quotaexceeded" in text or "skunotavailable" in text:
        return False
    if "401" in text or "403" in text or "unauthorized" in text:
        return False
    return True


def _parse_image_urn(urn: str) -> tuple[str, str, str, str]:
    """Parse 'publisher:offer:sku:version' image URN."""
    parts = urn.split(":")
    if len(parts) != 4:
        # Fall back to DSVM Ubuntu 22.04 if the URN is malformed.
        return ("microsoft-dsvm", "ubuntu-hpc", "2204", "latest")
    return parts[0], parts[1], parts[2], parts[3]


def _ephemeral_rg_name(config: SandboxConfig) -> str:
    return f"reprolab-{_safe_name(config.run_id)}-{uuid.uuid4().hex[:6]}"


def _vm_name(config: SandboxConfig) -> str:
    # Azure VM names: max 64 chars, must start with a letter, alphanumeric + hyphen.
    base = f"reprolab-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}"
    return base[:60] or "reprolab-vm"


def _safe_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    # Azure resource names cannot end in a hyphen.
    safe = safe.rstrip("-")
    return safe[:32] or "run"


def _safe_tag(value: str) -> str:
    """Tag values: max 256 chars, no leading/trailing whitespace."""
    return str(value or "").strip()[:256]


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
                    f"Azure artifact archive contains unsafe path: {member.name}",
                )
            archive.extract(member, root, filter="data")


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
    if ":" in raw and "\\" in raw and len(raw) >= 3 and raw[1] == ":":
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        mapped = Path(f"/mnt/{drive}/{tail}")
        return mapped.expanduser()
    return expanded


__all__ = [
    "AzureBackend",
    "AZURE_SKU_BY_SHORT_NAME",
    "DEFAULT_AZURE_IMAGE",
    "DEFAULT_AZURE_REGION",
    "DEFAULT_AZURE_VM_SIZE",
    "ensure_azure_available",
    "resolve_azure_vm_size",
]
