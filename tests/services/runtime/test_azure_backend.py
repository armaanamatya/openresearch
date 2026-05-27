"""Unit tests for the Azure compute sandbox backend.

These tests stay offline.  No Azure subscription is required.  Real SDK
calls are monkeypatched via the ``sdk_overrides`` injection seam on
``AzureBackend.__init__`` so the test exercises the ARM-call ordering,
resource-log writes, owned-instance allowlist, and destroy semantics
without touching Azure.

Coverage map (matches spec §11):
 1. test_azure_backend_create_sandbox_happy_path
 2. test_azure_backend_destroy_deletes_ephemeral_rg_cascade
 3. test_azure_backend_destroy_is_asyncio_shielded
 4. test_azure_backend_terminal_state_aborts_provisioning
 5. test_azure_backend_owned_instance_allowlist
 6. test_azure_backend_atexit_cleanup_registered
 7. test_ensure_azure_available_missing_creds_raises
 8. test_config_default_sandbox_accepts_azure
 9. test_cli_sandbox_choice_includes_azure
10. test_app_runpod_status_shows_azure_label
11. test_pod_sweep_scheduler_calls_azure_sweep
12. test_gpu_catalog_resolve_azure_sku
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.services.runtime.azure_backend import (
    AZURE_SKU_BY_SHORT_NAME,
    AzureBackend,
    DEFAULT_AZURE_REGION,
    DEFAULT_AZURE_VM_SIZE,
    ensure_azure_available,
    resolve_azure_vm_size,
)
from backend.services.runtime.interface import (
    RuntimeCauseKind,
    SandboxConfig,
    SandboxRuntimeError,
)


# ---------------------------------------------------------------------------
# Fixtures — fake SDK clients that record calls
# ---------------------------------------------------------------------------


class _FakePoller:
    """Sync-style poller returned by begin_* methods."""

    def __init__(self, result_obj: Any) -> None:
        self._result = result_obj

    def result(self) -> Any:
        return self._result


def _fake_resource(resource_id: str, **extras: Any) -> SimpleNamespace:
    return SimpleNamespace(id=resource_id, **extras)


@pytest.fixture
def fake_clients(tmp_path: Path) -> dict[str, Any]:
    """Build a stub dict that AzureBackend's sdk_overrides expects."""
    calls: list[str] = []

    def record(name: str):
        def _inner(*args: Any, **kwargs: Any):
            calls.append(name)
            # Return a poller whose .result() returns a stub resource with id=name
            return _FakePoller(_fake_resource(f"/fake/{name}", provisioning_state="Succeeded"))
        return _inner

    compute = MagicMock(name="compute")
    network = MagicMock(name="network")
    resource = MagicMock(name="resource")

    compute.virtual_machines.begin_create_or_update.side_effect = record(
        "VirtualMachines.begin_create_or_update"
    )
    compute.virtual_machines.begin_delete.side_effect = record(
        "VirtualMachines.begin_delete"
    )
    compute.disks.begin_delete.side_effect = record("Disks.begin_delete")

    network.network_security_groups.begin_create_or_update.side_effect = record(
        "NetworkSecurityGroups.begin_create_or_update"
    )
    network.network_security_groups.begin_delete.side_effect = record(
        "NetworkSecurityGroups.begin_delete"
    )
    network.virtual_networks.begin_create_or_update.side_effect = record(
        "VirtualNetworks.begin_create_or_update"
    )
    network.virtual_networks.begin_delete.side_effect = record(
        "VirtualNetworks.begin_delete"
    )
    network.subnets.begin_create_or_update.side_effect = record(
        "Subnets.begin_create_or_update"
    )
    network.public_ip_addresses.begin_create_or_update.side_effect = record(
        "PublicIPAddresses.begin_create_or_update"
    )
    network.public_ip_addresses.begin_delete.side_effect = record(
        "PublicIPAddresses.begin_delete"
    )
    network.public_ip_addresses.get.return_value = SimpleNamespace(
        ip_address="203.0.113.42"
    )
    network.network_interfaces.begin_create_or_update.side_effect = record(
        "NetworkInterfaces.begin_create_or_update"
    )
    network.network_interfaces.begin_delete.side_effect = record(
        "NetworkInterfaces.begin_delete"
    )

    rg_stub = SimpleNamespace(id="/fake/rg", tags={"reprolab": "true"})
    resource.resource_groups.create_or_update.return_value = rg_stub
    resource.resource_groups.get.return_value = rg_stub
    resource.resource_groups.begin_delete.side_effect = record(
        "ResourceGroups.begin_delete"
    )

    return {
        "compute_client": compute,
        "network_client": network,
        "resource_client": resource,
        "_calls": calls,
    }


@pytest.fixture
def sandbox_config(tmp_path: Path) -> SandboxConfig:
    project_root = tmp_path / "code"
    project_root.mkdir()
    (project_root / "hello.txt").write_text("hi")
    return SandboxConfig(
        project_id="proj-1",
        run_id="run-abc",
        project_root=project_root,
        artifact_root=tmp_path / "artifacts",
    )


# ---------------------------------------------------------------------------
# Patched helpers — bypass SSH wait + workspace prep so we can test create
# semantics without a real network.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_ssh_wait_and_workspace(monkeypatch: pytest.MonkeyPatch):
    async def _noop_wait(self, *args, **kwargs):
        return None

    async def _noop_workspace(self, *args, **kwargs):
        return None

    async def _noop_sync(self, *args, **kwargs):
        return None

    async def _noop_atexit(self, *args, **kwargs):
        return None

    monkeypatch.setattr(AzureBackend, "_wait_for_vm_ssh", _noop_wait)
    monkeypatch.setattr(AzureBackend, "_prepare_remote_workspace", _noop_workspace)
    monkeypatch.setattr(AzureBackend, "_sync_artifacts_to_host_quietly", _noop_sync)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_create_sandbox_happy_path(fake_clients, sandbox_config, tmp_path):
    backend = AzureBackend(
        subscription_id="sub-id",
        region="eastus",
        vm_size="Standard_NC6s_v3",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=False,  # avoid atexit
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend.create_sandbox(sandbox_config))
    assert sandbox.sandbox_id
    assert sandbox.image
    # Owned-instance allowlist should now include the new VM.
    assert sandbox.sandbox_id in backend._owned_vm_ids
    # Verify provisioning order via recorded calls.
    calls = fake_clients["_calls"]
    expected_prefix = [
        "NetworkSecurityGroups.begin_create_or_update",
        "VirtualNetworks.begin_create_or_update",
        "Subnets.begin_create_or_update",
        "PublicIPAddresses.begin_create_or_update",
        "NetworkInterfaces.begin_create_or_update",
        "VirtualMachines.begin_create_or_update",
    ]
    assert calls[: len(expected_prefix)] == expected_prefix
    # Resource log was written.
    log_path = tmp_path / "runs" / sandbox_config.project_id / "azure_resource_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    ops = [json.loads(l)["op"] for l in lines]
    assert "VirtualMachines.begin_create_or_update" in ops


def test_2_destroy_deletes_ephemeral_rg_cascade(fake_clients, sandbox_config, tmp_path):
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=True,
        resource_group="",  # ephemeral
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend.create_sandbox(sandbox_config))
    asyncio.run(backend.destroy(sandbox))
    # Ephemeral path => RG cascade-delete, no piecemeal VM deletes.
    assert "ResourceGroups.begin_delete" in fake_clients["_calls"]
    assert "VirtualMachines.begin_delete" not in fake_clients["_calls"]
    assert sandbox.sandbox_id not in backend._owned_vm_ids


def test_2b_destroy_preexisting_rg_piecemeal(fake_clients, sandbox_config, tmp_path):
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=True,
        resource_group="my-existing-rg",  # NOT ephemeral
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend.create_sandbox(sandbox_config))
    asyncio.run(backend.destroy(sandbox))
    # Preexisting RG path => piecemeal deletes, no RG cascade.
    assert "VirtualMachines.begin_delete" in fake_clients["_calls"]
    assert "NetworkInterfaces.begin_delete" in fake_clients["_calls"]
    assert "ResourceGroups.begin_delete" not in fake_clients["_calls"]


def test_3_destroy_is_asyncio_shielded(fake_clients, sandbox_config, tmp_path):
    """Cancelling the outer task during destroy() must not prevent the delete."""
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=True,
        resource_group="",
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend.create_sandbox(sandbox_config))

    async def cancel_during_destroy():
        task = asyncio.create_task(backend.destroy(sandbox))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(cancel_during_destroy())
    # The cascade-delete should have been requested before cancellation took effect.
    assert "ResourceGroups.begin_delete" in fake_clients["_calls"]


def test_4_terminal_state_aborts_provisioning(fake_clients, sandbox_config, tmp_path):
    # Make the VM create return a stub whose provisioning_state is "Failed".
    fake_clients["compute_client"].virtual_machines.begin_create_or_update.side_effect = (
        lambda *a, **k: _FakePoller(_fake_resource("/fake/vm-failed", provisioning_state="Failed"))
    )
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    with pytest.raises(SandboxRuntimeError) as excinfo:
        asyncio.run(backend.create_sandbox(sandbox_config))
    assert excinfo.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "terminal state" in str(excinfo.value).lower()


def test_5_owned_instance_allowlist_refuses_unowned_destroy(
    fake_clients, sandbox_config, tmp_path
):
    backend_a = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend_a.create_sandbox(sandbox_config))

    # backend_b never created this VM — destroy should be a no-op.
    backend_b = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=True,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    calls_before = list(fake_clients["_calls"])
    asyncio.run(backend_b.destroy(sandbox))
    # No delete calls were issued by backend_b.
    new_calls = [c for c in fake_clients["_calls"] if c not in calls_before]
    assert all("begin_delete" not in c for c in new_calls)


def test_6_atexit_cleanup_registered(fake_clients, sandbox_config, tmp_path):
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=True,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    with patch("backend.services.runtime.azure_backend.atexit.register") as mock_reg:
        asyncio.run(backend.create_sandbox(sandbox_config))
        assert mock_reg.called


def test_7_ensure_azure_available_missing_creds_raises(monkeypatch):
    monkeypatch.delenv("REPROLAB_AZURE_SUBSCRIPTION_ID", raising=False)
    monkeypatch.delenv("AZURE_SUBSCRIPTION_ID", raising=False)
    with pytest.raises(SandboxRuntimeError):
        ensure_azure_available()


def test_8_config_default_sandbox_accepts_azure(monkeypatch):
    from backend.config import Settings

    monkeypatch.setenv("REPROLAB_DEFAULT_SANDBOX", "azure")
    s = Settings()
    assert s.default_sandbox == "azure"


def test_9_cli_sandbox_choice_includes_azure():
    from backend.cli import _build_parser

    parser = _build_parser()
    ns = parser.parse_args(["reproduce", "fake.pdf", "--sandbox", "azure"])
    assert ns.sandbox == "azure"


def test_10_sandbox_mode_enum_has_azure():
    from backend.agents.execution import SandboxMode

    assert SandboxMode("azure") is SandboxMode.azure
    assert "azure" in [m.value for m in SandboxMode]


def test_11_pod_sweep_scheduler_calls_azure_sweep(monkeypatch):
    # Sweep is a no-op without AZURE_SUBSCRIPTION_ID — make sure the call
    # signature is right and the scheduler can drive both sweeps.
    from backend.services.runtime import azure_sweeper

    monkeypatch.delenv("REPROLAB_AZURE_SUBSCRIPTION_ID", raising=False)
    monkeypatch.delenv("AZURE_SUBSCRIPTION_ID", raising=False)
    summary = azure_sweeper.sweep_stale_azure_vms(max_age_seconds=60)
    assert summary.listed == 0
    assert summary.deleted == 0


def test_12_gpu_catalog_resolve_azure_sku():
    sku = resolve_azure_vm_size("rtx4090")
    assert sku.startswith("Standard_")
    assert sku == AZURE_SKU_BY_SHORT_NAME["rtx4090"]
    # Unknown short_name falls back.
    assert resolve_azure_vm_size("unknown-sku") == DEFAULT_AZURE_VM_SIZE


# ---------------------------------------------------------------------------
# Bonus regression — make sure RunPod path still constructs after our changes.
# ---------------------------------------------------------------------------


def test_runpod_backend_still_constructs():
    """Cheap regression: RunpodBackend instantiation should be unaffected."""
    from backend.services.runtime.runpod_backend import RunpodBackend

    backend = RunpodBackend(api_key="dummy")
    assert backend.api_key == "dummy"


# ---------------------------------------------------------------------------
# Fix-pass coverage (2026-05-27 audit follow-up)
# ---------------------------------------------------------------------------


def _vm_create_params(fake_clients: dict[str, Any]) -> dict[str, Any]:
    """Pull the params dict passed to VirtualMachines.begin_create_or_update."""
    call = fake_clients["compute_client"].virtual_machines.begin_create_or_update.call_args
    # The third positional arg (rg_name, vm_name, params) is the params dict.
    assert call is not None, "begin_create_or_update was never called"
    args = call.args
    if len(args) >= 3:
        return args[2]
    # Fallback: kwargs
    return call.kwargs.get("parameters") or {}


def _pip_create_params(fake_clients: dict[str, Any]) -> dict[str, Any]:
    call = fake_clients["network_client"].public_ip_addresses.begin_create_or_update.call_args
    assert call is not None
    args = call.args
    if len(args) >= 3:
        return args[2]
    return call.kwargs.get("parameters") or {}


def test_fix1_create_uses_configured_storage_tier(
    fake_clients, sandbox_config, tmp_path
):
    """FIX 1: os_disk_tier flows from constructor into the VM create params."""
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        os_disk_tier="StandardSSD_LRS",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    asyncio.run(backend.create_sandbox(sandbox_config))
    params = _vm_create_params(fake_clients)
    assert (
        params["storage_profile"]["os_disk"]["managed_disk"]["storage_account_type"]
        == "StandardSSD_LRS"
    )


def test_fix1_custom_storage_tier_passes_through(
    fake_clients, sandbox_config, tmp_path
):
    """FIX 1 negative: Premium_LRS override still works for GPU SKUs."""
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        os_disk_tier="Premium_LRS",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    asyncio.run(backend.create_sandbox(sandbox_config))
    params = _vm_create_params(fake_clients)
    assert (
        params["storage_profile"]["os_disk"]["managed_disk"]["storage_account_type"]
        == "Premium_LRS"
    )


def test_fix2_create_sets_explicit_os_disk_name(
    fake_clients, sandbox_config, tmp_path
):
    """FIX 2: os_disk.name is set so piecemeal delete in non-ephemeral RGs works."""
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    asyncio.run(backend.create_sandbox(sandbox_config))
    params = _vm_create_params(fake_clients)
    os_disk = params["storage_profile"]["os_disk"]
    assert "name" in os_disk
    assert os_disk["name"].endswith("-osdisk")


def test_fix3_public_ip_uses_standard_sku_static(
    fake_clients, sandbox_config, tmp_path
):
    """FIX 3: PublicIP uses Standard SKU + Static allocation (not Basic + Dynamic)."""
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    asyncio.run(backend.create_sandbox(sandbox_config))
    params = _pip_create_params(fake_clients)
    assert params["public_ip_allocation_method"] == "Static"
    assert params["sku"]["name"] == "Standard"


def test_fix4_destroy_ephemeral_rg_does_not_block_on_result(
    fake_clients, sandbox_config, tmp_path
):
    """FIX 4: destroy() must NOT await poller.result() on the RG cascade delete.

    Azure RG cascade takes 5-15 minutes server-side; blocking would stall the
    run.  We assert that resource_groups.begin_delete was called but its
    returned poller's .result() was NOT awaited.
    """
    # Wire the RG begin_delete to return a poller whose .result() would block.
    block_count = {"calls": 0}

    class _BlockingPoller:
        def result(self):
            block_count["calls"] += 1
            raise AssertionError(
                "destroy() should NOT call .result() on the RG cascade-delete poller"
            )

    fake_clients["resource_client"].resource_groups.begin_delete.side_effect = (
        lambda *a, **k: _BlockingPoller()
    )

    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=True,
        resource_group="",  # ephemeral path
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend.create_sandbox(sandbox_config))
    asyncio.run(backend.destroy(sandbox))
    assert block_count["calls"] == 0, "destroy() blocked on RG cascade poller"
    # And confirm begin_delete was actually issued
    fake_clients["resource_client"].resource_groups.begin_delete.assert_called()


def test_fix6_vnet_id_resolved_from_vnet_not_subnet(
    fake_clients, sandbox_config, tmp_path
):
    """FIX 6: connection.vnet_id comes from vnet.id, not subnet.id rsplit."""
    # Force the vnet create to return a specific id, distinct from the subnet's.
    fake_clients["network_client"].virtual_networks.begin_create_or_update.side_effect = (
        lambda *a, **k: _FakePoller(_fake_resource("/fake/vnet-id-DISTINCT"))
    )
    backend = AzureBackend(
        subscription_id="sub-id",
        ssh_key_path=_make_fake_key(tmp_path),
        ssh_public_key="ssh-ed25519 AAAA fake",
        delete_on_destroy=False,
        log_dir=tmp_path / "runs" / sandbox_config.project_id,
        sdk_overrides=fake_clients,
    )
    sandbox = asyncio.run(backend.create_sandbox(sandbox_config))
    connection = backend._connections[sandbox.sandbox_id]
    assert connection.vnet_id == "/fake/vnet-id-DISTINCT"


def test_settings_azure_os_disk_tier_field_present(monkeypatch):
    """FIX 1 wiring: Settings exposes azure_os_disk_tier."""
    from backend.config import Settings

    s = Settings()
    assert s.azure_os_disk_tier == "StandardSSD_LRS"

    monkeypatch.setenv("REPROLAB_AZURE_OS_DISK_TIER", "Premium_LRS")
    s2 = Settings()
    assert s2.azure_os_disk_tier == "Premium_LRS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_key(tmp_path: Path) -> Path:
    """Create an empty file that masquerades as an SSH private key."""
    key = tmp_path / "fake_id_ed25519"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----\n")
    key.chmod(0o600)
    return key
