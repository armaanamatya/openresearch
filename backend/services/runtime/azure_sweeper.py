"""Periodic sweep of stale Azure VMs tagged ``reprolab=true``.

Sibling of ``pod_sweeper.py``.  Deletes Azure VMs tagged with our reprolab
marker whose ``creation_time`` is older than ``max_age_seconds``.  Used by
both the FastAPI lifespan startup hook and the
``PodSweepScheduler`` background loop.

Auth: ``azure.identity.DefaultAzureCredential`` — same precedence as
``AzureBackend``.  Skipped silently when ``REPROLAB_AZURE_SUBSCRIPTION_ID``
is unset, so this is a no-op for environments that never use the Azure
sandbox.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class AzureSweepSummary:
    """One run of the sweeper, suitable for logging or telemetry."""

    listed: int = 0
    eligible: int = 0
    deleted: int = 0
    failed: int = 0
    skipped_unowned: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"azure_sweeper: listed={self.listed} eligible={self.eligible} "
            f"deleted={self.deleted} failed={self.failed} "
            f"skipped_unowned={self.skipped_unowned}"
        )


def _subscription_id() -> str:
    return (
        os.environ.get("REPROLAB_AZURE_SUBSCRIPTION_ID")
        or os.environ.get("AZURE_SUBSCRIPTION_ID")
        or ""
    ).strip()


def sweep_stale_azure_vms(
    *,
    max_age_seconds: int = 7200,
    dry_run: bool = False,
    sdk_overrides: dict | None = None,
) -> AzureSweepSummary:
    """Delete reprolab-tagged Azure VMs older than ``max_age_seconds``.

    Returns an ``AzureSweepSummary`` describing what was found / deleted.
    Fail-soft: any per-VM failure is logged but the sweep continues.

    Parameters
    ----------
    max_age_seconds:
        Age threshold; VMs younger than this are left alone.  Default 7200
        (2h) matches the pod_sweeper default.
    dry_run:
        When True, list and classify but skip the actual delete call.
    sdk_overrides:
        Test injection.  Dict with ``compute_client``, ``resource_client``
        keys overrides the real SDK clients.
    """
    summary = AzureSweepSummary()
    subscription_id = _subscription_id()
    if not subscription_id:
        return summary  # no Azure usage configured — silent no-op

    try:
        if sdk_overrides:
            compute_client = sdk_overrides["compute_client"]
            resource_client = sdk_overrides["resource_client"]
        else:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.mgmt.resource import ResourceManagementClient

            credential = DefaultAzureCredential()
            compute_client = ComputeManagementClient(credential, subscription_id)
            resource_client = ResourceManagementClient(credential, subscription_id)
    except Exception as exc:  # noqa: BLE001
        summary.errors.append(f"client_init_failed: {exc}")
        return summary

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)

    try:
        vms = list(compute_client.virtual_machines.list_all())
    except Exception as exc:  # noqa: BLE001
        summary.errors.append(f"list_failed: {exc}")
        return summary

    for vm in vms:
        summary.listed += 1
        tags = dict(getattr(vm, "tags", None) or {})
        if tags.get("reprolab") != "true":
            summary.skipped_unowned += 1
            continue

        # Resolve creation time from the VM's read details (list_all does not
        # include time_created on every API version).
        vm_id = str(getattr(vm, "id", "") or "")
        rg_name, vm_name = _parse_vm_id(vm_id)
        if not rg_name or not vm_name:
            continue

        try:
            detail = compute_client.virtual_machines.get(
                rg_name, vm_name, expand="instanceView"
            )
        except Exception as exc:  # noqa: BLE001
            summary.errors.append(f"get_failed:{vm_name}: {exc}")
            continue

        created = getattr(detail, "time_created", None) or _extract_created_from_view(detail)
        if created is None:
            # If we can't determine age, skip — we'd rather leak a VM than
            # delete a fresh one.
            continue
        if not isinstance(created, datetime):
            try:
                created = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        if created > cutoff:
            continue  # too young

        summary.eligible += 1
        if dry_run:
            continue

        # Ephemeral RG?  Tag policy says ephemeral resource groups created by
        # AzureBackend have the same reprolab=true marker AND share a name
        # prefix with the VM's RG.  When the RG itself is tagged reprolab,
        # cascade-delete the whole RG (cheapest).  Otherwise piecemeal-delete
        # the VM only.
        try:
            rg_obj = resource_client.resource_groups.get(rg_name)
            rg_tags = dict(getattr(rg_obj, "tags", None) or {})
        except Exception:  # noqa: BLE001
            rg_tags = {}

        try:
            if rg_tags.get("reprolab") == "true":
                resource_client.resource_groups.begin_delete(rg_name)
            else:
                compute_client.virtual_machines.begin_delete(rg_name, vm_name)
            summary.deleted += 1
        except Exception as exc:  # noqa: BLE001
            summary.failed += 1
            summary.errors.append(f"delete_failed:{vm_name}: {exc}")

    return summary


def _parse_vm_id(vm_id: str) -> tuple[str, str]:
    """Extract (resource_group, vm_name) from an ARM resource id."""
    if "/resourceGroups/" not in vm_id or "/virtualMachines/" not in vm_id:
        return ("", "")
    try:
        after_rg = vm_id.split("/resourceGroups/", 1)[1]
        rg_name = after_rg.split("/", 1)[0]
        after_vm = vm_id.split("/virtualMachines/", 1)[1]
        vm_name = after_vm.split("/", 1)[0]
        return (rg_name, vm_name)
    except (IndexError, ValueError):
        return ("", "")


def _extract_created_from_view(detail) -> datetime | None:
    """Best-effort extraction of creation time from VM instanceView."""
    view = getattr(detail, "instance_view", None)
    if view is None:
        return None
    statuses = getattr(view, "statuses", None) or []
    for status in statuses:
        code = str(getattr(status, "code", "") or "")
        if code.startswith("ProvisioningState/succeeded"):
            t = getattr(status, "time", None)
            if t is not None:
                return t if isinstance(t, datetime) else None
    return None


__all__ = ["AzureSweepSummary", "sweep_stale_azure_vms"]
