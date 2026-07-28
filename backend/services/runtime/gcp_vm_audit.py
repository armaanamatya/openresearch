"""Stray-billing audit for the single-VM GCP campaign path.

``VmComputeProvider.teardown()``/``release_gpu()`` (see ``vm_compute_provider.py``)
deliberately only STOP the named GCP VM, never delete it -- it is treated as a
persistent, reusable resource across runs. That is a reasonable design, but it
means the only thing standing between a forgotten run and an indefinitely
RUNNING (GPU-billed) or STOPPED (disk-billed) VM is an operator remembering to
run ``gcloud compute instances list`` by hand -- exactly what
``docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md`` tells them to do every
run, manually. This module makes that check a single command instead.

Pure argv-builder + injected-runner shape, mirroring
``vm_compute_provider.py``'s own convention: nothing here executes a real
subprocess in a unit test.  Read-only -- this module NEVER stops or deletes a
VM; it only reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from backend.services.runtime.vm_compute_provider import (
    _DEFAULT_INSTANCE,
    _DEFAULT_PROJECT,
    _DEFAULT_ZONE,
    VmExecResult,
    _default_subprocess_runner,
    _env,
)

# The GPU instance is always audited; the CPU staging instance (Phase 1d
# cpu_warm_disk_then_gpu_attach) only exists as a distinct resource when that
# tiering strategy is in use, but auditing it too is harmless (NOT_FOUND is a
# valid, non-alarming outcome for an operator who never uses that strategy).
_INSTANCE_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("gpu", "OPENRESEARCH_GCP_INSTANCE"),
    ("cpu-staging", "OPENRESEARCH_GCP_CPU_INSTANCE"),
)


@dataclass(frozen=True)
class VmAuditTarget:
    label: str
    project: str
    zone: str
    instance: str


@dataclass(frozen=True)
class VmAuditFinding:
    target: VmAuditTarget
    status: str  # gcloud instance status, "NOT_FOUND", or "ERROR:<detail>"
    active_local_runs: int
    level: str  # "warn" | "info" | "error"
    message: str


def resolve_audit_targets() -> list[VmAuditTarget]:
    """The configured instance(s) to audit, deduplicated by (project, zone, instance).

    Reads the SAME env vars (and falls back to the SAME literal defaults)
    ``VmComputeProvider`` uses, so this audit never drifts from what a live
    campaign would actually provision.
    """
    project = _env("OPENRESEARCH_GCP_PROJECT", _DEFAULT_PROJECT)
    zone = _env("OPENRESEARCH_GCP_ZONE", _DEFAULT_ZONE)
    gpu_instance = _env("OPENRESEARCH_GCP_INSTANCE", _DEFAULT_INSTANCE)
    cpu_instance = _env("OPENRESEARCH_GCP_CPU_INSTANCE", f"{gpu_instance}-cpu")

    seen: set[tuple[str, str, str]] = set()
    targets: list[VmAuditTarget] = []
    for label, instance in (("gpu", gpu_instance), ("cpu-staging", cpu_instance)):
        key = (project, zone, instance)
        if key in seen:
            continue
        seen.add(key)
        targets.append(VmAuditTarget(label=label, project=project, zone=zone, instance=instance))
    return targets


def build_describe_argv(target: VmAuditTarget) -> list[str]:
    return [
        "gcloud", "compute", "instances", "describe", target.instance,
        "--zone", target.zone,
        "--project", target.project,
        "--format=value(status)",
    ]


def count_active_local_gcp_runs(runs_root: Path) -> int:
    """Best-effort count of locally-tracked runs that look still in-flight.

    Not a precise cross-reference to a specific VM (demo_status.json's
    sandboxMode records the IN-VM execution sandbox, typically "local" for
    the campaign single-VM path, not "gcp" itself) -- this is deliberately
    conservative context an operator can weigh against a RUNNING VM, not a
    hard gate. A read/parse error on any one run dir is skipped, never fatal.
    """
    if not runs_root.is_dir():
        return 0
    count = 0
    for status_path in runs_root.glob("*/demo_status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") in ("running", "queued"):
            count += 1
    return count


def audit(
    *,
    runs_root: Path,
    runner: Callable[[list[str]], VmExecResult] = _default_subprocess_runner,
    targets: list[VmAuditTarget] | None = None,
) -> list[VmAuditFinding]:
    """Describe every audit target and classify the stray-billing risk.

    Pure given an injected ``runner`` -- the default only runs a real
    ``gcloud`` subprocess outside a test. Never stops or deletes anything.
    """
    active_local_runs = count_active_local_gcp_runs(runs_root)
    resolved_targets = targets if targets is not None else resolve_audit_targets()
    findings: list[VmAuditFinding] = []
    for target in resolved_targets:
        result = runner(build_describe_argv(target))
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if "not found" in stderr.lower() or "NOT_FOUND" in stderr:
                findings.append(VmAuditFinding(
                    target=target, status="NOT_FOUND", active_local_runs=active_local_runs,
                    level="info",
                    message=f"[{target.label}] {target.instance}: does not exist — nothing to audit.",
                ))
            else:
                findings.append(VmAuditFinding(
                    target=target, status=f"ERROR:{stderr[:200]}", active_local_runs=active_local_runs,
                    level="error",
                    message=f"[{target.label}] {target.instance}: could not check status ({stderr[:200] or 'unknown error'}).",
                ))
            continue
        status = (result.stdout or "").strip()
        if status == "RUNNING":
            level = "warn"
            note = (
                f"{active_local_runs} local run(s) currently tracked as running/queued — "
                "verify this VM is actually in use before assuming stray billing."
                if active_local_runs
                else "NO local run is tracked as running/queued — likely orphaned and "
                     "actively billing the GPU meter. Investigate before stopping."
            )
            message = f"[{target.label}] {target.instance}: RUNNING. {note}"
        elif status in ("TERMINATED", "STOPPED"):
            level = "info"
            message = (
                f"[{target.label}] {target.instance}: {status} — not GPU-billed, but its "
                "persistent disk still bills indefinitely until stopped VMs are reviewed."
            )
        else:
            level = "info"
            message = f"[{target.label}] {target.instance}: {status or 'UNKNOWN'}."
        findings.append(VmAuditFinding(
            target=target, status=status or "UNKNOWN", active_local_runs=active_local_runs,
            level=level, message=message,
        ))
    return findings


__all__ = [
    "VmAuditFinding",
    "VmAuditTarget",
    "audit",
    "build_describe_argv",
    "count_active_local_gcp_runs",
    "resolve_audit_targets",
]
