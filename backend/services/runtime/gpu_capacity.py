"""Backend-agnostic GPU capacity descriptor (spec 2026-05-31).

Single source of truth for the question every stage of OOM/capacity handling
asks: *how many GPUs do I have, how big is each, which can I pin a cell to, and
can I escalate to a bigger one?*  Consumers:

* the capacity gate (PREVENT) — clamp the training matrix to ``per_gpu_vram_gb``
* ``implement_baseline`` guidance — tell the agent its GPU budget
* ``run_experiment`` — size the ``gpu_cell_runner`` pool + decide the escalate axis

Per-backend providers (``describe_capacity`` dispatches on ``ctx.sandbox_mode``):

==================  =========================================  ============
backend             capacity source                            can_escalate
==================  =========================================  ============
local / docker      local_gpu_allocator.discover_gpus()        False
runpod / brev       provisioned pod SKU (ctx.gpu_plan)          True
azure               AKS settings + gpu_plan.json (plan-aware)  False (see _describe_azure)
==================  =========================================  ============

The descriptor reports **raw physical** capacity; the headroom multiplier
(``OPENRESEARCH_DYNAMIC_GPU_HEADROOM``) is applied by the capacity *gate*, not here,
so this stays a pure observation of the hardware.  Every nvidia-smi touch is
wrapped so a probe failure degrades to "no GPUs" rather than crashing a run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from backend.services.runtime import local_gpu_allocator as _alloc

logger = logging.getLogger(__name__)

__all__ = ["GpuCapacity", "describe_capacity"]

_MB_PER_GB = 1024.0


@dataclass(frozen=True)
class GpuCapacity:
    """A backend-agnostic snapshot of usable GPU capacity for one run.

    Attributes:
        backend_kind:    ``"local"`` | ``"docker"`` | ``"runpod"`` | ``"brev"`` | ``"azure"``.
        num_gpus:        Usable GPU count — the leased/free cards (local) or the
                         provisioned pod's GPU count (cloud).
        per_gpu_vram_gb: VRAM of the *smallest* usable card — the binding per-cell
                         budget the capacity gate clamps against.  ``0.0`` means
                         "unknown" (nvidia-smi unavailable); the gate must not
                         block on unknown capacity.
        free_gpu_ids:    Device ids to pin cells to (UUIDs on local, indices on
                         cloud).  Consumed by ``gpu_cell_runner.run_matrix(gpus=...)``.
        can_escalate:    Whether a bigger SKU is reachable (cloud catalog ladder).
        total_vram_gb:   Sum across usable cards (informational).
        detail:          Backend-specific extras (sku name, leased flag, ...).
    """

    backend_kind: str
    num_gpus: int
    per_gpu_vram_gb: float
    free_gpu_ids: tuple[str, ...]
    can_escalate: bool
    total_vram_gb: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.num_gpus <= 0

    def fits(self, required_gb: float, *, headroom: float = 1.0) -> bool:
        """True iff a workload needing ``required_gb`` fits one card with ``headroom``.

        Unknown capacity (``per_gpu_vram_gb <= 0``) returns True — the gate must
        not block when nvidia-smi could not report sizes.
        """
        if self.per_gpu_vram_gb <= 0:
            return True
        return required_gb * headroom <= self.per_gpu_vram_gb


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def describe_capacity(ctx: Any) -> GpuCapacity:
    """Return the :class:`GpuCapacity` for the run described by ``ctx``.

    Dispatches on ``ctx.sandbox_mode``.  Reads ``ctx.gpu_device_ids`` (the leased
    UUID/index list) on local, ``ctx.gpu_plan`` on cloud.  All ``ctx`` access is
    via ``getattr`` so a partial/duck-typed context (or a test namespace) works.
    """
    kind = _backend_kind(ctx)
    if kind in ("runpod", "brev"):
        return _describe_cloud(ctx, kind)
    if kind == "azure":
        return _describe_azure(ctx)
    return _describe_local(ctx, kind)


def _backend_kind(ctx: Any) -> str:
    raw = getattr(ctx, "sandbox_mode", None)
    name = (getattr(raw, "value", None) or getattr(raw, "name", None) or str(raw or "")).lower()
    for k in ("runpod", "brev", "azure", "docker"):
        if k in name:
            return k
    return "local"


# ---------------------------------------------------------------------------
# Local / docker (host GPUs)
# ---------------------------------------------------------------------------

def _describe_local(ctx: Any, kind: str) -> GpuCapacity:
    leased = _normalize_ids(getattr(ctx, "gpu_device_ids", None))
    override = _vram_override_gb(ctx)
    devices = _safe_discover()
    by_id: dict[str, Any] = {}
    for d in devices:
        by_id[d.uuid] = d
        by_id[str(d.index)] = d

    if leased:
        # The run already holds a lease — those ARE our usable cards (do NOT
        # re-run free_devices; our own training procs would look "foreign").
        matched = [by_id[i] for i in leased if i in by_id]
        if matched:
            per_gpu = min(d.memory_total_mb for d in matched) / _MB_PER_GB
            total = sum(d.memory_total_mb for d in matched) / _MB_PER_GB
        else:
            # smi failed or id-form mismatch — trust the lease count, size from
            # the manual override (or 0 = unknown).
            per_gpu = override
            total = override * len(leased)
        return GpuCapacity(kind, len(leased), per_gpu, leased, can_escalate=False,
                           total_vram_gb=total, detail={"leased": True})

    # No lease yet (planning-time query) — what is free right now?
    try:
        free = _alloc.free_devices(devices) if devices else []
    except Exception as exc:  # nvidia-smi quirk must never crash a run
        logger.warning("describe_capacity: free_devices failed (%s)", exc)
        free = []
    if not free:
        return GpuCapacity(kind, 0, override, (), can_escalate=False, detail={"leased": False})
    ids = tuple(d.uuid for d in free)
    per_gpu = min(d.memory_total_mb for d in free) / _MB_PER_GB
    total = sum(d.memory_total_mb for d in free) / _MB_PER_GB
    return GpuCapacity(kind, len(free), per_gpu, ids, can_escalate=False,
                       total_vram_gb=total, detail={"leased": False})


# ---------------------------------------------------------------------------
# Cloud (runpod / brev) — provisioned pod
# ---------------------------------------------------------------------------

def _describe_cloud(ctx: Any, kind: str) -> GpuCapacity:
    plan = getattr(ctx, "gpu_plan", None)
    vram = _plan_attr(plan, "vram_gb") or _vram_override_gb(ctx)
    try:
        count = int(_plan_attr(plan, "gpu_count") or 1)
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)
    # On the pod, gpu_cell_runner re-discovers via nvidia-smi; indices suffice.
    ids = tuple(str(i) for i in range(count))
    vram_f = float(vram or 0.0)
    return GpuCapacity(kind, count, vram_f, ids, can_escalate=True,
                       total_vram_gb=vram_f * count,
                       detail={"sku": _plan_attr(plan, "short_name")})


def _describe_azure(ctx: Any) -> GpuCapacity:
    """Plan-aware Azure AKS GPU capacity descriptor.

    Tries to load the resolved ``gpu_plan.json`` from
    ``<ctx.project_dir>/rlm_state/gpu_plan.json`` (same fail-soft pattern as
    ``run_experiment``).  When the plan is present *and* is an Azure plan
    (``cloud_type == "ONDEMAND"`` or ``short_name.startswith("azure_")``),
    ``per_gpu_vram_gb`` is taken from ``plan.vram_gb``.  In all other cases
    (no plan, unreadable file, non-azure plan, missing ctx attribute) the
    settings defaults are used so capacity queries made before
    ``resolve_gpu_requirements`` runs are still valid.

    ``num_gpus`` always comes from ``azure_max_nodes`` (the AKS node-pool
    concurrency cap) — the plan's ``gpu_count`` is a per-node count and the
    azure runner scales horizontally, not per-GPU.

    ``can_escalate=False``: the field specifically guards the *run_experiment
    monolithic SKU-ladder escalation loop* (the per-cell OOM retry path that
    advances through ``gpu_plan.ladder_remaining`` inside the RunPod backend).
    That loop does NOT apply to AKS — the azure runner dispatches Kubernetes Jobs
    that self-escalate via a different mechanism (node-pool SKU selection at Job
    dispatch time).  Setting ``can_escalate=True`` here would mislead that RunPod
    loop into attempting ladder escalation on an azure pod, which would fail.
    The azure runner's own SKU escalation operates independently and is correct
    regardless of this flag.

    Full design: docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md.
    """
    import json
    from pathlib import Path

    from backend.config import get_settings

    s = get_settings()
    num_gpus = max(1, int(s.azure_max_nodes))
    per_gpu_vram_gb = float(s.azure_per_gpu_vram_gb)

    # Try to load the resolved gpu_plan and refine per_gpu_vram_gb from it.
    try:
        project_dir = getattr(ctx, "project_dir", None)
        if project_dir is not None:
            plan_path = Path(project_dir) / "rlm_state" / "gpu_plan.json"
            if plan_path.exists():
                raw = json.loads(plan_path.read_text(encoding="utf-8"))
                short_name = raw.get("short_name", "") or ""
                cloud_type = raw.get("cloud_type", "") or ""
                is_azure_plan = (
                    cloud_type == "ONDEMAND"
                    or (isinstance(short_name, str) and short_name.startswith("azure_"))
                )
                if is_azure_plan:
                    plan_vram = raw.get("vram_gb")
                    if plan_vram is not None:
                        per_gpu_vram_gb = float(plan_vram)
    except Exception:  # noqa: BLE001 — capacity probe must never raise
        logger.warning(
            "_describe_azure: gpu_plan.json present but unreadable; using settings defaults",
            exc_info=True,
        )

    ids = tuple(str(i) for i in range(num_gpus))
    return GpuCapacity(
        "azure",
        num_gpus=num_gpus,
        per_gpu_vram_gb=per_gpu_vram_gb,
        free_gpu_ids=ids,
        can_escalate=False,
        total_vram_gb=per_gpu_vram_gb * num_gpus,
        detail={"node_pool": s.azure_node_pool_name},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ids(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(",", " ").split()]
    else:
        parts = [str(v).strip() for v in value]
    return tuple(p for p in parts if p)


def _vram_override_gb(ctx: Any) -> float:
    raw = getattr(ctx, "vram_override", None)
    if raw in (None, ""):
        raw = os.environ.get("OPENRESEARCH_VRAM_OVERRIDE_GB")
    try:
        return float(raw) if raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_discover() -> list:
    try:
        return _alloc.discover_gpus()
    except Exception as exc:  # nvidia-smi quirk must never crash a run
        logger.warning("describe_capacity: discover_gpus failed (%s)", exc)
        return []


def _plan_attr(plan: Any, key: str) -> Any:
    if plan is None:
        return None
    if isinstance(plan, dict):
        return plan.get(key)
    return getattr(plan, key, None)
