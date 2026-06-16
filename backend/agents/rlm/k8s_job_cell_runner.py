"""AKS Job cell runner — drop-in replacement for gpu_cell_runner.run_matrix.

Submits one Kubernetes Job per training cell, watches status via the K8s API,
pulls per-cell ``metrics.json`` from Azure Blob, and returns the identical
``{cell_id → CellResult.to_dict()}`` shape consumed by
``cell_matrix.aggregate_cell_metrics``.

Key design choices (locked 2026-06-03):

* **Drop-in signature** — ``run_matrix`` is byte-for-byte compatible with
  ``gpu_cell_runner.run_matrix``; the caller in ``primitives._execute_cell_matrix``
  selects the runner by sandbox mode and calls it with identical args.
* **gpus ignored** — K8s scheduler places each Job; ``gpus`` is accepted but
  silently ignored.  ``gpus_per_cell != 1`` → every non-skipped cell returns
  ``"error"`` (Azure Jobs are 1 GPU each by design; multi-GPU is via gpu_plan).
* **OOM in-Job** — the Job wrapper owns the shrink ladder; the orchestrator
  resubmits on OOM only when a ``gpu_plan`` with ``ladder_remaining`` is bound.
  ``max_oom_retries`` is forwarded to the Job as
  ``OPENRESEARCH_CELL_MAX_OOM_RETRIES``.
* **Blob artifact bus** — code is uploaded once from
  ``Path(cell_script).parent``; per-cell ``metrics.json`` is downloaded after
  Job completion.
* **Resume parity** — mirrors ``gpu_cell_runner``'s Track-B resume: when
  ``OPENRESEARCH_RESUME_CELLS`` is truthy, a prior ``cell_manifest.json`` with
  ``status=="ok"`` + matching fingerprint + not in ``force_cells`` → skip.
* **Budget** — before each submit, reserved GPU-seconds are checked against
  ``RunBudget.max_pod_seconds`` and ``RunBudget.max_run_gpu_usd``; caps
  sourced from a ``bind_run_context``-injected context var.
* **Dynamic gpu_plan** — when a ``GpuPlan`` is bound via ``bind_run_context``,
  the Job manifest targets the plan's SKU node pool and GPU count.  On
  ``oom_failed``, the runner escalates through ``plan.ladder_remaining``
  (bounded by ``settings.dynamic_gpu_max_escalations``, default 2), emitting
  ``gpu_escalated`` events.  No crash/loop on empty ladder or unprovisioned SKU.
* **Lazy K8s imports** — ``kubernetes`` is NOT installed in the dev venv.
  All ``kubernetes.*`` calls go through ``_k8s_factory()`` which may be
  monkeypatched in tests.  Module import always succeeds.
* **Concurrent-safe context** — ``bind_run_context`` uses a ``ContextVar``
  so multiple concurrent runs (threads) each see their own budget/sink/plan.
* **DRY cell helpers** — ``CellResult``, ``headline_metric``, ``load_cell_manifest``,
  ``should_skip_cell``, ``write_cell_manifest``, ``is_resume_armed``,
  ``deadline_from_timeout``, ``clamp_cell_timeout``, and the ``STATUS_*``
  constants are all imported from ``cell_scheduler`` (the shared module).

Status mapping:

    Job Complete + wrapper exit 0 + valid artifact  →  "ok"
    wrapper exit 42 / sentinel outcome oom_shrink_exhausted  →  "oom_failed"
    exit 40/41/43/44, Job Failed, deadline, overall timeout  →  "error"
    Pending beyond azure_pending_timeout_seconds  →  "error" (prefix capacity_exhausted:)
    Resume hit  →  "skipped"

SSE events — only EXISTING types are emitted (``run_warning``, ``gpu_escalated``).
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

# ---------------------------------------------------------------------------
# Shared cell-scheduler symbols (DRY adoption)
# ---------------------------------------------------------------------------

from backend.agents.rlm.cell_scheduler import (  # noqa: E402
    CELL_MANIFEST_NAME,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_OOM_FAILED,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,  # noqa: F401  re-exported for callers (tests assert kjcr.STATUS_TIMEOUT)
    CellResult,
    clamp_cell_timeout,  # noqa: F401  re-exported for callers
    deadline_from_timeout,
    headline_metric,  # noqa: F401  re-exported for callers
    is_resume_armed,
    load_cell_manifest,
    should_skip_cell,
    write_cell_manifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CELL_MANIFEST_NAME",
    "run_matrix",
    "bind_run_context",
]

# Exit codes from the in-Job wrapper that map to specific statuses.
_EXIT_OOM_EXHAUSTED = 42
_EXIT_TERMINAL = frozenset({40, 41, 43, 44})

# Blob sub-paths (relative to run prefix).
_BLOB_CODE_PREFIX = "code"
_BLOB_CELLS_PREFIX = "cells"

# Sentinel outcome written by the wrapper when OOM ladder is exhausted.
_SENTINEL_OOM_OUTCOME = "oom_shrink_exhausted"

# Default fallback values used when a settings attribute is absent (defensive,
# so the module imports + tests run against a partial/older config).
_SETTINGS_DEFAULTS: dict[str, Any] = {
    # --- Azure / AKS ---
    "azure_namespace": "reprolab",
    "azure_service_account": "reprolab-sa",
    "azure_node_pool_name": "gpunodes",
    # P1-fix-5: empty string fallback — ERROR clearly at submit if blank rather
    # than silently using a floating :latest tag.
    "azure_base_image": "",
    "azure_storage_account": "",
    "azure_blob_container": "reprolab-artifacts",
    "azure_files_share": "reprolab-cache",
    # P1-fix-5: aligns with config.py default of 4.
    "azure_max_nodes": 4,
    "azure_per_gpu_vram_gb": 80.0,
    "azure_gpu_usd_per_hour": 3.67,
    "azure_pending_timeout_seconds": 900,
    "azure_boot_timeout_seconds": 900,
    "azure_gpu_skus": [],        # provisioned SKU short_names (list[str])
    "dynamic_gpu_max_escalations": 2,
    # P1-fix-8: configurable knobs (config.py additions by another agent).
    "azure_ttl_seconds_after_finished": 3600,
    "azure_job_backoff_limit": 0,
    "azure_cache_mount_path": "/mnt/reprolab-cache",
    "azure_watch_poll_interval_s": 5.0,
    "azure_cell_oom_batch_scale_step1": 0.5,
    "azure_cell_oom_batch_scale_floor": 0.25,
    "azure_bootstrap_pip_timeout_s": 600,
    # --- GCP / GKE ---
    "gcp_namespace": "reprolab",
    "gcp_service_account": "reprolab-sa",
    "gcp_node_pool_name": "gpua100",
    "gcp_base_image": "",
    "gcp_max_nodes": 4,
    "gcp_gpu_usd_per_hour": 3.67,
    "gcp_pending_timeout_seconds": 900,
    "gcp_gpu_skus": ["gcp_a100_80"],
    "gcp_ttl_seconds_after_finished": 3600,
    "gcp_job_backoff_limit": 0,
    "gcp_cache_mount_path": "/mnt/reprolab-cache",
    "gcp_watch_poll_interval_s": 5.0,
    "gcp_cell_oom_batch_scale_step1": 0.5,
    "gcp_cell_oom_batch_scale_floor": 0.25,
    "gcp_bootstrap_pip_timeout_s": 600,
}

# ---------------------------------------------------------------------------
# Context vars — concurrent-safe budget + event-sink + gpu_plan + prefix injection
# ---------------------------------------------------------------------------

_RUN_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "k8s_job_cell_runner_context", default={}
)

# Separate ContextVar for the cloud prefix so that callers that patch
# bind_run_context in tests (with 3-arg fakes) remain unaffected.
_SETTINGS_PREFIX_CTX: ContextVar[str] = ContextVar(
    "k8s_job_cell_runner_settings_prefix", default="azure"
)


@contextmanager
def _bind_settings_prefix(prefix: str) -> Iterator[None]:
    """Set the cloud-provider settings prefix for the duration of the ``with`` block.

    Used by ``primitives._execute_cell_matrix`` alongside ``bind_run_context`` to
    activate the GCP path without breaking existing tests that patch
    ``bind_run_context`` with 3-argument fakes.

    Example::

        with _bind_settings_prefix("gcp"), bind_run_context(...):
            run_matrix(...)
    """
    token = _SETTINGS_PREFIX_CTX.set(prefix)
    try:
        yield
    finally:
        _SETTINGS_PREFIX_CTX.reset(token)


@contextmanager
def bind_run_context(
    *,
    run_budget: Any | None = None,
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    gpu_plan: Any | None = None,
    settings_prefix: str | None = None,
) -> Iterator[None]:
    """Bind a ``RunBudget``, event sink, and/or a ``GpuPlan`` for the duration of the
    ``with`` block.

    Uses a ``ContextVar`` so concurrent runs on different threads each see their
    own context without interfering.  The wiring layer (``primitives.py``) calls
    this around ``run_matrix`` to inject the budget, an SSE event sink, and the
    resolved GPU plan without changing ``run_matrix``'s signature.

    Args:
        run_budget:       A ``RunBudget`` instance (or None to disable checks).
        event_sink:       ``(event_type, payload) → None`` called to emit EXISTING SSE
                          events (``primitive_call``, ``repl_iteration``, ``run_warning``,
                          ``gpu_escalated``).  Default no-op when None.
        gpu_plan:         A ``GpuPlan`` instance (or None to use default pool + 1 GPU).
                          When bound, Jobs target the plan's SKU node pool; OOM cells
                          escalate through ``plan.ladder_remaining``.
        settings_prefix:  Cloud provider prefix for settings reads, e.g. ``"azure"``
                          or ``"gcp"``.  ``None`` (the default) means "do not change
                          the prefix" — it preserves whatever ``_bind_settings_prefix``
                          already bound, so the ``primitives.py`` nesting
                          ``with _bind_settings_prefix("gcp"), bind_run_context():``
                          resolves to ``"gcp"`` instead of being clobbered back to the
                          azure default.  Pass an explicit value only when binding the
                          prefix directly through this context manager.

    Example::

        with bind_run_context(
            run_budget=ctx.run_budget,
            event_sink=ctx.emit,
            gpu_plan=ctx.gpu_plan,
        ):
            results = k8s_job_cell_runner.run_matrix(cells, cell_script, ...)
    """
    token = _RUN_CONTEXT.set({
        "run_budget": run_budget,
        "event_sink": event_sink,
        "gpu_plan": gpu_plan,
    })
    # Only (re)bind the cloud prefix when one is explicitly passed; a None prefix
    # preserves whatever _bind_settings_prefix already set. This is what makes the
    # primitives nesting resolve correctly and lets test fakes that patch
    # bind_run_context (without a settings_prefix kwarg) keep working.
    prefix_token = (
        _SETTINGS_PREFIX_CTX.set(settings_prefix) if settings_prefix is not None else None
    )
    try:
        yield
    finally:
        _RUN_CONTEXT.reset(token)
        if prefix_token is not None:
            _SETTINGS_PREFIX_CTX.reset(prefix_token)


def _get_run_budget() -> Any | None:
    return _RUN_CONTEXT.get({}).get("run_budget")


def _get_event_sink() -> Callable[[str, dict[str, Any]], None]:
    sink = _RUN_CONTEXT.get({}).get("event_sink")
    return sink if callable(sink) else lambda _t, _p: None


def _get_gpu_plan() -> Any | None:
    """Return the bound GpuPlan (or None when none was bound)."""
    return _RUN_CONTEXT.get({}).get("gpu_plan")


def _get_settings_prefix() -> str:
    """Return the active cloud-provider settings prefix (default ``"azure"``).

    ``_SETTINGS_PREFIX_CTX`` is the single source of truth — set either by
    ``_bind_settings_prefix`` (the ``primitives.py`` path) or by an explicit
    ``settings_prefix=`` on ``bind_run_context``.
    """
    return _SETTINGS_PREFIX_CTX.get("azure")


def _cloud_setting(logical: str, default: Any = None) -> Any:
    """Read ``<prefix>_<logical>`` from settings, using ``_SETTINGS_DEFAULTS`` as fallback.

    Equivalent to ``_setting(f"{_get_settings_prefix()}_{logical}", default)`` but
    centralises the prefix-composition so callers stay readable.
    """
    return _setting(f"{_get_settings_prefix()}_{logical}", default)


# ---------------------------------------------------------------------------
# Lazy K8s client factory — monkeypatchable for tests
# ---------------------------------------------------------------------------

class _K8sClients:
    """Thin container for lazily-initialised K8s client objects."""
    __slots__ = ("batch", "core", "watch_cls")

    def __init__(self, batch: Any, core: Any, watch_cls: Any) -> None:
        self.batch = batch
        self.core = core
        self.watch_cls = watch_cls


_k8s_clients_override: _K8sClients | None = None  # test seam


def _k8s_factory() -> _K8sClients:
    """Return initialised K8s client objects.  Import kubernetes lazily."""
    if _k8s_clients_override is not None:
        return _k8s_clients_override

    from kubernetes import client as k8s_client  # type: ignore[import]
    from kubernetes import config as k8s_config  # type: ignore[import]
    from kubernetes.watch import Watch as K8sWatch  # type: ignore[import]

    # P1-fix-6: try incluster first (inside a pod), fall back to kubeconfig for
    # local/dev use.  This mirrors aks_job_backend and avoids spurious file-not-
    # found errors when running inside a cluster pod.
    try:
        k8s_config.load_incluster_config()
    except Exception:
        try:
            k8s_config.load_kube_config()
        except Exception as exc:
            raise RuntimeError(f"k8s_job_cell_runner: cannot load kubeconfig: {exc}") from exc

    return _K8sClients(
        batch=k8s_client.BatchV1Api(),
        core=k8s_client.CoreV1Api(),
        watch_cls=K8sWatch,
    )


# ---------------------------------------------------------------------------
# Azure Blob helpers (thin wrappers over backend.services.runtime.azure_blob)
# ---------------------------------------------------------------------------

def _blob_upload_prefix(
    local_root: str | Path,
    *,
    blob_prefix: str,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> list[str]:
    """Delegate to the active ObjectStore's upload_prefix.

    ``account_name`` and ``container_name`` are kept in the signature for
    call-site compatibility but are now unused for routing — the ObjectStore
    encapsulates those details.
    """
    return _object_store().upload_prefix(local_root, blob_prefix=blob_prefix)


def _blob_download_bytes(
    blob_name: str,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> bytes:
    """Delegate to the active ObjectStore's download_bytes."""
    return _object_store().download_bytes(blob_name)


def _blob_download_artifact(
    blob_name: str,
    destination: str | Path,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> Path:
    """Delegate to the active ObjectStore's download_artifact."""
    return _object_store().download_artifact(blob_name, destination)


# ---------------------------------------------------------------------------
# Cloud ObjectStore factory — monkeypatchable for tests
# ---------------------------------------------------------------------------

def _object_store() -> Any:
    """Return the active ObjectStore for the current cloud prefix.

    Resolved lazily from the bound ``settings_prefix`` (``"azure"`` → AzureBlobStore
    via ``_AZURE_CLOUD``; ``"gcp"`` → GcsStore via ``_GCP_CLOUD``).  Lazy imports
    prevent circular-import issues; ``_AZURE_CLOUD``/``_GCP_CLOUD`` live in their
    respective thin-adapter modules which already import ``k8s_job_backend``.

    Monkeypatch this symbol in tests via::

        monkeypatch.setattr(kjcr, "_object_store", lambda: FakeStore())
    """
    prefix = _get_settings_prefix()
    if prefix == "gcp":
        from backend.services.runtime.gke_job_backend import _GCP_CLOUD  # type: ignore[import]
        return _GCP_CLOUD.make_object_store(_get_settings(), None)
    else:
        # Default to Azure (covers "azure" and any unknown prefix).
        account = _setting("azure_storage_account", "") or ""
        container = _setting("azure_blob_container", "reprolab-artifacts") or "reprolab-artifacts"
        from backend.services.runtime.k8s_job_backend import AzureBlobStore  # type: ignore[import]
        return AzureBlobStore(account, container, None)


# ---------------------------------------------------------------------------
# Azure Blob ContainerClient factory — monkeypatchable for tests
# ---------------------------------------------------------------------------

def _make_blob_client(
    account_name: str,
    container_name: str,
) -> Any | None:
    """Build one ``ContainerClient`` using ``DefaultAzureCredential``.

    **Scale fix (P0-scale-2):** called ONCE per ``run_matrix`` invocation so
    that all cells share a single authenticated connection rather than each
    ``_try_download_*`` / ``_try_reconcile_status`` call constructing its own
    client (and triggering a fresh MSI credential probe each time).

    Returns ``None`` when:
    * ``account_name`` is empty (storage not configured — local/test path).
    * The azure SDK is absent (test environments without the extra dep).

    Tests monkeypatch this symbol via ``kjcr._make_blob_client = …`` to return
    a fake ContainerClient (or None) without touching the azure SDK at all.
    The ``_try_*`` helpers gracefully forward ``None`` to the azure_blob layer,
    which then also receives ``client=None`` and tries its own construction —
    but those helpers already catch all exceptions, so the worst case is a
    logged debug warning.
    """
    # GCP uses its own client internally via GcsStore; no shared ContainerClient.
    if _get_settings_prefix() == "gcp":
        return None
    if not account_name:
        return None
    try:
        from backend.services.runtime import azure_blob  # type: ignore[import]
        return azure_blob._make_container_client(account_name, container_name)
    except Exception as exc:
        logger.debug(
            "k8s_job_cell_runner: could not build ContainerClient "
            "(account=%s container=%s): %s",
            account_name, container_name, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Settings helper
# ---------------------------------------------------------------------------

def _get_settings() -> Any:
    try:
        from backend.config import get_settings  # type: ignore[import]
        return get_settings()
    except Exception:
        return None


def _setting(name: str, default: Any = None) -> Any:
    s = _get_settings()
    if s is not None:
        val = getattr(s, name, _SETTINGS_DEFAULTS.get(name, default))
        if val is not None:
            return val
    return _SETTINGS_DEFAULTS.get(name, default)


# ---------------------------------------------------------------------------
# DNS-safe Job name
# ---------------------------------------------------------------------------

_DNS_SAFE_RE = re.compile(r"[^a-z0-9-]")

_JOB_NAME_PREFIX = "reprolab-cell-"
_JOB_NAME_MAX = 63


def _job_name(cell_id: str, run_id: str = "") -> str:
    """Return a deterministic DNS-safe K8s Job name for ``cell_id``."""
    safe_run = _DNS_SAFE_RE.sub("-", run_id.lower())[:16] if run_id else ""
    safe_cell = _DNS_SAFE_RE.sub("-", cell_id.lower())
    suffix = f"{safe_run}-{safe_cell}" if safe_run else safe_cell
    # Trim to fit the 63-char limit, keeping the prefix.
    max_suffix = _JOB_NAME_MAX - len(_JOB_NAME_PREFIX)
    suffix = suffix[:max_suffix].strip("-")
    return f"{_JOB_NAME_PREFIX}{suffix}"


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def _check_budget(
    *,
    run_budget: Any | None,
    reserved_gpu_seconds: float,
    gpu_usd_per_hour: float,
    cell_id: str,
) -> str | None:
    """Return an error string if budget caps are exceeded, else None."""
    if run_budget is None:
        return None
    # GPU-USD cap
    max_run_gpu_usd = getattr(run_budget, "max_run_gpu_usd", None)
    if max_run_gpu_usd and max_run_gpu_usd > 0:
        projected_usd = reserved_gpu_seconds * gpu_usd_per_hour / 3600.0
        if projected_usd >= max_run_gpu_usd:
            return (
                f"budget: projected GPU spend ${projected_usd:.4f} "
                f">= max_run_gpu_usd ${max_run_gpu_usd:.4f} for cell={cell_id}"
            )
    # Pod-seconds cap
    max_pod_seconds = getattr(run_budget, "max_pod_seconds", None)
    if max_pod_seconds and max_pod_seconds > 0:
        if reserved_gpu_seconds >= max_pod_seconds:
            return (
                f"budget: reserved_gpu_seconds {reserved_gpu_seconds:.0f}s "
                f">= max_pod_seconds {max_pod_seconds:.0f}s for cell={cell_id}"
            )
    return None


# ---------------------------------------------------------------------------
# K8s Job manifest builder
# ---------------------------------------------------------------------------

def _build_job_manifest(
    *,
    job_name: str,
    namespace: str,
    service_account: str,
    node_pool_name: str,
    base_image: str,
    storage_account: str,
    blob_container: str,
    files_share: str,
    cell_id: str,
    cell_params_json: str,
    output_blob_prefix: str,
    code_blob_prefix: str,
    active_deadline_seconds: int,
    max_oom_retries: int,
    fingerprint: str | None,
    now_iso: str | None,
    gpu_plan: Any | None = None,
    # P1-fix-8: configurable knobs injected by the caller from settings.
    ttl_seconds_after_finished: int = 3600,
    backoff_limit: int = 0,
    cache_mount_path: str = "/mnt/reprolab-cache",
    # P1-fix-9: OOM shrink ratios forwarded to the in-Job wrapper.
    oom_batch_scale_step1: float = 0.5,
    oom_batch_scale_floor: float = 0.25,
    # pip bootstrap timeout
    bootstrap_pip_timeout_s: int = 600,
    default_sku: str | None = None,
    # Cloud-specific pod template labels (e.g. AKS Workload Identity).
    # When None the function uses the azure default ({"azure.workload.identity/use": "true"})
    # to preserve backward compatibility for callers that don't pass this param.
    pod_template_extra_labels: dict | None = None,
) -> dict[str, Any]:
    """Build the K8s Job manifest dict for a single training cell.

    When ``gpu_plan`` is provided the manifest uses:
    - ``nodeSelector = {"reprolab/sku": plan.short_name}`` (infra pool label contract)
    - GPU resource request/limit ``nvidia.com/gpu = plan.gpu_count``
    - taint toleration for ``nvidia.com/gpu`` (operator Exists)

    Without ``gpu_plan`` the manifest falls back to ``{"reprolab/sku": default_sku}``
    (P0-fix-3) so the pod is always placed on a GPU node in the correct pool.

    ``pod_template_extra_labels``:
        Additional labels merged into the pod template metadata.  When ``None``
        (the default) the Azure Workload Identity label is applied, preserving
        byte-for-byte compatibility for existing callers.  Pass ``{}`` explicitly
        for GCP (or any cloud that does not need WI labels).
    """
    # P1-fix-5: refuse to submit with an empty image tag rather than silently
    # using whatever :latest resolves to at runtime.
    if not base_image:
        _img_setting = f"{_get_settings_prefix()}_base_image"
        raise ValueError(
            f"k8s_job_cell_runner: {_img_setting} is empty — set the "
            f"OPENRESEARCH_{_img_setting.upper()} config field before submitting K8s Jobs."
        )

    # Resolve default_sku lazily from the active cloud context when not supplied.
    # This ensures the correct cloud-specific default (gcp_a100_80 vs azure_a100_80)
    # is used even when _build_job_manifest is called directly without default_sku.
    if default_sku is None:
        _gpu_skus_for_default: list = _cloud_setting("gpu_skus", []) or []
        default_sku = str(_gpu_skus_for_default[0]) if _gpu_skus_for_default else "azure_a100_80"

    pvc_name = f"{namespace}-files-pvc"

    # P0-fix-1: env-var NAMES must exactly match what aks_cell_entrypoint.py reads.
    # Canonical contract (runner injects → entrypoint reads):
    #   OPENRESEARCH_CELL_ID               → os.environ.get("OPENRESEARCH_CELL_ID")
    #   OPENRESEARCH_CELL_PARAMS           → os.environ.get("OPENRESEARCH_CELL_PARAMS")
    #   OPENRESEARCH_CELL_OUTPUT_DIR       → env["OPENRESEARCH_CELL_OUTPUT_DIR"] in subprocess
    #   OPENRESEARCH_CELL_MAX_OOM_RETRIES  → os.environ.get("OPENRESEARCH_CELL_MAX_OOM_RETRIES")
    #   OPENRESEARCH_AZURE_STORAGE_ACCOUNT → os.environ.get("OPENRESEARCH_AZURE_STORAGE_ACCOUNT")
    #   OPENRESEARCH_AZURE_BLOB_CONTAINER  → os.environ.get("OPENRESEARCH_AZURE_BLOB_CONTAINER")
    #   OPENRESEARCH_BLOB_CODE_PREFIX      → os.environ.get("OPENRESEARCH_BLOB_CODE_PREFIX")
    #   OPENRESEARCH_BLOB_OUTPUT_PREFIX    → os.environ.get("OPENRESEARCH_BLOB_OUTPUT_PREFIX")
    #   OPENRESEARCH_CACHE_MOUNT           → os.environ.get("OPENRESEARCH_CACHE_MOUNT")
    #   OPENRESEARCH_CELL_OOM_BATCH_SCALE_STEP1  → (entrypoint plan_attempts)
    #   OPENRESEARCH_CELL_OOM_BATCH_SCALE_FLOOR  → (entrypoint plan_attempts)
    #   OPENRESEARCH_BOOTSTRAP_PIP_TIMEOUT_S     → (entrypoint _bootstrap pip install)
    # Cloud-neutral env vars (both clouds).
    env_vars = [
        {"name": "OPENRESEARCH_CELL_ID",               "value": cell_id},
        {"name": "OPENRESEARCH_CELL_PARAMS",            "value": cell_params_json},
        {"name": "OPENRESEARCH_CELL_OUTPUT_DIR",        "value": f"/mnt/outputs/{cell_id}"},
        {"name": "OPENRESEARCH_CELL_MAX_OOM_RETRIES",   "value": str(max_oom_retries)},
        {"name": "OPENRESEARCH_BLOB_CODE_PREFIX",       "value": code_blob_prefix},
        {"name": "OPENRESEARCH_BLOB_OUTPUT_PREFIX",     "value": output_blob_prefix},
        {"name": "OPENRESEARCH_CACHE_MOUNT",            "value": cache_mount_path},
        # P1-fix-9: OOM shrink ratios + pip timeout forwarded from settings.
        {"name": "OPENRESEARCH_CELL_OOM_BATCH_SCALE_STEP1",
         "value": str(oom_batch_scale_step1)},
        {"name": "OPENRESEARCH_CELL_OOM_BATCH_SCALE_FLOOR",
         "value": str(oom_batch_scale_floor)},
        {"name": "OPENRESEARCH_BOOTSTRAP_PIP_TIMEOUT_S",
         "value": str(bootstrap_pip_timeout_s)},
    ]
    # Cloud-specific object-store env vars.
    # Azure: frozen contract with the baked ACR image — keep injecting the exact names.
    # GCP: inject OPENRESEARCH_GCP_GCS_BUCKET (the cloud-neutral pair stays above).
    _prefix = _get_settings_prefix()
    if _prefix == "gcp":
        env_vars.append(
            {"name": "OPENRESEARCH_GCP_GCS_BUCKET", "value": _cloud_setting("gcs_bucket", "")}
        )
    else:
        # Default / azure: P0-fix-1 standardised on OPENRESEARCH_AZURE_* names.
        env_vars.extend([
            {"name": "OPENRESEARCH_AZURE_STORAGE_ACCOUNT",  "value": storage_account},
            {"name": "OPENRESEARCH_AZURE_BLOB_CONTAINER",   "value": blob_container},
        ])
    if fingerprint:
        env_vars.append({"name": "OPENRESEARCH_CELL_FINGERPRINT", "value": fingerprint})
    if now_iso:
        env_vars.append({"name": "OPENRESEARCH_CELL_NOW_ISO", "value": now_iso})
    # Image-compat shim (audit 2026-06-10): the entrypoint baked into a PINNED
    # ACR image may predate the REPROLAB_->OPENRESEARCH_ rename and read the
    # legacy names. Inject both spellings until every base image is rebuilt
    # (the in-repo aks_cell_entrypoint.py reads the canonical names).
    env_vars.extend(
        {"name": "OPENRESEARCH_" + e["name"][len("OPENRESEARCH_"):], "value": e["value"]}
        for e in list(env_vars)
        if e["name"].startswith("OPENRESEARCH_")
    )

    # GPU count: from plan when available, else 1.
    gpu_count_str: str
    if gpu_plan is not None:
        gpu_count_str = str(getattr(gpu_plan, "gpu_count", 1))
    else:
        gpu_count_str = "1"

    # P0-fix-3: node selector uses the infra pool label reprolab/sku in ALL paths.
    # With gpu_plan → target that SKU's pool; without → fall back to the default SKU.
    if gpu_plan is not None:
        node_selector: dict[str, str] = {
            "reprolab/sku": str(getattr(gpu_plan, "short_name", default_sku))
        }
    else:
        node_selector = {"reprolab/sku": default_sku}

    # Toleration for the nvidia.com/gpu taint (always present; required by AKS GPU nodes).
    gpu_toleration = {
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule",
    }

    # Pod template labels: base labels + cloud-specific extras.
    # Default to Azure Workload Identity label when pod_template_extra_labels is
    # not explicitly provided, preserving byte-for-byte backward compatibility.
    _pod_extra: dict[str, str] = (
        {"azure.workload.identity/use": "true"}
        if pod_template_extra_labels is None
        else pod_template_extra_labels
    )
    _pod_labels: dict[str, str] = {
        "app": "reprolab-cell",
        "cell-id": cell_id[:63],
    }
    _pod_labels.update(_pod_extra)

    pod_template: dict[str, Any] = {
        "metadata": {
            "labels": _pod_labels,
        },
        "spec": {
            "serviceAccountName": service_account,
            "restartPolicy": "Never",
            "tolerations": [gpu_toleration],
            "nodeSelector": node_selector,
            "volumes": [
                {
                    "name": "reprolab-cache",
                    "persistentVolumeClaim": {"claimName": pvc_name},
                }
            ],
            "containers": [
                {
                    "name": "cell",
                    "image": base_image,
                    "env": env_vars,
                    "resources": {
                        "requests": {"nvidia.com/gpu": gpu_count_str},
                        "limits": {"nvidia.com/gpu": gpu_count_str},
                    },
                    "volumeMounts": [
                        {
                            "name": "reprolab-cache",
                            "mountPath": cache_mount_path,
                        }
                    ],
                }
            ],
        },
    }

    # podFailurePolicy: FailJob immediately for terminal wrapper exits 40-44.
    # This prevents backoffLimit from retrying application failures.
    pod_failure_rules = [
        {
            "action": "FailJob",
            "onExitCodes": {
                "containerName": "cell",
                "operator": "In",
                "values": [40, 41, 42, 43, 44],
            },
        }
    ]

    manifest: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "reprolab-cell"},
        },
        "spec": {
            # P1-fix-8: configurable knobs from settings.
            "backoffLimit": backoff_limit,
            "activeDeadlineSeconds": active_deadline_seconds,
            "ttlSecondsAfterFinished": ttl_seconds_after_finished,
            "podFailurePolicy": {"rules": pod_failure_rules},
            "template": pod_template,
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# Job watcher
# ---------------------------------------------------------------------------

def _watch_job(
    *,
    k8s: _K8sClients,
    job_name: str,
    namespace: str,
    overall_deadline: float | None,
    active_deadline_seconds: int,
    pending_timeout_s: float,
) -> dict[str, Any]:
    """Watch a K8s Job until terminal or timeout.

    **Scale fix (P0-scale-1):** The common path — a job that is actively
    running — makes exactly ONE API call per poll
    (``read_namespaced_job_status``).  ``list_namespaced_pod`` is called only
    when genuinely needed:

    * **Pending-timeout detection**: only while the job has reported zero
      active, succeeded, *and* failed pods — i.e. the scheduler has not yet
      placed the pod.  Once any of those counters is non-zero the job is past
      Pending and we stop querying pod phase entirely.
    * **Terminal log/exit-code collection**: once at the point we decide the
      job has completed (succeeded or failed), via ``_collect_pod_info``.

    With P parallel cells this reduces calls from 3P/poll to 1P/poll during
    normal execution.

    Returns a dict with keys:
        ``status``  — ``"succeeded"`` | ``"failed"`` | ``"deadline"``
                      | ``"overall_timeout"`` | ``"pending_timeout"``
        ``exit_code``  — int from container state (best-effort, may be None)
        ``node_name``  — str (best-effort, may be None)
        ``log``        — str (captured from K8s pod log, best-effort)
        ``outcome``    — str from Blob status.json, or None
        ``retries``    — int from Blob status.json, or 0
    """
    job_deadline = time.monotonic() + active_deadline_seconds
    pending_since: float | None = None

    while True:
        now = time.monotonic()

        # Overall timeout guard.
        if overall_deadline is not None and now >= overall_deadline:
            return _watch_result("overall_timeout")

        # Per-cell deadline.
        if now >= job_deadline:
            return _watch_result("deadline")

        # --- Single API call: read Job status ---
        _poll_interval: float = _cloud_setting("watch_poll_interval_s", 5.0)
        try:
            job = k8s.batch.read_namespaced_job_status(job_name, namespace)
        except Exception as exc:
            logger.warning("k8s_job_cell_runner: read_namespaced_job_status failed: %s", exc)
            time.sleep(_poll_interval)
            continue

        status = getattr(job, "status", None)
        # P1-fix-7: guard against status being None in transitional states.
        if status is None:
            time.sleep(_poll_interval)
            continue
        conditions = getattr(status, "conditions", None) or []
        succeeded = getattr(status, "succeeded", 0) or 0
        failed = getattr(status, "failed", 0) or 0
        active = getattr(status, "active", 0) or 0

        # --- Terminal detection from Job-level fields only (no pod list) ---

        # Check for terminal condition first (most reliable signal).
        for cond in conditions:
            ctype = getattr(cond, "type", "")
            cstatus = getattr(cond, "status", "")
            if ctype == "Complete" and cstatus == "True":
                node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
                return _watch_result(
                    "succeeded",
                    exit_code=exit_code,
                    node_name=node,
                    log=log,
                )
            if ctype == "Failed" and cstatus == "True":
                node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
                return _watch_result(
                    "failed",
                    exit_code=exit_code,
                    node_name=node,
                    log=log,
                )

        # Counters-based terminal detection (no conditions yet from the
        # controller, but the counts are definitive).
        if succeeded:
            node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
            return _watch_result("succeeded", exit_code=exit_code, node_name=node, log=log)

        # failed>0 and active==0 means all pods have exited and none are still
        # running — the job is done.  We derive this from Job-level counters
        # alone (no list_namespaced_pod) to avoid the per-poll thundering herd.
        if failed and active == 0:
            node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
            return _watch_result("failed", exit_code=exit_code, node_name=node, log=log)

        # --- Pending-timeout: pod-list query ONLY when no pods are active yet ---
        # If active>0 or succeeded>0 or failed>0, the pod was scheduled — skip.
        if active == 0 and succeeded == 0 and failed == 0:
            # Job has no activity at all → still in Pending or pre-scheduling.
            # Use a single list_namespaced_pod call to confirm/deny Pending.
            phase = _get_pod_phase(k8s, job_name, namespace)
            if phase in ("Pending", None):
                if pending_since is None:
                    pending_since = time.monotonic()
                elif time.monotonic() - pending_since >= pending_timeout_s:
                    return _watch_result("pending_timeout")
            else:
                # Pod moved past Pending (Running/Succeeded/Failed) — counters
                # will catch the terminal state on the next poll.
                pending_since = None
        else:
            # Pod is or was active — no longer in Pending limbo.
            pending_since = None

        # P1-fix-8: poll interval from settings.
        time.sleep(_poll_interval)


def _watch_result(
    status: str,
    *,
    exit_code: int | None = None,
    node_name: str | None = None,
    log: str = "",
    outcome: str | None = None,
    retries: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "exit_code": exit_code,
        "node_name": node_name,
        "log": log,
        "outcome": outcome,
        "retries": retries,
    }


def _collect_pod_info(
    k8s: _K8sClients, job_name: str, namespace: str
) -> tuple[str | None, int | None, str]:
    """Fetch node name, exit code, and log from the most recent Job pod."""
    node_name: str | None = None
    exit_code: int | None = None
    log = ""
    try:
        pods = k8s.core.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}"
        )
        items = getattr(pods, "items", []) or []
        if not items:
            return node_name, exit_code, log
        # Pick the most recent pod.
        pod = items[-1]
        node_name = getattr(pod.spec, "node_name", None) if pod.spec else None
        # Extract exit code from container state.
        cs_list = (
            getattr(pod.status, "container_statuses", None)
            if pod.status else None
        ) or []
        for cs in cs_list:
            term = getattr(getattr(cs, "state", None), "terminated", None)
            if term is not None:
                exit_code = getattr(term, "exit_code", None)
                break
        # Fetch logs (best-effort).
        try:
            log = k8s.core.read_namespaced_pod_log(
                pod.metadata.name, namespace, container="cell"
            ) or ""
        except Exception:
            log = ""
    except Exception as exc:
        logger.debug("k8s_job_cell_runner: _collect_pod_info failed: %s", exc)
    return node_name, exit_code, log


def _has_active_pods(k8s: _K8sClients, job_name: str, namespace: str) -> bool:
    try:
        pods = k8s.core.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}"
        )
        for pod in (getattr(pods, "items", []) or []):
            phase = getattr(getattr(pod, "status", None), "phase", "")
            if phase in ("Pending", "Running"):
                return True
    except Exception:
        pass
    return False


def _get_pod_phase(k8s: _K8sClients, job_name: str, namespace: str) -> str | None:
    try:
        pods = k8s.core.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}"
        )
        items = getattr(pods, "items", []) or []
        if items:
            return getattr(getattr(items[-1], "status", None), "phase", None)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Blob artifact reconciliation
# ---------------------------------------------------------------------------

def _try_download_metrics(
    *,
    cell_id: str,
    output_blob_prefix: str,
    account_name: str,
    container_name: str,
    output_dir: Path,
    client: Any | None = None,
) -> dict[str, Any] | None:
    """Download ``metrics.json`` from Blob into ``output_dir/<cell_id>/metrics.json``.

    ``client`` is the shared ``ContainerClient`` built once per ``run_matrix``
    invocation (P0-scale-2).  Passing it here avoids a fresh MSI credential
    probe on each call.

    Returns the parsed dict, or None on any failure.
    """
    blob_name = f"{output_blob_prefix}/{cell_id}/metrics.json"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=account_name,
            container_name=container_name,
            client=client,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_bytes(data)
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        logger.debug(
            "k8s_job_cell_runner: metrics download failed cell=%s: %s", cell_id, exc
        )
        return None


def _try_download_log(
    *,
    cell_id: str,
    output_blob_prefix: str,
    account_name: str,
    container_name: str,
    log_path: Path,
    fallback_log: str,
    client: Any | None = None,
) -> None:
    """Download pod log from Blob (best-effort); fall back to ``fallback_log``.

    ``client`` is the shared ``ContainerClient`` (P0-scale-2).
    """
    blob_name = f"{output_blob_prefix}/{cell_id}/logs/attempt-0.log"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=account_name,
            container_name=container_name,
            client=client,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(data)
        return
    except Exception:
        pass
    if fallback_log:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(fallback_log, encoding="utf-8", errors="replace")
        except OSError:
            pass


def _try_reconcile_status(
    *,
    cell_id: str,
    output_blob_prefix: str,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> dict[str, Any] | None:
    """Fetch Blob ``status.json`` to reconcile a TTL-deleted Job.

    ``client`` is the shared ``ContainerClient`` (P0-scale-2).
    """
    blob_name = f"{output_blob_prefix}/{cell_id}/status.json"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=account_name,
            container_name=container_name,
            client=client,
        )
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Map watch result → CellResult status
# ---------------------------------------------------------------------------

def _map_status(
    watch: dict[str, Any],
    *,
    cell_id: str,
    blob_status: dict[str, Any] | None,
) -> tuple[str, str | None, int]:
    """Return ``(status, error, retries)`` from a watch result + optional Blob sentinel.

    Priority: watch failure path → explicit sentinel outcome.
    """
    w_status = watch["status"]
    exit_code = watch.get("exit_code")
    outcome = watch.get("outcome") or (blob_status or {}).get("outcome")
    retries = watch.get("retries") or (blob_status or {}).get("retries") or 0
    log = watch.get("log", "")

    if w_status == "pending_timeout":
        return STATUS_ERROR, f"capacity_exhausted: Job {cell_id} stuck in Pending", retries

    if w_status in ("overall_timeout", "deadline"):
        return STATUS_ERROR, f"timeout: watch_status={w_status}", retries

    if w_status == "failed":
        # Check for terminal OOM sentinel.
        if outcome == _SENTINEL_OOM_OUTCOME or exit_code == _EXIT_OOM_EXHAUSTED:
            return STATUS_OOM_FAILED, log[-2000:] if log else f"exit_code={exit_code}", retries
        return STATUS_ERROR, log[-2000:] if log else f"exit_code={exit_code}", retries

    if w_status == "succeeded":
        if exit_code is None or exit_code == 0:
            return STATUS_OK, None, retries
        if exit_code == _EXIT_OOM_EXHAUSTED or outcome == _SENTINEL_OOM_OUTCOME:
            return STATUS_OOM_FAILED, log[-2000:] if log else f"exit_code={exit_code}", retries
        if exit_code in _EXIT_TERMINAL:
            return STATUS_ERROR, f"wrapper exit {exit_code}", retries
        return STATUS_OK, None, retries

    # Fallback.
    return STATUS_ERROR, f"unexpected watch_status={w_status}", retries


# ---------------------------------------------------------------------------
# Core: submit one cell Job and wait
# ---------------------------------------------------------------------------

def _run_cell_job(
    *,
    cell: dict[str, Any],
    k8s: _K8sClients,
    namespace: str,
    service_account: str,
    node_pool_name: str,
    base_image: str,
    storage_account: str,
    blob_container: str,
    code_blob_prefix: str,
    output_blob_prefix: str,
    output_root: Path,
    active_deadline_seconds: int,
    overall_deadline: float | None,
    pending_timeout_s: float,
    max_oom_retries: int,
    fingerprint: str | None,
    now_iso: str | None,
    run_id: str,
    gpu_plan: Any | None = None,
    blob_client: Any | None = None,
    pod_template_extra_labels: dict | None = None,
) -> CellResult:
    """Submit a K8s Job for ``cell`` and block until terminal, then return a CellResult.

    ``blob_client`` is the shared ``ContainerClient`` built once per
    ``run_matrix`` invocation (P0-scale-2).  When ``None`` the helpers fall
    back to constructing their own client (pre-fix behaviour — tolerated in
    tests and for any call site that does not supply one).
    """
    cell_id: str = cell.get("id", f"cell_{id(cell)}")
    job_name = _job_name(cell_id, run_id)
    output_dir = output_root / cell_id
    log_path = output_root / f"{cell_id}.log"

    cell_params_json = json.dumps(cell)

    try:
        manifest = _build_job_manifest(
            job_name=job_name,
            namespace=namespace,
            service_account=service_account,
            node_pool_name=node_pool_name,
            base_image=base_image,
            storage_account=storage_account,
            blob_container=blob_container,
            files_share=_cloud_setting("files_share", "reprolab-cache"),
            cell_id=cell_id,
            cell_params_json=cell_params_json,
            output_blob_prefix=output_blob_prefix,
            code_blob_prefix=code_blob_prefix,
            active_deadline_seconds=active_deadline_seconds,
            max_oom_retries=max_oom_retries,
            fingerprint=fingerprint,
            now_iso=now_iso,
            gpu_plan=gpu_plan,
            # P1-fix-8: configurable knobs from settings.
            ttl_seconds_after_finished=int(_cloud_setting("ttl_seconds_after_finished", 3600)),
            backoff_limit=int(_cloud_setting("job_backoff_limit", 0)),
            cache_mount_path=str(_cloud_setting("cache_mount_path", "/mnt/reprolab-cache")),
            # P1-fix-9: OOM shrink ratios forwarded from settings.
            oom_batch_scale_step1=float(
                _cloud_setting("cell_oom_batch_scale_step1", 0.5)
            ),
            oom_batch_scale_floor=float(
                _cloud_setting("cell_oom_batch_scale_floor", 0.25)
            ),
            bootstrap_pip_timeout_s=int(
                _cloud_setting("bootstrap_pip_timeout_s", 600)
            ),
            # P0-fix-3: default SKU for no-plan fallback.
            default_sku=str(
                (_cloud_setting("gpu_skus", []) or ["azure_a100_80"])[0]
            ),
            pod_template_extra_labels=pod_template_extra_labels,
        )
    except ValueError as exc:
        # P1-fix-5: manifest builder raises ValueError on empty base_image.
        # Treat as a job submission failure — cell becomes "error" with a clear message.
        logger.error(
            "k8s_job_cell_runner: manifest build failed cell=%s: %s", cell_id, exc
        )
        _cs = "gke" if _get_settings_prefix() == "gcp" else "aks"
        return CellResult(
            cell_id=cell_id,
            status=STATUS_ERROR,
            metrics=None,
            gpu=f"{_cs}:unassigned",
            retries=0,
            error=f"manifest build failed: {exc}",
        )

    # Submit the Job.
    _cs = "gke" if _get_settings_prefix() == "gcp" else "aks"
    try:
        k8s.batch.create_namespaced_job(namespace, manifest)
        logger.info(
            "k8s_job_cell_runner: submitted Job=%s for cell=%s (deadline=%ds sku=%s)",
            job_name, cell_id, active_deadline_seconds,
            getattr(gpu_plan, "short_name", "default") if gpu_plan else "default",
        )
    except Exception as exc:
        logger.error("k8s_job_cell_runner: create_namespaced_job failed cell=%s: %s", cell_id, exc)
        return CellResult(
            cell_id=cell_id,
            status=STATUS_ERROR,
            metrics=None,
            gpu=f"{_cs}:unassigned",
            retries=0,
            error=f"job submission failed: {exc}",
        )

    # Watch Job until terminal.
    watch = _watch_job(
        k8s=k8s,
        job_name=job_name,
        namespace=namespace,
        overall_deadline=overall_deadline,
        active_deadline_seconds=active_deadline_seconds,
        pending_timeout_s=pending_timeout_s,
    )

    node_name = watch.get("node_name")
    gpu_label = f"{_cs}:{node_name}" if node_name else f"{_cs}:unassigned"

    # Reconcile Blob status.json if Job was TTL-deleted before we could read it.
    blob_status: dict[str, Any] | None = None
    if watch["status"] in ("succeeded", "failed") and watch.get("exit_code") is None:
        blob_status = _try_reconcile_status(
            cell_id=cell_id,
            output_blob_prefix=output_blob_prefix,
            account_name=storage_account,
            container_name=blob_container,
            client=blob_client,
        )

    status, error, retries = _map_status(watch, cell_id=cell_id, blob_status=blob_status)

    # Pull metrics from Blob.
    metrics: dict[str, Any] | None = None
    if status in (STATUS_OK, STATUS_OOM_FAILED, STATUS_ERROR):
        metrics = _try_download_metrics(
            cell_id=cell_id,
            output_blob_prefix=output_blob_prefix,
            account_name=storage_account,
            container_name=blob_container,
            output_dir=output_dir,
            client=blob_client,
        )

    # Persist log.
    _try_download_log(
        cell_id=cell_id,
        output_blob_prefix=output_blob_prefix,
        account_name=storage_account,
        container_name=blob_container,
        log_path=log_path,
        fallback_log=watch.get("log", ""),
        client=blob_client,
    )

    # Write local resume manifest.
    write_cell_manifest(
        output_dir,
        caller="k8s_job_cell_runner",
        cell_id=cell_id,
        status=status,
        fingerprint=fingerprint,
        metrics=metrics,
        retries=retries,
        now_iso=now_iso,
    )

    logger.info(
        "k8s_job_cell_runner: cell=%s done status=%s gpu=%s retries=%d",
        cell_id, status, gpu_label, retries,
    )
    return CellResult(
        cell_id=cell_id,
        status=status,
        metrics=metrics,
        gpu=gpu_label,
        retries=retries,
        error=error,
    )


# ---------------------------------------------------------------------------
# SKU escalation helpers
# ---------------------------------------------------------------------------

def _resolve_escalation_sku(
    ladder_remaining: tuple[str, ...] | list[str],
    provisioned_skus: list[str],
) -> str | None:
    """Return the first ladder SKU that is also provisioned, or None."""
    provisioned_set = set(provisioned_skus)
    for short_name in ladder_remaining:
        if short_name in provisioned_set:
            return short_name
    return None


def _lookup_sku_by_short_name(short_name: str) -> Any | None:
    """Return the GpuSku for ``short_name`` from the catalog, or None."""
    try:
        from backend.services.runtime.gpu_catalog import CATALOG  # type: ignore[import]
        for sku in CATALOG:
            if sku.short_name == short_name:
                return sku
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_matrix(
    cells: list[dict[str, Any]],
    cell_script: str | Path,
    *,
    output_root: str | Path,
    gpus: list[str] | None = None,
    max_parallel: int | None = None,
    max_oom_retries: int = 2,
    per_cell_timeout_s: float | None = None,
    overall_timeout_s: float | None = None,
    gpus_per_cell: int = 1,
    fingerprints: dict[str, str] | None = None,
    force_cells: set[str] | None = None,
    now_iso: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Submit one AKS Job per cell and return every cell in the CellResult.to_dict() shape.

    Drop-in replacement for ``gpu_cell_runner.run_matrix``.  The return value
    is ``{cell_id → {"status", "metrics", "gpu", "retries", "error"}}`` consumed
    unchanged by ``cell_matrix.aggregate_cell_metrics``.

    Args:
        cells:               List of cell-description dicts (each must have ``"id"``).
        cell_script:         Path to the single-cell trainer.  Its parent directory
                             is uploaded to Blob once.
        output_root:         Local root; per-cell artifacts are downloaded here.
        gpus:                Accepted for signature parity; ignored (K8s schedules).
        max_parallel:        Orchestrator-side concurrency cap.  Defaults to
                             ``azure_max_nodes``.
        max_oom_retries:     Forwarded to the in-Job wrapper as
                             ``OPENRESEARCH_CELL_MAX_OOM_RETRIES``.
        per_cell_timeout_s:  Maps to Job ``activeDeadlineSeconds``.
        overall_timeout_s:   Wall-clock cap for the WHOLE matrix.
        gpus_per_cell:       Must be 1; any other value returns ``"error"`` for
                             all non-skipped cells (Azure Jobs are 1-GPU each by
                             default; multi-GPU flows through ``gpu_plan``).
        fingerprints:        ``{cell_id: fingerprint}`` for resume parity.
        force_cells:         Cell ids that must always re-run.
        now_iso:             ISO-8601 timestamp stamped into ``cell_manifest.json``.

    Returns:
        Dict mapping ``cell["id"]`` → ``CellResult.to_dict()``.  Every input cell
        is present regardless of outcome; never raises on single-cell failure.

    Dynamic gpu_plan + SKU escalation:
        When a ``GpuPlan`` is bound via ``bind_run_context(gpu_plan=...)`` the
        submitted Job targets ``plan.short_name`` (infra pool label) and
        ``plan.gpu_count`` GPUs.  On ``oom_failed`` (in-Job shrink exhausted) the
        runner picks the first ``plan.ladder_remaining`` SKU that is present in
        ``settings.azure_gpu_skus`` (provisioned pools), looks it up in
        ``gpu_catalog``, and resubmits the same cell targeting that bigger SKU.
        Each resubmission emits a ``gpu_escalated`` event.  Escalations are capped
        at ``settings.dynamic_gpu_max_escalations`` (default 2).  If the ladder is
        empty, no candidate is provisioned, or the cap is reached, the cell stays
        ``oom_failed`` — never crashes, never loops.
    """
    # Fast path — no SDK imports needed.
    if not cells:
        return {}

    # Read settings defensively — cloud-specific keys via _cloud_setting(),
    # cloud-agnostic keys via _setting() directly.
    namespace: str = _cloud_setting("namespace", "reprolab")
    service_account: str = _cloud_setting("service_account", "reprolab-sa")
    node_pool_name: str = _cloud_setting("node_pool_name", "gpunodes")
    # P1-fix-5: default is "" — _build_job_manifest raises clearly if still empty.
    base_image: str = _cloud_setting("base_image", "")
    storage_account: str = _cloud_setting("storage_account", "") or ""
    blob_container: str = _cloud_setting("blob_container", "reprolab-artifacts") or "reprolab-artifacts"
    # P1-fix-5: align with config.py default of 4 (was incorrectly 8 here).
    cloud_max_nodes: int = int(_cloud_setting("max_nodes", 4))
    gpu_usd_per_hour: float = float(_cloud_setting("gpu_usd_per_hour", 3.67))
    pending_timeout_s: float = float(_cloud_setting("pending_timeout_seconds", 900))
    provisioned_skus: list[str] = list(_cloud_setting("gpu_skus", []) or [])
    max_escalations: int = int(_setting("dynamic_gpu_max_escalations", 2))
    # Derive cloud prefix short label for gpu result fields ("aks" for azure, "gke" for gcp).
    _prefix = _get_settings_prefix()
    _cloud_short = "gke" if _prefix == "gcp" else "aks"
    # Pod template extra labels: AKS needs WI label, GKE does not.
    _pod_extra_labels: dict = (
        {} if _prefix == "gcp" else {"azure.workload.identity/use": "true"}
    )

    _fingerprints: dict[str, str] = fingerprints or {}
    _force_cells: set[str] = force_cells or set()
    _resume_armed: bool = is_resume_armed()

    run_budget = _get_run_budget()
    event_sink = _get_event_sink()
    gpu_plan = _get_gpu_plan()

    # Derive a run_id from the output_root path (last two segments).
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = output_root.name  # best-effort

    # gpus_per_cell guard — Azure Jobs are 1-GPU each (multi-GPU flows via gpu_plan).
    if gpus_per_cell != 1:
        msg = f"k8s_job_cell_runner: gpus_per_cell={gpus_per_cell} != 1; all cells error"
        logger.error(msg)
        event_sink("run_warning", {"code": "k8s_gpus_per_cell", "message": msg})
        return {
            cell.get("id", f"cell_{i}"): CellResult(
                cell_id=cell.get("id", f"cell_{i}"),
                status=STATUS_ERROR,
                metrics=None,
                gpu=f"{_cloud_short}:unassigned",
                retries=0,
                error=msg,
            ).to_dict()
            for i, cell in enumerate(cells)
        }

    # Overall deadline.
    overall_deadline: float | None = deadline_from_timeout(overall_timeout_s)

    # Parallelism.
    parallelism = min(
        max_parallel or cloud_max_nodes,
        cloud_max_nodes,
        len(cells),
    )
    parallelism = max(1, parallelism)

    # Lazily initialise K8s clients (inside function so tests can monkeypatch).
    try:
        k8s = _k8s_factory()
    except Exception as exc:
        err = f"k8s_job_cell_runner: cannot init K8s clients: {exc}"
        logger.error(err)
        return {
            cell.get("id", f"cell_{i}"): CellResult(
                cell_id=cell.get("id", f"cell_{i}"),
                status=STATUS_ERROR,
                metrics=None,
                gpu=f"{_cloud_short}:unassigned",
                retries=0,
                error=err,
            ).to_dict()
            for i, cell in enumerate(cells)
        }

    # Upload code once (parent of cell_script).
    cell_script = Path(cell_script)
    code_dir = cell_script.parent
    code_blob_prefix = f"runs/{run_id}/{_BLOB_CODE_PREFIX}"
    output_blob_prefix = f"runs/{run_id}/{_BLOB_CELLS_PREFIX}"

    try:
        uploaded = _blob_upload_prefix(
            code_dir,
            blob_prefix=code_blob_prefix,
            account_name=storage_account,
            container_name=blob_container,
        )
        logger.info(
            "k8s_job_cell_runner: uploaded %d code files to %s", len(uploaded), code_blob_prefix
        )
    except Exception as exc:
        err = f"k8s_job_cell_runner: code upload failed: {exc}"
        logger.error(err)
        return {
            cell.get("id", f"cell_{i}"): CellResult(
                cell_id=cell.get("id", f"cell_{i}"),
                status=STATUS_ERROR,
                metrics=None,
                gpu=f"{_cloud_short}:unassigned",
                retries=0,
                error=err,
            ).to_dict()
            for i, cell in enumerate(cells)
        }

    # P0-scale-2: build ONE shared ContainerClient for all per-cell blob calls
    # (metrics, log, status.json).  This avoids a fresh DefaultAzureCredential
    # MSI probe on every download across potentially 16+ cells.  Falls back to
    # None (each helper constructs its own client) when storage is unconfigured
    # or the azure SDK is absent.
    shared_blob_client: Any | None = _make_blob_client(storage_account, blob_container)

    # Budget tracking: sum of reserved GPU-seconds for active + completed cells.
    reserved_gpu_seconds = 0.0
    budget_lock = threading.Lock()

    results: dict[str, CellResult] = {}
    results_lock = threading.Lock()

    def _process_cell(cell: dict[str, Any]) -> None:
        nonlocal reserved_gpu_seconds

        # P0-fix-4: per-cell copy so escalation can update the rate for THIS cell
        # without affecting other concurrent cells (shared outer var would be a race
        # AND would cause UnboundLocalError since Python sees the assignment below
        # as making the name local throughout the whole function body).
        cell_gpu_usd_per_hour: float = gpu_usd_per_hour

        cell_id: str = cell.get("id", f"cell_{id(cell)}")
        output_dir = output_root / cell_id

        # --- Resume skip (Track B) ---
        if _resume_armed and should_skip_cell(cell_id, output_dir, _fingerprints, _force_cells):
            manifest = load_cell_manifest(output_dir)
            prior_metrics: dict[str, Any] | None = None
            if manifest:
                mf = output_dir / "metrics.json"
                if mf.is_file():
                    try:
                        prior_metrics = json.loads(mf.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            logger.info(
                "k8s_job_cell_runner: cell=%s SKIPPED (resume: prior ok + fingerprint match)",
                cell_id,
            )
            with results_lock:
                results[cell_id] = CellResult(
                    cell_id=cell_id,
                    status=STATUS_SKIPPED,
                    metrics=prior_metrics,
                    gpu=f"{_cloud_short}:unassigned",
                    retries=0,
                    error=None,
                )
            return

        # --- Overall deadline ---
        if overall_deadline is not None and time.monotonic() >= overall_deadline:
            with results_lock:
                results[cell_id] = CellResult(
                    cell_id=cell_id,
                    status=STATUS_ERROR,
                    metrics=None,
                    gpu=f"{_cloud_short}:unassigned",
                    retries=0,
                    error="overall matrix timeout — cell not submitted",
                )
            return

        # --- Budget check ---
        # Effective deadline for this cell.
        remaining_s: float | None = (
            overall_deadline - time.monotonic() if overall_deadline is not None else None
        )
        if per_cell_timeout_s is not None and remaining_s is not None:
            eff_cell_s = min(per_cell_timeout_s, remaining_s)
        elif per_cell_timeout_s is not None:
            eff_cell_s = per_cell_timeout_s
        elif remaining_s is not None:
            eff_cell_s = remaining_s
        else:
            eff_cell_s = 86400.0  # 24h fallback — Job still has activeDeadlineSeconds

        active_deadline_seconds = max(1, math.ceil(eff_cell_s))

        with budget_lock:
            new_reserved = reserved_gpu_seconds + eff_cell_s
            budget_err = _check_budget(
                run_budget=run_budget,
                reserved_gpu_seconds=new_reserved,
                gpu_usd_per_hour=cell_gpu_usd_per_hour,
                cell_id=cell_id,
            )
            if budget_err:
                logger.warning("k8s_job_cell_runner: budget exceeded: %s", budget_err)
                event_sink("run_warning", {"code": "k8s_budget_exceeded", "message": budget_err})
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status=STATUS_ERROR,
                        metrics=None,
                        gpu=f"{_cloud_short}:unassigned",
                        retries=0,
                        error=budget_err,
                    )
                return
            reserved_gpu_seconds = new_reserved

        # --- Submit and watch (with optional SKU escalation on oom_failed) ---
        current_plan = gpu_plan  # may become a lighter stub on escalation
        escalation_count = 0
        # P0-fix-2: track the effective run_id per attempt so each escalated Job
        # gets a unique name (avoids 409 AlreadyExists on resubmit).
        current_run_id = run_id

        while True:
            result = _run_cell_job(
                cell=cell,
                k8s=k8s,
                namespace=namespace,
                service_account=service_account,
                node_pool_name=node_pool_name,
                base_image=base_image,
                storage_account=storage_account,
                blob_container=blob_container,
                code_blob_prefix=code_blob_prefix,
                output_blob_prefix=output_blob_prefix,
                output_root=output_root,
                active_deadline_seconds=active_deadline_seconds,
                overall_deadline=overall_deadline,
                pending_timeout_s=pending_timeout_s,
                max_oom_retries=max_oom_retries,
                fingerprint=_fingerprints.get(cell_id),
                now_iso=now_iso,
                # P0-fix-2: use the (possibly suffixed) run_id for Job naming.
                run_id=current_run_id,
                gpu_plan=current_plan,
                # P0-scale-2: shared client avoids per-call MSI probe.
                blob_client=shared_blob_client,
                pod_template_extra_labels=_pod_extra_labels,
            )

            # Escalation check: only if oom_failed + plan available + cap not hit.
            if (
                result.status != STATUS_OOM_FAILED
                or current_plan is None
                or escalation_count >= max_escalations
            ):
                break

            ladder = getattr(current_plan, "ladder_remaining", None) or ()
            if not ladder:
                logger.info(
                    "k8s_job_cell_runner: cell=%s oom_failed, ladder empty → degrade",
                    cell_id,
                )
                break

            next_sku_name = _resolve_escalation_sku(ladder, provisioned_skus)
            if next_sku_name is None:
                logger.info(
                    "k8s_job_cell_runner: cell=%s oom_failed, no provisioned ladder candidate → degrade",
                    cell_id,
                )
                break

            next_sku = _lookup_sku_by_short_name(next_sku_name)
            from_sku = (
                getattr(current_plan, "short_name", "unknown") if current_plan else "default"
            )

            # P0-fix-4: update the GPU rate to the escalated SKU's catalog price
            # BEFORE the budget re-check, so the guard uses the correct (higher) rate.
            escalated_usd_per_hour: float = cell_gpu_usd_per_hour
            if next_sku is not None:
                escalated_usd_per_hour = float(
                    getattr(next_sku, "approx_usd_per_hr", cell_gpu_usd_per_hour)
                )

            # P0-fix-4 (cont.): re-check budget with escalated rate + already-
            # reserved seconds before committing to the resubmit.
            with budget_lock:
                escalated_budget_err = _check_budget(
                    run_budget=run_budget,
                    reserved_gpu_seconds=reserved_gpu_seconds,
                    gpu_usd_per_hour=escalated_usd_per_hour,
                    cell_id=cell_id,
                )
            if escalated_budget_err:
                logger.warning(
                    "k8s_job_cell_runner: escalation budget exceeded, stopping: %s",
                    escalated_budget_err,
                )
                event_sink("run_warning", {
                    "code": "k8s_escalation_budget_exceeded",
                    "message": escalated_budget_err,
                })
                # Return oom_failed cleanly rather than resubmitting over budget.
                break

            logger.info(
                "k8s_job_cell_runner: cell=%s oom_failed → escalate %s→%s (escalation %d/%d)",
                cell_id, from_sku, next_sku_name, escalation_count + 1, max_escalations,
            )
            event_sink("gpu_escalated", {
                "cell_id": cell_id,
                "from_sku": from_sku,
                "to_sku": next_sku_name,
                "reason": STATUS_OOM_FAILED,
            })

            # Build a lightweight plan stub for the next attempt.
            # We only need short_name and gpu_count to build the next manifest.
            escalation_count += 1
            current_plan = _EscalationPlan(
                short_name=next_sku_name,
                gpu_count=getattr(next_sku, "gpu_count", 1) if next_sku else 1,
                # Trim the ladder: drop everything up to and including next_sku_name.
                ladder_remaining=_trim_ladder(ladder, next_sku_name),
            )
            # P0-fix-2: suffix the run_id so the escalated Job gets a unique name.
            current_run_id = f"{run_id}-e{escalation_count}"
            # P0-fix-4: carry the escalated rate forward for any further budget checks
            # in this cell's loop.  cell_gpu_usd_per_hour is cell-local (not shared).
            cell_gpu_usd_per_hour = escalated_usd_per_hour
            active_deadline_seconds = max(
                1,
                math.ceil(
                    (overall_deadline - time.monotonic())
                    if overall_deadline is not None else eff_cell_s
                ),
            )

        with results_lock:
            results[cell_id] = result

    # Run cells with thread-pool parallelism.
    cell_queue: list[dict[str, Any]] = list(cells)
    cell_idx = 0
    cell_lock = threading.Lock()

    def _worker_thread() -> None:
        while True:
            with cell_lock:
                nonlocal cell_idx
                if cell_idx >= len(cell_queue):
                    return
                cell = cell_queue[cell_idx]
                cell_idx += 1
            _process_cell(cell)

    threads: list[threading.Thread] = []
    for _ in range(parallelism):
        t = threading.Thread(target=_worker_thread, daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Defensive completeness: ensure every input cell has a result.
    for i, cell in enumerate(cells):
        cid = cell.get("id", f"cell_{i}")
        if cid not in results:
            results[cid] = CellResult(
                cell_id=cid,
                status=STATUS_ERROR,
                metrics=None,
                gpu=f"{_cloud_short}:unassigned",
                retries=0,
                error="worker exited without recording a result",
            )

    return {cid: r.to_dict() for cid, r in results.items()}


# ---------------------------------------------------------------------------
# Escalation plan stub
# ---------------------------------------------------------------------------

class _EscalationPlan:
    """Lightweight plan stub used for escalation resubmits.

    Carries only the fields _build_job_manifest reads from a GpuPlan:
    ``short_name``, ``gpu_count``, and ``ladder_remaining`` (for further
    escalations).  Never used outside this module.
    """

    __slots__ = ("short_name", "gpu_count", "ladder_remaining")

    def __init__(
        self,
        *,
        short_name: str,
        gpu_count: int,
        ladder_remaining: tuple[str, ...],
    ) -> None:
        self.short_name = short_name
        self.gpu_count = gpu_count
        self.ladder_remaining = ladder_remaining


def _trim_ladder(
    ladder: tuple[str, ...] | list[str],
    used_name: str,
) -> tuple[str, ...]:
    """Return ladder entries that come AFTER ``used_name``."""
    items = list(ladder)
    try:
        idx = items.index(used_name)
        return tuple(items[idx + 1:])
    except ValueError:
        return tuple(items)
