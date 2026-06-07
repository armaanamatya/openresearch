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
  ``"error"`` (Azure Jobs are 1 GPU each by design).
* **OOM in-Job** — the Job wrapper owns the shrink ladder; the orchestrator
  does NOT resubmit on OOM.  ``max_oom_retries`` is forwarded to the Job as
  ``REPROLAB_CELL_MAX_OOM_RETRIES``.
* **Blob artifact bus** — code is uploaded once from
  ``Path(cell_script).parent``; per-cell ``metrics.json`` is downloaded after
  Job completion.
* **Resume parity** — mirrors ``gpu_cell_runner``'s Track-B resume: when
  ``REPROLAB_RESUME_CELLS`` is truthy, a prior ``cell_manifest.json`` with
  ``status=="ok"`` + matching fingerprint + not in ``force_cells`` → skip.
* **Budget** — before each submit, reserved GPU-seconds are checked against
  ``RunBudget.max_pod_seconds`` and ``RunBudget.max_run_gpu_usd``; caps
  sourced from a ``bind_run_context``-injected context var.
* **Lazy K8s imports** — ``kubernetes`` is NOT installed in the dev venv.
  All ``kubernetes.*`` calls go through ``_k8s_factory()`` which may be
  monkeypatched in tests.  Module import always succeeds.
* **Concurrent-safe context** — ``bind_run_context`` uses a ``ContextVar``
  so multiple concurrent runs (threads) each see their own budget/sink.

Status mapping:

    Job Complete + wrapper exit 0 + valid artifact  →  "ok"
    wrapper exit 42 / sentinel outcome oom_shrink_exhausted  →  "oom_failed"
    exit 40/41/43/44, Job Failed, deadline, overall timeout  →  "error"
    Pending beyond azure_pending_timeout_seconds  →  "error" (prefix capacity_exhausted:)
    Resume hit  →  "skipped"

SSE events — only EXISTING types are emitted (``run_warning``).  No new types.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "CELL_MANIFEST_NAME",
    "run_matrix",
    "bind_run_context",
]

# Name of the per-cell resume manifest (same as gpu_cell_runner).
CELL_MANIFEST_NAME = "cell_manifest.json"

# Exit codes from the in-Job wrapper that map to specific statuses.
_EXIT_OOM_EXHAUSTED = 42
_EXIT_TERMINAL = frozenset({40, 41, 43, 44})

# Blob sub-paths (relative to run prefix).
_BLOB_CODE_PREFIX = "code"
_BLOB_CELLS_PREFIX = "cells"

# Sentinel outcome written by the wrapper when OOM ladder is exhausted.
_SENTINEL_OOM_OUTCOME = "oom_shrink_exhausted"

# Default fallback values for settings that may not exist yet (W2 adds them).
_SETTINGS_DEFAULTS: dict[str, Any] = {
    "azure_namespace": "reprolab",
    "azure_service_account": "reprolab-sa",
    "azure_node_pool_name": "gpunodes",
    "azure_base_image": "reprolab.azurecr.io/reprolab-aks-cell:latest",
    "azure_storage_account": "",
    "azure_blob_container": "reprolab-artifacts",
    "azure_files_share": "reprolab-cache",
    "azure_max_nodes": 8,
    "azure_per_gpu_vram_gb": 80.0,
    "azure_gpu_usd_per_hour": 3.67,
    "azure_pending_timeout_seconds": 900,
    "azure_boot_timeout_seconds": 900,
}

# ---------------------------------------------------------------------------
# Context var — concurrent-safe budget + event-sink injection
# ---------------------------------------------------------------------------

_RUN_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "k8s_job_cell_runner_context", default={}
)


@contextmanager
def bind_run_context(
    *,
    run_budget: Any | None = None,
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
) -> Iterator[None]:
    """Bind a ``RunBudget`` and/or an event sink for the duration of the ``with`` block.

    Uses a ``ContextVar`` so concurrent runs on different threads each see their
    own context without interfering.  The wiring layer (``primitives.py``) calls
    this around ``run_matrix`` to inject the budget and an SSE event sink without
    changing ``run_matrix``'s signature.

    Args:
        run_budget:  A ``RunBudget`` instance (or None to disable checks).
        event_sink:  ``(event_type, payload) → None`` called to emit EXISTING SSE
                     events (``primitive_call``, ``repl_iteration``, ``run_warning``).
                     Default no-op when None.

    Example::

        with bind_run_context(run_budget=ctx.run_budget, event_sink=ctx.emit):
            results = k8s_job_cell_runner.run_matrix(cells, cell_script, ...)
    """
    token = _RUN_CONTEXT.set({"run_budget": run_budget, "event_sink": event_sink})
    try:
        yield
    finally:
        _RUN_CONTEXT.reset(token)


def _get_run_budget() -> Any | None:
    return _RUN_CONTEXT.get({}).get("run_budget")


def _get_event_sink() -> Callable[[str, dict[str, Any]], None]:
    sink = _RUN_CONTEXT.get({}).get("event_sink")
    return sink if callable(sink) else lambda _t, _p: None


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

    try:
        k8s_config.load_kube_config()
    except Exception:
        try:
            k8s_config.load_incluster_config()
        except Exception as exc:
            raise RuntimeError(f"k8s_job_cell_runner: cannot load kubeconfig: {exc}") from exc

    return _K8sClients(
        batch=k8s_client.BatchV1Api(),
        core=k8s_client.CoreV1Api(),
        watch_cls=K8sWatch,
    )


# ---------------------------------------------------------------------------
# Azure Blob helpers (code-against-contract; W1-A owns the real file)
# ---------------------------------------------------------------------------

def _blob_upload_prefix(
    local_root: str | Path,
    *,
    blob_prefix: str,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> list[str]:
    """Delegate to ``backend.services.runtime.azure_blob.upload_prefix``."""
    from backend.services.runtime import azure_blob  # type: ignore[import]
    return azure_blob.upload_prefix(
        local_root,
        blob_prefix=blob_prefix,
        account_name=account_name,
        container_name=container_name,
        client=client,
    )


def _blob_download_bytes(
    blob_name: str,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> bytes:
    from backend.services.runtime import azure_blob  # type: ignore[import]
    return azure_blob.download_bytes(
        blob_name,
        account_name=account_name,
        container_name=container_name,
        client=client,
    )


def _blob_download_artifact(
    blob_name: str,
    destination: str | Path,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> Path:
    from backend.services.runtime import azure_blob  # type: ignore[import]
    return azure_blob.download_artifact(
        blob_name,
        destination,
        account_name=account_name,
        container_name=container_name,
        client=client,
    )


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
# Cell result / manifest helpers (mirrors gpu_cell_runner)
# ---------------------------------------------------------------------------

class CellResult:
    """Result record for a single K8s-dispatched training cell."""

    __slots__ = ("cell_id", "status", "metrics", "gpu", "retries", "error")

    def __init__(
        self,
        *,
        cell_id: str,
        status: str,
        metrics: dict[str, Any] | None,
        gpu: str,
        retries: int,
        error: str | None,
    ) -> None:
        self.cell_id = cell_id
        self.status = status
        self.metrics = metrics
        self.gpu = gpu
        self.retries = retries
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "gpu": self.gpu,
            "retries": self.retries,
            "error": self.error,
        }


def _headline_metric(metrics: dict[str, Any] | None) -> Any:
    if not isinstance(metrics, dict):
        return None
    for key in ("metric", "reward_mean", "accuracy"):
        val = metrics.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            return val
    return None


def _load_cell_manifest(output_dir: Path) -> dict[str, Any] | None:
    mf = output_dir / CELL_MANIFEST_NAME
    if not mf.is_file():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _should_skip_cell(
    cell_id: str,
    output_dir: Path,
    fingerprints: dict[str, str],
    force_cells: set[str],
) -> bool:
    """Return True iff a prior manifest authorises skipping this cell (Track B)."""
    if cell_id in force_cells:
        return False
    current_fp = fingerprints.get(cell_id)
    if not current_fp:
        return False
    manifest = _load_cell_manifest(output_dir)
    if not manifest or manifest.get("status") != "ok":
        return False
    return manifest.get("fingerprint") == current_fp


def _write_cell_manifest(
    output_dir: Path,
    *,
    cell_id: str,
    status: str,
    fingerprint: str | None,
    metrics: dict[str, Any] | None,
    retries: int,
    now_iso: str | None,
) -> None:
    manifest: dict[str, Any] = {
        "cell_id": cell_id,
        "status": status,
        "fingerprint": fingerprint,
        "metric": _headline_metric(metrics),
        "retries": retries,
    }
    if now_iso is not None:
        manifest["completed_at"] = now_iso
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CELL_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning(
            "k8s_job_cell_runner: could not write %s for cell=%s: %s",
            CELL_MANIFEST_NAME, cell_id, exc,
        )


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
) -> dict[str, Any]:
    """Build the K8s Job manifest dict for a single training cell."""
    pvc_name = f"{namespace}-files-pvc"

    env_vars = [
        {"name": "REPROLAB_CELL_ID", "value": cell_id},
        {"name": "REPROLAB_CELL_PARAMS", "value": cell_params_json},
        {"name": "REPROLAB_CELL_OUTPUT_DIR", "value": f"/mnt/outputs/{cell_id}"},
        {"name": "REPROLAB_CELL_MAX_OOM_RETRIES", "value": str(max_oom_retries)},
        {"name": "REPROLAB_BLOB_ACCOUNT", "value": storage_account},
        {"name": "REPROLAB_BLOB_CONTAINER", "value": blob_container},
        {"name": "REPROLAB_BLOB_CODE_PREFIX", "value": code_blob_prefix},
        {"name": "REPROLAB_BLOB_OUTPUT_PREFIX", "value": output_blob_prefix},
    ]
    if fingerprint:
        env_vars.append({"name": "REPROLAB_CELL_FINGERPRINT", "value": fingerprint})
    if now_iso:
        env_vars.append({"name": "REPROLAB_CELL_NOW_ISO", "value": now_iso})

    pod_template: dict[str, Any] = {
        "metadata": {
            "labels": {
                "azure.workload.identity/use": "true",
                "app": "reprolab-cell",
                "cell-id": cell_id[:63],
            },
        },
        "spec": {
            "serviceAccountName": service_account,
            "restartPolicy": "Never",
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
            "nodeSelector": {"agentpool": node_pool_name},
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
                        "requests": {"nvidia.com/gpu": "1"},
                        "limits": {"nvidia.com/gpu": "1"},
                    },
                    "volumeMounts": [
                        {
                            "name": "reprolab-cache",
                            "mountPath": "/mnt/reprolab-cache",
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
            "backoffLimit": 2,
            "activeDeadlineSeconds": active_deadline_seconds,
            "ttlSecondsAfterFinished": 3600,
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
    last_phase: str = "Pending"

    while True:
        now = time.monotonic()

        # Overall timeout guard.
        if overall_deadline is not None and now >= overall_deadline:
            return _watch_result("overall_timeout")

        # Per-cell deadline.
        if now >= job_deadline:
            return _watch_result("deadline")

        # Poll Job status.
        try:
            job = k8s.batch.read_namespaced_job_status(job_name, namespace)
        except Exception as exc:
            logger.warning("k8s_job_cell_runner: read_namespaced_job_status failed: %s", exc)
            time.sleep(5)
            continue

        status = getattr(job, "status", None)
        conditions = getattr(status, "conditions", None) or []
        succeeded = getattr(status, "succeeded", 0) or 0
        failed = getattr(status, "failed", 0) or 0

        # Check for terminal condition.
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

        if succeeded:
            node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
            return _watch_result("succeeded", exit_code=exit_code, node_name=node, log=log)
        if failed and not _has_active_pods(k8s, job_name, namespace):
            node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
            return _watch_result("failed", exit_code=exit_code, node_name=node, log=log)

        # Pending timeout check.
        phase = _get_pod_phase(k8s, job_name, namespace)
        if phase in ("Pending", None):
            if pending_since is None:
                pending_since = time.monotonic()
            elif time.monotonic() - pending_since >= pending_timeout_s:
                return _watch_result("pending_timeout")
        else:
            pending_since = None  # reset once past Pending
            last_phase = phase or last_phase

        time.sleep(5)


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
) -> dict[str, Any] | None:
    """Download ``metrics.json`` from Blob into ``output_dir/<cell_id>/metrics.json``.

    Returns the parsed dict, or None on any failure.
    """
    blob_name = f"{output_blob_prefix}/{cell_id}/metrics.json"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=account_name,
            container_name=container_name,
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
) -> None:
    """Download pod log from Blob (best-effort); fall back to ``fallback_log``."""
    blob_name = f"{output_blob_prefix}/{cell_id}/logs/attempt-0.log"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=account_name,
            container_name=container_name,
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
) -> dict[str, Any] | None:
    """Fetch Blob ``status.json`` to reconcile a TTL-deleted Job."""
    blob_name = f"{output_blob_prefix}/{cell_id}/status.json"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=account_name,
            container_name=container_name,
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
        return "error", f"capacity_exhausted: Job {cell_id} stuck in Pending", retries

    if w_status in ("overall_timeout", "deadline"):
        return "error", f"timeout: watch_status={w_status}", retries

    if w_status == "failed":
        # Check for terminal OOM sentinel.
        if outcome == _SENTINEL_OOM_OUTCOME or exit_code == _EXIT_OOM_EXHAUSTED:
            return "oom_failed", log[-2000:] if log else f"exit_code={exit_code}", retries
        return "error", log[-2000:] if log else f"exit_code={exit_code}", retries

    if w_status == "succeeded":
        if exit_code is None or exit_code == 0:
            return "ok", None, retries
        if exit_code == _EXIT_OOM_EXHAUSTED or outcome == _SENTINEL_OOM_OUTCOME:
            return "oom_failed", log[-2000:] if log else f"exit_code={exit_code}", retries
        if exit_code in _EXIT_TERMINAL:
            return "error", f"wrapper exit {exit_code}", retries
        return "ok", None, retries

    # Fallback.
    return "error", f"unexpected watch_status={w_status}", retries


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
) -> CellResult:
    """Submit a K8s Job for ``cell`` and block until terminal, then return a CellResult."""
    cell_id: str = cell.get("id", f"cell_{id(cell)}")
    job_name = _job_name(cell_id, run_id)
    output_dir = output_root / cell_id
    log_path = output_root / f"{cell_id}.log"

    cell_params_json = json.dumps(cell)

    manifest = _build_job_manifest(
        job_name=job_name,
        namespace=namespace,
        service_account=service_account,
        node_pool_name=node_pool_name,
        base_image=base_image,
        storage_account=storage_account,
        blob_container=blob_container,
        files_share=_setting("azure_files_share", "reprolab-cache"),
        cell_id=cell_id,
        cell_params_json=cell_params_json,
        output_blob_prefix=output_blob_prefix,
        code_blob_prefix=code_blob_prefix,
        active_deadline_seconds=active_deadline_seconds,
        max_oom_retries=max_oom_retries,
        fingerprint=fingerprint,
        now_iso=now_iso,
    )

    # Submit the Job.
    try:
        k8s.batch.create_namespaced_job(namespace, manifest)
        logger.info(
            "k8s_job_cell_runner: submitted Job=%s for cell=%s (deadline=%ds)",
            job_name, cell_id, active_deadline_seconds,
        )
    except Exception as exc:
        logger.error("k8s_job_cell_runner: create_namespaced_job failed cell=%s: %s", cell_id, exc)
        return CellResult(
            cell_id=cell_id,
            status="error",
            metrics=None,
            gpu="aks:unassigned",
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
    gpu_label = f"aks:{node_name}" if node_name else "aks:unassigned"

    # Reconcile Blob status.json if Job was TTL-deleted before we could read it.
    blob_status: dict[str, Any] | None = None
    if watch["status"] in ("succeeded", "failed") and watch.get("exit_code") is None:
        blob_status = _try_reconcile_status(
            cell_id=cell_id,
            output_blob_prefix=output_blob_prefix,
            account_name=storage_account,
            container_name=blob_container,
        )

    status, error, retries = _map_status(watch, cell_id=cell_id, blob_status=blob_status)

    # Pull metrics from Blob.
    metrics: dict[str, Any] | None = None
    if status in ("ok", "oom_failed", "error"):
        metrics = _try_download_metrics(
            cell_id=cell_id,
            output_blob_prefix=output_blob_prefix,
            account_name=storage_account,
            container_name=blob_container,
            output_dir=output_dir,
        )

    # Persist log.
    _try_download_log(
        cell_id=cell_id,
        output_blob_prefix=output_blob_prefix,
        account_name=storage_account,
        container_name=blob_container,
        log_path=log_path,
        fallback_log=watch.get("log", ""),
    )

    # Write local resume manifest.
    _write_cell_manifest(
        output_dir,
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
                             ``REPROLAB_CELL_MAX_OOM_RETRIES``.
        per_cell_timeout_s:  Maps to Job ``activeDeadlineSeconds``.
        overall_timeout_s:   Wall-clock cap for the WHOLE matrix.
        gpus_per_cell:       Must be 1; any other value returns ``"error"`` for
                             all non-skipped cells (Azure Jobs are 1-GPU each).
        fingerprints:        ``{cell_id: fingerprint}`` for resume parity.
        force_cells:         Cell ids that must always re-run.
        now_iso:             ISO-8601 timestamp stamped into ``cell_manifest.json``.

    Returns:
        Dict mapping ``cell["id"]`` → ``CellResult.to_dict()``.  Every input cell
        is present regardless of outcome; never raises on single-cell failure.
    """
    # Fast path — no SDK imports needed.
    if not cells:
        return {}

    # Read settings defensively.
    namespace: str = _setting("azure_namespace", "reprolab")
    service_account: str = _setting("azure_service_account", "reprolab-sa")
    node_pool_name: str = _setting("azure_node_pool_name", "gpunodes")
    base_image: str = _setting("azure_base_image", "reprolab.azurecr.io/reprolab-aks-cell:latest")
    storage_account: str = _setting("azure_storage_account", "")
    blob_container: str = _setting("azure_blob_container", "reprolab-artifacts")
    azure_max_nodes: int = int(_setting("azure_max_nodes", 8))
    gpu_usd_per_hour: float = float(_setting("azure_gpu_usd_per_hour", 3.67))
    pending_timeout_s: float = float(_setting("azure_pending_timeout_seconds", 900))

    _fingerprints: dict[str, str] = fingerprints or {}
    _force_cells: set[str] = force_cells or set()
    _resume_armed = bool(os.environ.get("REPROLAB_RESUME_CELLS", "").strip())

    run_budget = _get_run_budget()
    event_sink = _get_event_sink()

    # Derive a run_id from the output_root path (last two segments).
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = output_root.name  # best-effort

    # gpus_per_cell guard — Azure Jobs are 1-GPU each.
    if gpus_per_cell != 1:
        msg = f"k8s_job_cell_runner: gpus_per_cell={gpus_per_cell} != 1; all cells error"
        logger.error(msg)
        event_sink("run_warning", {"code": "k8s_gpus_per_cell", "message": msg})
        return {
            cell.get("id", f"cell_{i}"): CellResult(
                cell_id=cell.get("id", f"cell_{i}"),
                status="error",
                metrics=None,
                gpu="aks:unassigned",
                retries=0,
                error=msg,
            ).to_dict()
            for i, cell in enumerate(cells)
        }

    # Overall deadline.
    overall_deadline: float | None = (
        time.monotonic() + overall_timeout_s
        if overall_timeout_s is not None and overall_timeout_s > 0
        else None
    )

    # Parallelism.
    parallelism = min(
        max_parallel or azure_max_nodes,
        azure_max_nodes,
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
                status="error",
                metrics=None,
                gpu="aks:unassigned",
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
                status="error",
                metrics=None,
                gpu="aks:unassigned",
                retries=0,
                error=err,
            ).to_dict()
            for i, cell in enumerate(cells)
        }

    # Budget tracking: sum of reserved GPU-seconds for active + completed cells.
    reserved_gpu_seconds = 0.0
    budget_lock = threading.Lock()

    results: dict[str, CellResult] = {}
    results_lock = threading.Lock()

    def _process_cell(cell: dict[str, Any]) -> None:
        nonlocal reserved_gpu_seconds

        cell_id: str = cell.get("id", f"cell_{id(cell)}")
        output_dir = output_root / cell_id

        # --- Resume skip (Track B) ---
        if _resume_armed and _should_skip_cell(cell_id, output_dir, _fingerprints, _force_cells):
            manifest = _load_cell_manifest(output_dir)
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
                    status="skipped",
                    metrics=prior_metrics,
                    gpu="aks:unassigned",
                    retries=0,
                    error=None,
                )
            return

        # --- Overall deadline ---
        if overall_deadline is not None and time.monotonic() >= overall_deadline:
            with results_lock:
                results[cell_id] = CellResult(
                    cell_id=cell_id,
                    status="error",
                    metrics=None,
                    gpu="aks:unassigned",
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
                gpu_usd_per_hour=gpu_usd_per_hour,
                cell_id=cell_id,
            )
            if budget_err:
                logger.warning("k8s_job_cell_runner: budget exceeded: %s", budget_err)
                event_sink("run_warning", {"code": "k8s_budget_exceeded", "message": budget_err})
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="error",
                        metrics=None,
                        gpu="aks:unassigned",
                        retries=0,
                        error=budget_err,
                    )
                return
            reserved_gpu_seconds = new_reserved

        # --- Submit and watch ---
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
            run_id=run_id,
        )
        with results_lock:
            results[cell_id] = result

    # Run cells with thread-pool parallelism.
    cell_queue: list[dict[str, Any]] = list(cells)
    active_threads: list[threading.Thread] = []
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
                status="error",
                metrics=None,
                gpu="aks:unassigned",
                retries=0,
                error="worker exited without recording a result",
            )

    return {cid: r.to_dict() for cid, r in results.items()}
