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

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Mapping
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
from backend.agents.rlm import cpu_class  # noqa: E402
from backend.agents.rlm.feature_flags import env_truthy  # noqa: E402
from backend.agents.rlm.run_controller import durable_controller_enabled  # noqa: E402
from backend.services.runtime import deadline  # noqa: E402
from backend.services.runtime.gcs_blob import upload_bytes as _gcs_upload_bytes  # noqa: E402
from backend.services.runtime.job_fence import (  # noqa: E402
    adopt_or_submit,
    fenced_blob_prefix,
    fenced_job_name,
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

# Literal name of the persistent-cache PVC. Must match the Helm-rendered PVC
# ("reprolab-cache" in infra/gcp/helm/templates/pvc-cache.yaml and its Azure
# mirror) -- NOT the backing Filestore/Azure-Files share name, which is a
# separate, operator-configurable string (files_share).
_CACHE_PVC_NAME = "reprolab-cache"

# Dataset/asset-access credentials CredentialBroker may hand out for injection
# into a training-cell pod env (backend/services/runtime/credential_broker.py
# is the canonical resolver -- read its registry, don't hand-roll a second
# one). Deliberately EXCLUDES LLM-provider keys (anthropic/openai/runpod/
# azure_foundry api keys): a cell pod runs train_cell.py, never an LLM call,
# so injecting those would only widen the secret blast radius for no
# functional benefit.
_CELL_POD_SECRET_NAMES: tuple[str, ...] = (
    "hf_token",
    "kaggle_key",
    "aws_access_key_id",
    "aws_secret_access_key",
)

# Default fallback values used when a settings attribute is absent (defensive,
# so the module imports + tests run against a partial/older config).
_SETTINGS_DEFAULTS: dict[str, Any] = {
    # --- AWS / EKS ---
    # All GPU metadata is deliberately inert.  EKS pool labels, VRAM, and
    # effective prices are deployment facts and must be declared before the
    # runner may submit a cell Job.
    "aws_namespace": "reprolab",
    "aws_service_account": "reprolab-sa",
    "aws_base_image": "",
    "aws_s3_bucket": "",
    "aws_region": "",
    "aws_max_nodes": 0,
    "aws_gpus_per_node": 0,
    "aws_per_gpu_vram_gb": 0.0,
    "aws_gpu_usd_per_hour": 0.0,
    "aws_pending_timeout_seconds": 1500,
    "aws_gpu_skus": [],
    "aws_ttl_seconds_after_finished": 3600,
    "aws_job_backoff_limit": 0,
    "aws_use_spot": False,
    "aws_spot_backoff_limit": 3,
    "aws_cell_preempt_grace_s": 20,
    "aws_cache_mount_path": "/mnt/reprolab-cache",
    "aws_files_cache_enabled": False,
    "aws_files_share": "",
    "aws_watch_poll_interval_s": 5.0,
    "aws_cell_oom_batch_scale_step1": 0.5,
    "aws_cell_oom_batch_scale_floor": 0.25,
    "aws_bootstrap_pip_timeout_s": 600,
    # --- Azure / AKS ---
    "azure_namespace": "reprolab",
    "azure_service_account": "reprolab-sa",
    "azure_node_pool_name": "gpua100",
    # P1-fix-5: empty string fallback — ERROR clearly at submit if blank rather
    # than silently using a floating :latest tag.
    "azure_base_image": "",
    "azure_storage_account": "",
    "azure_blob_container": "reprolab-artifacts",
    "azure_files_share": "reprolab-cache",
    "azure_files_cache_enabled": True,
    # P1-fix-5: aligns with config.py default of 4.
    "azure_max_nodes": 4,
    "azure_gpus_per_node": 1,
    "azure_per_gpu_vram_gb": 80.0,
    "azure_gpu_usd_per_hour": 3.67,
    "azure_pending_timeout_seconds": 1500,
    "azure_boot_timeout_seconds": 900,
    "azure_gpu_skus": [],        # provisioned SKU short_names (list[str])
    "dynamic_gpu_max_escalations": 2,
    # P1-fix-8: configurable knobs (config.py additions by another agent).
    "azure_ttl_seconds_after_finished": 3600,
    "azure_job_backoff_limit": 0,
    # Spot/preemptible data plane (opt-in; default off → on-demand behavior).
    # use_spot adds the cloud spot-taint toleration to the cell Pod; spot_backoff_limit
    # lets a preempted cell Job reschedule onto a fresh spot node (an evicted Pod exits
    # outside the 40-44 FailJob codes, so backoff applies to preemptions, not app errors).
    "azure_use_spot": False,
    "azure_spot_backoff_limit": 3,
    # Grace window the cell entrypoint gets on a SIGTERM (spot preemption / node
    # drain) to flush its checkpoint + partial metrics before the kubelet SIGKILLs.
    # Injected as both the OPENRESEARCH_CELL_PREEMPT_GRACE_S env (entrypoint reads it)
    # and the pod terminationGracePeriodSeconds (else the kubelet's 30s default would
    # truncate a longer flush). Clamped to [1, 120].
    "azure_cell_preempt_grace_s": 20,
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
    "gcp_gpus_per_node": 1,
    "gcp_gpu_usd_per_hour": 3.67,
    "gcp_pending_timeout_seconds": 900,
    "gcp_gpu_skus": ["gcp_a100_80x8"],  # dormant fallback; keep == config.gcp_gpu_skus default (live Settings always wins; this only guards the get_settings()-failed path from re-introducing the nodeSelector mismatch)
    "gcp_ttl_seconds_after_finished": 3600,
    "gcp_job_backoff_limit": 0,
    "gcp_use_spot": False,
    "gcp_spot_backoff_limit": 3,
    "gcp_cell_preempt_grace_s": 20,
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

# Kept separate from bind_run_context so older tests/callers that monkeypatch
# its historical three-argument shape remain compatible.
_PROJECT_ID_CTX: ContextVar[str] = ContextVar(
    "k8s_job_cell_runner_project_id", default=""
)


class _BudgetReservationLedger:
    """One controller-owned reservation pool shared by a logical matrix.

    A staged search invokes the runner once for candidates and again for full
    cells.  Its caps apply to the logical matrix, not each invocation, so the
    same lock and accumulated reservations must span both phases.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.gpu_seconds = 0.0
        self.gpu_usd = 0.0


_BUDGET_RESERVATION_LEDGER_CTX: ContextVar[_BudgetReservationLedger | None] = ContextVar(
    "k8s_job_cell_runner_budget_reservation_ledger", default=None
)


@contextmanager
def _bind_budget_reservation_ledger(ledger: _BudgetReservationLedger) -> Iterator[None]:
    """Bind a shared reservation pool across multiple runner invocations."""
    token = _BUDGET_RESERVATION_LEDGER_CTX.set(ledger)
    try:
        yield
    finally:
        _BUDGET_RESERVATION_LEDGER_CTX.reset(token)


def _new_budget_reservation_ledger() -> _BudgetReservationLedger:
    """Create an unshared reservation pool for one logical cloud matrix."""
    return _BudgetReservationLedger()


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
def _bind_project_id(project_id: str) -> Iterator[None]:
    """Bind a controller project id for collision-safe S3 object prefixes."""
    token = _PROJECT_ID_CTX.set(str(project_id).strip())
    try:
        yield
    finally:
        _PROJECT_ID_CTX.reset(token)


@contextmanager
def bind_run_context(
    *,
    run_budget: Any | None = None,
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    gpu_plan: Any | None = None,
    settings_prefix: str | None = None,
    fence_generation: int | None = None,
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
        fence_generation: WS3 durable-controller lease generation (``LeaseToken
                          .generation``), or ``None`` (the default) when no durable
                          controller drives this run.  This is the ONLY channel that
                          threads a controller-restart fence token into ``run_matrix``
                          without adding a kwarg to its signature (pinned byte-for-byte
                          against ``gpu_cell_runner.run_matrix`` by ``TestSignatureParity``).
                          Consumed only when ``run_controller.durable_controller_enabled()``
                          is also true — otherwise inert.

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
        "fence_generation": fence_generation,
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


def _get_fence_generation() -> int | None:
    """Return the bound WS3 controller-generation fence token (or None when unbound).

    Mirrors ``_get_gpu_plan`` — same ``ContextVar`` pattern, same default-None
    behaviour. ``run_matrix`` reads this ONCE at the top of the call (like
    ``gpu_plan``) and threads the resolved value down as an explicit parameter;
    a raw ``threading.Thread``/``ContextVar`` does NOT propagate into the
    worker threads ``run_matrix`` spawns, so re-reading this accessor from
    inside a worker thread would silently see ``None`` even when a generation
    is bound — always thread the value explicitly instead of calling this a
    second time deeper in the stack.

    Env fallback: inside the durable controller Pod nothing binds the
    ContextVar, so when it is unbound this reads the stable fence epoch the
    submit stamped into the Pod env (``OPENRESEARCH_CELL_FENCE_EPOCH``). An
    explicit ContextVar binding wins; a missing/non-integer env yields ``None``
    (byte-identical to before for every non-controller caller).
    """
    bound = _RUN_CONTEXT.get({}).get("fence_generation")
    if bound is not None:
        return bound
    raw = os.environ.get("OPENRESEARCH_CELL_FENCE_EPOCH", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_project_id() -> str:
    """Return the controller-supplied project id for collision-safe S3 prefixes."""
    return _PROJECT_ID_CTX.get("")


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

    Resolved lazily from the bound ``settings_prefix`` (``"azure"`` → AzureBlobStore,
    ``"gcp"`` → GcsStore, ``"aws"`` → S3Store). Lazy imports prevent circular-import
    issues; provider CloudSpecs live in their thin adapter modules.

    Monkeypatch this symbol in tests via::

        monkeypatch.setattr(kjcr, "_object_store", lambda: FakeStore())
    """
    prefix = _get_settings_prefix()
    if prefix == "gcp":
        from backend.services.runtime.gke_job_backend import _GCP_CLOUD  # type: ignore[import]
        return _GCP_CLOUD.make_object_store(_get_settings(), None)
    if prefix == "aws":
        from backend.services.runtime.eks_job_backend import _AWS_CLOUD  # type: ignore[import]
        return _AWS_CLOUD.make_object_store(_get_settings(), None)
    if prefix == "azure":
        account = _setting("azure_storage_account", "") or ""
        container = _setting("azure_blob_container", "reprolab-artifacts") or "reprolab-artifacts"
        from backend.services.runtime.k8s_job_backend import AzureBlobStore  # type: ignore[import]
        return AzureBlobStore(account, container, None)
    # Fail closed: an unrecognised prefix must NOT silently route a run's blob I/O to
    # the wrong cloud (the prior `else: azure` failed OPEN). The ContextVar default is
    # "azure" and both real backends bind explicitly via _bind_settings_prefix, so this
    # only fires on a genuine typo or a missing binding — where a loud error is correct.
    raise ValueError(
        f"k8s_job_cell_runner: unknown settings prefix {prefix!r}; expected 'gcp', 'aws', or 'azure'"
    )


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
    # GCP and AWS use their own object-store clients internally; no shared Azure
    # ContainerClient.  AWS auth is exclusively the pod's IRSA identity.
    if _get_settings_prefix() in ("gcp", "aws"):
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
# Credential injection (CredentialBroker is the canonical resolver -- this is
# the ONLY seam that turns a logical secret name into a cell-pod env var; see
# backend/services/runtime/credential_broker.py::resolve_env)
# ---------------------------------------------------------------------------

def _credential_env_vars() -> list[dict[str, str]]:
    """Resolve dataset/asset credentials for injection into the cell Job env.

    Uses ``CredentialBroker.resolve_env`` -- the canonical secret resolver --
    so the cell-pod env seam shares resolution order/semantics with the
    host-side asset-gating seam (``asset_resolver.py``'s ``gated_exclusion``).
    Only secrets that actually resolve to a non-empty value are returned; an
    unconfigured credential is silently omitted (byte-identical env when
    nothing is configured, matching every other flag/knob in this module).
    Fail-soft: any resolver error yields an empty list rather than aborting
    manifest construction over a credential lookup.
    """
    try:
        from backend.services.runtime.credential_broker import CredentialBroker  # type: ignore[import]
        pairs = CredentialBroker().resolve_env(_CELL_POD_SECRET_NAMES)
    except Exception as exc:  # noqa: BLE001 — a credential lookup must never abort the manifest
        logger.debug("k8s_job_cell_runner: credential resolution failed: %s", exc)
        return []
    if _get_settings_prefix() == "aws":
        # EKS cell pods authenticate only through IRSA.  A developer shell's
        # static key is neither needed nor allowed to cross this boundary.
        pairs = [
            (name, value)
            for name, value in pairs
            if name not in _AWS_STATIC_CREDENTIAL_ENV_NAMES
        ]
    if pairs:
        # Log only the NAMES that were injected -- never the resolved values.
        logger.info(
            "k8s_job_cell_runner: injecting credentials into cell env: %s",
            [name for name, _value in pairs],
        )
    return [{"name": name, "value": value} for name, value in pairs]


# ---------------------------------------------------------------------------
# DNS-safe Job name
# ---------------------------------------------------------------------------

_DNS_SAFE_RE = re.compile(r"[^a-z0-9-]")

_JOB_NAME_PREFIX = "reprolab-cell-"
_JOB_NAME_MAX = 63
_CODE_BUNDLE_EXCLUDED_PARTS = frozenset({"outputs", ".git", "__pycache__", ".venv", "repo"})


def _k8s_identity_digest(value: str) -> str:
    """A DNS/label-safe stable identity for a full run or cell identifier."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _job_config_digest(intent: Mapping[str, Any]) -> str:
    """Return an immutable identity for the exact Job configuration to submit."""
    encoded = json.dumps(intent, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _collision_guard_enabled() -> bool:
    """Whether non-fenced Jobs carry identity hashes and may adopt on a 409.

    EKS has no legacy production workload, so it always uses the stronger
    ownership protocol.  GKE/AKS opt in with the explicit default-OFF flag;
    their existing default manifests and 409 behaviour remain byte-identical.
    """
    return _get_settings_prefix() == "aws" or (
        os.environ.get("OPENRESEARCH_K8S_COLLISION_GUARD", "").strip().lower()
        in ("1", "true", "yes")
    )


def _code_bundle_digest(code_dir: Path) -> str:
    """Hash the exact eligible code bundle sent to the cell artifact store.

    The filter intentionally mirrors the three cloud object-store uploaders:
    generated output/VCS/venv state and bytecode are excluded, while an in-root
    symlink is represented by the bytes that the uploader dereferences.  This
    binds 409 adoption to trainer source, not only to its mutable remote prefix.
    """
    root = code_dir.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not (path.is_file() or path.is_symlink()) or not path.exists():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
            relative = path.relative_to(root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file():
            continue
        if relative.suffix == ".pyc" or _CODE_BUNDLE_EXCLUDED_PARTS.intersection(relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _job_name(cell_id: str, run_id: str = "", gen: int | None = None) -> str:
    """Return a deterministic, collision-resistant K8s Job name.

    A durable controller generation gets its own job identity. Legacy short
    IDs retain their historic spelling; only a truncation-prone long run ID
    gains the full-ID digest needed to avoid a collision.
    """
    if durable_controller_enabled() and gen is not None:
        return fenced_job_name(run_id, cell_id, gen)
    safe_cell = _DNS_SAFE_RE.sub("-", cell_id.lower())
    if not run_id:
        return f"{_JOB_NAME_PREFIX}{safe_cell[:_JOB_NAME_MAX - len(_JOB_NAME_PREFIX)].strip('-')}"
    safe_run = _DNS_SAFE_RE.sub("-", run_id.lower())
    legacy_suffix = f"{safe_run[:16]}-{safe_cell}"
    legacy = (
        f"{_JOB_NAME_PREFIX}"
        f"{legacy_suffix[:_JOB_NAME_MAX - len(_JOB_NAME_PREFIX)].strip('-')}"
    )
    if len(safe_run) <= 16 or not _collision_guard_enabled():
        return legacy
    safe_cell = safe_cell.strip("-") or "cell"
    safe_run = safe_run.strip("-") or "run"
    # 14-byte prefix + (12 + '-' + 10) run token + '-' + (14 + '-' + 10)
    # cell token = 63-byte Kubernetes DNS-label maximum.
    run_token = f"{safe_run[:12]}-{_k8s_identity_digest(run_id)}"
    cell_token = f"{safe_cell[:14]}-{_k8s_identity_digest(cell_id)}"
    return f"{_JOB_NAME_PREFIX}{run_token}-{cell_token}"


def _api_status(exc: Exception) -> int | None:
    """Best-effort Kubernetes-client status extraction without importing it."""
    for name in ("status", "status_code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def _job_is_terminal(job: Any) -> tuple[bool, bool]:
    """Return ``(terminal, succeeded)`` from a Kubernetes Job object.

    We reclaim only a Job whose Job-level condition is authoritative.  A
    transient `failed` counter with retryable Pods must be adopted and watched,
    never treated as permission to launch duplicate GPU work.
    """
    status = getattr(job, "status", None)
    for condition in getattr(status, "conditions", None) or ():
        if str(getattr(condition, "status", "")) != "True":
            continue
        kind = str(getattr(condition, "type", ""))
        if kind == "Complete":
            return True, True
        if kind == "Failed":
            return True, False
    return False, False


def _owned_conflict_job(
    job: Any, *, run_id: str, cell_id: str, config_sha256: str,
) -> bool:
    """Return whether a 409 Job is exactly this controller's cell attempt.

    The short digests make K8s labels and names readable but are *not* an
    authority boundary. Exact controller-generated run/cell IDs and a digest of
    the complete Job intent live in annotations and must agree too; otherwise a
    same-name, spoofed, stale-configuration, or digest-collision Job is treated
    as foreign and never adopted, replaced, or deleted. Namespace RBAC must
    separately restrict Job creation/patching to the campaign controller.
    """
    metadata = getattr(job, "metadata", None)
    labels = getattr(metadata, "labels", None)
    annotations = getattr(metadata, "annotations", None)
    if not isinstance(labels, Mapping) or not isinstance(annotations, Mapping):
        return False
    return (
        labels.get("app") == "reprolab-cell"
        and labels.get("reprolab/run-sha256") == _k8s_identity_digest(run_id)
        and labels.get("reprolab/cell-sha256") == _k8s_identity_digest(cell_id)
        and annotations.get("reprolab.openresearch/run-id") == run_id
        and annotations.get("reprolab.openresearch/cell-id") == cell_id
        and annotations.get("reprolab.openresearch/config-sha256") == config_sha256
    )


def _sanitize_label_token(value: str) -> str:
    """DNS-1123-safe K8s label VALUE derived from ``value`` (a run token).

    Reuses this module's ``_DNS_SAFE_RE`` — the same character class
    ``_job_name`` applies to its run/cell segments — but (unlike
    ``_job_name``'s 16-char-capped run segment) keeps the full 63-char K8s
    label-value budget, since a standalone label value isn't sharing space
    with a name prefix + cell segment. Falls back to ``"unknown"`` on an
    empty/all-unsafe input so a fenced Job never gets an invalid empty label.
    """
    safe = _DNS_SAFE_RE.sub("-", value.lower())[:63].strip("-")
    return safe or "unknown"


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def _check_budget(
    *,
    run_budget: Any | None,
    projected_gpu_usd: float,
    projected_pod_seconds: float,
    cell_id: str,
) -> str | None:
    """Return a ``budget_exhausted:`` error string if either cap is exceeded, else None.

    Takes two independent projected totals (the caller maintains both under
    ``budget_lock``), because the two caps measure different quantities:

    * ``projected_gpu_usd`` — Σ over reserved attempts of
      ``wall_clock_s × gpu_count × $/GPU-hour / 3600``. The ``× gpu_count`` is the
      load-bearing fix: ``gpu_usd_per_hour`` is a PER-GPU rate, so an 8-GPU cell costs
      8× a 1-GPU cell for the same wall-clock. Folding ``gpu_count`` in here (not into
      the seconds accumulator) keeps the dollar cap correct for heterogeneous
      multi-GPU matrices without distorting the pod-seconds cap.
    * ``projected_pod_seconds`` — Σ of wall-clock seconds (NOT × gpu_count), the right
      quantity for ``max_pod_seconds`` (a wall-clock pod-time cap, not GPU-seconds).

    The ``budget_exhausted:`` prefix is the terminal-stop contract: when EVERY remaining
    cell is refused with it, _execute_cell_matrix promotes the matrix to a terminal
    budget_exhausted stop_reason (no re-loop). Comparisons stay ``>=`` (fail-closed: a
    cell that would land exactly on the cap is refused, not admitted).
    """
    if run_budget is None:
        return None
    # GPU-USD cap
    max_run_gpu_usd = getattr(run_budget, "max_run_gpu_usd", None)
    if max_run_gpu_usd and max_run_gpu_usd > 0 and projected_gpu_usd >= max_run_gpu_usd:
        return (
            f"budget_exhausted: projected GPU spend ${projected_gpu_usd:.4f} "
            f">= max_run_gpu_usd ${max_run_gpu_usd:.4f} for cell={cell_id}"
        )
    # Pod-seconds cap (wall-clock)
    max_pod_seconds = getattr(run_budget, "max_pod_seconds", None)
    if max_pod_seconds and max_pod_seconds > 0 and projected_pod_seconds >= max_pod_seconds:
        return (
            f"budget_exhausted: reserved pod-seconds {projected_pod_seconds:.0f}s "
            f">= max_pod_seconds {max_pod_seconds:.0f}s for cell={cell_id}"
        )
    return None


def _accrued_gpu_usd(*, elapsed_s: float, usd_per_hr_per_gpu: float, gpu_count: int) -> float:
    """USD accrued so far for one running cell = hours x per-GPU rate x GPUs."""
    return (elapsed_s / 3600.0) * float(usd_per_hr_per_gpu) * max(1, int(gpu_count))


def _over_gpu_budget(*, accrued: float, cap: float | None) -> bool:
    """True when a positive cap exists and accrued spend meets/exceeds it."""
    if not cap or cap <= 0:
        return False
    return accrued >= cap


# ---------------------------------------------------------------------------
# K8s Job manifest builder
# ---------------------------------------------------------------------------

def _cache_volume_spec(
    *, namespace: str, files_share: str, files_cache_enabled: bool
) -> dict[str, Any]:
    """Return the K8s volume dict named 'reprolab-cache'.

    PVC (<namespace>-files-pvc) when the Azure Files cache is enabled AND a
    share name is configured; otherwise an ephemeral emptyDir so the cell Pod
    never blocks on a missing PVC (spec 2026-06-14 §4.1, blob-only path).
    """
    if files_cache_enabled and files_share.strip():
        return {
            "name": "reprolab-cache",
            # claimName MUST match the Helm PVC metadata.name in
            # infra/azure/helm/templates/pvc-cache.yaml (and the smoke-job
            # claimNames) — both are the literal "reprolab-cache".
            "persistentVolumeClaim": {"claimName": "reprolab-cache"},
        }
    return {"name": "reprolab-cache", "emptyDir": {}}


def _cache_pvc_exists(k8s: _K8sClients, namespace: str, pvc_name: str = _CACHE_PVC_NAME) -> bool:
    """Best-effort LIVE check: does the persistent-cache PVC actually exist?

    A single GET, called ONCE per ``run_matrix`` invocation (not per-cell) --
    mirrors the P0-scale-2 shared-client pattern elsewhere in this module.
    Fail-closed: any error (404 not-found, RBAC denial, a transient API
    hiccup, or an older test double that doesn't implement this method) is
    treated as "not available", so the caller falls back to an HONEST
    emptyDir + loud warning instead of referencing a PVC that may never bind
    (which would strand the cell pod in Pending until the timeout fires --
    worse than an emptyDir, not better).
    """
    try:
        k8s.core.read_namespaced_persistent_volume_claim(pvc_name, namespace)
    except Exception:
        return False
    return True


_AWS_STATIC_CREDENTIAL_ENV_NAMES: frozenset[str] = frozenset({
    "AWS_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN", "AWS_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
})


def _assert_no_static_aws_credentials(env_vars: list[dict[str, str]]) -> None:
    """Reject accidental static AWS credential injection into an EKS cell pod.

    IRSA populates short-lived web-identity plumbing through the Kubernetes
    ServiceAccount.  The controller must never serialize a developer's static
    key, profile, or credential-file path into the Job manifest.
    """
    leaked = sorted({entry.get("name", "") for entry in env_vars} & _AWS_STATIC_CREDENTIAL_ENV_NAMES)
    if leaked:
        raise ValueError(
            "k8s_job_cell_runner: refusing static AWS credential env vars in EKS pod: "
            + ", ".join(leaked)
        )


def _aws_cell_configuration_error(gpu_plan: Any | None) -> str | None:
    """Validate the EKS-only metadata required to meter and target a GPU cell."""
    skus = tuple(str(v).strip() for v in (_cloud_setting("gpu_skus", []) or []) if str(v).strip())
    try:
        max_nodes = int(_cloud_setting("max_nodes", 0) or 0)
        gpus_per_node = int(_cloud_setting("gpus_per_node", 0) or 0)
        vram = float(_cloud_setting("per_gpu_vram_gb", 0.0) or 0.0)
        rate = float(_cloud_setting("gpu_usd_per_hour", 0.0) or 0.0)
    except (TypeError, ValueError):
        return "AWS GPU metadata is malformed"
    errors: list[str] = []
    if not skus:
        errors.append("aws_gpu_skus is empty")
    if max_nodes <= 0:
        errors.append("aws_max_nodes must be > 0")
    if gpus_per_node != 1:
        errors.append("aws_gpus_per_node must equal 1 (v1 EKS meters whole nodes)")
    if vram <= 0:
        errors.append("aws_per_gpu_vram_gb must be > 0")
    if rate <= 0:
        errors.append("aws_gpu_usd_per_hour must be > 0")
    if gpu_plan is not None:
        short_name = getattr(gpu_plan, "short_name", None)
        if isinstance(gpu_plan, dict):
            short_name = gpu_plan.get("short_name")
        if short_name not in skus:
            errors.append("resolved gpu_plan is not an aws_gpu_skus label")
        try:
            count = int(
                gpu_plan.get("gpu_count", 1) if isinstance(gpu_plan, dict)
                else getattr(gpu_plan, "gpu_count", 1)
            )
        except (TypeError, ValueError):
            count = 0
        if count != 1:
            errors.append("resolved gpu_plan must request exactly one GPU for v1 EKS")
    return "; ".join(errors) if errors else None


def _safe_prefix_component(value: str) -> str:
    """Return a path-safe nonempty S3 key component without importing runtime SDKs."""
    safe = _DNS_SAFE_RE.sub("-", value.lower()).strip("-")[:63]
    return safe or "unknown"


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
    now_iso: str | None = None,
    code_bundle_sha256: str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    gpu_plan: Any | None = None,
    # WS3 durable-controller fencing: the run token + lease generation stamped
    # onto the Job's metadata.labels (only when durable_controller_enabled()
    # AND fence_generation is not None — see body). run_id here is whatever
    # token the caller used to build this specific Job's name (may carry the
    # pre-existing escalation "-eN" suffix); unrelated to gpu_plan/escalation.
    run_id: str = "",
    fence_generation: int | None = None,
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
    # Spec 2026-06-14 §4.1: blob-only fallback. When False (or files_share is
    # empty) the cache volume is an ephemeral emptyDir, not the Azure Files PVC.
    files_cache_enabled: bool = True,
    # Phase D (CPU cloud lane, OPENRESEARCH_CPU_CLOUD_CELLS): "gpu" (default)
    # preserves every byte of the manifest below; "cpu" swaps the GPU node
    # selector/toleration/resources for a CPU pool. See the accelerator=="cpu"
    # branch below for the exact substitutions.
    accelerator: str = "gpu",
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

    ``accelerator``:
        ``"gpu"`` (the default) leaves every line below byte-identical to the
        pre-Phase-D manifest. ``"cpu"`` targets the CPU pool label parsed from
        ``OPENRESEARCH_CPU_POOL_LABEL`` (cloud-specific system-pool default),
        drops all GPU tolerations (including the spot toleration — a CPU pool
        is never the spot GPU pool), swaps the container's ``nvidia.com/gpu``
        resources for a plain CPU/memory request, and omits
        ``OPENRESEARCH_CELL_GPU_COUNT`` (meaningless without a GPU).
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
        if _gpu_skus_for_default:
            default_sku = str(_gpu_skus_for_default[0])
        elif _get_settings_prefix() == "aws":
            raise ValueError(
                "k8s_job_cell_runner: aws_gpu_skus is empty; refusing unlabelled EKS GPU Job"
            )
        else:
            default_sku = "azure_a100_80"

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
    _preempt_grace_s = max(1, min(120, int(_cloud_setting("cell_preempt_grace_s", 20))))
    env_vars = [
        {"name": "OPENRESEARCH_CELL_ID",               "value": cell_id},
        {"name": "OPENRESEARCH_CELL_PARAMS",            "value": cell_params_json},
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
        # Preemption grace (spot/drain): the entrypoint flushes checkpoint + partial
        # metrics on SIGTERM within this window; paired with the pod
        # terminationGracePeriodSeconds below so the kubelet can't truncate the flush.
        {"name": "OPENRESEARCH_CELL_PREEMPT_GRACE_S",
         "value": str(_preempt_grace_s)},
    ]
    # Cloud-specific object-store env vars.
    # Azure: frozen contract with the baked ACR image — keep injecting the exact names.
    # GCP: inject OPENRESEARCH_GCP_GCS_BUCKET (the cloud-neutral pair stays above).
    _prefix = _get_settings_prefix()
    if _prefix == "gcp":
        env_vars.append(
            {"name": "OPENRESEARCH_GCP_GCS_BUCKET", "value": _cloud_setting("gcs_bucket", "")}
        )
    elif _prefix == "aws":
        # EKS pods obtain S3 access exclusively through their IRSA-bound Service
        # Account.  Only the bucket and non-secret region cross this boundary.
        env_vars.extend([
            {"name": "OPENRESEARCH_AWS_S3_BUCKET", "value": _cloud_setting("s3_bucket", "")},
            {"name": "AWS_REGION", "value": _cloud_setting("region", "")},
            {"name": "AWS_DEFAULT_REGION", "value": _cloud_setting("region", "")},
            # Refuse the node instance-profile fallback.  EKS worker Pods must
            # receive IRSA's projected web-identity variables or fail in boto3.
            {"name": "AWS_EC2_METADATA_DISABLED", "value": "true"},
        ])
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
    # GPU count: from plan when available, else 1.
    gpu_count_str: str
    if gpu_plan is not None:
        gpu_count_str = str(getattr(gpu_plan, "gpu_count", 1))
    else:
        gpu_count_str = "1"

    # Plumb the leased GPU count into the cell so the in-pod entrypoint can
    # torchrun-wrap a >1-GPU distributed cell
    # (gke_cell_entrypoint.resolve_cell_gpu_count). Additive + default 1 → every
    # single-GPU cell (incl. the AKS path) is byte-for-byte unchanged where unread.
    # Phase D: a CPU-class cell has no GPU to count — the entrypoint never reads
    # this var on that path, so it is simply omitted rather than stamped "1".
    if accelerator != "cpu":
        env_vars.append({"name": "OPENRESEARCH_CELL_GPU_COUNT", "value": gpu_count_str})

    # Dataset/asset credentials (HF_TOKEN etc.) -- resolved via CredentialBroker,
    # the canonical secret resolver. Only injects what is actually configured;
    # byte-identical env when nothing resolves (the common case today).
    env_vars.extend(_credential_env_vars())
    if _prefix == "aws":
        _assert_no_static_aws_credentials(env_vars)

    if accelerator == "cpu":
        # Phase D (OPENRESEARCH_CPU_CLOUD_CELLS): target the CPU pool label
        # instead of the GPU reprolab/sku pool. "key=value" → {key: value}.
        _cpu_default = (
            "reprolab/node-type=system"
            if _prefix == "gcp"
            else "kubernetes.azure.com/mode=system"
        )
        _cpu_label = os.environ.get("OPENRESEARCH_CPU_POOL_LABEL", _cpu_default)
        _cpu_key, _, _cpu_value = _cpu_label.partition("=")
        if not _cpu_key.strip() or not _cpu_value.strip():
            raise ValueError(
                "OPENRESEARCH_CPU_POOL_LABEL must use non-empty key=value form"
            )
        node_selector = {_cpu_key: _cpu_value}
        # No GPU taint to tolerate on a CPU pool — never the spot GPU pool either.
        _tolerations = []
    else:
        # P0-fix-3: node selector uses the infra pool label reprolab/sku in ALL paths.
        # With gpu_plan → target that SKU's pool; without → fall back to the default SKU.
        if gpu_plan is not None:
            node_selector: dict[str, str] = {
                "reprolab/sku": str(getattr(gpu_plan, "short_name", default_sku))
            }
        else:
            node_selector = {"reprolab/sku": default_sku}

        if _prefix == "aws":
            allowed = {str(v) for v in (_cloud_setting("gpu_skus", []) or [])}
            selected = node_selector["reprolab/sku"]
            if selected not in allowed:
                raise ValueError(
                    f"k8s_job_cell_runner: EKS selector {selected!r} is not in aws_gpu_skus"
                )

        # Toleration for the nvidia.com/gpu taint (always present; required by AKS GPU nodes).
        gpu_toleration = {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
        }

        # Spot toleration (opt-in via <prefix>_use_spot): a spot/preemptible GPU pool gets a
        # cloud-specific taint, so the cell Pod must tolerate it or it never schedules onto a
        # spot node. Cloud-specific key (GKE vs AKS). Default (use_spot off) → tolerations is
        # exactly [gpu_toleration], byte-identical to the on-demand path.
        _tolerations = [gpu_toleration]
        if _cloud_setting("use_spot", False):
            if _prefix == "gcp":
                _tolerations.append({
                    "key": "cloud.google.com/gke-spot",
                    "operator": "Equal", "value": "true", "effect": "NoSchedule",
                })
            elif _prefix == "azure":
                _tolerations.append({
                    "key": "kubernetes.azure.com/scalesetpriority",
                    "operator": "Equal", "value": "spot", "effect": "NoSchedule",
                })

    # Pod template labels: base labels + cloud-specific extras.
    # Default to Azure Workload Identity label when pod_template_extra_labels is
    # not explicitly provided, preserving byte-for-byte backward compatibility.
    _pod_extra: dict[str, str] = (
        {"azure.workload.identity/use": "true"}
        if pod_template_extra_labels is None and _prefix == "azure"
        else (pod_template_extra_labels or {})
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
            # +10s buffer over the entrypoint grace so the kubelet doesn't SIGKILL
            # mid-flush (the entrypoint's metrics/sentinel upload runs AFTER the
            # trainer child exits). Without this the kubelet default (30s) caps it.
            "terminationGracePeriodSeconds": _preempt_grace_s + 10,
            "tolerations": _tolerations,
            "nodeSelector": node_selector,
            "volumes": [
                _cache_volume_spec(
                    namespace=namespace,
                    files_share=files_share,
                    files_cache_enabled=files_cache_enabled,
                )
            ],
            "containers": [
                {
                    "name": "cell",
                    "image": base_image,
                    "env": env_vars,
                    "resources": (
                        # Phase D: plain CPU/memory request, no nvidia.com/gpu anywhere.
                        {"requests": {"cpu": "2", "memory": "8Gi"}}
                        if accelerator == "cpu"
                        else {
                            "requests": {"nvidia.com/gpu": gpu_count_str},
                            "limits": {"nvidia.com/gpu": gpu_count_str},
                        }
                    ),
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

    # WS3 fencing labels: merged into the base Job labels (never dropping
    # "app") only when the durable controller is enabled AND a generation is
    # bound — these let a future reaper find every Job for a run via
    # `list_namespaced_job(label_selector="reprolab-run-id=<run>")` and
    # compare `reprolab-generation` against the live lease token. OFF/no-gen
    # ⇒ labels unchanged (byte-identical to before this field existed).
    _job_labels: dict[str, str] = {"app": "reprolab-cell"}
    if durable_controller_enabled() and fence_generation is not None:
        _job_labels["reprolab-run-id"] = _sanitize_label_token(run_id)
        _job_labels["reprolab-generation"] = str(fence_generation)
    job_annotations: dict[str, str] = {}
    if run_id and _collision_guard_enabled():
        if len(code_bundle_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in code_bundle_sha256):
            raise ValueError("k8s_job_cell_runner: code_bundle_sha256 must be a SHA-256 digest")
        # Kubernetes label values are length bounded, so keep abbreviated
        # digests there for listing/filtering and the full controller-owned
        # identities in annotations for 409 ownership verification.
        _job_labels.update({
            "reprolab/run-sha256": _k8s_identity_digest(run_id),
            "reprolab/cell-sha256": _k8s_identity_digest(cell_id),
        })
        job_annotations.update({
            "reprolab.openresearch/run-id": run_id,
            "reprolab.openresearch/cell-id": cell_id,
            # Bind 409 adoption to the generated training configuration too.
            # A stable run/cell ID alone must not reuse a stale Job after the
            # trainer parameters, fingerprint, image, placement, or artifacts
            # changed.
            "reprolab.openresearch/config-sha256": _job_config_digest({
                "cell_params_json": cell_params_json,
                "fingerprint": fingerprint or "",
                "code_bundle_sha256": code_bundle_sha256,
                "active_deadline_seconds": active_deadline_seconds,
                "ttl_seconds_after_finished": ttl_seconds_after_finished,
                "backoff_limit": backoff_limit,
                "pod_failure_policy": pod_failure_rules,
                # `now_iso` is excluded intentionally: it is provenance text,
                # not trainer behavior, and changes on a controller restart.
                "pod_template": {
                    "metadata": pod_template["metadata"],
                    "spec": {
                        **pod_template["spec"],
                        "containers": [{
                            **pod_template["spec"]["containers"][0],
                            "env": [
                                entry for entry in env_vars
                                if entry["name"] != "OPENRESEARCH_CELL_NOW_ISO"
                            ],
                        }],
                    },
                },
            }),
        })

    manifest: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": _job_labels,
            "annotations": job_annotations,
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
    backoff_limit: int = 0,
    gpu_budget_cap: float | None = None,
    gpu_usd_per_hr_per_gpu: float = 0.0,
    gpu_count: int = 1,
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
                      | ``"gpu_budget_exceeded"``
        ``exit_code``  — int from container state (best-effort, may be None)
        ``node_name``  — str (best-effort, may be None)
        ``log``        — str (captured from K8s pod log, best-effort)
        ``outcome``    — str from Blob status.json, or None
        ``retries``    — int from Blob status.json, or 0

    **Mid-cell GPU-$ heartbeat.** ``run_experiment``'s ``RunBudget.check_run_gpu_usd``
    only fires when a cell RETURNS, so a cell wedged on a slow download holds its
    GPU(s) for the full ``active_deadline`` and can silently breach the run's GPU-$
    cap. Each poll re-estimates the cell's accrued GPU spend (elapsed x per-GPU rate
    x gpus) and returns ``"gpu_budget_exceeded"`` the moment it meets/exceeds
    ``gpu_budget_cap``. NO-OP (byte-identical to the prior loop) when ``gpu_budget_cap``
    is falsy or ``gpu_usd_per_hr_per_gpu`` <= 0. The three billing values are threaded
    EXPLICITLY from ``_process_cell`` — ContextVars (``_get_run_budget``/``_get_gpu_plan``)
    do NOT propagate into the ``run_matrix`` worker threads this loop runs in.
    """
    job_deadline = time.monotonic() + active_deadline_seconds
    job_started = time.monotonic()
    pending_since: float | None = None

    while True:
        now = time.monotonic()

        # Overall timeout guard.
        if overall_deadline is not None and now >= overall_deadline:
            return _watch_result("overall_timeout")

        # Per-cell deadline.
        if now >= job_deadline:
            return _watch_result("deadline")

        # Mid-cell GPU-$ heartbeat: kill a cell that would breach the run's GPU-$ cap
        # before its deadline. NO-OP when there is no cap or no positive rate, so a
        # run without a GPU-$ cap is byte-for-byte the prior loop.
        if gpu_usd_per_hr_per_gpu > 0:
            _accrued = _accrued_gpu_usd(
                elapsed_s=now - job_started,
                usd_per_hr_per_gpu=gpu_usd_per_hr_per_gpu,
                gpu_count=gpu_count,
            )
            if _over_gpu_budget(accrued=_accrued, cap=gpu_budget_cap):
                logger.warning(
                    "k8s_job_cell_runner: cell job=%s exceeded GPU-$ cap "
                    "$%.4f >= $%.4f — terminating",
                    job_name, _accrued, gpu_budget_cap,
                )
                node, exit_code, log = _collect_pod_info(k8s, job_name, namespace)
                return _watch_result(
                    "gpu_budget_exceeded",
                    exit_code=exit_code,
                    node_name=node,
                    log=log,
                )

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
        # running.  But with a spot backoffLimit>0 the Job controller RESCHEDULES a
        # preempted Pod, and there is a window where failed>=1 while active==0 BEFORE
        # the replacement Pod appears — classifying that as terminal would defeat the
        # spot retry. Only the Failed *condition* (checked above) is authoritative for
        # backoff exhaustion; from counters alone we wait until the retries are spent
        # (failed > backoffLimit). With backoff_limit=0 (on-demand, the default) this
        # is byte-for-byte the prior behavior: the first failure (failed=1 > 0) is
        # terminal. App failures (exit 40-44) FailJob immediately via podFailurePolicy,
        # so they still terminate at the condition check regardless of backoffLimit.
        if failed and active == 0 and failed > backoff_limit:
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
# WS3 fencing: persisted absolute-epoch deadline (adopt-on-409 companion)
# ---------------------------------------------------------------------------

def _persist_fenced_deadline(
    *, run_id: str, gen: int, cell_id: str, active_deadline_seconds: int
) -> None:
    """Best-effort persist an absolute-epoch deadline record for a fenced cell.

    Keyed by ``fenced_blob_prefix(run_id, gen, cell_id=...) + "deadline.json"``
    — the SAME ``(run_id, gen, cell_id)`` triple that determines this cell's
    fenced Job name, so a controller-restart resubmit of the identical Job
    (the adopt-on-409 branch in ``_run_cell_job``) can re-read exactly this
    record and inherit the REMAINING wall-clock budget instead of a fresh
    full one — otherwise every restart would double the GPU wall-clock spend.

    ``time.time()`` (absolute epoch) is used deliberately, NOT
    ``time.monotonic()`` — monotonic has no fixed reference across process
    restarts, so persisting it would be meaningless (mirrors ``blob_lease``'s
    clock discipline: the caller supplies "now", the module never guesses).

    Fail-soft: any error (unset bucket setting, transient network blip, ...)
    is logged and swallowed — a deadline-persist failure must never fail an
    otherwise-successful Job submit.
    """
    try:
        record = deadline.make_deadline(time.time(), float(active_deadline_seconds))
        blob_name = fenced_blob_prefix(run_id, gen, cell_id=cell_id) + "deadline.json"
        _gcs_upload_bytes(
            deadline.serialize(record),
            blob_name=blob_name,
            bucket=_cloud_setting("gcs_bucket", "") or "",
            project=_setting("gcp_project", None) or None,
        )
    except Exception as exc:
        logger.warning(
            "k8s_job_cell_runner: failed to persist fenced deadline cell=%s gen=%s: %s",
            cell_id, gen, exc,
        )


def _adopted_active_deadline_seconds(
    *,
    run_id: str,
    gen: int,
    cell_id: str,
    storage_account: str,
    blob_container: str,
    blob_client: Any | None,
    fallback_active_deadline_seconds: int,
) -> int:
    """Recompute the REMAINING ``active_deadline_seconds`` for an adopted Job.

    Re-reads the deadline record ``_persist_fenced_deadline`` wrote at the
    original submit (via the existing ``_blob_download_bytes`` seam, same as
    every other per-cell Blob read in this module) and returns
    ``max(1, deadline.remaining_s(record, time.time()))``.

    Falls back to ``fallback_active_deadline_seconds`` (the caller's freshly
    computed value) whenever the blob is missing, unreadable, or corrupt —
    never raises, never blocks an adopt on a persistence hiccup.
    """
    blob_name = fenced_blob_prefix(run_id, gen, cell_id=cell_id) + "deadline.json"
    try:
        data = _blob_download_bytes(
            blob_name,
            account_name=storage_account,
            container_name=blob_container,
            client=blob_client,
        )
        record = deadline.parse(data)
        remaining = deadline.remaining_s(record, time.time())
        return max(1, int(remaining))
    except Exception as exc:
        logger.info(
            "k8s_job_cell_runner: no persisted deadline for cell=%s gen=%s (%s); "
            "adopting with a fresh active_deadline_seconds=%d",
            cell_id, gen, exc, fallback_active_deadline_seconds,
        )
        return fallback_active_deadline_seconds


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

    # Mid-cell GPU-$ heartbeat kill. Terminal + NON-retryable (STATUS_ERROR is not
    # STATUS_OOM_FAILED, so the escalation loop in _process_cell never re-submits it):
    # a cell that already breached the run's GPU-$ cap must not be retried onto a
    # bigger/pricier SKU. Mirrors the "deadline" hard-failure path.
    if w_status == "gpu_budget_exceeded":
        return (
            STATUS_ERROR,
            f"budget_exhausted: cell exceeded run GPU-$ cap ({log[-1500:]})"
            if log else "budget_exhausted: cell exceeded run GPU-$ cap",
            retries,
        )

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
    code_bundle_sha256: str,
    now_iso: str | None,
    run_id: str,
    gpu_plan: Any | None = None,
    fence_generation: int | None = None,
    blob_client: Any | None = None,
    pod_template_extra_labels: dict | None = None,
    files_cache_enabled_override: bool | None = None,
    # Phase D (OPENRESEARCH_CPU_CLOUD_CELLS): "gpu" (default) is byte-identical
    # to before this param existed; "cpu" threads through to
    # _build_job_manifest's accelerator="cpu" branch.
    accelerator: str = "gpu",
    gpu_budget_cap: float | None = None,
    gpu_usd_per_hr_per_gpu: float = 0.0,
    gpu_count: int = 1,
) -> CellResult:
    """Submit a K8s Job for ``cell`` and block until terminal, then return a CellResult.

    ``blob_client`` is the shared ``ContainerClient`` built once per
    ``run_matrix`` invocation (P0-scale-2).  When ``None`` the helpers fall
    back to constructing their own client (pre-fix behaviour — tolerated in
    tests and for any call site that does not supply one).

    ``files_cache_enabled_override`` is the FINAL cache decision already
    computed once per ``run_matrix`` invocation (settings flag AND a live PVC
    existence check — see ``_cache_pvc_exists``). When ``None`` (a direct call
    bypassing ``run_matrix``, e.g. in tests), falls back to the settings-only
    read (pre-existing behaviour, no live check).
    """
    cell_id: str = cell.get("id", f"cell_{id(cell)}")
    job_name = _job_name(cell_id, run_id, gen=fence_generation)
    output_dir = output_root / cell_id
    log_path = output_root / f"{cell_id}.log"

    cell_params_json = json.dumps(cell)

    # Spot-aware backoffLimit: on a spot pool a preempted Pod exits OUTSIDE the 40-44
    # FailJob codes, so a >0 backoffLimit lets the Job controller reschedule that one
    # cell onto a fresh spot node (bounding a preemption's cost to one cell's redo).
    # Off spot, or with an explicit non-zero job_backoff_limit, the configured value wins.
    _backoff_limit = int(_cloud_setting("job_backoff_limit", 0))
    if _backoff_limit == 0 and _cloud_setting("use_spot", False):
        _backoff_limit = int(_cloud_setting("spot_backoff_limit", 3))

    _effective_files_cache_enabled: bool = (
        bool(_cloud_setting("files_cache_enabled", True))
        if files_cache_enabled_override is None
        else bool(files_cache_enabled_override)
    )

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
            run_id=run_id,
            code_bundle_sha256=code_bundle_sha256,
            gpu_plan=gpu_plan,
            # WS3 fencing: threaded through to the manifest's metadata.labels.
            fence_generation=fence_generation,
            # P1-fix-8: configurable knobs from settings.
            ttl_seconds_after_finished=int(_cloud_setting("ttl_seconds_after_finished", 3600)),
            backoff_limit=_backoff_limit,
            cache_mount_path=str(_cloud_setting("cache_mount_path", "/mnt/reprolab-cache")),
            files_cache_enabled=_effective_files_cache_enabled,
            # P1-fix-9: OOM shrink ratios forwarded from settings.
            oom_batch_scale_step1=float(
                _cloud_setting("oom_batch_scale_step1", 0.5)
            ),
            oom_batch_scale_floor=float(
                _cloud_setting("oom_batch_scale_floor", 0.25)
            ),
            bootstrap_pip_timeout_s=int(
                _cloud_setting("bootstrap_pip_timeout_s", 600)
            ),
            # P0-fix-3: default SKU for no-plan fallback.
            default_sku=str(
                (_cloud_setting("gpu_skus", []) or ["azure_a100_80"])[0]
            ),
            pod_template_extra_labels=pod_template_extra_labels,
            accelerator=accelerator,
        )
    except ValueError as exc:
        # P1-fix-5: manifest builder raises ValueError on empty base_image.
        # Treat as a job submission failure — cell becomes "error" with a clear message.
        # Redact defensively: the manifest env may now carry injected credentials
        # (HF_TOKEN etc.), so an exception string must never round-trip one into
        # this CellResult.error field, which IS persisted to cell_manifest.json.
        from backend.services.runtime.credential_broker import CredentialBroker  # type: ignore[import]
        _safe_exc = CredentialBroker.redact_text(str(exc)) or "error (redacted)"
        logger.error(
            "k8s_job_cell_runner: manifest build failed cell=%s: %s", cell_id, _safe_exc
        )
        _cs = {"gcp": "gke", "aws": "eks", "azure": "aks"}.get(
            _get_settings_prefix(), _get_settings_prefix()
        )
        return CellResult(
            cell_id=cell_id,
            status=STATUS_ERROR,
            metrics=None,
            gpu=f"{_cs}:unassigned",
            retries=0,
            error=f"manifest build failed: {_safe_exc}",
        )

    # Submit the Job.  A retry of the same campaign/cell may encounter a
    # terminal Job left behind by Kubernetes TTL lag. Never blindly delete or
    # replace it: first prove full controller ownership, then only adopt an
    # active/succeeded Job. A terminal failure is fail-closed: a second
    # controller must not launch unreserved duplicate GPU work.
    _cs = {"gcp": "gke", "aws": "eks", "azure": "aks"}.get(
        _get_settings_prefix(), _get_settings_prefix()
    )
    # WS3: resolved ONCE per submit attempt — gates both the persisted-deadline
    # write (success path, just below) and the adopt-on-409 branch (except,
    # just below). Uses the `fence_generation` PARAMETER (threaded explicitly
    # from run_matrix's main thread), never `_get_fence_generation()` here —
    # `_run_cell_job` runs on a worker thread, where that ContextVar accessor
    # would silently read back None (see `_get_fence_generation`'s docstring).
    _durable_fenced = durable_controller_enabled() and fence_generation is not None
    submitted_job_name = job_name
    try:
        k8s.batch.create_namespaced_job(namespace, manifest)
        logger.info(
            "k8s_job_cell_runner: submitted Job=%s for cell=%s (deadline=%ds sku=%s)",
            job_name, cell_id, active_deadline_seconds,
            getattr(gpu_plan, "short_name", "default") if gpu_plan else "default",
        )
        if _durable_fenced:
            # Edit 2: persist the absolute-epoch deadline for this fenced
            # submit so a later controller-restart adopt (below) inherits the
            # REMAINING budget instead of a fresh full one. Fail-soft internally
            # — can never turn a successful submit into a failure.
            _persist_fenced_deadline(
                run_id=run_id,
                gen=fence_generation,
                cell_id=cell_id,
                active_deadline_seconds=active_deadline_seconds,
            )
    except Exception as exc:
        # Redact: the submitted manifest may carry injected credentials, and some
        # API-server validation errors echo back request fields — never let a
        # secret value survive into a log line or this CellResult.error field.
        from backend.services.runtime.credential_broker import CredentialBroker  # type: ignore[import]

        _safe_exc = CredentialBroker.redact_text(str(exc)) or "error (redacted)"
        if _api_status(exc) != 409:
            logger.error(
                "k8s_job_cell_runner: create_namespaced_job failed cell=%s: %s",
                cell_id,
                _safe_exc,
            )
            return CellResult(
                cell_id=cell_id,
                status=STATUS_ERROR,
                metrics=None,
                gpu=f"{_cs}:unassigned",
                retries=0,
                error=f"job submission failed: {_safe_exc}",
            )
        if _durable_fenced:
            # The generation-fenced path is separately gated and preserves its
            # legacy adopt/skip semantics.  A generation in the name is the
            # ownership boundary; the generic hash annotations are deliberately
            # absent so OFF and no-generation artifacts stay stable.
            try:
                existing = k8s.batch.read_namespaced_job_status(job_name, namespace)
            except Exception as read_exc:
                safe_read = CredentialBroker.redact_text(str(read_exc)) or "error (redacted)"
                logger.error(
                    "k8s_job_cell_runner: fenced 409 inspection failed cell=%s: %s",
                    cell_id, safe_read,
                )
                return CellResult(
                    cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                    gpu=f"{_cs}:unassigned", retries=0,
                    error=f"job submission failed: {safe_read}",
                )
            existing_status = getattr(existing, "status", None)
            if getattr(existing_status, "succeeded", 0):
                existing_phase = "done"
            elif getattr(existing_status, "active", 0):
                existing_phase = "Running"
            else:
                existing_phase = None
            reconciled = _try_reconcile_status(
                cell_id=cell_id,
                output_blob_prefix=output_blob_prefix,
                account_name=storage_account,
                container_name=blob_container,
                client=blob_client,
            )
            already_succeeded = bool(
                reconciled
                and (reconciled.get("outcome") == STATUS_OK or reconciled.get("exit_code") == 0)
            )
            decision = adopt_or_submit(existing_phase, already_succeeded=already_succeeded)
            if decision == "skip":
                metrics = _try_download_metrics(
                    cell_id=cell_id,
                    output_blob_prefix=output_blob_prefix,
                    account_name=storage_account,
                    container_name=blob_container,
                    output_dir=output_dir,
                    client=blob_client,
                )
                retries = (reconciled or {}).get("retries") or 0
                write_cell_manifest(
                    output_dir, caller="k8s_job_cell_runner", cell_id=cell_id,
                    status=STATUS_OK, fingerprint=fingerprint, metrics=metrics,
                    retries=retries, now_iso=now_iso,
                )
                return CellResult(
                    cell_id=cell_id, status=STATUS_OK, metrics=metrics,
                    gpu=f"{_cs}:unassigned", retries=retries, error=None,
                )
            if decision != "adopt":
                return CellResult(
                    cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                    gpu=f"{_cs}:unassigned", retries=0,
                    error=f"job submission failed: {_safe_exc}",
                )
            active_deadline_seconds = _adopted_active_deadline_seconds(
                run_id=run_id, gen=fence_generation, cell_id=cell_id,
                storage_account=storage_account, blob_container=blob_container,
                blob_client=blob_client,
                fallback_active_deadline_seconds=active_deadline_seconds,
            )
        elif durable_controller_enabled() or fence_generation is not None:
            # Durable controller is off or unbound: preserve the old fail-closed
            # error rather than probing a potentially unrelated legacy Job.
            return CellResult(
                cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                gpu=f"{_cs}:unassigned", retries=0,
                error=f"job submission failed: {_safe_exc}",
            )
        elif not _collision_guard_enabled():
            return CellResult(
                cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                gpu=f"{_cs}:unassigned", retries=0,
                error=f"job submission failed: {_safe_exc}",
            )
        else:
            try:
                existing = k8s.batch.read_namespaced_job_status(job_name, namespace)
            except Exception as read_exc:
                logger.error(
                    "k8s_job_cell_runner: 409 but cannot inspect existing Job cell=%s: %s",
                    cell_id, read_exc,
                )
                return CellResult(
                    cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                    gpu=f"{_cs}:unassigned", retries=0,
                    error=f"job submission conflict could not be inspected: {read_exc}",
                )
            expected_config = str(
                (manifest.get("metadata", {}).get("annotations", {}) or {}).get(
                    "reprolab.openresearch/config-sha256", ""
                )
            )
            if not expected_config or not _owned_conflict_job(
                existing, run_id=run_id, cell_id=cell_id, config_sha256=expected_config,
            ):
                logger.error(
                    "k8s_job_cell_runner: refusing foreign 409 Job reuse cell=%s job=%s",
                    cell_id, job_name,
                )
                return CellResult(
                    cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                    gpu=f"{_cs}:unassigned", retries=0,
                    error="job submission conflict belongs to a different run/cell; refusing adoption",
                )
            terminal, succeeded = _job_is_terminal(existing)
            if not terminal or succeeded:
                logger.info(
                    "k8s_job_cell_runner: adopting owned %s Job=%s for cell=%s after 409",
                    "succeeded" if succeeded else "active", job_name, cell_id,
                )
            elif is_resume_armed():
                # Resume/retry is explicitly armed (OPENRESEARCH_RESUME_CELLS):
                # the operator has already signalled "retry incomplete/failed
                # cells". A Job the Job-level condition confirms Failed (not a
                # transient retryable-pod count — _job_is_terminal only
                # returns True on an authoritative Complete/Failed condition)
                # is definitively not running any GPU work, so deleting it and
                # resubmitting fresh is NOT "unreserved duplicate GPU work" —
                # there is nothing duplicate to race against. Default (resume
                # NOT armed) behavior below is unchanged: fail closed and let
                # an operator inspect/delete manually.
                try:
                    k8s.batch.delete_namespaced_job(
                        job_name, namespace,
                        propagation_policy="Foreground",
                    )
                    k8s.batch.create_namespaced_job(namespace, manifest)
                    logger.info(
                        "k8s_job_cell_runner: deleted+resubmitted owned terminal-failed "
                        "Job=%s for cell=%s (resume armed)", job_name, cell_id,
                    )
                except Exception as retry_exc:
                    safe_retry_exc = CredentialBroker.redact_text(str(retry_exc)) or "error (redacted)"
                    logger.error(
                        "k8s_job_cell_runner: delete+resubmit failed for owned terminal-failed "
                        "Job=%s cell=%s: %s", job_name, cell_id, safe_retry_exc,
                    )
                    return CellResult(
                        cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                        gpu=f"{_cs}:unassigned", retries=0,
                        error=f"job resubmit after owned terminal failure failed: {safe_retry_exc}",
                    )
            else:
                logger.error(
                    "k8s_job_cell_runner: owned terminal failed Job=%s blocks resubmit cell=%s",
                    job_name, cell_id,
                )
                return CellResult(
                    cell_id=cell_id, status=STATUS_ERROR, metrics=None,
                    gpu=f"{_cs}:unassigned", retries=0,
                    error=("job submission conflict is an owned terminal failure; "
                           "refusing unreserved duplicate GPU retry"),
                )

    # Watch Job until terminal.
    watch = _watch_job(
        k8s=k8s,
        job_name=submitted_job_name,
        namespace=namespace,
        overall_deadline=overall_deadline,
        active_deadline_seconds=active_deadline_seconds,
        pending_timeout_s=pending_timeout_s,
        backoff_limit=_backoff_limit,
        gpu_budget_cap=gpu_budget_cap,
        gpu_usd_per_hr_per_gpu=gpu_usd_per_hr_per_gpu,
        gpu_count=gpu_count,
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


def _resolve_framework_image(
    cells: list[dict[str, Any]], base_image: str, mapping: dict[str, str]
) -> str:
    """Pure: resolve the framework->image floor for a cell matrix.

    A cell declares its framework via ``image_key`` (preferred) then
    ``framework`` (fallback); the first that maps to a non-empty image in
    ``mapping`` selects that image. Returns the mapped image ONLY when every
    matching cell agrees on exactly ONE distinct image (the deterministic
    single-framework synth case). Zero matches, an empty mapping, or an
    ambiguous mix of two+ distinct images all fall back to ``base_image`` —
    never guess a run-level image for a heterogeneous matrix.
    """
    if not mapping:
        return base_image
    resolved: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        for field in ("image_key", "framework"):
            val = cell.get(field)
            if isinstance(val, str) and val:
                image = mapping.get(val)
                if image:
                    resolved.add(image)
                    break
    if len(resolved) == 1:
        return next(iter(resolved))
    return base_image


def _maybe_framework_image(cells: list[dict[str, Any]], base_image: str) -> str:
    """E1 gate: apply the framework->image floor when
    ``OPENRESEARCH_FRAMEWORK_IMAGES`` is on. Off/unmapped => ``base_image``
    unchanged (byte-identical). The mapping is the cloud-prefixed
    ``<prefix>_framework_images`` setting (gcp_framework_images on GKE)."""
    if not env_truthy("OPENRESEARCH_FRAMEWORK_IMAGES"):
        return base_image
    mapping = _cloud_setting("framework_images", {}) or {}
    if not isinstance(mapping, dict):
        return base_image
    return _resolve_framework_image(cells, base_image, mapping)


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
    # E1: framework->validated-image floor (OPENRESEARCH_FRAMEWORK_IMAGES,
    # default off). A verl cell auto-targets gke-cell-verl instead of the single
    # gcp_base_image; off/unmapped => base_image unchanged.
    base_image = _maybe_framework_image(cells, base_image)
    _prefix = _get_settings_prefix()
    # ``storage_account`` / ``blob_container`` are legacy Azure-shaped helper
    # parameters.  For AWS they carry the S3 bucket only for call-site parity;
    # actual routing goes through S3Store and does not serialize credentials.
    if _prefix == "aws":
        storage_account = str(_cloud_setting("s3_bucket", "") or "")
        blob_container = ""
    else:
        storage_account = _cloud_setting("storage_account", "") or ""
        blob_container = _cloud_setting("blob_container", "reprolab-artifacts") or "reprolab-artifacts"
    # P1-fix-5: align with config.py default of 4 (was incorrectly 8 here).
    cloud_max_nodes: int = int(_cloud_setting("max_nodes", 4))
    cloud_gpus_per_node: int = int(_cloud_setting("gpus_per_node", 1))
    gpu_usd_per_hour: float = float(_cloud_setting("gpu_usd_per_hour", 3.67))
    pending_timeout_s: float = float(_cloud_setting("pending_timeout_seconds", 900))
    provisioned_skus: list[str] = list(_cloud_setting("gpu_skus", []) or [])
    max_escalations: int = int(_setting("dynamic_gpu_max_escalations", 2))
    # Derive cloud prefix short label for gpu result fields.
    _cloud_short = {"gcp": "gke", "aws": "eks", "azure": "aks"}.get(_prefix, _prefix)
    # Pod template extra labels: only AKS needs this workload-identity label.
    _pod_extra_labels: dict = (
        {"azure.workload.identity/use": "true"} if _prefix == "azure" else {}
    )

    _fingerprints: dict[str, str] = fingerprints or {}
    _force_cells: set[str] = force_cells or set()
    _resume_armed: bool = is_resume_armed()

    # Phase D (CPU cloud lane): read ONCE here, main thread — mirrors the
    # gpu_plan/fence_generation precedent above (ContextVars/env reads that
    # gate per-cell worker-thread behavior are always resolved up front, never
    # re-read inside a worker thread). Off (default) ⇒ every cell below keeps
    # accelerator="gpu" and the post-matrix fallback block never runs.
    _cpu_cloud_enabled: bool = env_truthy("OPENRESEARCH_CPU_CLOUD_CELLS")

    run_budget = _get_run_budget()
    event_sink = _get_event_sink()
    gpu_plan = _get_gpu_plan()
    # WS3 fencing: read ONCE here in the main thread (mirrors gpu_plan above) —
    # ContextVars do NOT propagate into the worker threads this function spawns
    # (see _get_fence_generation's docstring / the settings-prefix precedent
    # below), so every downstream consumer receives this resolved value as an
    # explicit parameter/closure capture, never by re-calling the accessor.
    fence_generation = _get_fence_generation()

    if _prefix == "aws":
        aws_error = _aws_cell_configuration_error(gpu_plan)
        if aws_error:
            msg = f"k8s_job_cell_runner: AWS EKS cell route blocked: {aws_error}"
            logger.error(msg)
            event_sink("run_warning", {"code": "aws_gpu_configuration", "message": msg})
            return {
                cell.get("id", f"cell_{i}"): CellResult(
                    cell_id=cell.get("id", f"cell_{i}"),
                    status=STATUS_ERROR,
                    metrics=None,
                    gpu="eks:unassigned",
                    retries=0,
                    error=msg,
                ).to_dict()
                for i, cell in enumerate(cells)
            }

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

    # Parallelism. Total schedulable GPUs = max_nodes × gpus_per_node; each cell
    # Job requests _cell_gpu_count GPUs (gpu_plan, default 1), so up to
    # total_gpus // _cell_gpu_count cells run at once — K8s packs single-GPU cells
    # onto a multi-GPU node. Default gpus_per_node=1 ⇒ min(max_nodes, len(cells)),
    # byte-identical to before.
    _total_gpus = max(1, cloud_max_nodes * cloud_gpus_per_node)
    _cell_gpu_count = max(1, int(getattr(gpu_plan, "gpu_count", 1) or 1))
    parallelism = min(
        max_parallel or _total_gpus,
        _total_gpus // _cell_gpu_count,
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

    # --- GPU SKU / node-pool preflight: ONE live check per run_matrix call, not
    # per-cell (same "once per call" shape as the persistent-cache check below).
    # A configured SKU with no matching provisioned node pool leaves every cell
    # that resolves to it Pending until capacity_exhausted (~15-25 min wasted,
    # near-$0 apparent spend) -- catch it here, before the first Job is ever
    # submitted, instead of once per cell. GCP-only for now (mirrors gke_check.sh);
    # see backend/services/runtime/gpu_pool_preflight.py for the verified-absent
    # vs cannot-verify distinction (a transient API blip or an idle scale-to-zero
    # cluster must never hard-block a run).
    if _prefix == "gcp" and provisioned_skus:
        from backend.services.runtime import gpu_pool_preflight
        try:
            gpu_pool_preflight.enforce_gpu_pool_preflight(
                provisioned_skus,
                core_api=k8s.core,
                provider_label="GCP",
                settings_var_name="OPENRESEARCH_GCP_GPU_SKUS",
            )
        except Exception as exc:
            err = f"k8s_job_cell_runner: GPU SKU/node-pool preflight failed: {exc}"
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
    if _prefix == "aws":
        project_id = _get_project_id()
        if not project_id:
            msg = "k8s_job_cell_runner: AWS EKS requires controller project_id for collision-safe S3 prefixes"
            logger.error(msg)
            return {
                cell.get("id", f"cell_{i}"): CellResult(
                    cell_id=cell.get("id", f"cell_{i}"), status=STATUS_ERROR,
                    metrics=None, gpu="eks:unassigned", retries=0, error=msg,
                ).to_dict()
                for i, cell in enumerate(cells)
            }
        object_root = (
            f"projects/{_safe_prefix_component(project_id)}/"
            f"runs/{_safe_prefix_component(run_id)}"
        )
    else:
        object_root = f"runs/{run_id}"
    code_blob_prefix = f"{object_root}/{_BLOB_CODE_PREFIX}"
    # WS3 fencing scopes a GCP/Azure controller generation under a distinct
    # output prefix. EKS uses the project-scoped S3 root above for collision
    # safety and has no GCS fence record.
    if _prefix != "aws" and durable_controller_enabled() and fence_generation is not None:
        output_blob_prefix = (
            fenced_blob_prefix(run_id, fence_generation).rstrip("/")
            + "/" + _BLOB_CELLS_PREFIX
        )
    else:
        output_blob_prefix = f"{object_root}/{_BLOB_CELLS_PREFIX}"

    try:
        code_bundle_sha256 = _code_bundle_digest(code_dir)
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
        err = f"k8s_job_cell_runner: code bundle/upload failed: {exc}"
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

    # --- Persistent-cache availability: ONE live check per run_matrix call,
    # not per-cell (mirrors the P0-scale-2 shared-client pattern above).
    # gcp_files_cache_enabled now defaults True, so by default every cell tries
    # to use the shared PVC cache instead of an ephemeral emptyDir. But a
    # config flag flipped on does not mean the PVC actually exists (Terraform
    # filestore_enabled + the matching Helm storage.filestoreShare/filestoreIp
    # values are a separate, operator-run step) -- referencing a nonexistent
    # claimName would strand every cell pod Pending until the timeout. So:
    # check once, and if the cache is configured-on but genuinely unreachable,
    # fall back to an HONEST emptyDir with a loud run_warning naming the cost
    # impact, instead of either silently using emptyDir (the old behaviour) or
    # hanging forever.
    _configured_cache_enabled: bool = bool(_cloud_setting("files_cache_enabled", True))
    _files_share_for_check: str = str(_cloud_setting("files_share", "reprolab-cache") or "")
    _cache_confirmed: bool = False
    if _configured_cache_enabled and _files_share_for_check.strip():
        _cache_confirmed = _cache_pvc_exists(k8s, namespace, _CACHE_PVC_NAME)
    if _configured_cache_enabled and not _cache_confirmed:
        _cloud_label = "Filestore" if _prefix == "gcp" else "Azure Files"
        _flag_name = f"OPENRESEARCH_{_prefix.upper()}_FILES_CACHE_ENABLED"
        _cache_warning = (
            f"persistent cache PVC '{_CACHE_PVC_NAME}' not found in "
            f"namespace={namespace!r} — falling back to an ephemeral emptyDir. "
            f"EVERY cell (and every re-run) will now re-download model weights, "
            f"datasets, and pip wheels while the GPU meters — a multi-GB model "
            f"pulled across a wide grid is real recurring cost, not a one-time "
            f"charge. Provision the {_cloud_label} instance + PVC (Terraform "
            f"filestore_enabled=true + the matching Helm storage.filestoreShare/"
            f"filestoreIp values — see infra/gcp/README.md), or set "
            f"{_flag_name}=false to silence this warning as a deliberate opt-out."
        )
        logger.warning("k8s_job_cell_runner: %s", _cache_warning)
        event_sink("run_warning", {
            "code": "persistent_cache_unavailable",
            "message": _cache_warning,
        })
    _use_persistent_cache: bool = _configured_cache_enabled and _cache_confirmed

    # Budget tracking: sum of reserved GPU-seconds for active + completed cells.
    # A caller may bind one ledger across staged candidate/full invocations;
    # ordinary runs receive a fresh matrix-local ledger.
    reservation_ledger = _BUDGET_RESERVATION_LEDGER_CTX.get()
    if reservation_ledger is None:
        reservation_ledger = _new_budget_reservation_ledger()
    budget_lock = reservation_ledger.lock

    results: dict[str, CellResult] = {}
    results_lock = threading.Lock()

    # ContextVars do NOT propagate into worker threads. Capture the active cloud
    # settings prefix HERE — in the main thread, inside the caller's
    # `_bind_settings_prefix(...)` — and re-pin it at the top of `_process_cell`
    # so each per-cell manifest resolves the right cloud. Without this, a gcp
    # run's cell threads fall back to the ContextVar default ("azure") →
    # `reprolab/sku=azure_a100_80` nodeSelector + the `reprolab-cache` Azure-Files
    # PVC → the pod is unschedulable and sits Pending forever (prj_618, 2026-07-07).
    _active_prefix = _get_settings_prefix()

    def _process_cell(cell: dict[str, Any]) -> None:
        _SETTINGS_PREFIX_CTX.set(_active_prefix)

        # P0-fix-4: per-cell copy so escalation can update the rate for THIS cell
        # without affecting other concurrent cells (shared outer var would be a race
        # AND would cause UnboundLocalError since Python sees the assignment below
        # as making the name local throughout the whole function body).
        cell_gpu_usd_per_hour: float = gpu_usd_per_hour
        # GPUs this cell occupies — the dollar cap must bill ALL of them, because
        # gpu_usd_per_hour is a PER-GPU rate and the manifest requests plan.gpu_count
        # GPUs. From the plan when available, else 1. (The fix for the multi-GPU
        # undercount where an 8-GPU cell was billed as 1.)
        _cell_gpu_count: int = max(1, int(getattr(gpu_plan, "gpu_count", 1) or 1))

        cell_id: str = cell.get("id", f"cell_{id(cell)}")
        output_dir = output_root / cell_id

        # Phase D: per-cell CPU-vs-GPU routing. Off (default), or any cell
        # carrying a hard/soft GPU signal, keeps accelerator="gpu" — the
        # manifest stays byte-identical. Only a CPU-class cell with the flag
        # on gets accelerator="cpu".
        _cell_accelerator: str = (
            "cpu" if _cpu_cloud_enabled and not cpu_class.requires_gpu(cell) else "gpu"
        )

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

        # --- Cross-pod resume (Blob) ---
        # The local should_skip_cell above only sees the orchestrator pod's ephemeral
        # filesystem. When armed and that local manifest is absent — a fresh orchestrator
        # pod after a control-plane preemption, or any rescheduled run under a stable
        # run_id (OPENRESEARCH_STABLE_RUN_ID) — consult the DURABLE Blob status.json the
        # cell entrypoint wrote. A prior success (exit_code 0 / outcome "ok") under the
        # SAME run_id means the cell already completed: skip it and reuse its Blob metrics,
        # bounding a preemption's cost to the in-flight cell. Trust model: a stable run_id
        # pins the same code prefix, so the cell definition is unchanged. Gated on
        # _resume_armed → default (unarmed) runs submit normally, byte-identical.
        if _resume_armed:
            _blob_status = _try_reconcile_status(
                cell_id=cell_id,
                output_blob_prefix=output_blob_prefix,
                account_name=storage_account,
                container_name=blob_container,
                client=shared_blob_client,
            )
            if _blob_status and (
                _blob_status.get("exit_code") == 0 or _blob_status.get("outcome") == "ok"
            ):
                _resumed_metrics = _try_download_metrics(
                    cell_id=cell_id,
                    output_blob_prefix=output_blob_prefix,
                    account_name=storage_account,
                    container_name=blob_container,
                    output_dir=output_dir,
                    client=shared_blob_client,
                )
                # R1: an "ok" status.json with NO downloadable/parseable metrics blob is
                # untrustworthy — the entrypoint ignores a metrics-upload failure on its
                # success path, so a transient upload error leaves ok+no-metrics. Skipping
                # here would store metrics=None and SILENTLY LOSE the cell's result. Only
                # skip when real metrics came back; otherwise fall through and resubmit
                # (re-running one cell is cheap; a lost result is not).
                if _resumed_metrics is not None:
                    logger.info(
                        "k8s_job_cell_runner: cell=%s SKIPPED (Blob resume: prior ok status.json)",
                        cell_id,
                    )
                    with results_lock:
                        results[cell_id] = CellResult(
                            cell_id=cell_id,
                            status=STATUS_SKIPPED,
                            metrics=_resumed_metrics,
                            gpu=f"{_cloud_short}:unassigned",
                            retries=0,
                            error=None,
                        )
                    return
                logger.warning(
                    "k8s_job_cell_runner: cell=%s blob status ok but metrics "
                    "missing/unreadable — NOT skipping, resubmitting to avoid silent "
                    "result loss",
                    cell_id,
                )

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
            _cell_usd = eff_cell_s * _cell_gpu_count * cell_gpu_usd_per_hour / 3600.0
            new_reserved_s = reservation_ledger.gpu_seconds + eff_cell_s
            new_reserved_usd = reservation_ledger.gpu_usd + _cell_usd
            budget_err = _check_budget(
                run_budget=run_budget,
                projected_gpu_usd=new_reserved_usd,
                projected_pod_seconds=new_reserved_s,
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
            reservation_ledger.gpu_seconds = new_reserved_s
            reservation_ledger.gpu_usd = new_reserved_usd

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
                code_bundle_sha256=code_bundle_sha256,
                now_iso=now_iso,
                # P0-fix-2: use the (possibly suffixed) run_id for Job naming.
                run_id=current_run_id,
                gpu_plan=current_plan,
                # WS3 fencing: closure-captured from the main thread (see the
                # fence_generation = _get_fence_generation() comment above) —
                # never re-read via the ContextVar accessor inside this
                # worker thread.
                fence_generation=fence_generation,
                # P0-scale-2: shared client avoids per-call MSI probe.
                blob_client=shared_blob_client,
                pod_template_extra_labels=_pod_extra_labels,
                files_cache_enabled_override=_use_persistent_cache,
                # Phase D: closure-captured per-cell routing decision (see
                # _cell_accelerator above) — never re-derived here.
                accelerator=_cell_accelerator,
                # Mid-cell GPU-$ heartbeat: thread the run's GPU-$ cap + this cell's
                # (possibly escalated) billing rate + gpu_count EXPLICITLY — the watch
                # loop runs in this worker thread where ContextVars don't propagate, so
                # _get_run_budget()/_get_gpu_plan() would read None there. No-op when
                # no cap or no rate.
                gpu_budget_cap=(
                    getattr(run_budget, "max_run_gpu_usd", None)
                    if run_budget is not None else None
                ),
                gpu_usd_per_hr_per_gpu=cell_gpu_usd_per_hour,
                gpu_count=_cell_gpu_count,
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
            # The rate MUST be PER-GPU: _check_budget multiplies it by the cell's
            # gpu_count (see its docstring), and the catalog's approx_usd_per_hr is
            # the WHOLE-MACHINE rate. Feeding the machine rate in raw billed an
            # escalation to a2-ultragpu-8g at $31.44 × 8 = $251.52/hr — an 8×
            # phantom cost that spuriously trips budget_exhausted and aborts a
            # legitimate escalation. usd_per_gpu_hr is the canonical divisor and is
            # a no-op for every 1-GPU SKU.
            escalated_usd_per_hour: float = cell_gpu_usd_per_hour
            if next_sku is not None:
                try:
                    from backend.services.runtime.gpu_catalog import usd_per_gpu_hr  # type: ignore[import]
                    escalated_usd_per_hour = float(usd_per_gpu_hr(next_sku))
                except Exception:  # noqa: BLE001 — a pricing lookup must never abort the escalation
                    escalated_usd_per_hour = float(
                        getattr(next_sku, "approx_usd_per_hr", cell_gpu_usd_per_hour)
                    )

            # P0-fix-4 (cont.): RESERVE the escalated retry's ADDITIONAL budget before
            # committing — the escalated attempt runs another ~deadline on a bigger SKU,
            # so it must bill its own wall-clock × the escalated SKU's gpu_count × the
            # (higher) escalated rate, not merely re-price the already-reserved seconds.
            _esc_gpu_count = max(
                1, int(getattr(next_sku, "gpu_count", _cell_gpu_count) or _cell_gpu_count)
            )
            _esc_eff_s = (
                max(0.0, overall_deadline - time.monotonic())
                if overall_deadline is not None else eff_cell_s
            )
            with budget_lock:
                _esc_usd = _esc_eff_s * _esc_gpu_count * escalated_usd_per_hour / 3600.0
                _esc_new_s = reservation_ledger.gpu_seconds + _esc_eff_s
                _esc_new_usd = reservation_ledger.gpu_usd + _esc_usd
                escalated_budget_err = _check_budget(
                    run_budget=run_budget,
                    projected_gpu_usd=_esc_new_usd,
                    projected_pod_seconds=_esc_new_s,
                    cell_id=cell_id,
                )
                if not escalated_budget_err:
                    # Commit the reservation only when we're actually going to resubmit.
                    reservation_ledger.gpu_seconds = _esc_new_s
                    reservation_ledger.gpu_usd = _esc_new_usd
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

    result_dicts = {cid: r.to_dict() for cid, r in results.items()}

    # Phase D (OPENRESEARCH_CPU_CLOUD_CELLS): when this ENTIRE matrix is
    # CPU-class and EVERY cell came back infra-failed on the cluster path (a
    # single real result — ok/oom_failed/skipped/anything but "error" —
    # suppresses this branch so a genuine result is never discarded), fall
    # back to running the identical cells locally in-process via
    # gpu_cell_runner.run_matrix — the durable controller Pod always has CPU,
    # so a CPU-class matrix should never be stuck behind a broken CPU node
    # pool. Off (default) ⇒ this block never executes.
    if _cpu_cloud_enabled and cpu_class.run_is_cpu_class(cells) and (
        cpu_class.all_cells_infra_failed(result_dicts)
    ):
        _msg = (
            f"k8s_job_cell_runner: all {len(cells)} CPU-class cells infra-failed on "
            "the K8s cluster path — falling back to local in-process "
            "gpu_cell_runner.run_matrix"
        )
        logger.warning(_msg)
        event_sink("run_warning", {"code": "cpu_cloud_fallback", "message": _msg})
        from backend.agents.rlm import gpu_cell_runner as _gpu_cell_runner
        return _gpu_cell_runner.run_matrix(
            cells,
            cell_script,
            output_root=output_root,
            gpus=gpus,
            max_parallel=max_parallel,
            max_oom_retries=max_oom_retries,
            per_cell_timeout_s=per_cell_timeout_s,
            overall_timeout_s=overall_timeout_s,
            gpus_per_cell=gpus_per_cell,
            fingerprints=fingerprints,
            force_cells=force_cells,
            now_iso=now_iso,
        )

    return result_dicts


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
