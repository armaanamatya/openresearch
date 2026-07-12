"""Shared Kubernetes Job-native RuntimeBackend base for cloud providers.

This module provides the generalised ``_KubernetesJobBackend`` base class plus
all K8s helpers that are cloud-agnostic.  Cloud-specific thin adapters (AKS,
GKE) subclass ``_KubernetesJobBackend`` by passing a ``CloudSpec`` descriptor.

All Kubernetes imports are **lazy** (behind factory functions) so the module can
be imported and the full test suite can run without those optional packages
installed.  Tests inject fake ``batch_api``, ``core_api``, and ``blob_client``
via the constructor.

Design reference:
  - Azure AKS: docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md
  - GCP GKE:   docs/superpowers/specs/2026-06-16-gcp-gke-execution-backend-design.md
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import shlex
import time
import typing
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeBackend,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all importable from here; aks_job_backend re-exports the ones
# tests reference directly.
# ---------------------------------------------------------------------------

_JOB_PREFIX = "reprolab-exec"
_DEFAULT_NAMESPACE = "reprolab"
_DEFAULT_PENDING_TIMEOUT_S = 900
_DEFAULT_TTL_AFTER_FINISHED_S = 3600
_DEFAULT_BACKOFF_LIMIT = 0
# No floating :latest default — callers must supply a pinned image setting.
_DEFAULT_BASE_IMAGE = ""


# ---------------------------------------------------------------------------
# Lazy SDK factories — the ONLY places that import optional packages.
# ---------------------------------------------------------------------------


def _load_kubeconfig(kubeconfig: str | None = None) -> None:
    """Load kubeconfig once: in-cluster first, kubeconfig file fallback.

    Extracted to avoid repeating the same incluster-first logic in every API
    factory.  Callers must have already imported ``kubernetes.config`` and
    pass it as an argument to avoid re-importing.

    Raises ``SandboxRuntimeError(backend_unavailable, ...)`` if the
    ``kubernetes`` package is not installed (propagated from callers).
    """
    from kubernetes import config as k8s_config  # type: ignore[import]

    if kubeconfig:
        k8s_config.load_kube_config(config_file=kubeconfig)
    else:
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()


def _load_kubernetes_batch_api(kubeconfig: str | None = None) -> Any:
    """Lazily load and return a kubernetes BatchV1Api instance."""
    try:
        from kubernetes import client as k8s_client  # type: ignore[import]
    except ImportError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "The 'kubernetes' Python package is not installed. "
            "Install it with: pip install kubernetes",
        ) from exc
    _load_kubeconfig(kubeconfig)
    return k8s_client.BatchV1Api()


def _load_kubernetes_core_api(kubeconfig: str | None = None) -> Any:
    """Lazily load and return a kubernetes CoreV1Api instance."""
    try:
        from kubernetes import client as k8s_client  # type: ignore[import]
    except ImportError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "The 'kubernetes' Python package is not installed. "
            "Install it with: pip install kubernetes",
        ) from exc
    _load_kubeconfig(kubeconfig)
    return k8s_client.CoreV1Api()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_name(value: str) -> str:
    """Return a DNS-safe lowercase slug (≤48 chars) from *value*."""
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return safe[:48] or "run"


def _job_name(run_id: str, suffix: str = "") -> str:
    """Generate a deterministic, DNS-safe K8s Job name."""
    base = f"{_JOB_PREFIX}-{_safe_name(run_id)}"
    if suffix:
        base = f"{base}-{_safe_name(suffix)}"
    # Kubernetes names must be ≤63 chars; trim + append a short UUID to ensure
    # uniqueness while staying within the limit.
    uid = uuid.uuid4().hex[:8]
    name = f"{base[:48]}-{uid}"
    return name


def _blob_code_prefix(project_id: str, run_id: str) -> str:
    """Return the blob prefix for uploaded code."""
    return f"runs/{_safe_name(run_id)}/code/"


def _blob_artifact_key(project_id: str, run_id: str, path: str) -> str:
    """Map an in-sandbox path to a run-scoped blob key."""
    clean = path.lstrip("/")
    return f"runs/{_safe_name(run_id)}/artifacts/{clean}"


def _settings_get(settings: Any, attr: str, default: Any = None) -> Any:
    """Defensively read a settings attribute, falling back when it is absent."""
    return getattr(settings, attr, default)


def _extract_exit_code(job: Any) -> int | None:
    """Best-effort: read the exit code from job.status.conditions or pods."""
    try:
        conditions = job.status.conditions or []
        for cond in conditions:
            if cond.type == "Failed" and cond.status == "True":
                return 1
        return None
    except Exception:
        return None


# WS2 guard (docs/superpowers/specs — Phase-3 durable-controller fan-out):
# ``exec()`` submits the base image with NO code-staging step (the
# ``blob_prefix`` uploaded in ``create_sandbox`` is never read back here) — see
# the module docstring gap this closes. This predicate decides, conservatively,
# whether a shell *command* reads project code that this path never stages.
_PY_INTERPRETER_RE = re.compile(r"^python3?(?:\.\d+)?$")
# "-m <module>" invocations that are common environment/tooling bootstrap
# steps, not a project entrypoint — e.g. ``python -m pip install -r
# requirements.txt`` (the dominant "-m" shape actually emitted by this
# codebase's own environment setup, see primitives.py/env_pin.py). Keeping
# this list short and well-known avoids the false positives this guard must
# not produce; an unrecognised module conservatively counts as project code.
_NON_PROJECT_PY_MODULES = frozenset(
    {
        "pip",
        "venv",
        "ensurepip",
        "http.server",
        "pytest",
        "unittest",
        "compileall",
        "site",
        "pdb",
        "cProfile",
        "profile",
        "json.tool",
    }
)


# Shell chain operators that start a new (sub)command when scanning a flat
# shlex token stream (shlex has no notion of shell control operators; each
# becomes its own whitespace-delimited token, e.g. "a && b" -> ["a", "&&",
# "b"]). Used ONLY to gate the bare-".py"-token branch below: a review
# finding (phase3-owner2-job_backend-ws2guard.md) confirmed that branch
# previously fired for ANY ".py"-suffixed token anywhere in the command
# (e.g. "cat code/train.py", "git diff -- train.py") rather than only the
# program actually being invoked.
_SHELL_OPERATORS = frozenset({"&&", "||", ";", "|"})


def _command_needs_staged_code(command: str) -> bool:
    """Conservatively decide whether *command* needs project code on disk.

    True for a ``python``/``python3``/``python3.NN`` invocation of a project
    ``.py`` file (``python train.py``) or an unrecognised ``-m`` module
    (``python -m project.train``); also True for a bare ``*.py`` script
    token that is itself the **invoked program** of a (sub)command
    (``./train.py``, ``pip install -r requirements.txt && ./train.py``) — a
    "train/run-style script" invocation per the design. False for a
    ``python -c "..."`` inline snippet (no file is read), a known
    environment-bootstrap ``-m`` module (``pip``, ...), a ``.py``-suffixed
    token that is merely an ARGUMENT to some other program rather than the
    thing being executed (``cat code/train.py``, ``echo train.py``,
    ``wc -l code/*.py``, ``black train.py``, ``git diff -- train.py``), and
    any command with no python/`.py` reference at all (``nvidia-smi``,
    ``pip install torch``, ``echo``, ``ls``, ``nvcc --version``).

    Deliberately errs toward **False** on anything ambiguous: a false
    negative just preserves today's (already-broken) behavior, whereas a
    false positive would block a legitimate non-code exec. Scans the whole
    (possibly ``&&``/``;``/``|``-chained) command string — a token-by-token
    ``shlex`` split is used so quoting is respected; a command that is not
    validly shell-quoted falls back to a plain whitespace split rather than
    raising. The bare-``.py``-token check only fires at a command-start
    position (index 0, or immediately after a ``_SHELL_OPERATORS`` token) —
    a ``.py``-suffixed token appearing anywhere else in the same (sub)command
    is an argument to whatever program was actually invoked, not code this
    path would execute.
    """
    if not command or not command.strip():
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    at_command_start = True
    for i, tok in enumerate(tokens):
        if tok in _SHELL_OPERATORS:
            at_command_start = True
            continue
        base = tok.rsplit("/", 1)[-1]
        if _PY_INTERPRETER_RE.match(base):
            rest = tokens[i + 1 :]
            for j, arg in enumerate(rest):
                if arg == "-c":
                    break  # inline snippet — no file read from disk
                if arg == "-m":
                    module = rest[j + 1] if j + 1 < len(rest) else ""
                    if module and module not in _NON_PROJECT_PY_MODULES:
                        return True
                    break
                if arg.startswith("-"):
                    continue  # some other interpreter flag — keep scanning
                if arg.endswith(".py"):
                    return True
                break  # first unrecognised non-flag arg — stop guessing
            at_command_start = False
            continue
        if at_command_start and base.endswith(".py"):
            return True  # direct script execution, e.g. "./train.py --seed 1"
        at_command_start = False

    return False


# ---------------------------------------------------------------------------
# Job manifest builder
# ---------------------------------------------------------------------------


def _build_job_manifest(
    *,
    job_name: str,
    namespace: str,
    image: str,
    service_account: str,
    command: str,
    environment: dict[str, str],
    active_deadline_seconds: int,
    ttl_seconds: int,
    backoff_limit: int,
    gpu_sku: str | None = None,
    gpu_count: int = 1,
    pod_template_extra_labels: dict | None = None,
    sandbox_label: str = "aks",
) -> dict:
    """Return a Kubernetes Job manifest dict for a short exec command.

    Uses a plain JSON-serialisable dict so this module does not depend on the
    kubernetes SDK at import time; the SDK materialises it via ``create_namespaced_job``.

    When *gpu_sku* is set (from a resolved GpuPlan), the pod spec gains a
    ``nodeSelector`` targeting that SKU's node pool (``reprolab/sku: <short_name>``)
    and requests ``nvidia.com/gpu: <gpu_count>`` resources.  When *gpu_sku* is None
    the spec falls back to the cluster default (1 GPU, no explicit nodeSelector).

    Parameters
    ----------
    pod_template_extra_labels:
        Additional labels merged into the pod template's metadata.labels.
        Cloud adapters use this for provider-specific annotations (e.g.
        ``{"azure.workload.identity/use": "true"}`` for AKS Workload Identity).
    sandbox_label:
        Value for the ``reprolab/sandbox`` metadata label (e.g. ``"aks"`` or ``"gke"``).
    """
    # OPENRESEARCH_EXEC_COMMAND is already injected by the caller via ``environment``;
    # do NOT append it again here — that would produce a duplicate env var whose
    # behaviour is unspecified by Kubernetes.
    env_list = [{"name": k, "value": str(v)} for k, v in sorted(environment.items())]

    gpu_count = max(1, int(gpu_count or 1))

    # GPU taint toleration — required so the pod can be scheduled on GPU nodes
    # (which carry the ``nvidia.com/gpu:NoSchedule`` taint).
    gpu_toleration = {
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule",
    }

    pod_spec: dict[str, Any] = {
        "serviceAccountName": service_account,
        "restartPolicy": "Never",
        "tolerations": [gpu_toleration],
        "containers": [
            {
                "name": "exec",
                "image": image,
                "command": ["/bin/sh", "-c", command],
                "env": env_list,
                "resources": {
                    "limits": {"nvidia.com/gpu": str(gpu_count)},
                    "requests": {"nvidia.com/gpu": str(gpu_count)},
                },
            }
        ],
    }

    if gpu_sku:
        pod_spec["nodeSelector"] = {"reprolab/sku": gpu_sku}

    # Build pod template labels: start with base, merge cloud-specific extras.
    pod_template_labels: dict[str, str] = {"app": "reprolab-exec"}
    if pod_template_extra_labels:
        pod_template_labels.update(pod_template_extra_labels)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app": "reprolab-exec",
                "reprolab/sandbox": sandbox_label,
            },
        },
        "spec": {
            "backoffLimit": backoff_limit,
            "activeDeadlineSeconds": math.ceil(active_deadline_seconds),
            "ttlSecondsAfterFinished": ttl_seconds,
            "template": {
                "metadata": {
                    "labels": pod_template_labels,
                },
                "spec": pod_spec,
            },
        },
    }


# ---------------------------------------------------------------------------
# ObjectStore protocol + cloud-specific implementations
# ---------------------------------------------------------------------------


class ObjectStore(typing.Protocol):
    """Duck-type interface for a cloud object store (GCS / Azure Blob)."""

    def upload_prefix(self, local_root: Any, *, blob_prefix: str) -> list[str]: ...
    def upload_bytes(self, data: bytes, *, blob_name: str) -> None: ...
    def download_bytes(self, blob_name: str) -> bytes: ...
    def download_artifact(self, blob_name: str, destination: Any) -> Path: ...


class AzureBlobStore:
    """ObjectStore backed by Azure Blob Storage.

    Each method does ``from backend.services.runtime import azure_blob`` at
    call time (NOT at module level or in __init__) so that tests can patch
    ``backend.services.runtime.azure_blob`` via
    ``patch("backend.services.runtime.azure_blob", fake_module)`` and have the
    patch take effect for the duration of the ``with`` block.
    """

    def __init__(self, account_name: str, container_name: str, client: Any) -> None:
        self._account_name = account_name
        self._container_name = container_name
        self._client = client

    def upload_prefix(self, local_root: Any, *, blob_prefix: str) -> list[str]:
        from backend.services.runtime import azure_blob  # type: ignore[import]

        return azure_blob.upload_prefix(
            local_root,
            blob_prefix=blob_prefix,
            account_name=self._account_name,
            container_name=self._container_name,
            client=self._client,
        )

    def upload_bytes(self, data: bytes, *, blob_name: str) -> None:
        from backend.services.runtime import azure_blob  # type: ignore[import]

        azure_blob.upload_bytes(
            data,
            blob_name=blob_name,
            account_name=self._account_name,
            container_name=self._container_name,
            client=self._client,
        )

    def download_bytes(self, blob_name: str) -> bytes:
        from backend.services.runtime import azure_blob  # type: ignore[import]

        return azure_blob.download_bytes(
            blob_name,
            account_name=self._account_name,
            container_name=self._container_name,
            client=self._client,
        )

    def download_artifact(self, blob_name: str, destination: Any) -> Path:
        from backend.services.runtime import azure_blob  # type: ignore[import]

        return azure_blob.download_artifact(
            blob_name,
            destination,
            account_name=self._account_name,
            container_name=self._container_name,
            client=self._client,
        )


class GcsStore:
    """ObjectStore backed by Google Cloud Storage.

    Each method does ``from backend.services.runtime import gcs_blob`` at call
    time so that tests can patch the module via
    ``patch("backend.services.runtime.gcs_blob", fake_module)``.
    """

    def __init__(self, bucket: str, project: str | None, client: Any) -> None:
        self._bucket = bucket
        self._project = project
        self._client = client

    def upload_prefix(self, local_root: Any, *, blob_prefix: str) -> list[str]:
        from backend.services.runtime import gcs_blob  # type: ignore[import]

        return gcs_blob.upload_prefix(
            local_root,
            blob_prefix=blob_prefix,
            bucket=self._bucket,
            project=self._project,
            client=self._client,
        )

    def upload_bytes(self, data: bytes, *, blob_name: str) -> None:
        from backend.services.runtime import gcs_blob  # type: ignore[import]

        gcs_blob.upload_bytes(
            data,
            blob_name=blob_name,
            bucket=self._bucket,
            project=self._project,
            client=self._client,
        )

    def download_bytes(self, blob_name: str) -> bytes:
        from backend.services.runtime import gcs_blob  # type: ignore[import]

        return gcs_blob.download_bytes(
            blob_name,
            bucket=self._bucket,
            project=self._project,
            client=self._client,
        )

    def download_artifact(self, blob_name: str, destination: Any) -> Path:
        from backend.services.runtime import gcs_blob  # type: ignore[import]

        return gcs_blob.download_artifact(
            blob_name,
            destination,
            bucket=self._bucket,
            project=self._project,
            client=self._client,
        )


# ---------------------------------------------------------------------------
# CloudSpec — descriptor passed from thin adapters to _KubernetesJobBackend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudSpec:
    """Descriptor that parameterises ``_KubernetesJobBackend`` for a specific cloud.

    Attributes
    ----------
    provider:
        Short lowercase name, e.g. ``"azure"`` or ``"gcp"``.  Used in log
        messages and error strings.
    settings_prefix:
        Prefix for settings attribute lookups, e.g. ``"azure"`` → reads
        ``azure_namespace``, ``azure_base_image``, etc.
    sandbox_prefix:
        Prefix for ``sandbox_id`` construction, e.g. ``"aks"`` or ``"gke"``.
    sandbox_label:
        Value for the ``reprolab/sandbox`` K8s label on the Job metadata.
    pod_template_extra_labels:
        Additional labels merged into every Job's pod template metadata.
        AKS uses ``{"azure.workload.identity/use": "true"}``; GKE uses ``{}``.
    base_image_setting:
        The exact settings attribute name that holds the base container image,
        e.g. ``"azure_base_image"`` or ``"gcp_base_image"``.
    make_object_store:
        Factory ``(settings, client) -> ObjectStore``; called once on first use.
    ensure_available:
        Cloud-specific preflight check; called by ``ensure_*_available`` helpers.
    """

    provider: str
    settings_prefix: str
    sandbox_prefix: str
    sandbox_label: str
    pod_template_extra_labels: dict
    base_image_setting: str
    make_object_store: Callable  # (settings, client) -> ObjectStore
    ensure_available: Callable  # () -> None


# ---------------------------------------------------------------------------
# _KubernetesJobBackend — generalised base class
# ---------------------------------------------------------------------------


class _KubernetesJobBackend(RuntimeBackend):
    """Generalised RuntimeBackend backed by short-lived Kubernetes Jobs.

    Cloud-specific thin adapters (``AksJobBackend``, ``GkeJobBackend``) subclass
    this and pass a ``CloudSpec`` as the first positional argument.

    Constructor keyword arguments (all optional; test doubles injected here):
      run_budget   – ``RunBudget`` for pod-second / GPU-USD caps.
      gpu_plan     – ``GpuPlan`` (or a plain dict with the same keys) resolved by
                     ``resolve_gpu_requirements``; when set, Jobs are scheduled on
                     the matching node pool via ``nodeSelector={"reprolab/sku": plan.short_name}``
                     and request ``nvidia.com/gpu=plan.gpu_count`` resources.  When
                     ``None`` (default) the backend falls back to the settings node
                     pool and requests 1 GPU.
      batch_api    – pre-built ``kubernetes.client.BatchV1Api`` (or fake).
      core_api     – pre-built ``kubernetes.client.CoreV1Api`` (or fake).
      blob_client  – pre-built cloud storage client (or fake); passed through to
                     the ``ObjectStore`` constructed by ``cloud.make_object_store``.
      settings     – settings object; defaults to ``get_settings()`` lazily.
    """

    def __init__(
        self,
        cloud: CloudSpec,
        *,
        run_budget: Any | None = None,
        gpu_plan: Any | None = None,
        batch_api: Any | None = None,
        core_api: Any | None = None,
        blob_client: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        self._cloud = cloud
        self._run_budget = run_budget
        self._gpu_plan = gpu_plan
        self._batch_api = batch_api
        self._core_api = core_api
        self._blob_client = blob_client
        self._settings = settings
        # Lazy-built ObjectStore; created on first use via _get_store().
        self._store: ObjectStore | None = None
        # Track Job names submitted during this backend instance's lifetime.
        # destroy() will clean these up.
        self._active_jobs: dict[str, list[str]] = {}  # sandbox_id → [job_name, ...]

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _get_settings(self) -> Any:
        if self._settings is not None:
            return self._settings
        from backend.config import get_settings  # lazy import

        return get_settings()

    def _namespace(self) -> str:
        return (
            _settings_get(
                self._get_settings(),
                f"{self._cloud.settings_prefix}_namespace",
                _DEFAULT_NAMESPACE,
            )
            or _DEFAULT_NAMESPACE
        )

    def _base_image(self) -> str:
        """Return the resolved base image tag, or raise if unset (no :latest fallback)."""
        img = (
            _settings_get(
                self._get_settings(),
                self._cloud.base_image_setting,
                _DEFAULT_BASE_IMAGE,
            )
            or _DEFAULT_BASE_IMAGE
        )
        if not img.strip():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"{self._cloud.provider.upper()} backend: "
                f"OPENRESEARCH_{self._cloud.base_image_setting.upper()} "
                f"(or {self._cloud.base_image_setting} setting) is not set. "
                f"Set it to a pinned registry tag.",
            )
        return img

    def _service_account(self) -> str:
        return (
            _settings_get(
                self._get_settings(),
                f"{self._cloud.settings_prefix}_service_account",
                "reprolab-sa",
            )
            or "reprolab-sa"
        )

    def _pending_timeout(self) -> int:
        return int(
            _settings_get(
                self._get_settings(),
                f"{self._cloud.settings_prefix}_pending_timeout_seconds",
                _DEFAULT_PENDING_TIMEOUT_S,
            )
            or _DEFAULT_PENDING_TIMEOUT_S
        )

    def _ttl_after_finished(self) -> int:
        return int(
            _settings_get(
                self._get_settings(),
                f"{self._cloud.settings_prefix}_ttl_seconds_after_finished",
                _DEFAULT_TTL_AFTER_FINISHED_S,
            )
            or _DEFAULT_TTL_AFTER_FINISHED_S
        )

    def _job_backoff_limit(self) -> int:
        val = _settings_get(
            self._get_settings(),
            f"{self._cloud.settings_prefix}_job_backoff_limit",
            None,
        )
        return int(val) if val is not None else _DEFAULT_BACKOFF_LIMIT

    # ------------------------------------------------------------------
    # Object store lazy accessor
    # ------------------------------------------------------------------

    def _get_store(self) -> ObjectStore:
        """Return the cached ObjectStore, constructing it on first use."""
        if self._store is None:
            self._store = self._cloud.make_object_store(self._get_settings(), self._blob_client)
        return self._store

    # ------------------------------------------------------------------
    # API accessors (lazy — fall back to constructing real clients)
    # ------------------------------------------------------------------

    def _gpu_plan_short_name(self) -> str | None:
        """Return the SKU short_name from the stored gpu_plan (GpuPlan or dict), or None."""
        plan = self._gpu_plan
        if plan is None:
            return None
        try:
            if isinstance(plan, dict):
                return plan.get("short_name") or None
            return getattr(plan, "short_name", None) or None
        except Exception:
            return None

    def _default_gpu_sku(self) -> str | None:
        """Return the first entry of the cloud's configured ``<prefix>_gpu_skus`` setting.

        Fallback node-pool target for ``exec()`` when a GPU Job needs to be submitted
        but no ``GpuPlan`` was resolved (``_gpu_plan_short_name()`` is None) — mirrors
        the cell-matrix runner's P0-fix-3 default (``k8s_job_cell_runner._build_job_manifest``).
        This cluster's GPU node pools are scale-to-zero and labeled
        ``reprolab/sku=<short_name>``; a Job submitted with no nodeSelector at all can
        land on no particular pool, so the autoscaler can never satisfy it (scales a
        GPU node up, the pod never binds, the node scales back down — a thrash loop
        that bills the cloud while the run shows $0). Returns None (never invents a
        label) when the setting is unset/empty, preserving the cluster-default
        no-nodeSelector behavior.
        """
        skus = _settings_get(
            self._get_settings(), f"{self._cloud.settings_prefix}_gpu_skus", None
        )
        if not skus:
            return None
        try:
            first = skus[0]
        except (IndexError, TypeError, KeyError):
            return None
        return str(first) if first else None

    def _gpu_plan_gpu_count(self) -> int:
        """Return gpu_count from the stored gpu_plan (GpuPlan or dict), defaulting to 1."""
        plan = self._gpu_plan
        if plan is None:
            return 1
        try:
            if isinstance(plan, dict):
                return int(plan.get("gpu_count") or 1)
            return int(getattr(plan, "gpu_count", None) or 1)
        except Exception:
            return 1

    def _gpu_plan_total_usd_per_hr(self) -> float:
        """Return total_usd_per_hr (per-GPU rate * gpu_count) from the stored gpu_plan, else 0.0."""
        plan = self._gpu_plan
        if plan is None:
            return 0.0
        try:
            if isinstance(plan, dict):
                return float(plan.get("total_usd_per_hr") or 0.0)
            return float(getattr(plan, "total_usd_per_hr", None) or 0.0)
        except Exception:
            return 0.0

    def _enforce_run_budget(self, sandbox: Sandbox) -> None:
        """Fail-closed per-run pod-second + GPU-USD ceiling BEFORE launching a Job.

        The GKE exec path previously stored ``self._run_budget`` but never checked
        it, so a ``sandbox="gcp"`` run had NO per-run dollar ceiling at exec time
        (only the cell-matrix path enforced ``OPENRESEARCH_MAX_RUN_GPU_USD``).
        Mirrors ``runpod_backend.exec``: refuse to submit a new GPU Job once the
        run's cumulative pod-time cost (``elapsed_hr * plan.total_usd_per_hr``)
        reaches the cap. Ephemeral Jobs self-terminate (activeDeadline + TTL), so
        unlike RunPod there is no persistent pod to destroy — refusing new launches
        IS the protection. Raises ``BudgetExhausted`` (same contract as RunPod); a
        None budget / unset caps / missing ``created_at`` are a no-op (byte-identical
        to before)."""
        if self._run_budget is None:
            return
        self._run_budget.check_pod_seconds(
            pod_started_at=sandbox.created_at,
            agent_id="experiment-runner",
        )
        rate = self._gpu_plan_total_usd_per_hr()
        if rate > 0 and sandbox.created_at is not None:
            elapsed_hr = (
                datetime.now(timezone.utc) - sandbox.created_at
            ).total_seconds() / 3600.0
            self._run_budget.check_run_gpu_usd(
                cumulative_pod_usd=elapsed_hr * rate,
                agent_id="experiment-runner",
            )

    def _get_batch_api(self) -> Any:
        if self._batch_api is not None:
            return self._batch_api
        return _load_kubernetes_batch_api()

    def _get_core_api(self) -> Any:
        if self._core_api is not None:
            return self._core_api
        return _load_kubernetes_core_api()

    # ------------------------------------------------------------------
    # RuntimeBackend ABC
    # ------------------------------------------------------------------

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        """Upload the project directory to object storage and return a logical Sandbox token.

        No pod or persistent resource is created.  The image build is a no-op
        because the base image is pre-baked in the cloud registry.
        """
        project_root = config.project_root.resolve()
        if not project_root.exists():
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"{self._cloud.provider.upper()} backend: project root does not exist: {project_root}",
            )

        image = self._base_image()
        blob_prefix = _blob_code_prefix(config.project_id, config.run_id)

        # Upload project files to object storage (run in executor — sync SDK).
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                self._upload_project_sync,
                project_root,
                blob_prefix,
                config,
            )
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"{self._cloud.provider.upper()} backend: failed to upload project to "
                f"blob prefix {blob_prefix!r}: {exc}",
            ) from exc

        sandbox_id = (
            f"{self._cloud.sandbox_prefix}-"
            f"{_safe_name(config.project_id)}-"
            f"{_safe_name(config.run_id)}"
        )
        self._active_jobs[sandbox_id] = []
        _log.info(
            "%s backend: sandbox created (blob_prefix=%s, image=%s, sandbox_id=%s)",
            self._cloud.provider.upper(),
            blob_prefix,
            image,
            sandbox_id,
        )
        return Sandbox(
            sandbox_id=sandbox_id,
            name=f"reprolab-{_safe_name(config.project_id)}-{_safe_name(config.run_id)}",
            image=image,
            config=config,
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        """Submit a short K8s Job running *command*, wait, capture logs, return ExecResult.

        The Job runs the base-image wrapper in exec mode (passes the command via
        the ``OPENRESEARCH_EXEC_COMMAND`` environment variable).  On infra failures
        raises ``SandboxRuntimeError(backend_unavailable, ...)``.
        """
        # Fail-closed per-run $ / pod-second ceiling before spending on a new Job.
        self._enforce_run_budget(sandbox)
        started_at = datetime.now(timezone.utc)
        ns = self._namespace()
        image = self._base_image()
        job_name = _job_name(sandbox.config.run_id, suffix="exec")

        env_vars = dict(sandbox.config.environment)
        env_vars["OPENRESEARCH_EXEC_COMMAND"] = command
        env_vars["OPENRESEARCH_EXEC_MODE"] = "1"

        # WS2 fail-loud guard (flag-gated, default OFF/byte-identical): this
        # path never stages project code into the pod (see
        # ``_command_needs_staged_code`` above) — on gcp, once the durable
        # controller is enabled, refuse a doomed code-dependent Job instead
        # of silently running it against an absent checkout. Local import —
        # ``run_controller`` imports only ``feature_flags`` (cycle-safe) but
        # this keeps the module's own import graph untouched when unused.
        from backend.agents.rlm import run_controller

        if (
            run_controller.durable_controller_enabled()
            and self._cloud.provider == "gcp"
            and _command_needs_staged_code(command)
        ):
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"{self._cloud.provider.upper()} backend: monolithic_exec_unstaged — the "
                f"k8s_job_backend.exec path does not stage project code into the pod; route "
                f"training via cells.json + train_cell.py (OPENRESEARCH_GKE_SYNTH_CELL). "
                f"Refusing to submit a code-less Job for: {command!r}"
            )

        gpu_count = self._gpu_plan_gpu_count()
        gpu_sku = self._gpu_plan_short_name()
        if gpu_sku is None and gpu_count >= 1:
            # P0-fix-3 parity (k8s_job_cell_runner._build_job_manifest): without this,
            # a GPU Job with no resolved GpuPlan gets NO nodeSelector at all, so on a
            # scale-to-zero cluster the autoscaler cannot bind it to the pool it just
            # scaled up (FailedScheduling/untolerated-taint) — thrashing the node up
            # then back down while billing the cloud. Fall back to the first
            # configured/provisioned SKU so the pod always targets a real pool; an
            # empty gpu_skus setting keeps today's no-nodeSelector/cluster-default
            # behavior (never invent a label).
            gpu_sku = self._default_gpu_sku()

        job_manifest = _build_job_manifest(
            job_name=job_name,
            namespace=ns,
            image=image,
            service_account=self._service_account(),
            command=command,
            environment=env_vars,
            active_deadline_seconds=timeout,
            ttl_seconds=self._ttl_after_finished(),
            backoff_limit=self._job_backoff_limit(),
            gpu_sku=gpu_sku,
            gpu_count=gpu_count,
            pod_template_extra_labels=self._cloud.pod_template_extra_labels,
            sandbox_label=self._cloud.sandbox_label,
        )

        batch_api = self._get_batch_api()

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: batch_api.create_namespaced_job(namespace=ns, body=job_manifest),
            )
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"{self._cloud.provider.upper()} backend: failed to submit exec Job {job_name!r}: {exc}",
            ) from exc

        # Track for cleanup in destroy().
        self._active_jobs.setdefault(sandbox.sandbox_id, []).append(job_name)

        # Poll until Job completes, times out, or Pending deadline exceeded.
        exit_code, stdout, stderr, timed_out, cause = await self._watch_job(
            job_name=job_name,
            namespace=ns,
            timeout=timeout,
        )

        finished_at = datetime.now(timezone.utc)
        duration = (finished_at - started_at).total_seconds()

        if timed_out:
            return ExecResult(
                command=command,
                exit_code=None,
                stdout=stdout,
                stderr=stderr or f"{self._cloud.provider.upper()} Job {job_name!r} timed out after {timeout}s.",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                timed_out=True,
                cause_kind=RuntimeCauseKind.exec_timeout,
            )

        return ExecResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            timed_out=False,
            cause_kind=cause if exit_code != 0 else None,
        )

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        """Download a file from the run-scoped object store prefix and return its bytes."""
        blob_key = _blob_artifact_key(sandbox.config.project_id, sandbox.config.run_id, path)
        try:
            store = self._get_store()
            data = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: store.download_bytes(blob_key),
            )
            return data
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"{self._cloud.provider.upper()} backend: copy_out failed for path {path!r} "
                f"(blob_key={blob_key!r}): {exc}",
            ) from exc

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        """Upload *data* to the run-scoped object store prefix at *path*."""
        blob_key = _blob_artifact_key(sandbox.config.project_id, sandbox.config.run_id, path)
        try:
            store = self._get_store()
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: store.upload_bytes(data, blob_name=blob_key),
            )
        except SandboxRuntimeError:
            raise
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.copy_failed,
                f"{self._cloud.provider.upper()} backend: copy_in failed for path {path!r} "
                f"(blob_key={blob_key!r}): {exc}",
            ) from exc

    async def destroy(self, sandbox: Sandbox) -> None:
        """Delete tracked active Jobs.  Blob artifacts are preserved intentionally."""
        ns = self._namespace()
        batch_api = self._get_batch_api()
        job_names = self._active_jobs.pop(sandbox.sandbox_id, [])
        for job_name in job_names:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda jn=job_name: self._delete_job_quietly(batch_api, ns, jn),
            )
        _log.info(
            "%s backend: destroy() cleaned up %d Jobs for sandbox %s (blob artifacts preserved).",
            self._cloud.provider.upper(),
            len(job_names),
            sandbox.sandbox_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upload_project_sync(
        self,
        project_root: Path,
        blob_prefix: str,
        config: SandboxConfig,
    ) -> None:
        """Synchronous: upload project_root to the object store under blob_prefix.

        Falls back to a best-effort warning when the object store module's optional
        SDK deps are absent (e.g. test env without cloud storage packages installed).
        """
        try:
            self._get_store().upload_prefix(project_root, blob_prefix=blob_prefix)
        except ImportError:
            _log.warning(
                "%s backend: object store module not available; skipping code upload (test/stub mode).",
                self._cloud.provider.upper(),
            )

    def _delete_job_quietly(self, batch_api: Any, namespace: str, job_name: str) -> None:
        """Best-effort synchronous Job deletion; logs failures, never raises.

        Uses a plain dict body so this method does not require the kubernetes
        package to be installed — injected test doubles receive the call directly.
        """
        try:
            # V1DeleteOptions as a plain dict is accepted by both the real
            # kubernetes client (which materialises it automatically) and fake
            # batch_api doubles used in tests.
            body = {"propagationPolicy": "Foreground"}
            batch_api.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=body,
            )
            _log.info(
                "%s backend: deleted Job %s/%s.",
                self._cloud.provider.upper(),
                namespace,
                job_name,
            )
        except Exception as exc:
            _log.warning(
                "%s backend: failed to delete Job %s/%s (ignored): %s",
                self._cloud.provider.upper(),
                namespace,
                job_name,
                exc,
            )

    async def _watch_job(
        self,
        job_name: str,
        namespace: str,
        timeout: int,
    ) -> tuple[int | None, str, str, bool, RuntimeCauseKind | None]:
        """Poll the Job until completion or deadline.

        Returns ``(exit_code, stdout, stderr, timed_out, cause_kind)``.
        On infra failure raises ``SandboxRuntimeError(backend_unavailable, ...)``.
        """
        batch_api = self._get_batch_api()
        core_api = self._get_core_api()
        ns = namespace
        deadline = time.monotonic() + timeout
        pending_start: float | None = None
        pending_timeout = self._pending_timeout()
        poll_interval = 5  # seconds

        while True:
            now = time.monotonic()
            if now >= deadline:
                # Timed out — best-effort log capture.
                stdout, stderr = await self._capture_job_logs(core_api, ns, job_name)
                return (None, stdout, stderr, True, RuntimeCauseKind.exec_timeout)

            # Poll job status (run in executor — sync SDK).
            try:
                job = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: batch_api.read_namespaced_job_status(
                        name=job_name, namespace=ns
                    ),
                )
            except Exception as exc:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"{self._cloud.provider.upper()} backend: failed to read Job status for {job_name!r}: {exc}",
                ) from exc

            status = job.status
            conditions = status.conditions or []
            condition_types = {c.type for c in conditions if c.status == "True"}

            if "Complete" in condition_types:
                # Job succeeded.
                stdout, stderr = await self._capture_job_logs(core_api, ns, job_name)
                return (0, stdout, stderr, False, None)

            if "Failed" in condition_types:
                # Job failed — retrieve logs and exit code heuristic.
                stdout, stderr = await self._capture_job_logs(core_api, ns, job_name)
                exit_code = _extract_exit_code(job) or 1
                return (exit_code, stdout, stderr, False, RuntimeCauseKind.command_failed)

            # Check for stuck-Pending (node pool not scaling → capacity_exhausted).
            pod_phase = await self._get_pod_phase(core_api, ns, job_name)
            if pod_phase == "Pending":
                if pending_start is None:
                    pending_start = now
                elif (now - pending_start) > pending_timeout:
                    _log.warning(
                        "%s backend: Job %s stuck Pending for >%ds → capacity_exhausted.",
                        self._cloud.provider.upper(),
                        job_name,
                        pending_timeout,
                    )
                    stdout, stderr = await self._capture_job_logs(core_api, ns, job_name)
                    err_msg = (
                        f"capacity_exhausted: Job {job_name!r} stuck Pending for "
                        f">{pending_timeout}s — node pool may be at quota."
                    )
                    return (1, stdout, err_msg, False, RuntimeCauseKind.backend_unavailable)
            else:
                # Pod is running (or gone); reset pending clock.
                pending_start = None

            remaining = deadline - now
            await asyncio.sleep(min(poll_interval, max(remaining, 0.1)))

    async def _capture_job_logs(
        self, core_api: Any, namespace: str, job_name: str
    ) -> tuple[str, str]:
        """Best-effort: retrieve stdout/stderr from the Job's pod logs."""
        try:
            pods = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: core_api.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"job-name={job_name}",
                ),
            )
            if not pods.items:
                return ("", "")
            pod_name = pods.items[0].metadata.name
            logs = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    _preload_content=True,
                ),
            )
            return (logs or "", "")
        except Exception as exc:
            _log.debug(
                "%s backend: log capture failed for Job %s: %s",
                self._cloud.provider.upper(),
                job_name,
                exc,
            )
            return ("", f"[log capture failed: {exc}]")

    async def _get_pod_phase(self, core_api: Any, namespace: str, job_name: str) -> str | None:
        """Return the phase of the first pod for *job_name*, or None on error."""
        try:
            pods = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: core_api.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"job-name={job_name}",
                ),
            )
            if not pods.items:
                return None
            return pods.items[0].status.phase
        except Exception:
            return None


__all__ = [
    "_KubernetesJobBackend",
    "_build_job_manifest",
    "_safe_name",
    "_job_name",
    "_blob_code_prefix",
    "_blob_artifact_key",
    "_settings_get",
    "_extract_exit_code",
    "_load_kubeconfig",
    "_load_kubernetes_batch_api",
    "_load_kubernetes_core_api",
    "_DEFAULT_NAMESPACE",
    "_DEFAULT_PENDING_TIMEOUT_S",
    "_DEFAULT_TTL_AFTER_FINISHED_S",
    "_DEFAULT_BACKOFF_LIMIT",
    "_DEFAULT_BASE_IMAGE",
    "_JOB_PREFIX",
    "AzureBlobStore",
    "GcsStore",
    "ObjectStore",
    "CloudSpec",
]
