"""GPU SKU <-> node-pool preflight — fail LOUD on config/infra drift, before money moves.

The hazard this guards against: cloud K8s cell Jobs (GKE/AKS) are scheduled onto
scale-to-zero GPU node pools via a ``nodeSelector={"reprolab/sku": <short_name>}``
(``k8s_job_backend.py``, ``k8s_job_cell_runner.py``). The set of SKUs the resolver
is allowed to pick from comes from config (``Settings.gcp_gpu_skus`` /
``OPENRESEARCH_GCP_GPU_SKUS``); Terraform provisions the actual node pools
(``infra/gcp/variables.tf``'s ``gpu_skus`` variable). These two CAN drift — a
``.env`` override or a stale config value can name a SKU no node pool actually
provides. When that happens, a cell resolving to the drifted SKU submits a Job
whose nodeSelector matches NO node in the cluster: the pod sits Pending, the
autoscaler cannot help (there is nothing to scale), and the run silently burns
the pending-timeout (~15-25 min, see ``gcp_pending_timeout_seconds``) before
failing as ``capacity_exhausted`` — slow, expensive, and easy to miss because
the cost ledger shows ~$0.

Two evidence sources, in strict precedence order
-------------------------------------------------
**Tier 1 — the GKE node-pool API (AUTHORITATIVE).** A node pool *exists at zero
nodes*: every pool here is scale-to-zero by design
(``infra/gcp/modules/gpu_nodepool/main.tf``), so the cluster is COLD most of the
time — and the first run of the day, when drift bites hardest, is exactly when a
Node-listing check goes blind. The GKE control-plane API
(``ClusterManagerClient.list_node_pools``, or the equivalent REST call) returns
every *configured* pool regardless of its current node count, so it answers
definitively even on a fully cold cluster. Pools are matched to SKUs by reading
the SAME ``reprolab/sku`` label the nodeSelector uses, off each pool's
``config.labels`` — ONE label scheme, not two.

**Tier 2 — live Node observation (HEURISTIC, fallback).** ``core_api.list_node()``
sees only pools that currently have >=1 node. Used when tier 1 is unavailable
(no GKE-API IAM permission, SDK/credentials absent, network error, or
project/region/cluster unset). Strong right after GPU activity; blind on a cold
cluster.

**Tier 3 — cannot_verify.** Warn loudly, proceed. Never a hard block.

A tier-1 failure NEVER hard-blocks and never errors the run: it degrades to
tier 2, then to tier 3.

The corroboration invariant (bounds the blast radius)
-----------------------------------------------------
At BOTH evidence tiers, a configured SKU is hard-blocked ONLY when the
``reprolab/sku`` label scheme is demonstrably observable — i.e. at least one
OTHER SKU was seen carrying that label — and the configured SKU is not among
them (``status="verified_absent"``). If we observe the cluster but see the label
on *nothing at all*, that is ``cannot_verify``, not "every configured SKU is
absent".

This is deliberate, and it is what makes the guard safe to run in front of every
GCP run:

* At tier 2 it is *required for correctness*: an idle scale-to-zero cluster has
  ZERO Node objects, so "no labels seen" is the normal between-runs state, and
  treating it as absence would hard-block every run whenever the cluster is idle
  — a constant false alarm that trains operators to bypass the guard.
* At tier 1 it is a *blast-radius bound*: if a response-shape change ever broke
  label parsing, the failure mode is a loud "could not verify" warning rather
  than a spurious hard block of every GCP run. The
  cluster-has-node-pools-but-none-is-labelled case still surfaces loudly (naming
  the pools actually seen), it just does not gate the run on a signal we could
  not corroborate.

IAM (operator prerequisite for tier 1)
---------------------------------------
The GKE node-pool API needs ``container.clusters.get`` on the target cluster —
GCP IAM, NOT the Kubernetes RBAC Role. ``roles/container.clusterViewer`` grants
it. A local operator authed via ``gcloud auth application-default login``
normally has this through their project role; the in-cluster orchestrator GSA
(``infra/gcp/modules/identity/main.tf``) currently does NOT — it holds only
``storage.objectAdmin``, ``secretmanager.secretAccessor``, and
``iam.workloadIdentityUser``. Without the permission the API returns 403 and this
module degrades to tier 2 (a warning), never an error.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError

_log = logging.getLogger(__name__)

_SKU_LABEL = "reprolab/sku"

# Sentinel: "build the tier-1 pool lister from settings yourself" — distinct from
# an explicitly-injected lister (tests) and from None (tier 1 disabled).
_AUTO = object()

__all__ = [
    "GpuPoolPreflightResult",
    "check_gpu_pools_provisioned",
    "enforce_gpu_pool_preflight",
    "list_gke_node_pools",
    "make_gke_pool_lister",
]


@dataclass(frozen=True)
class GpuPoolPreflightResult:
    """Outcome of one :func:`check_gpu_pools_provisioned` call.

    ``status``:
        ``"ok"``              — every configured SKU was corroborated, or there
                                 was nothing configured to check.
        ``"cannot_verify"``   — no tier produced a corroborated observation (API
                                 unreachable/denied AND no live node carries the
                                 label). Never a reason to block a run.
        ``"verified_absent"`` — the label scheme WAS observed, and ``missing_skus``
                                 names the configured SKU(s) with no matching pool.

    ``source``: which evidence tier produced the verdict — ``"node_pool_api"``
    (authoritative), ``"live_nodes"`` (heuristic), or ``"none"``.
    """

    status: str
    configured_skus: tuple[str, ...] = ()
    observed_skus: frozenset[str] = field(default_factory=frozenset)
    missing_skus: tuple[str, ...] = ()
    reason: str = ""
    source: str = "none"


# ---------------------------------------------------------------------------
# Tier 1 — GKE node-pool API (authoritative; sees a pool at ZERO nodes)
# ---------------------------------------------------------------------------


def _settings_attr(settings: Any, name: str) -> str:
    try:
        return str(getattr(settings, name, "") or "").strip()
    except Exception:  # noqa: BLE001 — a hostile settings double must not break preflight
        return ""


def _pool_labels(pool: Any) -> dict[str, Any]:
    """Label dict for one node pool — typed client object OR raw REST JSON.

    ``container_v1.NodePool`` exposes ``.config.labels``; the REST payload is
    ``{"config": {"labels": {...}}}``. Any unexpected shape yields ``{}``, which
    under the corroboration invariant degrades to cannot_verify — never to a
    false "absent".
    """
    if isinstance(pool, dict):
        return (pool.get("config") or {}).get("labels") or {}
    config = getattr(pool, "config", None)
    if config is not None:
        return dict(getattr(config, "labels", None) or {})
    return {}


def _pool_name(pool: Any) -> str:
    if isinstance(pool, dict):
        return str(pool.get("name") or "?")
    return str(getattr(pool, "name", None) or "?")


def _normalize_pools(pools: Any) -> list[dict[str, Any]]:
    """Normalize any tier-1 response to ``[{"name": str, "labels": {...}}, ...]``."""
    return [{"name": _pool_name(p), "labels": _pool_labels(p)} for p in pools or []]


def list_gke_node_pools(*, project: str, location: str, cluster: str) -> list[dict[str, Any]]:
    """Return EVERY node pool configured on the target GKE cluster — nodes or not.

    Two transports, tried in order. Both authenticate via Application Default
    Credentials — the same ADC the rest of the GCP path already relies on
    (``gcloud auth application-default login`` locally, Workload Identity
    in-cluster):

    1. ``google.cloud.container_v1.ClusterManagerClient`` — the typed GKE client,
       used when the optional ``google-cloud-container`` package is installed.
    2. The GKE REST API via ``google.auth`` + ``AuthorizedSession`` — needs NO
       package beyond what the repo ALREADY depends on (``google-auth`` ships with
       the required ``google-cloud-storage``). This keeps the authoritative tier
       live TODAY instead of inert behind an uninstalled optional dependency.

    Raises on any failure (SDK absent, no credentials, HTTP 403 from a missing IAM
    permission, network error). Callers convert that to ``cannot_verify`` and fall
    back to the live-Node heuristic — a tier-1 failure must NEVER hard-block or
    error a run.

    IAM: needs ``container.clusters.get`` on the cluster (e.g.
    ``roles/container.clusterViewer``) — GCP IAM, not Kubernetes RBAC.
    """
    # Transport 1 — typed client (optional dependency).
    try:
        from google.cloud import container_v1  # type: ignore[import]
    except Exception:  # noqa: BLE001 — not installed => fall through to REST
        pass
    else:
        client = container_v1.ClusterManagerClient()
        parent = f"projects/{project}/locations/{location}/clusters/{cluster}"
        response = client.list_node_pools(parent=parent)
        return _normalize_pools(getattr(response, "node_pools", None))

    # Transport 2 — REST via ADC (no dependency beyond google-auth).
    import google.auth  # type: ignore[import]
    from google.auth.transport.requests import AuthorizedSession  # type: ignore[import]

    credentials, _project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)
    url = (
        "https://container.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/clusters/{cluster}/nodePools"
    )
    response = session.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"GKE nodePools API returned HTTP {response.status_code}: {response.text[:300]}"
        )
    return _normalize_pools(response.json().get("nodePools"))


def make_gke_pool_lister(settings: Any) -> Callable[[], list[dict[str, Any]]] | None:
    """Build the tier-1 pool lister from settings, or None when it can't be addressed.

    Returns ``None`` (tier 1 unavailable -> heuristic only) when ``gcp_project``,
    ``gcp_region``, or ``gcp_gke_cluster`` is unset: we cannot name the cluster, so
    we cannot ask the authoritative question. A silent, correct degradation — not
    an error.
    """
    project = _settings_attr(settings, "gcp_project")
    location = _settings_attr(settings, "gcp_region")
    cluster = _settings_attr(settings, "gcp_gke_cluster")
    if not (project and location and cluster):
        return None

    def _lister() -> list[dict[str, Any]]:
        return list_gke_node_pools(project=project, location=location, cluster=cluster)

    return _lister


def _safe_get_settings() -> Any:
    try:
        from backend.config import get_settings

        return get_settings()
    except Exception:  # noqa: BLE001 — settings unavailable => tier 1 off, not an error
        return None


# ---------------------------------------------------------------------------
# Tier 2 — live Node observation (heuristic; blind to a scaled-to-zero pool)
# ---------------------------------------------------------------------------


def _node_labels(node: Any) -> dict[str, Any]:
    """Best-effort label dict for one node, real client object or plain-dict fake."""
    metadata = getattr(node, "metadata", None)
    if metadata is not None:
        labels = getattr(metadata, "labels", None)
        return labels or {}
    if isinstance(node, dict):
        return (node.get("metadata") or {}).get("labels") or {}
    return {}


def _observed_skus_from_node_list(node_list: Any) -> frozenset[str]:
    """Extract the ``reprolab/sku`` label VALUES present on any live node.

    Accepts the real ``kubernetes`` client's ``V1NodeList`` (``.items``) or a plain
    ``{"items": [...]}`` dict double. Any unexpected shape degrades to an empty set
    rather than raising — an empty set means "cannot_verify", never "every SKU
    confirmed absent" (the corroboration invariant).

    Dict-shaped fakes are checked FIRST: ``dict`` objects have their own built-in
    ``.items()`` METHOD, so ``getattr(a_dict, "items", None)`` would return that
    bound method (truthy, not the node list) rather than falling through.
    """
    if isinstance(node_list, dict):
        items = node_list.get("items")
    else:
        items = getattr(node_list, "items", None)
    observed: set[str] = set()
    for node in items or []:
        sku = _node_labels(node).get(_SKU_LABEL)
        if sku:
            observed.add(str(sku))
    return frozenset(observed)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify(
    configured: tuple[str, ...],
    observed: frozenset[str],
    source: str,
) -> GpuPoolPreflightResult:
    """Apply the corroboration invariant to ONE tier's (non-empty) observation."""
    missing = tuple(s for s in configured if s not in observed)
    if missing:
        return GpuPoolPreflightResult(
            status="verified_absent",
            configured_skus=configured,
            observed_skus=observed,
            missing_skus=missing,
            source=source,
        )
    return GpuPoolPreflightResult(
        status="ok",
        configured_skus=configured,
        observed_skus=observed,
        source=source,
    )


def check_gpu_pools_provisioned(
    configured_skus: Sequence[str],
    *,
    core_api: Any,
    pool_lister: Callable[[], list[dict[str, Any]]] | None = None,
) -> GpuPoolPreflightResult:
    """Classify each configured SKU against the cluster. Never raises.

    Evidence precedence: ``pool_lister`` (tier 1, authoritative — sees pools at
    zero nodes) -> ``core_api.list_node()`` (tier 2, heuristic) -> cannot_verify.
    At most ONE query per tier, and tier 2 is skipped entirely once tier 1 answers
    conclusively — so a run makes one cluster query, never one per cell or per SKU.

    Every failure mode (SDK absent, credentials missing, HTTP 403 from a missing
    IAM permission, network error, malformed response) degrades to the next tier
    and ultimately to ``cannot_verify``; :func:`enforce_gpu_pool_preflight` owns
    the policy.
    """
    configured = tuple(dict.fromkeys(str(s) for s in configured_skus if s))
    if not configured:
        return GpuPoolPreflightResult(status="ok")

    reasons: list[str] = []

    # --- Tier 1: the GKE node-pool API (authoritative, cold-cluster-proof).
    if pool_lister is not None:
        try:
            pools = _normalize_pools(pool_lister())
        except Exception as exc:  # noqa: BLE001 — 403/network/SDK => next tier, never a block
            reasons.append(f"node-pool API unavailable ({type(exc).__name__}: {exc})")
        else:
            observed = frozenset(
                str(p["labels"][_SKU_LABEL]) for p in pools if p["labels"].get(_SKU_LABEL)
            )
            if observed:
                return _classify(configured, observed, source="node_pool_api")
            if pools:
                # Pools exist but not one carries reprolab/sku. Either no GPU pool is
                # provisioned at all, or we are pointed at the wrong cluster, or the
                # label scheme changed. Loud — but NOT corroborated, so it does not
                # gate the run (see the corroboration invariant).
                reasons.append(
                    f"node-pool API listed {[p['name'] for p in pools]} but NOT ONE "
                    f"carries a {_SKU_LABEL!r} label (no GPU pool provisioned, or "
                    "OPENRESEARCH_GCP_GKE_CLUSTER points at the wrong cluster)"
                )
            else:
                reasons.append("node-pool API returned no node pools for the target cluster")

    # --- Tier 2: live Node observation (blind to a scaled-to-zero pool).
    try:
        observed = _observed_skus_from_node_list(core_api.list_node())
    except Exception as exc:  # noqa: BLE001 — any transport/auth/shape error => cannot verify
        reasons.append(f"live-node query failed ({type(exc).__name__}: {exc})")
    else:
        if observed:
            return _classify(configured, observed, source="live_nodes")
        reasons.append(
            f"no live node carries a {_SKU_LABEL!r} label "
            "(all GPU pools may be scaled to zero right now)"
        )

    return GpuPoolPreflightResult(
        status="cannot_verify",
        configured_skus=configured,
        reason="; ".join(reasons),
        source="none",
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def enforce_gpu_pool_preflight(
    configured_skus: Sequence[str],
    *,
    core_api: Any,
    provider_label: str = "GCP",
    settings_var_name: str = "OPENRESEARCH_GCP_GPU_SKUS",
    pool_lister: Any = _AUTO,
    settings: Any = None,
) -> GpuPoolPreflightResult:
    """Run :func:`check_gpu_pools_provisioned` and FAIL LOUD on a verified-absent SKU.

    - ``verified_absent`` -> raises ``SandboxRuntimeError(backend_unavailable, ...)``
      naming the exact drifting SKU(s) and both fixes (provision in Terraform, or
      drop the SKU from settings). Call this BEFORE any Job is submitted.
    - ``cannot_verify``   -> logs a loud warning and RETURNS. A transient API blip,
      a missing IAM permission, or an idle cluster we could not corroborate must
      never become a hard run-blocker.
    - ``ok``              -> returns quietly (debug log only).

    ``pool_lister`` defaults to ``_AUTO``: the authoritative tier-1 lister is built
    from ``settings`` (or ``get_settings()``). Pass an explicit callable to inject
    one, or ``None`` to disable tier 1 (heuristic only).
    """
    if pool_lister is _AUTO:
        source_settings = settings if settings is not None else _safe_get_settings()
        try:
            pool_lister = make_gke_pool_lister(source_settings)
        except Exception:  # noqa: BLE001 — lister construction must never break a run
            pool_lister = None

    result = check_gpu_pools_provisioned(
        configured_skus, core_api=core_api, pool_lister=pool_lister
    )

    if result.status == "verified_absent":
        missing_sorted = sorted(result.missing_skus)
        observed_sorted = sorted(result.observed_skus) or ["none"]
        tf_path = (
            "infra/gcp/variables.tf `gpu_skus`"
            if provider_label.strip().upper() == "GCP"
            else "infra/azure/variables.tf `gpu_skus` (or infra/gcp/variables.tf, if GCP)"
        )
        evidence = (
            "the cluster's GKE node-pool API (authoritative — it lists a pool even at "
            "zero nodes)"
            if result.source == "node_pool_api"
            else "the cluster's live GPU nodes"
        )
        message = (
            f"{provider_label} GPU SKU/node-pool DRIFT detected -- refusing to start "
            f"(no money spent yet). {settings_var_name} configures {missing_sorted}, but "
            f"according to {evidence} the target cluster has NO node pool carrying a "
            f"matching {_SKU_LABEL} label (pools actually provisioned: {observed_sorted}). "
            f"A cell that resolves to {missing_sorted} would submit a Kubernetes Job whose "
            f"nodeSelector matches NO node pool: the pod sits Pending, the autoscaler cannot "
            f"help, and the run dies to capacity_exhausted after the pending timeout "
            f"(~15-25 min wasted, near-$0 apparent spend in the cost ledger). "
            f"Fix ONE of: (1) provision {missing_sorted} as a node pool in Terraform "
            f"({tf_path}) and apply, or "
            f"(2) remove {missing_sorted} from {settings_var_name} so the resolver never "
            f"targets a pool that doesn't exist."
        )
        _log.error(message)
        raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, message)

    if result.status == "cannot_verify":
        _log.warning(
            "%s GPU SKU/node-pool preflight could NOT verify configured SKUs %s against "
            "the cluster (%s) -- proceeding WITHOUT this guard. The authoritative check "
            "needs container.clusters.get (e.g. roles/container.clusterViewer) on the "
            "caller's GCP identity, plus OPENRESEARCH_GCP_{PROJECT,REGION,GKE_CLUSTER}; "
            "without it a genuinely-missing pool is still caught, just late and "
            "expensively (capacity_exhausted after the pending timeout). Verify manually "
            "with `scripts/gke_check.sh`.",
            provider_label,
            sorted(result.configured_skus),
            result.reason,
        )
        return result

    _log.debug(
        "%s GPU SKU/node-pool preflight: all %d configured SKU(s) verified via %s.",
        provider_label,
        len(result.configured_skus),
        result.source,
    )
    return result
