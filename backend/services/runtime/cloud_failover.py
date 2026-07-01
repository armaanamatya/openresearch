"""GCP<->Azure failover selector for cloud execution backends (Phase 1c, Unit A).

Provision-time failover only (v1): each cloud's ``ensure_*_available`` probe is
tried in ``preference`` order and the first one that passes wins; a cloud whose
probe raises a "cloud is down" signal (``SandboxRuntimeError`` whose
``cause_kind`` is ``backend_unavailable``/``capacity_exhausted``) is skipped in
favor of the next. Mid-run failover -- e.g. GCP dies partway through a training
job that has already started -- is explicitly OUT OF SCOPE for this module; a
live run that loses its cloud mid-flight is a concern for the provider/
run-lifecycle layer, not this selector.

Hermetic by construction: ``availability`` and ``backend_factory`` are
injectable, so every test in this module exercises pure control flow and never
touches a real cloud SDK or credential. The *default* (real) values
lazy-import ``ensure_gcp_available``/``ensure_azure_available`` and
``backend.agents.rlm.primitives._backend_for_sandbox_mode`` from inside the
function bodies -- never at module import time -- so importing this module
never pulls in the (heavy) cloud SDKs or the primitives module, and never
risks an import cycle with either.

Standalone in this unit: nothing in the live run path calls this module yet.
``failover_preference()`` reads ``OPENRESEARCH_CLOUD_FAILOVER`` and returns
``[]`` when unset/empty -- wiring a caller to treat an empty preference as
"skip failover, use today's single-cloud selection" is a later step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError

# SandboxRuntimeError.cause_kind values that mean "this cloud is down/
# unreachable right now, try the next one" rather than "a real bug". Compared
# by string VALUE (not enum identity/attribute access) so this stays correct
# even though "capacity_exhausted" is not currently a RuntimeCauseKind member
# -- referencing RuntimeCauseKind.capacity_exhausted directly would raise
# AttributeError today; comparing strings lets a future cause_kind member (or
# an injected test fake) participate without a code change here.
_CLOUD_DOWN_KINDS = frozenset({"backend_unavailable", "capacity_exhausted"})


def _cause_kind_str(cause_kind: object) -> str:
    """Normalize a SandboxRuntimeError.cause_kind (enum member or plain str) to str."""
    value = getattr(cause_kind, "value", None)
    return value if isinstance(value, str) else str(cause_kind)


@dataclass(frozen=True)
class BackendSelection:
    """The winning backend from a failover attempt, plus the full attempt log."""

    backend: object  # the constructed RuntimeBackend
    cloud: str  # "gcp" | "azure" -- the cloud that won
    attempts: tuple[tuple[str, str], ...]  # ((cloud, "ok"|"<err>"), ...) in order tried


def failover_preference() -> list[str]:
    """Parse ``OPENRESEARCH_CLOUD_FAILOVER`` (comma list, e.g. ``"gcp,azure"``).

    Returns ``[]`` when unset or empty -- the byte-identical-off state.
    """
    raw = os.environ.get("OPENRESEARCH_CLOUD_FAILOVER", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _real_availability_map() -> "dict[str, Callable[[], None]]":
    """The production ``ensure_*_available`` checks, lazy-imported per cloud."""
    from backend.services.runtime.aks_job_backend import ensure_azure_available
    from backend.services.runtime.gke_job_backend import ensure_gcp_available

    return {"gcp": ensure_gcp_available, "azure": ensure_azure_available}


def _real_backend_factory(
    cloud: str, *, run_budget: object = None, gpu_plan: object = None
) -> object:
    """Build the real ``RuntimeBackend`` for one cloud's sandbox mode, lazy-imported."""
    from backend.agents.execution import SandboxMode
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode

    return _backend_for_sandbox_mode(
        SandboxMode(cloud), run_budget=run_budget, gpu_plan=gpu_plan
    )


def select_backend_with_failover(
    preference: "list[str]",
    *,
    run_budget: object = None,
    gpu_plan: object = None,
    availability: "dict[str, Callable[[], None]] | None" = None,
    backend_factory: "Callable[..., object] | None" = None,
) -> BackendSelection:
    """Try each cloud in ``preference`` order; return the first that's up.

    For each cloud, call its ``availability`` probe (default: the real
    ``ensure_gcp_available``/``ensure_azure_available``, lazy-imported). A
    probe that raises ``SandboxRuntimeError`` with a "cloud down" cause_kind
    (see ``_CLOUD_DOWN_KINDS``) is recorded and the next cloud is tried. ANY
    OTHER exception -- a ``SandboxRuntimeError`` with a different cause_kind,
    or a wholly different exception type -- propagates immediately: a real bug
    must never be silently treated as "cloud down".

    Once a cloud's probe passes, its backend is built via ``backend_factory``
    (default: ``primitives._backend_for_sandbox_mode``, lazy-imported) and
    returned inside a ``BackendSelection``.

    Raises ``SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, ...)``
    (message lists the per-cloud reasons) only when every cloud in
    ``preference`` failed the "down" way (or ``preference`` is empty).
    """
    resolved_availability = (
        availability if availability is not None else _real_availability_map()
    )
    resolved_factory = backend_factory if backend_factory is not None else _real_backend_factory

    attempts: list[tuple[str, str]] = []
    for cloud in preference:
        probe = resolved_availability.get(cloud)
        try:
            if probe is not None:
                probe()
        except SandboxRuntimeError as exc:
            kind = _cause_kind_str(exc.cause_kind)
            if kind not in _CLOUD_DOWN_KINDS:
                raise
            attempts.append((cloud, f"{kind}: {exc}"))
            continue

        backend = resolved_factory(cloud, run_budget=run_budget, gpu_plan=gpu_plan)
        attempts.append((cloud, "ok"))
        return BackendSelection(backend=backend, cloud=cloud, attempts=tuple(attempts))

    if attempts:
        joined = "; ".join(f"{cloud}: {reason}" for cloud, reason in attempts)
        message = f"all clouds in failover preference are unavailable: {joined}"
    else:
        message = "no cloud backend available: failover preference is empty"
    raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, message)


__all__ = ["BackendSelection", "failover_preference", "select_backend_with_failover"]
