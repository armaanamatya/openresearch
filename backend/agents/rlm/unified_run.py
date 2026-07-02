"""Opt-in composition root wiring the Phase-1a..1e pieces into a runnable run (Phase 1f).

``backend/agents/rlm/reproduction_run.py`` (Phase 1c), ``compute_provider.py``/
``vm_compute_provider.py``/``cloud_profile.py`` (Phase 1c/1d), ``run_plan.py``
(Phase 1b), and ``feasibility_triage.py`` (Phase 1b) are each independently
built and unit-tested, but nothing on disk yet assembles them into one
constructible thing. This module is that assembler -- and ONLY that: it has
no side effects of its own (constructing a ``VmComputeProvider`` builds no
argv and makes no gcloud/ssh call; those only happen once the returned
``ReproductionRun.run()`` is actually invoked), and it is not called from
anywhere in the live run path (``backend/agents/rlm/run.py``, ``cli.py``).
Proving the whole stack composes into a working state machine is the
allocated Phase-1f scope; wiring it into the live path is a later, separate
step, exactly as ``reproduction_run.py``'s own module docstring anticipates
("Live-cloud wiring is a later, explicitly opt-in step
(``OPENRESEARCH_UNIFIED_RUN``)").

``unified_run_enabled()`` reads that env var so a future live-path caller has
a single canonical flag to gate on; today nothing reads its return value
except this module's own tests.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

from backend.agents.rlm.reproduction_run import ReproductionRun
from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
from backend.services.runtime.feasibility_triage import FeasibilityTriage
from backend.services.runtime.gpu_catalog import find_by_alias
from backend.services.runtime.run_plan import RunPlan, extract_required_assets
from backend.services.runtime.vm_compute_provider import VmComputeProvider

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import coupling at runtime
    from pathlib import Path

    from backend.agents.resilience.budget import RunBudget
    from backend.agents.schemas import ScopeSpec
    from backend.services.runtime.compute_provider import ComputeLease, ComputeProvider
    from backend.services.runtime.gpu_catalog import GpuSku
    from backend.services.runtime.vm_compute_provider import VmExecResult

__all__ = ["build_reproduction_run", "unified_run_enabled"]


def unified_run_enabled() -> bool:
    """Whether the opt-in unified composition-root run path is enabled.

    Reads ``OPENRESEARCH_UNIFIED_RUN`` from the environment on every call (no
    import-time capture, so tests can monkeypatch per-case); truthy values are
    ``"1"``/``"true"``/``"yes"`` (case-insensitive). Default OFF -- unset means
    no live code path consults this yet; ``build_reproduction_run`` itself
    does not read this flag either, since assembly and gating are separate
    concerns.
    """
    return os.environ.get("OPENRESEARCH_UNIFIED_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def build_reproduction_run(
    *,
    paper_id: str,
    state_dir: "Path",
    scope: "ScopeSpec | None" = None,
    budget: "RunBudget | None" = None,
    cloud: str = "gcp",
    vm_spec: "VmSpec | None" = None,
    sku: "GpuSku | None" = None,
    provider: "ComputeProvider | None" = None,
    triage: "FeasibilityTriage | None" = None,
    green_gate: "Callable[[ComputeLease], bool] | None" = None,
    contract: object = None,
    claim_map: object = None,
    rubric: object = None,
    runner: "Callable[[list[str]], VmExecResult] | None" = None,
) -> ReproductionRun:
    """Assemble the Phase-1a..1e pieces into one constructible ``ReproductionRun``.

    Pure assembler: every collaborator ``ReproductionRun`` needs is either
    the caller-supplied override or a real-but-inert default built here, so
    a caller with no injected fakes still gets a working (if live-cloud-
    capable once run) state machine. Callers that need hermetic behaviour
    (tests, dry runs) supply ``provider``/``triage``/``sku`` explicitly.

    ``required_assets`` is derived from whichever of ``contract``/
    ``claim_map``/``rubric`` is given via ``extract_required_assets`` --
    already fail-soft (returns ``[]`` on bad/absent input or internal error),
    so this function never needs its own try/except around that call.
    """
    required_assets = extract_required_assets(contract=contract, claim_map=claim_map, rubric=rubric)
    plan = RunPlan(
        paper_id=paper_id,
        scope=scope,
        budget=budget,
        required_assets=tuple(required_assets),
    )
    resolved_provider = provider or VmComputeProvider(
        CloudProfile(cloud=cloud, vm=vm_spec or VmSpec()), runner=runner
    )
    resolved_triage = triage or FeasibilityTriage()
    resolved_sku = sku or find_by_alias("rtx4090")
    return ReproductionRun(
        plan=plan,
        provider=resolved_provider,
        triage=resolved_triage,
        sku=resolved_sku,
        state_dir=state_dir,
        green_gate=green_gate,
    )
