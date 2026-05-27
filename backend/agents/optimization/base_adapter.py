"""Base protocol every optimization-lane adapter implements.

GEPA Lane B uses ``ReproLabGEPAAdapter`` (in gepa_adapter.py). A future
SkillOpt Lane E would add ``ReproLabSkillOptAdapter``. Both implement
this protocol so the unified CLI in ``scripts/optimize.py`` can dispatch
either without per-lane casing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from backend.agents.optimization.cost_tracker import LMCostTracker


@runtime_checkable
class BaseOptimizationAdapter(Protocol):
    """Minimum surface area shared across optimization lanes.

    ``surface_id`` distinguishes adapters that target different sets of
    mutable components (e.g. GEPA's ``improvement`` vs ``root_system``
    vs ``baseline_agent``). ``cost_tracker`` is the shared accumulator
    so callers can read final spend without lane-specific bookkeeping.

    ``evaluate`` mirrors the GEPA adapter contract — it returns
    whatever the lane's optimizer expects (e.g. ``gepa.EvaluationBatch``).
    The protocol is intentionally loose on return type because each
    optimization library defines its own batch shape.
    """

    surface_id: str
    cost_tracker: LMCostTracker

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any: ...


def opt_run_dir(lane: str, name: str | None = None) -> Path:
    """Standardized output directory for any optimization run.

    Layout:
        optimization_runs/<lane>/<name>/
            cost.json                  # LMCostTracker
            ... lane-specific artifacts ...
    """
    from datetime import datetime, timezone

    if name is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = Path("optimization_runs") / lane / name
    p.mkdir(parents=True, exist_ok=True)
    return p
