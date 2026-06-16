"""leaf_actuator — close the leaf-repair loop (L4/L5/L6, 2026-06-16).

``leaf_triage`` DIAGNOSES each weak leaf into a cost-ordered repair plan; today
that plan is *advisory only* — a directive string appended to the implementer
prompt that the agent may or may not act on. On the real Adam run (0.764) it
often didn't: leaf ``fe5e7900`` shipped with its figure un-rendered and
``ac4006bf`` shipped with its failed cell un-re-run, each a clean ``0.0``.

This module is the OPTIONAL actuator that closes the loop. When
``REPROLAB_LEAF_ACTUATE`` is on it turns the cheapest, most deterministic repair
classes into concrete repair ARTIFACTS the EXISTING execution routes consume:

* **L4** — a ``result_quality`` leaf (inverted paper ordering ≈ an untuned
  per-condition lr) → a synthesized per-condition ``search`` section
  (``staged_search.synthesize_search_from_leaf``) the staged-search route runs.
* **L5** — a leaf demanding variance/CI over seeds → a budget-gated seed-count
  plan (``plan_seed_expansion``), behind the GPU-cost ``REPROLAB_LEAF_ACTUATE_SEEDS``
  sub-gate, with ``expand_cells_for_seeds`` the pure replication the route applies.
* **L6** — an ``aggregation_gap`` leaf → a declared-vs-aggregated completeness
  audit (``cell_matrix.audit_aggregation_completeness``) surfacing silently-lost
  cells.

Design discipline (matches ``staged_search``/``leaf_triage``):
  * **Pure cores** (``plan_seed_expansion`` / ``expand_cells_for_seeds`` /
    ``_wants_variance``) are stdlib-only and unit-tested against plain dicts.
  * **STAGE, don't execute mid-verify:** the actuator writes the repair artifact
    to ``rlm_state/leaf_actuation.json``; the existing routes pick it up on the
    NEXT iteration (the spec's ordering seam — repaired evidence then flows back
    through the operator's A1 median-of-N → A3 floor → A4 champion pipeline).
  * **Default-OFF, fail-soft:** ``REPROLAB_LEAF_ACTUATE`` unset == today
    byte-for-byte; every entry point swallows and falls back to the advisory
    directive, so enabling the flag can only ADD a repair attempt.

Spec: ``docs/superpowers/specs/2026-06-16-leaf-frontier-out-of-scope-remediation-design.md``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_FLAG = "REPROLAB_LEAF_ACTUATE"               # master, default OFF
MAX_COST_FLAG = "REPROLAB_LEAF_ACTUATE_MAX_COST"  # "none" | "targeted_rerun"
SEEDS_FLAG = "REPROLAB_LEAF_ACTUATE_SEEDS"        # L5 GPU-cost sub-gate, default OFF
SEED_MAX_FLAG = "REPROLAB_LEAF_SEED_MAX"          # hard ceiling on auto seeds
STATE_FILE = "leaf_actuation.json"

_DEFAULT_SEED_MAX = 5
_COST_ORDER = {"none": 0, "targeted_rerun": 1, "review": 2}

# A leaf demands multi-seed variance only when it names a SEED / error-bar /
# confidence signal — NOT merely the word "variance" (which appears in plenty of
# single-run justifications). Kept tight to avoid a false multi-seed expansion.
_VARIANCE_RE = re.compile(
    r"(single|only one|just one|one)\s+seed|"
    r"\bseeds?\b.{0,40}(mean|std|standard deviation|average|variance|error.?bar|"
    r"confidence|ci\b|spread|deviation)|"
    r"(mean|std|standard deviation|average|error.?bar|confidence interval|±|\+/-)"
    r".{0,40}\bseeds?\b|"
    r"(no|missing|without|lacks?)\s+(error.?bars?|confidence interval|std|variance)|"
    r"\bn\s*=\s*\d+\s+seeds?|over\s+\d+\s+seeds?|across\s+(multiple|\d+)\s+seeds?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Flag accessors
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Master gate — default OFF (opposite of leaf_triage; actuation is opt-in)."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


def max_cost() -> str:
    """Cost ceiling: only actuate plan entries at or below this cost.

    ``none`` (default) actuates the free repairs (L6 audit, render/aggregate);
    ``targeted_rerun`` additionally actuates the one-cell re-runs (L4 search).
    """
    v = os.environ.get(MAX_COST_FLAG, "none").strip().lower()
    return v if v in ("none", "targeted_rerun") else "none"


def seeds_enabled() -> bool:
    """L5 sub-gate — default OFF (the only GPU-cost actuator)."""
    return os.environ.get(SEEDS_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


def seed_max() -> int:
    try:
        v = int(float(os.environ.get(SEED_MAX_FLAG, "") or _DEFAULT_SEED_MAX))
        return max(1, v)
    except (TypeError, ValueError):
        return _DEFAULT_SEED_MAX


def _cost_allowed(cost: str) -> bool:
    return _COST_ORDER.get(cost, 9) <= _COST_ORDER.get(max_cost(), 0)


# ---------------------------------------------------------------------------
# L5 — budget-gated seed planning (pure)
# ---------------------------------------------------------------------------


@dataclass
class SeedPlan:
    """How many seeds the run can afford for a variance-demanding leaf."""

    current_seeds: int
    target_seeds: int       # min(paper_n, seed_max)
    affordable_seeds: int   # what the remaining budget actually permits
    fits: bool              # affordable_seeds >= target_seeds
    expand: bool            # affordable_seeds > current_seeds (worth doing)
    reason: str


def _wants_variance(text: str) -> bool:
    return bool(_VARIANCE_RE.search(text or ""))


def plan_seed_expansion(
    *,
    current_seeds: int,
    paper_n: int,
    seed_max: int,
    est_seconds_per_seed: float | None,
    remaining_s: float | None,
    reserve_s: float = 0.0,
) -> SeedPlan:
    """Compute the affordable seed count for a variance-demanding leaf — L5.

    ``target = min(paper_n, seed_max)``. The number of EXTRA seeds beyond
    ``current`` that fit the remaining wall-clock is
    ``floor((remaining - reserve) / est_seconds_per_seed)``; when the cost or
    remaining budget is unknown the planner is fail-soft and grants the full
    target (an un-estimable run is never blocked, mirroring ``staged_search``'s
    ``budget_feasible``). Never raises; pure arithmetic.

    ``expand`` is the actionable bit: ``True`` only when the budget permits MORE
    seeds than we already ran (no-op expansions are not proposed). ``fits``
    distinguishes "ran the paper's full N" from "ran as many as the budget
    allowed" so the caller can log the shortfall (no silent cap).
    """
    cur = max(1, int(current_seeds or 1))
    target = max(1, min(int(paper_n or 1), int(seed_max or 1)))
    if target <= cur:
        return SeedPlan(cur, target, cur, True, False, "already at or above target")

    if not est_seconds_per_seed or est_seconds_per_seed <= 0 or remaining_s is None:
        # Un-estimable cost → fail-soft grant of the full target.
        return SeedPlan(cur, target, target, True, True, "no_estimate: granted target")

    budget = max(0.0, float(remaining_s) - max(0.0, reserve_s))
    extra_affordable = int(math.floor(budget / float(est_seconds_per_seed)))
    affordable = cur + max(0, min(target - cur, extra_affordable))
    fits = affordable >= target
    expand = affordable > cur
    if not expand:
        reason = (
            f"budget too tight: ~{int(est_seconds_per_seed)}s/seed, "
            f"~{int(budget)}s remaining — no extra seed fits"
        )
    elif fits:
        reason = f"fits target {target} seeds"
    else:
        reason = (
            f"budget-capped at {affordable}/{target} seeds "
            f"(~{int(est_seconds_per_seed)}s/seed, ~{int(budget)}s remaining)"
        )
    return SeedPlan(cur, target, affordable, fits, expand, reason)


def expand_cells_for_seeds(cells: list[dict], n_seeds: int) -> list[dict]:
    """Replicate each cell across ``n_seeds`` distinct seeds — pure, L5.

    Each replica is a deep copy with a distinct ``seed`` written to BOTH the top
    level and ``["params"]`` (the ``_set_both`` shape ``train_cell.py`` may read
    either) and a seed-suffixed ``id`` so the cells stay distinct in the matrix.
    A cell that already carries a ``seed`` keeps it as the FIRST replica's value
    so a resume re-uses prior work. Returns the original list unchanged when
    ``n_seeds <= 1`` or input is unusable. Never raises.

    NOTE: this produces the replicated MATRIX; computing cross-seed mean±std from
    the replicas' results is the aggregation step the consuming route owns (and
    the documented GPU-validated follow-on) — this helper only fans the cells out.
    """
    try:
        if not isinstance(cells, list) or not cells or int(n_seeds) <= 1:
            return cells if isinstance(cells, list) else []
        out: list[dict] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            base_id = str(cell.get("id") or "cell")
            base_seed = cell.get("seed")
            params = cell.get("params") if isinstance(cell.get("params"), dict) else {}
            base_seed = base_seed if isinstance(base_seed, int) else params.get("seed")
            start = int(base_seed) if isinstance(base_seed, int) else 0
            for i in range(int(n_seeds)):
                seed = start + i
                rep = copy.deepcopy(cell)
                rep["id"] = f"{base_id}__seed{seed}"
                rep["seed"] = seed
                p = rep.get("params")
                if not isinstance(p, dict):
                    p = {}
                p["seed"] = seed
                rep["params"] = p
                out.append(rep)
        return out or cells
    except Exception:  # noqa: BLE001 — replication must never break the run.
        logger.debug("expand_cells_for_seeds failed", exc_info=True)
        return cells if isinstance(cells, list) else []


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------


def _read_cells(project_dir: Path) -> list[dict]:
    try:
        doc = json.loads((project_dir / "code" / "cells.json").read_text(encoding="utf-8"))
        cells = doc.get("cells") if isinstance(doc, dict) else None
        return [c for c in cells if isinstance(c, dict)] if isinstance(cells, list) else []
    except Exception:  # noqa: BLE001
        return []


def _read_metrics(project_dir: Path) -> dict:
    try:
        m = json.loads((project_dir / "code" / "metrics.json").read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def actuate(
    plan: list[dict],
    project_dir: Path | str,
    *,
    weak_leaves: list[dict] | None = None,
    lr_grid: "list[float] | None" = None,
    est_seconds_per_seed: float | None = None,
    remaining_s: float | None = None,
    paper_n_seeds: int = 5,
) -> dict[str, Any]:
    """Turn the leaf_triage plan into staged repair artifacts. Never raises.

    Returns ``{"actuated": [...classes], "artifact": {...}, "summary": str}`` and
    persists the artifact to ``rlm_state/leaf_actuation.json``. A no-op (and no
    file write) when the master flag is off or nothing is actionable, so an
    enabled flag with an empty plan is still byte-for-byte today.
    """
    result: dict[str, Any] = {"actuated": [], "artifact": {}, "summary": ""}
    if not is_enabled():
        return result
    try:
        project_dir = Path(project_dir)
        plan = [p for p in (plan or []) if isinstance(p, dict)]
        artifact: dict[str, Any] = {}
        actuated: list[str] = []
        cells = _read_cells(project_dir)

        # L4 — synthesize a per-condition lr search from a result_quality leaf.
        if cells and any(
            p.get("repair_class") == "result_quality" and _cost_allowed(p.get("cost", "review"))
            for p in plan
        ):
            from backend.agents.rlm import staged_search as _ss

            search = _ss.synthesize_search_from_leaf(cells, lr_grid=lr_grid)
            if search:
                artifact["search"] = search
                actuated.append("result_quality")

        # L6 — declared-vs-aggregated completeness audit when an aggregation_gap
        # leaf is present (cost: none → fires at the default ceiling).
        if any(p.get("repair_class") == "aggregation_gap" for p in plan):
            from backend.agents.rlm.cell_matrix import audit_aggregation_completeness

            audit = audit_aggregation_completeness(cells, _read_metrics(project_dir))
            if audit.get("failed") or audit.get("unaccounted"):
                artifact["aggregation_audit"] = audit
                actuated.append("aggregation_gap")

        # L5 — budget-gated seed plan for a variance-demanding leaf (GPU sub-gate).
        if seeds_enabled() and weak_leaves:
            wants = any(
                isinstance(l, dict)
                and float(l.get("score") or 0.0) < 0.6
                and _wants_variance(str(l.get("justification") or l.get("requirement") or ""))
                for l in weak_leaves
            )
            if wants:
                sp = plan_seed_expansion(
                    current_seeds=1,
                    paper_n=paper_n_seeds,
                    seed_max=seed_max(),
                    est_seconds_per_seed=est_seconds_per_seed,
                    remaining_s=remaining_s,
                )
                if sp.expand:
                    artifact["seed_plan"] = asdict(sp)
                    actuated.append("variance_gap")

        if artifact:
            _persist(project_dir, {"artifact": artifact, "actuated": actuated})
        result["artifact"] = artifact
        result["actuated"] = actuated
        bits = []
        if "search" in artifact:
            bits.append(f"{len(artifact['search'])} per-condition lr search group(s)")
        if "seed_plan" in artifact:
            sp = artifact["seed_plan"]
            bits.append(f"seed expansion → {sp['affordable_seeds']} seeds ({sp['reason']})")
        if "aggregation_audit" in artifact:
            a = artifact["aggregation_audit"]
            bits.append(
                f"aggregation audit: {len(a.get('failed', []))} failed, "
                f"{len(a.get('unaccounted', []))} unaccounted cell(s)"
            )
        result["summary"] = "leaf actuation staged: " + "; ".join(bits) if bits else ""
    except Exception:  # noqa: BLE001 — actuation is advisory; never blocks verify.
        logger.debug("leaf_actuator.actuate failed", exc_info=True)
    return result


def _persist(project_dir: Path, payload: dict[str, Any]) -> None:
    try:
        state_dir = Path(project_dir) / "rlm_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / STATE_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        logger.debug("leaf_actuator: persist failed", exc_info=True)


# ---------------------------------------------------------------------------
# Consumption readers (called by the existing routes; flag-guarded by the caller)
# ---------------------------------------------------------------------------


def staged_search_override(project_dir: Path | str) -> list[dict] | None:
    """The synthesized ``search`` section from the last actuation, or ``None``.

    The staged-search route consults this (only when ``is_enabled()``) as a
    SECOND fallback — after the paper-hint synthesis — so a result_quality leaf's
    tuning fires even when neither the agent nor a hint supplied a search block.
    """
    if not is_enabled():
        return None
    try:
        doc = json.loads(
            (Path(project_dir) / "rlm_state" / STATE_FILE).read_text(encoding="utf-8")
        )
        search = (doc.get("artifact") or {}).get("search")
        return search if isinstance(search, list) and search else None
    except Exception:  # noqa: BLE001
        return None


def seed_plan_for(project_dir: Path | str) -> dict | None:
    """The staged seed plan from the last actuation, or ``None`` (flag-guarded)."""
    if not (is_enabled() and seeds_enabled()):
        return None
    try:
        doc = json.loads(
            (Path(project_dir) / "rlm_state" / STATE_FILE).read_text(encoding="utf-8")
        )
        sp = (doc.get("artifact") or {}).get("seed_plan")
        return sp if isinstance(sp, dict) and sp.get("expand") else None
    except Exception:  # noqa: BLE001
        return None


def guidance_block(project_dir: Path | str, *, max_chars: int = 1200) -> str:
    """Compact implementer-prompt block for the staged actuation (empty when N/A).

    Symmetric with ``leaf_triage.guidance_block``. L4's search runs automatically
    (route-consumed), so it's surfaced as informational; L5's seed plan + L6's
    audit are surfaced as ACTIONABLE directives the agent acts on next iteration.
    Self-guards on the master flag (default-OFF == empty == today).
    """
    if not is_enabled():
        return ""
    try:
        doc = json.loads(
            (Path(project_dir) / "rlm_state" / STATE_FILE).read_text(encoding="utf-8")
        )
        art = doc.get("artifact") if isinstance(doc, dict) else None
        if not isinstance(art, dict) or not art:
            return ""
        lines = ["\n\nLEAF ACTUATION (harness-staged repairs for the last verify's weak leaves):"]
        if art.get("search"):
            lines.append(
                f"  [auto] {len(art['search'])} per-condition lr search group(s) synthesized — "
                "the staged-search route will tune each condition at ITS OWN best lr and re-run "
                "automatically; do not hand-tune."
            )
        sp = art.get("seed_plan")
        if isinstance(sp, dict) and sp.get("expand"):
            lines.append(
                f"  [seeds] run {sp.get('affordable_seeds')} seeds for the variance leaves and "
                f"report mean±std across them ({sp.get('reason')}) — a single seed cannot satisfy "
                "a 'mean±std over N seeds' leaf."
            )
        aud = art.get("aggregation_audit")
        if isinstance(aud, dict):
            failed = aud.get("failed") or []
            unacc = aud.get("unaccounted") or []
            if failed:
                lines.append(
                    f"  [rerun] {len(failed)} declared cell(s) ran but produced no result "
                    f"({', '.join(failed[:6])}) — fix the cell error and re-run; do NOT exclude."
                )
            if unacc:
                lines.append(
                    f"  [missing] {len(unacc)} declared cell(s) never reached the aggregate "
                    f"({', '.join(unacc[:6])}) — re-run them or record an explicit gap."
                )
        if len(lines) == 1:
            return ""
        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[: max_chars - 15].rstrip() + "\n  (truncated)"
        return block + "\n"
    except Exception:  # noqa: BLE001
        logger.debug("leaf_actuator: guidance block failed", exc_info=True)
        return ""


__all__ = [
    "ENV_FLAG",
    "SeedPlan",
    "actuate",
    "expand_cells_for_seeds",
    "guidance_block",
    "is_enabled",
    "max_cost",
    "plan_seed_expansion",
    "seed_plan_for",
    "seeds_enabled",
    "staged_search_override",
]
