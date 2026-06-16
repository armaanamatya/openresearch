"""staged_search — harness-owned two-phase tuning for the cells route.

The agent declares an optional ``search`` section in ``code/cells.json`` (a
bounded candidate grid + a select rule + a promote template per group); the
HARNESS deterministically (1) runs the candidate phase, (2) selects each group's
winner by the declared metric, (3) budget-preflights the full phase against the
remaining wall-clock, and (4) materializes exactly ONE full cell per group at the
winner's tuned params, then runs it.

This converts "tune-then-run" from unenforceable prose into harness-GUARANTEED
behavior (2026-06-14 Codex review of the Adam score plan): the LLM never owns
phase ordering, winner propagation, or the budget check — those are deterministic
Python here. It is paper-agnostic: ANY comparative paper that needs per-condition
hyperparameter tuning emits a ``search`` section and gets the same enforced
protocol.

Design:
  * **Pure core** (``parse_search_spec`` / ``select_winner`` /
    ``materialize_full_cells`` / ``estimate_full_seconds`` / ``budget_feasible``)
    is stdlib-only and unit-tested against plain dicts — no GPU, no clock.
  * **Orchestration** (``run_staged_search``) measures the candidate phase's
    wall-clock and wires the pure core to ``gpu_cell_runner.run_matrix``.
  * **Shape-gated:** a ``cells.json`` with no ``search`` key returns ``[]`` from
    ``parse_search_spec`` and the caller runs the legacy single-phase path
    byte-for-byte unchanged.

``search`` schema (one entry per (family, condition, optimizer) group)::

    {"search": [{
        "group": "mnist_mlp_dropout_adam",
        "select_metric": "final_train_loss",   # key in the candidate's metrics.json
        "select_objective": "min",             # "min" | "max"
        "candidates": [                          # bounded short-budget cells
            {"id": "mnist_mlp_dropout_adam__lr3e-4", "params": {"lr": 3e-4, "epochs": 3, ...}},
            {"id": "mnist_mlp_dropout_adam__lr1e-3", "params": {"lr": 1e-3, "epochs": 3, ...}},
            {"id": "mnist_mlp_dropout_adam__lr3e-3", "params": {"lr": 3e-3, "epochs": 3, ...}}],
        "promote": {                             # the ONE full cell for the winner
            "id": "mnist_mlp_dropout_adam",
            "params": {"epochs": 200, "model_key": "mnist_mlp", "env": "dropout", "baseline": "adam"},
            "param_from_winner": ["lr"]}}]}      # winner params copied into the full cell
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Harness-enforced bounds on the tuning phase — the agent CANNOT blow these up
# into a wall-clock-killing cross-product (the exact failure this module exists
# to prevent). Excess is dropped with a logged warning.
_MAX_CANDIDATES_PER_GROUP = 5
_MAX_GROUPS = 40
_MAX_TOTAL_CANDIDATES = 80

# The candidate phase's short epochs run with high fixed-cost overhead per epoch;
# full cells (more epochs) amortize it, so a candidate per-epoch rate OVER-states
# the full rate. We still pad the estimate so the preflight errs toward caution.
_BUDGET_SAFETY_FACTOR = 1.25


@dataclass
class SearchGroup:
    """One tuning group: candidates → winner → one promoted full cell."""

    group: str
    select_metric: str
    select_objective: str  # "min" | "max"
    candidates: list[dict]
    promote: dict
    param_from_winner: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


def parse_search_spec(cells_json: Any) -> list[SearchGroup]:
    """Parse + validate the ``search`` section. Returns ``[]`` when absent or
    unusable (→ caller runs the legacy single-phase path). Never raises.

    Enforces the candidate-count caps deterministically so a malformed or
    over-eager manifest cannot launch an unbounded tuning grid.
    """
    if not isinstance(cells_json, dict):
        return []
    raw = cells_json.get("search")
    if not isinstance(raw, list) or not raw:
        return []

    groups: list[SearchGroup] = []
    total_candidates = 0
    for item in raw[:_MAX_GROUPS]:
        if not isinstance(item, dict):
            continue
        cands = item.get("candidates")
        promote = item.get("promote")
        if not isinstance(cands, list) or not isinstance(promote, dict):
            continue
        cands = [c for c in cands if isinstance(c, dict) and c.get("id")]
        cands = cands[:_MAX_CANDIDATES_PER_GROUP]
        if not cands or not promote.get("id"):
            continue
        if total_candidates + len(cands) > _MAX_TOTAL_CANDIDATES:
            logger.warning(
                "staged_search: total-candidate cap %d reached — dropping the "
                "remaining search groups (had %d, group '%s' would exceed).",
                _MAX_TOTAL_CANDIDATES, total_candidates,
                item.get("group") or promote.get("id"),
            )
            break
        total_candidates += len(cands)
        obj = str(item.get("select_objective") or "min").strip().lower()
        if obj not in ("min", "max"):
            obj = "min"
        groups.append(
            SearchGroup(
                group=str(item.get("group") or promote.get("id")),
                select_metric=str(item.get("select_metric") or "final_train_loss"),
                select_objective=obj,
                candidates=cands,
                promote=promote,
                param_from_winner=[str(k) for k in (item.get("param_from_winner") or [])],
            )
        )
    return groups


def _coerce_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_select_value(metrics: dict | None, metric_key: str) -> float | None:
    """Read the select metric from a cell's flat ``metrics.json``.

    Tries a flat key first, then a dotted path (``per_model.x.y``) for nested
    shapes. Returns ``None`` when absent or non-numeric (that candidate is then
    ineligible to win — a crashed/empty cell never wins by default).
    """
    if not isinstance(metrics, dict):
        return None
    flat = _coerce_float(metrics.get(metric_key))
    if flat is not None:
        return flat
    if "." in metric_key:
        node: Any = metrics
        for part in metric_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        return _coerce_float(node)
    return None


def _set_both(cell: dict, key: str, value: Any) -> None:
    """Write a param to BOTH ``cell[key]`` and ``cell["params"][key]``.

    The cell runner serializes the WHOLE cell to ``REPROLAB_CELL_PARAMS`` and a
    given paper's ``train_cell.py`` may read its lr/epochs from EITHER the top level
    (All-CNN) OR ``["params"]`` (Adam). Writing both makes the synthesized cell
    train at the intended value regardless of which shape the trainer reads, and
    lets ``materialize_full_cells`` (which copies from ``["params"]``) find it.
    """
    cell[key] = value
    p = cell.get("params")
    if not isinstance(p, dict):
        p = {}
    p[key] = value
    cell["params"] = p


def _read_either(cell: dict, key: str) -> Any:
    """Read ``cell[key]`` falling back to ``cell["params"][key]`` (None if neither)."""
    if key in cell:
        return cell[key]
    p = cell.get("params")
    if isinstance(p, dict) and key in p:
        return p[key]
    return None


def synthesize_search_from_hint(cells: list[dict], lr_search: dict) -> list[dict]:
    """Build a ``search`` array from a paper-hint ``lr_search`` spec + the agent's
    emitted full cells — Issue #1 (2026-06-15).

    One search GROUP per emitted cell: ``candidates`` = that cell cloned across the
    hint grid at ``probe_epochs`` (searched value written to BOTH shapes via
    ``_set_both``); ``promote`` = the agent's full cell (its real epochs preserved,
    normalized into ``["params"]`` so the budget preflight sees the true cost);
    ``param_from_winner = [param_key]`` so the harness promotes the TUNED value.

    Returns ``[]`` when the spec or cells are unusable — the caller then runs the
    legacy path (or whatever ``search`` the agent emitted). Never raises. This is
    the harness-synthesis half of the "both" decision: papers that declare
    ``lr_search`` get the staged route even when the agent emits a single fixed lr.
    """
    try:
        grid = [g for g in (lr_search.get("grid") or []) if isinstance(g, (int, float))]
        if not grid or not isinstance(cells, list) or not cells:
            return []
        param_key = str(lr_search.get("param_key") or "lr")
        epochs_key = str(lr_search.get("epochs_key") or "epochs")
        probe_epochs = lr_search.get("probe_epochs") or 8
        select_metric = str(lr_search.get("select_metric") or "final_train_loss")
        objective = str(lr_search.get("select_objective") or "min").lower()
        if objective not in ("min", "max"):
            objective = "min"
        search: list[dict] = []
        for cell in cells:
            if not isinstance(cell, dict) or not cell.get("id"):
                continue
            base_id = str(cell["id"])
            agent_epochs = _read_either(cell, epochs_key)
            candidates: list[dict] = []
            for val in grid:
                c = copy.deepcopy(cell)
                c["id"] = f"{base_id}__{param_key}_{val}"
                _set_both(c, param_key, val)
                _set_both(c, epochs_key, probe_epochs)
                candidates.append(c)
            promote = copy.deepcopy(cell)
            if agent_epochs is not None:
                _set_both(promote, epochs_key, agent_epochs)  # mirror for the budget calc
            search.append({
                "group": base_id,
                "select_metric": select_metric,
                "select_objective": objective,
                "candidates": candidates,
                "promote": promote,
                "param_from_winner": [param_key],
            })
        return search
    except Exception:  # noqa: BLE001 — synthesis must never break the run; fall back to legacy.
        logger.exception("synthesize_search_from_hint failed; falling back to legacy path")
        return []


# A modest default per-condition lr grid (3 candidates) used when a result_quality
# leaf demands tuning but no paper-hint grid is available. Kept small so N cells ×
# this grid stays well under _MAX_TOTAL_CANDIDATES; the existing parse_search_spec
# caps still apply on top.
_DEFAULT_LEAF_LR_GRID = (3e-4, 1e-3, 3e-3)


def synthesize_search_from_leaf(
    cells: list[dict],
    *,
    lr_grid: "list[float] | tuple[float, ...] | None" = None,
    param_key: str = "lr",
    epochs_key: str = "epochs",
    probe_epochs: int = 8,
    select_metric: str = "final_train_loss",
    select_objective: str = "min",
) -> list[dict]:
    """Build a ``search`` array from a weak-leaf DIAGNOSIS — L4 (2026-06-16).

    The complement to :func:`synthesize_search_from_hint`: that one is triggered
    by a paper-hint ``lr_search`` grid; this one is triggered by ``leaf_triage``
    classifying a leaf as ``result_quality`` (the paper's ordering is inverted —
    almost always an UNTUNED per-condition learning rate). When no hint grid
    exists, the harness still gets the staged tune-then-run by synthesizing a
    bounded default per-condition lr sweep over the agent's emitted cells.

    Each emitted cell IS a per-condition unit (one cell per optimizer/method/
    ablation), so the one-group-per-cell shape :func:`synthesize_search_from_hint`
    produces gives exactly per-condition tuning — every condition reported at ITS
    OWN best lr, the fix for the inverted-ordering leaves. Pure: delegates to
    :func:`synthesize_search_from_hint` with a derived ``lr_search`` dict.

    Returns ``[]`` when ``cells`` is unusable (→ caller keeps the legacy path).
    Never raises.
    """
    try:
        grid = list(lr_grid) if lr_grid else list(_DEFAULT_LEAF_LR_GRID)
        grid = [g for g in grid if isinstance(g, (int, float)) and not isinstance(g, bool)]
        if not grid or not isinstance(cells, list) or not cells:
            return []
        lr_search = {
            "grid": grid,
            "param_key": param_key,
            "epochs_key": epochs_key,
            "probe_epochs": probe_epochs,
            "select_metric": select_metric,
            "select_objective": select_objective,
        }
        return synthesize_search_from_hint(cells, lr_search)
    except Exception:  # noqa: BLE001 — synthesis must never break the run.
        logger.exception("synthesize_search_from_leaf failed; falling back to legacy path")
        return []


def select_winner(group: SearchGroup, candidate_results: dict[str, dict]) -> dict | None:
    """Deterministically pick the winning candidate CELL for ``group``.

    ``candidate_results`` maps ``cell_id`` → the ``run_matrix`` result dict (with
    a ``metrics`` key). Selection is by ``group.select_metric`` /
    ``select_objective``; ties break by candidate order (first wins). Returns the
    winning candidate cell dict, or ``None`` when no candidate produced a usable
    metric (caller then drops that group from the full phase).
    """
    best_cell: dict | None = None
    best_val: float | None = None
    for cand in group.candidates:
        cid = str(cand.get("id"))
        res = candidate_results.get(cid)
        metrics = res.get("metrics") if isinstance(res, dict) else None
        val = extract_select_value(metrics, group.select_metric)
        if val is None:
            continue
        if best_val is None or (
            val < best_val if group.select_objective == "min" else val > best_val
        ):
            best_val, best_cell = val, cand
    return best_cell


def materialize_full_cells(
    groups: list[SearchGroup], winners: dict[str, dict]
) -> list[dict]:
    """Build the full comparison cells: exactly ONE per group whose winner was
    found, the promote template with the winner's tuned params copied in.

    A group with no winner is omitted (its candidates all crashed/empty) — the
    caller logs it; it never silently materializes an untuned full cell.
    """
    full: list[dict] = []
    for g in groups:
        win = winners.get(g.group)
        if win is None:
            continue
        cell = dict(g.promote)
        params = dict(cell.get("params") or {})
        win_params = win.get("params") or {}
        for k in g.param_from_winner:
            if k in win_params:
                params[k] = win_params[k]
                cell[k] = win_params[k]  # Issue #1: also top-level, so a trainer that
                #                          reads the WHOLE cell (All-CNN) honors the tune.
        cell["params"] = params
        cell.setdefault("id", g.group)
        cell["_tuned_from"] = str(win.get("id") or "")  # provenance for the report
        full.append(cell)
    return full


def _total_epochs(cells: list[dict]) -> float:
    total = 0.0
    for c in cells:
        ep = _coerce_float((c.get("params") or {}).get("epochs")) or 1.0
        total += max(1.0, ep)
    return total


def candidate_rate(candidate_wall_s: float | None, candidate_cells: list[dict]) -> float | None:
    """Wall-seconds per epoch measured from the candidate phase (encodes this
    run's parallelism, since both phases share the GPU pool). ``None`` when
    unusable. The single throughput source for the estimate + the reducer."""
    cand_epochs = _total_epochs(candidate_cells)
    if not candidate_wall_s or candidate_wall_s <= 0 or cand_epochs <= 0:
        return None
    return candidate_wall_s / cand_epochs


def estimate_full_seconds(
    candidate_wall_s: float | None,
    candidate_cells: list[dict],
    full_cells: list[dict],
) -> float | None:
    """Estimate the full phase's wall-clock from the MEASURED candidate phase.

    Converts the measured candidate rate to the full phase's epochs, padded by
    ``_BUDGET_SAFETY_FACTOR`` (the candidate phase's short epochs over-state the
    per-epoch cost that full cells amortize, so this errs toward caution).
    Returns ``None`` when inputs are unusable (caller SKIPS the preflight —
    fail-soft, never a false abort).
    """
    rate = candidate_rate(candidate_wall_s, candidate_cells)
    full_epochs = _total_epochs(full_cells)
    if rate is None or full_epochs <= 0:
        return None
    return rate * full_epochs * _BUDGET_SAFETY_FACTOR


def affordable_full_cells(
    full_cells: list[dict],
    rate: float | None,
    remaining_s: float | None,
    reserve_s: float,
) -> tuple[list[dict], list[dict]]:
    """Greedily keep the CHEAPEST full cells that fit ``remaining_s - reserve_s``.

    Returns ``(kept, dropped)``. Cheapest-first preserves BREADTH (more families
    graded — breadth is the rubric's dominant lever) when the budget is tight,
    rather than a wall-clock death that loses everything. When the rate or
    remaining budget is unknown, keep ALL (fail-soft). Dropped cells are the
    caller's loud warning, never a silent truncation.
    """
    if rate is None or rate <= 0 or remaining_s is None or remaining_s <= 0:
        return list(full_cells), []
    budget = remaining_s - max(0.0, reserve_s)
    if budget <= 0:
        return [], list(full_cells)
    costed = sorted(
        full_cells,
        key=lambda c: max(1.0, _coerce_float((c.get("params") or {}).get("epochs")) or 1.0),
    )
    kept: list[dict] = []
    dropped: list[dict] = []
    spent = 0.0
    for c in costed:
        ep = max(1.0, _coerce_float((c.get("params") or {}).get("epochs")) or 1.0)
        cost = rate * ep * _BUDGET_SAFETY_FACTOR
        if spent + cost <= budget:
            kept.append(c)
            spent += cost
        else:
            dropped.append(c)
    return kept, dropped


def budget_feasible(
    est_full_s: float | None, remaining_s: float | None, reserve_s: float
) -> tuple[bool, str]:
    """Is there room to run the full phase within the remaining wall-clock?

    Returns ``(True, reason)`` when it fits OR when the estimate is unavailable
    (fail-soft — an un-estimable run is NOT blocked, mirroring the rest of the
    harness's fail-open posture). ``(False, reason)`` only on a CONFIDENT
    infeasibility, so the caller can stop before launching a doomed full grid.
    """
    if est_full_s is None or remaining_s is None or remaining_s <= 0:
        return True, "no_estimate"
    need = est_full_s + max(0.0, reserve_s)
    if need <= remaining_s:
        return True, f"fits (need ~{int(need)}s <= remaining {int(remaining_s)}s)"
    return False, (
        f"infeasible (need ~{int(need)}s > remaining {int(remaining_s)}s, "
        f"reserve {int(reserve_s)}s)"
    )


# ---------------------------------------------------------------------------
# Orchestration (the only non-pure function — calls run_matrix + a monotonic clock)
# ---------------------------------------------------------------------------


def run_staged_search(
    groups: list[SearchGroup],
    cell_script: Any,
    *,
    output_root: Any,
    gpus: list[str] | None = None,
    max_parallel: int | None = None,
    remaining_s: float | None = None,
    reserve_s: float = 0.0,
    per_cell_timeout_s: float | None = None,
    now_iso: str | None = None,
    emit: Any = None,
    **run_matrix_kwargs: Any,
) -> dict[str, Any]:
    """Run the two-phase tune-then-run and return the FULL-cell results.

    Phase 1 runs every group's candidates via ``run_matrix`` (measured); winners
    are selected deterministically; the full phase is budget-checked against
    ``remaining_s`` (minus the candidate wall-clock) and greedily reduced to what
    fits; Phase 2 runs the surviving full cells at the winners' tuned params.

    Returns ``{"results": {cell_id: result}, ...}`` where ``results`` is the
    full-cell results the caller aggregates exactly like a single-phase
    ``run_matrix`` return. ``emit(code, message, **extra)`` (optional) surfaces
    progress/warnings as SSE. Never raises on an individual cell failure.
    """
    import time

    from backend.agents.rlm.gpu_cell_runner import run_matrix

    run_matrix_kwargs.pop("overall_timeout_s", None)  # the staged budget owns this

    def _emit(code: str, msg: str, **extra: Any) -> None:
        if emit is not None:
            try:
                emit(code, msg, **extra)
            except Exception:  # noqa: BLE001 — emit is best-effort
                logger.debug("staged_search: emit failed", exc_info=True)
        logger.info("staged_search: %s — %s", code, msg)

    candidate_cells = [c for g in groups for c in g.candidates]
    _emit(
        "staged_search_candidates",
        f"phase 1: {len(candidate_cells)} candidate cells across {len(groups)} group(s)",
    )
    t0 = time.monotonic()
    candidate_results = run_matrix(
        candidate_cells, cell_script, output_root=output_root, gpus=gpus,
        max_parallel=max_parallel, per_cell_timeout_s=per_cell_timeout_s,
        now_iso=now_iso, **run_matrix_kwargs,
    )
    cand_wall = time.monotonic() - t0

    winners: dict[str, dict] = {}
    for g in groups:
        w = select_winner(g, candidate_results)
        if w is not None:
            winners[g.group] = w
        else:
            _emit(
                "staged_search_no_winner",
                f"group '{g.group}': no candidate produced metric '{g.select_metric}' "
                f"— group dropped from the full phase",
            )
    full_cells = materialize_full_cells(groups, winners)

    rate = candidate_rate(cand_wall, candidate_cells)
    remaining_after = (remaining_s - cand_wall) if remaining_s is not None else None
    kept, dropped = affordable_full_cells(full_cells, rate, remaining_after, reserve_s)
    if dropped:
        _emit(
            "staged_search_budget_reduced",
            f"{len(dropped)} full cell(s) dropped to fit the remaining budget "
            f"(~{int(remaining_after or 0)}s, rate {rate:.1f}s/epoch): "
            + ", ".join(str(c.get("id")) for c in dropped),
            dropped=[str(c.get("id")) for c in dropped],
        )

    _emit("staged_search_full", f"phase 2: {len(kept)} full cell(s) at tuned params")
    full_results: dict[str, Any] = {}
    if kept:
        full_results = run_matrix(
            kept, cell_script, output_root=output_root, gpus=gpus,
            max_parallel=max_parallel, per_cell_timeout_s=per_cell_timeout_s,
            overall_timeout_s=remaining_after, now_iso=now_iso, **run_matrix_kwargs,
        )

    return {
        "results": full_results,
        "kept_cells": kept,  # the materialized full-cell DICTS, for aggregation
        "candidate_results": candidate_results,
        "winners": {grp: str(w.get("id") or "") for grp, w in winners.items()},
        "full_cells": [str(c.get("id") or "") for c in kept],
        "dropped_cells": [str(c.get("id") or "") for c in dropped],
        "candidate_wall_s": cand_wall,
        "rate_s_per_epoch": rate,
    }


__all__ = [
    "SearchGroup",
    "parse_search_spec",
    "synthesize_search_from_hint",
    "synthesize_search_from_leaf",
    "extract_select_value",
    "select_winner",
    "materialize_full_cells",
    "candidate_rate",
    "estimate_full_seconds",
    "affordable_full_cells",
    "budget_feasible",
    "run_staged_search",
]
