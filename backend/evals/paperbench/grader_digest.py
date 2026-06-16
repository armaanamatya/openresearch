"""Count-based per-cell grader digest — make wide grids fully visible (A6).

``_compact_metrics_for_grader`` fixed the 14-model case by collapsing long
series, but post-compaction *scalar* volume is still unbounded: a wider grid hits
the raw ``[:96KB]`` slice (32KB on the exception fallback) and the TRAILING
headline cells silently fall off the grader prompt. The grader then scores a
genuinely-run cell "central claim unverified" because it never SAW it.

The fix is a deterministic, COUNT-BASED digest: for every cell/model/env/baseline
leaf in ``per_model`` we emit one compact record — ``{status, headline_metric,
n_epochs}`` — so the byte budget scales with the NUMBER of cells, not the size of
their histories, and **no cell can vanish regardless of grid width**.

We also fix the ranking foot-gun: ``_latest_metrics_path`` ranked metrics paths
on ``has_results`` *truthiness*, so a placeholder ``per_model: {m: {}}`` could
outrank genuinely-measured older data. :func:`per_model_has_measured_value`
ranks on whether a REAL measured numeric exists.

Pure: no network, no global state, no disk I/O.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "per_model_has_measured_value",
    "build_grader_digest",
]

# Cell fields that are bookkeeping, not the experiment's measured RESULT. Used
# only as a fallback "is there ANY measured value" probe and to pick a headline
# when no obvious result key is present — never to gate a cell out of the digest.
_NON_RESULT_KEYS = frozenset({
    "status", "model_key", "env", "environment", "baseline", "dataset",
    "variant", "letter", "device", "seed", "n", "depth", "param_count",
    "use_residual", "shortcut_option", "best_at_iter", "total_iters",
    "epochs_run", "wall_time_seconds", "metric",
})

# Preferred headline-metric keys, in priority order — the single scalar that most
# directly states the cell's reported RESULT. First present wins. Key-name
# agnostic fallback (any other numeric) follows, so an unseen paper still gets a
# headline.
_HEADLINE_PRIORITY = (
    "test_error_pct", "test_accuracy", "best_test_accuracy", "best_test_error_pct",
    "test_acc", "accuracy", "acc", "error_pct", "error",
    "final_reward", "reward", "success_rate", "win_rate",
    "f1", "em", "exact_match", "bleu", "perplexity", "ppl",
    "final_train_loss", "loss",
)

# Keys whose value is a per-epoch series we count for ``n_epochs``.
_EPOCH_SERIES_KEYS = (
    "epoch", "test_accuracy", "test_acc", "test_error", "train_loss",
    "train_error", "iteration", "lr", "reward",
)


def _is_real_number(v: Any) -> bool:
    """True for a genuine measured numeric (int/float, NOT bool, NOT None)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _cell_has_measured_value(cell: dict) -> bool:
    """True iff a cell dict carries at least one real measured numeric value.

    An empty ``{}`` placeholder is False. A dict whose result fields are all
    ``None`` (declared-but-unmeasured) is False. A nested ``history`` of numeric
    series counts as measured (the cell ran and produced curves).
    """
    if not cell:
        return False
    for k, v in cell.items():
        if k in _NON_RESULT_KEYS:
            continue
        if _is_real_number(v):
            return True
        if isinstance(v, dict):
            # e.g. history: {train_loss: [...], ...} — any non-empty numeric
            # series means the cell produced measurements.
            for sv in v.values():
                if isinstance(sv, list) and any(_is_real_number(x) for x in sv):
                    return True
                if _is_real_number(sv):
                    return True
        if isinstance(v, list) and any(_is_real_number(x) for x in v):
            return True
    return False


def _iter_cells(per_model: Any):
    """Yield ``(model_key, env, baseline, cell_dict)`` for every leaf cell.

    Handles BOTH metrics shapes the harness emits:

    * 3-level ``per_model[model][env][baseline] -> cell`` (the cells-route /
      All-CNN shape), and
    * flat ``per_model[model] -> cell`` (older shape where the model dict itself
      carries ``status`` and result scalars).

    A dict is treated as a CELL (a leaf) when it carries a ``status`` or any
    real measured numeric / history; otherwise it is treated as a CONTAINER and
    we descend one level. This guarantees every leaf surfaces exactly once and
    none is mistaken for a container (which would drop it from the digest).
    """
    if not isinstance(per_model, dict):
        return
    for mkey, mval in per_model.items():
        if not isinstance(mval, dict):
            # Degenerate model value — surface it as a status-less cell so it is
            # never silently dropped.
            yield (str(mkey), None, None, {"_value": mval})
            continue
        if _looks_like_cell(mval):
            yield (str(mkey), None, None, mval)
            continue
        # Container of environments.
        for env, eval_ in mval.items():
            if not isinstance(eval_, dict):
                yield (str(mkey), str(env), None, {"_value": eval_})
                continue
            if _looks_like_cell(eval_):
                yield (str(mkey), str(env), None, eval_)
                continue
            # Container of baselines.
            for bname, cell in eval_.items():
                if isinstance(cell, dict):
                    yield (str(mkey), str(env), str(bname), cell)
                else:
                    yield (str(mkey), str(env), str(bname), {"_value": cell})


def _looks_like_cell(d: dict) -> bool:
    """Heuristic: a dict is a leaf CELL (not a container) if it carries a
    ``status`` field or any real measured numeric / history, rather than being a
    pure mapping of sub-dicts.

    An EMPTY dict is also a leaf (a placeholder cell), never a container — a
    container always has children. Treating ``{}`` as a container would let a
    placeholder ``per_model: {m: {}}`` silently vanish from the digest, the very
    foot-gun A6 closes ("no cell may silently vanish")."""
    if not d:
        return True
    if "status" in d:
        return True
    for k, v in d.items():
        if _is_real_number(v):
            return True
        if k == "history" and isinstance(v, dict):
            return True
    return False


def per_model_has_measured_value(metrics_obj: dict) -> bool:
    """True iff SOME ``per_model`` leaf carries a real measured numeric value.

    Used to rank candidate metrics paths on MEASURED data rather than
    truthiness, so a placeholder ``per_model: {m: {}}`` (truthy but empty) never
    outranks genuinely-measured older data.
    """
    if not isinstance(metrics_obj, dict):
        return False
    per_model = metrics_obj.get("per_model")
    if not isinstance(per_model, dict):
        return False
    for _m, _e, _b, cell in _iter_cells(per_model):
        if isinstance(cell, dict) and _cell_has_measured_value(cell):
            return True
    return False


def _resolve_headline(cell: dict) -> dict | None:
    """Return ``{"name": str, "value": number}`` for the cell's headline metric.

    Tries the priority result keys first, then any other real measured numeric
    (key-name agnostic), so an unseen paper still gets a headline. Returns
    ``None`` when the cell carries no scalar result (e.g. a placeholder or a
    history-only cell).
    """
    for name in _HEADLINE_PRIORITY:
        v = cell.get(name)
        if _is_real_number(v):
            return {"name": name, "value": v}
    # Fallback: first non-bookkeeping real numeric, deterministic by key order.
    for k in sorted(cell.keys()):
        if k in _NON_RESULT_KEYS:
            continue
        v = cell.get(k)
        if _is_real_number(v):
            return {"name": k, "value": v}
    return None


def _resolve_n_epochs(cell: dict) -> int:
    """Best-effort epoch count for a cell.

    Prefers an explicit ``epochs_run`` integer; else the longest per-epoch
    series under ``history``; else a top-level epoch-like series; else 0.
    """
    explicit = cell.get("epochs_run")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    best = 0
    history = cell.get("history")
    if isinstance(history, dict):
        for k in _EPOCH_SERIES_KEYS:
            seq = history.get(k)
            if isinstance(seq, list):
                best = max(best, len(seq))
        if best == 0:
            for seq in history.values():
                if isinstance(seq, list):
                    best = max(best, len(seq))
    if best == 0:
        for k in _EPOCH_SERIES_KEYS:
            seq = cell.get(k)
            if isinstance(seq, list):
                best = max(best, len(seq))
    return best


def build_grader_digest(metrics_obj: dict) -> dict:
    """Return a deterministic count-based per-cell digest of ``per_model``.

    For EVERY cell/model/env/baseline leaf, emit one record::

        {"model_key", "env", "baseline", "status", "headline_metric", "n_epochs", "measured"}

    where ``headline_metric`` is ``{"name", "value"}`` or ``None``. The digest's
    size scales with the NUMBER of cells, not the size of their histories, so a
    wide grid (20+ models) is fully represented and no cell silently vanishes
    regardless of grid width.

    Cells are sorted by ``(model_key, env, baseline)`` for stable, deterministic
    output. The top-level result also carries the cell ``count`` and a
    ``measured_count`` so a downstream prompt-builder can budget against the
    grid size directly.
    """
    cells: list[dict] = []
    if isinstance(metrics_obj, dict):
        per_model = metrics_obj.get("per_model")
        for mkey, env, baseline, cell in _iter_cells(per_model):
            cell = cell if isinstance(cell, dict) else {}
            status = cell.get("status")
            measured = _cell_has_measured_value(cell)
            cells.append({
                "model_key": mkey,
                "env": env,
                "baseline": baseline,
                "status": str(status) if status is not None else None,
                "headline_metric": _resolve_headline(cell),
                "n_epochs": _resolve_n_epochs(cell),
                "measured": measured,
            })

    cells.sort(key=lambda c: (
        c["model_key"] or "",
        c["env"] or "",
        c["baseline"] or "",
    ))
    return {
        "count": len(cells),
        "measured_count": sum(1 for c in cells if c["measured"]),
        "cells": cells,
    }
