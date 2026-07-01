"""Harness-owned figure + training-curve sidecar emitter (cells route, 2026-06-27).

Called from ``primitives._persist_metrics`` after the aggregated ``metrics.json``
is written.  Produces two classes of grounded evidence the TEXT-ONLY grader reads:

* **``training_curves.json``** — per ``(model, env, baseline)`` reward/loss step
  series extracted from ``per_model`` leaves (keys searched: ``training_curves``
  nested dict/list, ``reward_history``, ``loss_history``, ``step``/``epoch``).
  Written only when ≥1 curve is found anywhere in ``per_model``; a run with no
  curve data writes NO file (fail-soft, byte-identical).

* **``fig_auto_<model_key>_<env>.json``** — one comparison sidecar per
  ``(model, env)`` group built from the primary on-disk metric across its
  baselines (a numeric array → downsampled curve; scalars → by-condition
  comparison).  Matches the shape ``leaf_scorer._gather_figure_sidecars`` reads.
  Not emitted when the agent already wrote its own ``fig_*.json`` sidecar
  (``_agent_emitted_sidecar`` guard keeps sidecars honest).

Flag: ``OPENRESEARCH_EMIT_FIGURE_SIDECARS`` (default **OFF** — unset ⇒ byte-
identical to the prior baseline).  Recommend ON for SDAR runs via ``--run-spec``
(the text-only grader cannot see PNGs; without sidecars it grades figure leaves
against no evidence, which bottoms the eval-protocol + artifact sub-scores).

Design discipline:
  * **Stdlib-only** (``json``, ``re``, ``os``, ``pathlib``, ``logging``) — safe
    to call from within the cells-route write path without adding import weight.
  * **Fail-soft** — every error is caught and logged at ``DEBUG``; never raises
    out to the caller.
  * **Grounded** — never fabricates: a group with no measured values emits no
    sidecar; a file with no curve rows writes no ``training_curves.json``.
  * **Additive** — does not modify ``metrics.json`` or any other existing file.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------

FLAG = "OPENRESEARCH_EMIT_FIGURE_SIDECARS"
_FIG_AUTO_PREFIX = "fig_auto_"


def is_enabled() -> bool:
    return os.environ.get(FLAG, "").strip().lower() in ("1", "true", "on", "yes")


# ---------------------------------------------------------------------------
# Internal helpers (stdlib-only, no side-effects)
# ---------------------------------------------------------------------------

# Keys that hold reward curves in a per_model leaf (ordered by priority).
_CURVE_REWARD_KEYS = ("reward_history", "reward", "rewards", "mean_reward")
# Keys that hold loss curves.
_CURVE_LOSS_KEYS = ("loss_history", "loss", "train_loss")
# Keys that hold step/epoch indices.
_CURVE_STEP_KEYS = ("step", "steps", "epoch", "epochs")

# Keys skipped when choosing the *primary* scalar metric for a figure sidecar.
_FIG_SKIP_KEYS = frozenset({"status", "steps_run", "wall_time_s", "seed", "epoch"})
# Ordered preference list: pick the first matching key from each leaf.
_FIG_METRIC_PREFERENCE = (
    "metric", "final_test_acc", "test_acc", "accuracy", "final_elbo", "elbo",
    "cumulative_return", "reward", "final_test_nll", "test_nll", "nll",
    "final_train_loss", "loss", "test_error_pct", "error", "score",
)
_FIG_LOG_HINTS = ("loss", "nll", "elbo", "error", "perplexity")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fig_axis_scale(metric_name: str) -> str:
    n = (metric_name or "").lower()
    return "log" if any(h in n for h in _FIG_LOG_HINTS) else "linear"


def _extract_series(value: Any) -> list[float] | None:
    """Return a non-empty list of floats from ``value`` iff it is a numeric sequence."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    out = [float(x) for x in value if _is_number(x)]
    return out if out else None


def _extract_curves_from_leaf(leaf: dict) -> dict[str, list[float]]:
    """Pull reward / loss / step series out of a per_model leaf dict."""
    if not isinstance(leaf, dict):
        return {}
    curves: dict[str, list[float]] = {}

    tc = leaf.get("training_curves")
    if isinstance(tc, dict):
        # Nested dict: {"reward": [...], "loss": [...], "step": [...]}
        for rk in _CURVE_REWARD_KEYS:
            s = _extract_series(tc.get(rk))
            if s:
                curves["reward"] = s
                break
        for lk in _CURVE_LOSS_KEYS:
            s = _extract_series(tc.get(lk))
            if s:
                curves["loss"] = s
                break
        for sk in _CURVE_STEP_KEYS:
            s = _extract_series(tc.get(sk))
            if s:
                curves["step"] = s
                break
    elif isinstance(tc, list) and tc:
        # List-of-dicts: [{"step": 0, "reward": 0.1, "loss": 2.3}, ...]
        rewards = _extract_series([d.get("reward") for d in tc if isinstance(d, dict)])
        losses = _extract_series([d.get("loss") for d in tc if isinstance(d, dict)])
        steps = _extract_series([d.get("step") for d in tc if isinstance(d, dict)])
        if rewards:
            curves["reward"] = rewards
        if losses:
            curves["loss"] = losses
        if steps:
            curves["step"] = steps

    # Flat-key fallback: reward_history / loss_history / step at the leaf root.
    if "reward" not in curves:
        for rk in _CURVE_REWARD_KEYS:
            s = _extract_series(leaf.get(rk))
            if s:
                curves["reward"] = s
                break
    if "loss" not in curves:
        for lk in _CURVE_LOSS_KEYS:
            s = _extract_series(leaf.get(lk))
            if s:
                curves["loss"] = s
                break
    if "step" not in curves:
        for sk in _CURVE_STEP_KEYS:
            s = _extract_series(leaf.get(sk))
            if s and len(s) > 1:   # a lone step index is not a meaningful curve
                curves["step"] = s
                break

    return curves


def _fig_primary_metric(cell: dict) -> "tuple[str, Any] | None":
    """The headline numeric (scalar or series) from a per_model cell, grounded."""
    if not isinstance(cell, dict):
        return None
    for key in _FIG_METRIC_PREFERENCE:
        v = cell.get(key)
        if _is_number(v):
            return key, float(v)
        if isinstance(v, list) and v and all(_is_number(x) for x in v):
            return key, [float(x) for x in v]
    for key, v in cell.items():
        if key not in _FIG_SKIP_KEYS and _is_number(v):
            return key, float(v)
    return None


def _downsample(series: list, max_points: int = 40) -> list:
    if len(series) <= max_points:
        return list(series)
    step = len(series) / float(max_points)
    return [series[min(len(series) - 1, int(i * step))] for i in range(max_points)]


def _agent_emitted_sidecar(code_dir: Path) -> bool:
    """True when a NON-backstop ``fig_*.json`` already exists in ``code_dir``."""
    try:
        for sc in code_dir.rglob("fig_*.json"):
            if sc.is_file() and not sc.name.startswith(_FIG_AUTO_PREFIX):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# Public emitters (called individually or via emit_sidecars)
# ---------------------------------------------------------------------------


def emit_training_curves(code_dir: Path, per_model: dict) -> bool:
    """Write ``training_curves.json`` from per_model leaf histories.

    Shape written::

        {
          "<model_key>": {
            "<env>": {
              "<baseline>": {"step": [...], "reward": [...], "loss": [...]}
            }
          }
        }

    Only keys that are actually present in the leaf are included; missing curve
    types are omitted (never padded with zeros).  Returns ``True`` if the file
    was written, ``False`` when no curves were found (no file written —
    byte-identical to the prior baseline).
    """
    if not isinstance(per_model, dict):
        return False
    result: dict[str, Any] = {}
    for mk, envs in per_model.items():
        if not isinstance(envs, dict):
            continue
        for env, bases in envs.items():
            if not isinstance(bases, dict):
                continue
            for baseline, leaf in bases.items():
                curves = _extract_curves_from_leaf(leaf)
                if curves:
                    result.setdefault(str(mk), {}).setdefault(str(env), {})[str(baseline)] = curves
    if not result:
        return False
    try:
        (code_dir / "training_curves.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return True
    except OSError as exc:
        logger.debug("figure_sidecars: failed to write training_curves.json: %s", exc)
        return False


def emit_figure_sidecars_harness(
    code_dir: Path,
    per_model: dict,
    *,
    max_sidecars: int = 6,
    max_points: int = 40,
    max_baselines: int = 12,
) -> list[str]:
    """Write ``fig_auto_<model_key>_<env>.json`` sidecars from measured per_model.

    Produces one comparison sidecar per ``(model_key, env)`` group built from the
    primary on-disk metric across its baselines.  Matches the JSON shape that
    ``leaf_scorer._gather_figure_sidecars`` reads and the grader consumes.

    No-op when the agent already emitted its own ``fig_*.json`` sidecar (the
    ``_agent_emitted_sidecar`` guard prevents piling on), when no measured values
    exist (grounded), or on any error.  Returns relative paths written (relative
    to ``code_dir.parent`` — the project root).
    """
    written: list[str] = []
    if not isinstance(per_model, dict) or not per_model:
        return written
    if _agent_emitted_sidecar(code_dir):
        return written  # don't pile on the agent's own sidecars

    project_dir = code_dir.parent
    for mk, envs in per_model.items():
        if len(written) >= max_sidecars:
            break
        if not isinstance(envs, dict):
            continue
        for env, bases in envs.items():
            if len(written) >= max_sidecars:
                break
            if not isinstance(bases, dict):
                continue
            series: dict[str, Any] = {}
            metric_name: str | None = None
            is_curve = False
            for baseline, cell in list(bases.items())[:max_baselines]:
                pm = _fig_primary_metric(cell)
                if pm is None:
                    continue
                metric_name = metric_name or pm[0]
                val = pm[1]
                if isinstance(val, list):
                    is_curve = True
                    series[str(baseline)] = _downsample(val, max_points)
                else:
                    series[str(baseline)] = val
            if not series:
                continue  # grounded: nothing measured for this group
            key = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{mk}_{env}")[:60]
            sidecar = {
                "figure": f"{_FIG_AUTO_PREFIX}{key}",
                "shows": (
                    f"Measured {metric_name} "
                    + ("trajectory" if is_curve else "by condition")
                    + f" for {mk} / {env} — comparison across "
                    f"{len(series)} baseline(s): {', '.join(sorted(series))}."
                ),
                "x_axis": {
                    "label": "training step" if is_curve else "condition / baseline",
                    "scale": "linear",
                },
                "y_axis": {
                    "label": metric_name,
                    "scale": _fig_axis_scale(metric_name or ""),
                },
                "series": series,
                "source": "code/metrics.json (measured on-disk values)",
                "note": (
                    "Harness-owned figure sidecar (figure_sidecars.py): the "
                    "text-only grader reads this instead of a PNG. Values are "
                    "measured per-condition results — grounded, never fabricated."
                ),
            }
            try:
                path = code_dir / f"{_FIG_AUTO_PREFIX}{key}.json"
                path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                written.append(str(path.relative_to(project_dir)))
            except OSError:
                continue

    return written


# ---------------------------------------------------------------------------
# Main entry point — wired into primitives._persist_metrics
# ---------------------------------------------------------------------------


def emit_sidecars(code_dir: "Path | str", metrics: dict) -> None:
    """Emit ``training_curves.json`` + ``fig_auto_*.json`` from aggregated metrics.

    Called from ``primitives._persist_metrics`` after the aggregated
    ``metrics.json`` is written.  No-op (fail-soft) when:

    * ``OPENRESEARCH_EMIT_FIGURE_SIDECARS`` is unset/off (byte-identical default),
    * ``metrics`` has no ``per_model``,
    * ``per_model`` contains no curve or scalar data (grounded).

    Never raises.
    """
    if not is_enabled():
        return
    try:
        code_dir = Path(code_dir)
        per_model = metrics.get("per_model") if isinstance(metrics, dict) else None
        if not isinstance(per_model, dict) or not per_model:
            return
        emit_training_curves(code_dir, per_model)
        emit_figure_sidecars_harness(code_dir, per_model)
    except Exception:  # noqa: BLE001 — sidecars are advisory, never block the run
        logger.debug("figure_sidecars.emit_sidecars failed", exc_info=True)


__all__ = [
    "FLAG",
    "is_enabled",
    "emit_training_curves",
    "emit_figure_sidecars_harness",
    "emit_sidecars",
]
