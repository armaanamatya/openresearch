"""
F3 no-learning-signal detector — deterministic curve-trend veto.

Pure stdlib module — no third-party imports.  Detects run_experiment results
whose training curves show NO learning signal: reward never rose AND loss never
descended.  Such a run trained real steps but produced an under-powered or
mis-wired result, so the replication verdict is forced to "inconclusive" rather
than claiming a false success.

Design mirrors zero_metrics_detection.py and eval_provenance.py:
  - Flag default-OFF: ``OPENRESEARCH_NO_LEARNING_SIGNAL_GATE``.
  - Fail-soft everywhere: any exception → safe fallback.
  - Conservative: if ANY leaf shows learning, returns (False, None) — the run
    learned somewhere, so it is a real (if weak) result.
  - Both the nested cells-route shape (per_model[m][e][b]=leaf) and the flat
    per_model[model]=leaf shape are handled.

DIVERGED CURVES (NaN/±Inf — P0 fix): non-finite readings used to be FILTERED OUT
of the curve, so a run whose loss went NaN and stayed NaN read as "no curve → not
judgeable → no data" — the single most degenerate training outcome was invisible to
the trend guard.  Non-finite points are now KEPT, and a curve that ENDS non-finite
is reported as DIVERGED (a no-learning leaf, with a "diverged" detail).  Deliberately
conservative — it is the TAIL that decides, so a transient early NaN/Inf (fp16
overflow that the grad-scaler recovers from) that returns to finite values is NOT
flagged, and the rise/descent math still runs on the finite sub-curve exactly as
before.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_POINTS: int = 5        # need at least this many curve points to judge
_TREND_EPS: float = 0.05    # "rose" means best >= first*(1+eps) for nonzero first;
                             # for first==0, "rose" means best > abs(_TREND_EPS)

# Statuses that mark a leaf as a real result (vs a failed/skipped cell).
_SUCCESS_STATUSES: frozenset[str] = frozenset({"ok", "success", "succeeded", "completed"})


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def no_learning_signal_enabled() -> bool:
    """True iff ``OPENRESEARCH_NO_LEARNING_SIGNAL_GATE`` is in {'1','true','yes','on'}."""
    return os.environ.get(
        "OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Curve extraction helpers
# ---------------------------------------------------------------------------

def _coerce_float_list(raw: Any) -> list[float]:
    """Coerce a raw value to a list of floats, NaN/±Inf INCLUDED.  Fail-soft → [].

    Non-finite readings are deliberately preserved: dropping them is what made an
    all-NaN (diverged) curve indistinguishable from "no curve at all".  Callers use
    :func:`_curve_diverged` to classify and :func:`_finite` for the trend math.
    """
    try:
        if not isinstance(raw, (list, tuple)):
            return []
        out: list[float] = []
        for v in raw:
            if isinstance(v, bool):
                continue  # structural flag, not a curve point
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
        return out
    except Exception:  # noqa: BLE001
        return []


def _finite(curve: list[float]) -> list[float]:
    """The finite sub-curve — what the rise/descent math is computed over.

    Keeps the trend semantics byte-identical to the pre-NaN-fix behaviour for every
    curve that has no non-finite readings, and keeps ``max``/``min`` sound (NaN makes
    them order-dependent) for curves that do.
    """
    return [v for v in curve if math.isfinite(v)]


def _curve_diverged(curve: list[float]) -> bool:
    """True iff the curve ENDS in a non-finite (NaN/±Inf) reading.

    "Went NaN and STAYED NaN" — the training blew up and never recovered, so the run
    has no result.  Keying on the TAIL (not "any non-finite point anywhere") is the
    conservative choice: a transient NaN/Inf that later returns to finite values is a
    recovered run, not a diverged one, and must never be flagged.
    """
    try:
        return bool(curve) and not math.isfinite(curve[-1])
    except (TypeError, ValueError):  # noqa: BLE001
        return False


def _leaf_diverged(leaf: dict[str, Any]) -> bool:
    """True iff either of this leaf's persisted curves ends non-finite."""
    try:
        return _curve_diverged(_reward_curve(leaf)) or _curve_diverged(_loss_curve(leaf))
    except Exception:  # noqa: BLE001
        return False


def _reward_curve(leaf: dict[str, Any]) -> list[float]:
    """Extract the reward training curve from a leaf dict.

    Precedence: leaf.training_curves.reward | .rewards | .mean_reward,
    else leaf.reward_history.  Returns [] when nothing usable is found.
    """
    try:
        tc = leaf.get("training_curves")
        if isinstance(tc, dict):
            for key in ("reward", "rewards", "mean_reward"):
                val = tc.get(key)
                if isinstance(val, (list, tuple)):
                    curve = _coerce_float_list(val)
                    if curve:
                        return curve
        rh = leaf.get("reward_history")
        if isinstance(rh, (list, tuple)):
            curve = _coerce_float_list(rh)
            if curve:
                return curve
        return []
    except Exception:  # noqa: BLE001
        return []


def _loss_curve(leaf: dict[str, Any]) -> list[float]:
    """Extract the loss training curve from a leaf dict.

    Precedence: leaf.training_curves.loss, else leaf.loss_history.
    Returns [] when nothing usable is found.
    """
    try:
        tc = leaf.get("training_curves")
        if isinstance(tc, dict):
            val = tc.get("loss")
            if isinstance(val, (list, tuple)):
                curve = _coerce_float_list(val)
                if curve:
                    return curve
        lh = leaf.get("loss_history")
        if isinstance(lh, (list, tuple)):
            curve = _coerce_float_list(lh)
            if curve:
                return curve
        return []
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Per-leaf no-learning predicate
# ---------------------------------------------------------------------------

def _leaf_no_learning(leaf: dict[str, Any]) -> bool | None:
    """Assess a single leaf for no-learning signal.

    Returns:
        None   — not judgeable (no curve with >= _MIN_POINTS points).
        True   — the leaf shows no learning: it DIVERGED (a curve ends NaN/±Inf), or
                 reward did NOT rise AND (loss absent OR did NOT descend).
        False  — the leaf shows learning (reward rose OR loss descended).

    Reward "rose" means: max(curve) > first*(1+eps) for nonzero first, OR
    max(curve) > abs(eps) for first==0.  Loss "descended" means:
    min(curve) < first*(1-eps) for nonzero first, OR
    first > abs(eps) and min(curve) < first - abs(eps).
    """
    try:
        reward = _reward_curve(leaf)
        loss = _loss_curve(leaf)

        # DIVERGED (P0): a curve that ends non-finite is a degenerate result, not
        # missing data — judgeable regardless of length (there is no "too short to
        # tell" about a NaN) and never a learning signal.  Checked BEFORE the trend
        # math, which is only defined over finite readings.
        if _curve_diverged(reward) or _curve_diverged(loss):
            return True

        reward = _finite(reward)
        loss = _finite(loss)

        has_reward = len(reward) >= _MIN_POINTS
        has_loss = len(loss) >= _MIN_POINTS

        if not has_reward and not has_loss:
            return None  # not judgeable

        reward_no_rise: bool | None = None
        if has_reward:
            first_r = reward[0]
            best_r = max(reward)
            if first_r != 0.0:
                threshold_r = abs(first_r) * (1.0 + _TREND_EPS)
                reward_no_rise = best_r <= threshold_r
            else:
                # first == 0: "rose" means best > absolute eps
                reward_no_rise = best_r <= abs(_TREND_EPS)

        loss_no_descent: bool | None = None
        if has_loss:
            first_l = loss[0]
            best_l = min(loss)
            if first_l != 0.0:
                threshold_l = abs(first_l) * (1.0 - _TREND_EPS)
                loss_no_descent = best_l >= threshold_l
            else:
                # first == 0: loss can only go up or stay; treat as no descent
                loss_no_descent = True

        # A leaf shows NO learning iff reward did NOT rise AND
        # (loss is absent OR loss did NOT descend).
        # If only loss is present (no reward curve) the loss alone drives the verdict.
        if reward_no_rise is None:
            # No reward curve; use loss only.
            return bool(loss_no_descent)
        if loss_no_descent is None:
            # No loss curve; use reward only.
            return bool(reward_no_rise)
        return bool(reward_no_rise and loss_no_descent)

    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Walk the metrics.json and collect judgeable leaves
# ---------------------------------------------------------------------------

def _is_leaf(obj: Any) -> bool:
    """Heuristic: a dict that has a status key OR any curve key is a leaf result."""
    if not isinstance(obj, dict):
        return False
    if "status" in obj:
        return True
    if "reward_history" in obj or "loss_history" in obj:
        return True
    tc = obj.get("training_curves")
    if isinstance(tc, dict) and tc:
        return True
    return False


def _walk_per_model(per_model: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Recursively collect leaf dicts from the per_model sub-tree.

    Handles:
      flat:   per_model[model] = leaf
      nested: per_model[model][env][baseline] = leaf  (depth ≤ 3)

    Returns all found leaves.
    """
    leaves: list[dict[str, Any]] = []
    try:
        if not isinstance(per_model, dict):
            return leaves
        for value in per_model.values():
            if _is_leaf(value):
                leaves.append(value)
            elif isinstance(value, dict) and depth < 3:
                leaves.extend(_walk_per_model(value, depth + 1))
    except Exception:  # noqa: BLE001
        pass
    return leaves


# ---------------------------------------------------------------------------
# Public detection API
# ---------------------------------------------------------------------------

def detect_no_learning_signal(code_dir: Any) -> tuple[bool, str | None]:
    """Walk ``code/metrics.json`` and detect if no leaf shows a learning signal.

    Args:
        code_dir: Path-like pointing to the ``code/`` directory of a run.

    Returns:
        ``(False, None)`` when:
          - the gate is disabled (default),
          - ``code/metrics.json`` is absent or unreadable,
          - no judgeable leaf exists (no curve with >= _MIN_POINTS points, and no
            diverged curve),
          - at least one judgeable leaf shows learning.
        ``(True, detail)`` iff there is ≥1 judgeable leaf AND EVERY judgeable
          leaf shows no learning signal — flat OR DIVERGED (a curve ending in
          NaN/±Inf).  ``detail`` names up to 3 leaves, with their first/best
          values (flat) or the non-finite tail reading (diverged).
    """
    if not no_learning_signal_enabled():
        return (False, None)

    try:
        code_path = Path(code_dir)
        metrics_path = code_path / "metrics.json"
        if not metrics_path.exists():
            return (False, None)

        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return (False, None)

        if not isinstance(data, dict):
            return (False, None)

        # Collect all leaves from the per_model sub-tree.
        per_model = data.get("per_model")
        if isinstance(per_model, dict):
            leaves = _walk_per_model(per_model)
        else:
            # Top-level IS the leaf (flat single-model shape).
            leaves = [data] if _is_leaf(data) else []

        # Filter to success-status leaves that are judgeable.
        no_learning_leaves: list[dict[str, Any]] = []
        learning_found = False

        for leaf in leaves:
            status = str(leaf.get("status", "")).lower()
            if status not in _SUCCESS_STATUSES:
                continue
            verdict = _leaf_no_learning(leaf)
            if verdict is None:
                continue  # not judgeable
            if verdict:
                no_learning_leaves.append(leaf)
            else:
                learning_found = True
                break  # conservative: any learning → not a no-signal run

        if learning_found or not no_learning_leaves:
            return (False, None)

        # All judgeable leaves show no learning.  Build a detail string.
        detail_parts: list[str] = []
        for leaf in no_learning_leaves[:3]:
            label = leaf.get("cell_id") or leaf.get("model_key") or leaf.get("model") or "leaf"
            # DIVERGED first — its curves contain NaN/±Inf, over which max()/min()
            # are order-dependent nonsense, so never run the trend summary on them.
            if _leaf_diverged(leaf):
                raw_r, raw_l = _reward_curve(leaf), _loss_curve(leaf)
                which, tail = (
                    ("reward", raw_r[-1]) if _curve_diverged(raw_r) else ("loss", raw_l[-1])
                )
                detail_parts.append(f"{label}(diverged: {which} curve ends {tail})")
                continue
            rc = _finite(_reward_curve(leaf))
            if rc:
                first_r = rc[0]
                best_r = max(rc)
                detail_parts.append(f"{label}(first_reward={first_r:.4g}, best_reward={best_r:.4g})")
            else:
                lc = _finite(_loss_curve(leaf))
                if lc:
                    first_l = lc[0]
                    best_l = min(lc)
                    detail_parts.append(f"{label}(first_loss={first_l:.4g}, best_loss={best_l:.4g})")
                else:
                    detail_parts.append("leaf(no curve)")

        detail = "; ".join(detail_parts) if detail_parts else f"{len(no_learning_leaves)} leaf(ves) with flat curves"
        return (True, detail)

    except Exception:  # noqa: BLE001 — fail-soft
        return (False, None)


# ---------------------------------------------------------------------------
# Repair message
# ---------------------------------------------------------------------------

def no_learning_repair_message(detail: str) -> str:
    """A run_warning-style message for the no-learning-signal veto.

    Covers both shapes the detector reports: FLAT curves (under-powered/mis-wired)
    and DIVERGED curves (loss/reward ended NaN/±Inf — a numerical blow-up, whose
    repair is different, so the message names it explicitly when it applies).
    """
    try:
        detail_str = str(detail).strip() if detail else "(no detail)"
        if "diverged" in detail_str:
            return (
                f"no_learning_signal: {detail_str} — training DIVERGED: the reward/loss "
                f"curve ends in a non-finite value (NaN/Inf), so no reported metric from "
                f"this run is a real measurement. replication_verdict is forced to "
                f"'inconclusive'. Fix the numerical instability (learning rate too high, "
                f"missing gradient clipping, log/div by zero, fp16 overflow, un-normalized "
                f"reward) and re-run until the curves stay finite."
            )
        return (
            f"no_learning_signal: {detail_str} — training ran but reward/loss show "
            f"no improvement; the run is under-powered or training is mis-wired. "
            f"replication_verdict is forced to 'inconclusive' (implementation may still "
            f"be faithful). Increase training budget (steps) or fix the reward wiring."
        )
    except Exception:  # noqa: BLE001
        return (
            "no_learning_signal: training curves show no improvement. "
            "replication_verdict is forced to 'inconclusive'. "
            "Increase training budget (steps) or fix the reward wiring."
        )
