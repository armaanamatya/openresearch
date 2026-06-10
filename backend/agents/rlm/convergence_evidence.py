"""convergence_evidence — structured convergence / sweep evidence for fidelity grading.

The motivating failure (2026-06-09 Adam reproduction, scored 0.7364 < the 0.8308 bar):
the agent computed the paper's convergence curves, LR sweeps, and a regret series, but
left them in stdout logs and final-scalar metrics — so the "Evaluation protocol and metric
correctness" rubric area collapsed to 0.21 (four 0.0 leaves: "no sweep results in metrics",
"regret … no time series", "fig_4 axis.x='Epoch' … not the Figure 4 protocol"). The paper's
HEADLINE claims are about convergence SPEED, which final scalars structurally cannot evidence.

This module is the *enforced* backstop the prior prompt-only guidance lacked. It is a pure,
copyable helper (stdlib only; numpy/torch never imported) so it copies into an agent sandbox
exactly like ``rubric_guard.py`` / ``gpu_cell_runner.py``. Two jobs:

1. ``derive_convergence_metrics`` — turn per-method curves into the comparison the grader
   needs (iterations-to-threshold, area-under-curve, final), so a faithful "Adam converges
   faster" claim becomes *verifiable evidence* rather than a near-tie of final scalars.
2. ``missing_structured_evidence`` — given the metrics dict + a declared evidence requirement,
   return the list of absent structured-evidence paths. ``rubric_guard.assert_metrics_schema``
   raises ``RubricGuardFailure`` on a non-empty list (→ next iteration's ``repair_context``),
   so the agent CANNOT silently ship final-scalars-only when the paper makes convergence/sweep
   claims.

Flag-gated: everything is inert unless ``OPENRESEARCH_FIDELITY_EVIDENCE`` is truthy. Off → the
checks return "nothing missing" and the derivations are still pure/no-ops, so non-convergence
papers and the byte-for-byte-unchanged path stay unaffected.
"""

from __future__ import annotations

import math
import os
from typing import Any

__all__ = [
    "is_enabled",
    "derive_convergence_metrics",
    "iterations_to_threshold",
    "area_under_curve",
    "missing_structured_evidence",
    "figure_axis_matches",
]

ENV_FLAG = "OPENRESEARCH_FIDELITY_EVIDENCE"


def is_enabled() -> bool:
    """True when the fidelity-evidence layer is armed (``OPENRESEARCH_FIDELITY_EVIDENCE`` truthy).

    Mirror of ``execution_smoke.is_enabled()`` — a single env read so the gate name lives
    in exactly one place. Default OFF: an unset/empty/``0``/``false`` value is disabled.
    """
    val = os.environ.get(ENV_FLAG, "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Derived convergence metrics (pure — safe to call regardless of the flag)
# ---------------------------------------------------------------------------

def _as_float_list(seq: Any) -> list[float]:
    """Coerce ``seq`` to a list of finite floats, dropping anything non-numeric/NaN/Inf."""
    out: list[float] = []
    if not isinstance(seq, (list, tuple)):
        return out
    for v in seq:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            f = float(v)
            if math.isfinite(f):
                out.append(f)
    return out


def iterations_to_threshold(
    x: list[float], y: list[float], threshold: float, *, descending: bool = True
) -> float | None:
    """Return the first x at which y crosses ``threshold``, or None if never crossed.

    ``descending=True`` (the default, for a loss curve) crosses when ``y <= threshold``;
    ``descending=False`` (for an accuracy curve) crosses when ``y >= threshold``. Linear
    interpolation between the bracketing points gives a sub-step crossing so two methods that
    both reach the target between the same two logged epochs are still ordered. None when the
    curve never reaches the target — an honest "did not converge to this level" signal, never
    a fabricated number.
    """
    xs, ys = _aligned(x, y)
    if not xs:
        return None
    crossed = (lambda v: v <= threshold) if descending else (lambda v: v >= threshold)
    if crossed(ys[0]):
        return xs[0]
    for i in range(1, len(ys)):
        if crossed(ys[i]):
            # linear interpolation between (xs[i-1], ys[i-1]) and (xs[i], ys[i])
            y0, y1 = ys[i - 1], ys[i]
            if y1 == y0:
                return xs[i]
            frac = (threshold - y0) / (y1 - y0)
            frac = min(1.0, max(0.0, frac))
            return xs[i - 1] + frac * (xs[i] - xs[i - 1])
    return None


def area_under_curve(x: list[float], y: list[float]) -> float | None:
    """Trapezoidal area under the (x, y) curve, or None when fewer than 2 aligned points.

    Lower AUC of a loss curve = faster overall descent — a single scalar that summarises a
    convergence advantage even when finals tie.
    """
    xs, ys = _aligned(x, y)
    if len(xs) < 2:
        return None
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return area


def _aligned(x: Any, y: Any) -> tuple[list[float], list[float]]:
    """Return the (x, y) prefix of equal length with both values finite-numeric."""
    xs, ys = _as_float_list(x), _as_float_list(y)
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def derive_convergence_metrics(
    history: dict[str, Any], *, threshold: float | None = None, y_key: str = "train_loss",
    x_key: str = "epoch", descending: bool = True,
) -> dict[str, dict[str, float | None]]:
    """Summarise a per-method ``history`` block into the grader's comparison table.

    ``history`` shape (what the prompt asks the agent to emit)::

        {"adam": {"epoch": [...], "train_loss": [...], "val_metric": [...]},
         "sgd_nesterov": {"epoch": [...], "train_loss": [...], ...}, ...}

    Returns, per method: ``{"iters_to_threshold": <x|None>, "auc": <float|None>,
    "final": <float|None>}``. When ``threshold`` is None it defaults to the worst (max for a
    loss) final value across methods × a small margin, so "iterations to a COMMON target the
    slowest method also reaches" is comparable across methods. Pure and total — never raises.
    """
    if not isinstance(history, dict) or not history:
        return {}
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for method, rec in history.items():
        if not isinstance(rec, dict):
            continue
        xs, ys = _aligned(rec.get(x_key), rec.get(y_key))
        if xs:
            curves[str(method)] = (xs, ys)
    if not curves:
        return {}
    if threshold is None:
        finals = [ys[-1] for _, ys in curves.values() if ys]
        if finals:
            # common target every method should reach: the slowest method's final, eased 1%
            threshold = (max(finals) if descending else min(finals))
            threshold *= (1.01 if descending else 0.99)
    out: dict[str, dict[str, float | None]] = {}
    for method, (xs, ys) in curves.items():
        out[method] = {
            "iters_to_threshold": (
                iterations_to_threshold(xs, ys, threshold, descending=descending)
                if threshold is not None else None
            ),
            "auc": area_under_curve(xs, ys),
            "final": ys[-1] if ys else None,
        }
    return out


# ---------------------------------------------------------------------------
# Structured-evidence enforcement (consulted by rubric_guard when the flag is on)
# ---------------------------------------------------------------------------

def _resolve_path(metrics: dict[str, Any], dotted: str) -> bool:
    """True iff ``dotted`` (e.g. ``history.adam.train_loss``) resolves to a non-empty value."""
    node: Any = metrics
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    if isinstance(node, (list, tuple, dict, str)):
        return len(node) > 0
    return node is not None


def missing_structured_evidence(
    metrics: dict[str, Any], requirement: dict[str, Any] | None
) -> list[str]:
    """Return the list of required structured-evidence paths absent from ``metrics``.

    ``requirement`` (supplied per-paper, e.g. from a PAPER_HINTS entry) declares which
    structured-evidence families the paper's claims demand::

        {"history_methods": ["adam", "sgd_nesterov", "adagrad"],  # need per-method curves
         "sweeps": ["vae_lr_sweep"],                              # need sweep results in metrics
         "series": ["regret"],                                    # need a time-series, not a scalar
         "require_history": true}

    Off-flag OR an empty/None requirement → ``[]`` (nothing enforced). Pure, total, never
    raises — a malformed requirement degrades to "nothing missing" (fail-soft, never blocks a
    run on a bad hint).
    """
    if not is_enabled() or not isinstance(requirement, dict) or not isinstance(metrics, dict):
        return []
    missing: list[str] = []

    methods = requirement.get("history_methods")
    if isinstance(methods, list) and methods:
        hist = metrics.get("history")
        for m in methods:
            # robust to BOTH a flat history.<method>.<curve> and a per-experiment
            # history.<experiment>.<method>.<curve> nesting (the Adam paper has many
            # experiments, each comparing the same optimizers).
            if not _history_has_method(hist, str(m)):
                missing.append(f"history.{m}.<curve> (per-epoch trajectory; flat or per-experiment)")

    for sweep in requirement.get("sweeps", []) or []:
        if not _resolve_path(metrics, str(sweep)):
            missing.append(f"{sweep} (sweep results in metrics.json, not only logs)")

    for series in requirement.get("series", []) or []:
        # a time-series must be an array somewhere under the named key, not a lone scalar
        if not _series_present(metrics, str(series)):
            missing.append(f"{series} time-series (array over t, not a single scalar)")

    return missing


_CURVE_KEYS = (
    "train_loss", "train_cost", "loss", "val_metric", "val_loss", "elbo", "nll",
    "train_nll", "test_acc", "test_accuracy", "accuracy", "regret",
)


def _norm(s: Any) -> str:
    """Identifier-normalise a method name for tolerant comparison (lower, strip non-alnum)."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _carries_curve(node: Any) -> bool:
    """True iff ``node`` is a dict holding at least one length>1 numeric curve array."""
    if not isinstance(node, dict):
        return False
    for k, v in node.items():
        if isinstance(v, (list, tuple)) and len(_as_float_list(v)) > 1:
            # a recognised curve key, or any numeric array alongside an x-axis
            if _norm(k) in {_norm(c) for c in _CURVE_KEYS} or "epoch" in str(k).lower() \
                    or "step" in str(k).lower() or "iter" in str(k).lower():
                return True
    # accept the degenerate case of a dict of two parallel arrays (x + one curve)
    arrays = [v for v in node.values() if isinstance(v, (list, tuple)) and len(_as_float_list(v)) > 1]
    return len(arrays) >= 2


def _history_has_method(hist: Any, method: str) -> bool:
    """Search ``hist`` for ``method`` as a key whose value carries a per-epoch curve.

    Matches a flat ``history.adam.train_loss`` and a nested
    ``history.mnist_lr.adam.train_loss`` alike — recursion finds the method key at any depth.
    """
    target = _norm(method)
    if isinstance(hist, dict):
        for k, v in hist.items():
            if _norm(k) == target and _carries_curve(v):
                return True
            if _history_has_method(v, method):
                return True
    return False


def _series_present(metrics: dict[str, Any], name: str) -> bool:
    """True iff a key containing ``name`` maps (possibly nested) to a length>1 numeric array.

    The match is *ancestral*: once we descend into a key containing ``name`` (e.g. ``regret``),
    any numeric array beneath it counts — so ``regret: {t: [...], cumulative: [...]}`` and a
    flat ``regret: [...]`` both pass, while a lone scalar ``regret_final_cumulative: 0.05``
    does not (a scalar is neither dict nor list).
    """
    name = name.lower()

    def walk(node: Any, matched: bool) -> bool:
        if isinstance(node, dict):
            for k, v in node.items():
                if walk(v, matched or (name in str(k).lower())):
                    return True
            return False
        if isinstance(node, (list, tuple)):
            return matched and len(_as_float_list(node)) > 1
        return False

    return walk(metrics, False)


def figure_axis_matches(sidecar: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    """True iff a figure's axis sidecar matches the paper's expected axes.

    ``expected`` names the axis labels/scales the paper uses, e.g.
    ``{"x": {"scale": "log"}, "y": {"label_contains": "loss"}}``. The Adam Fig-4 failure was
    ``axis.x='Epoch' (linear)`` where the paper plots loss vs ``log10(alpha)`` — a wrong-axis
    figure the grader scored 0.0. Only the keys present in ``expected`` are checked (lenient on
    everything unspecified). Off-flag → always True (no enforcement). Never raises.
    """
    if not is_enabled():
        return True
    if not isinstance(sidecar, dict):
        return False
    axes = sidecar.get("axis", sidecar)
    if not isinstance(axes, dict):
        return False
    for ax_name, want in expected.items():
        got = axes.get(ax_name)
        if not isinstance(got, dict) or not isinstance(want, dict):
            return False
        if "scale" in want and str(got.get("scale", "")).lower() != str(want["scale"]).lower():
            return False
        lc = want.get("label_contains")
        if lc and lc.lower() not in str(got.get("label", "")).lower():
            return False
    return True
