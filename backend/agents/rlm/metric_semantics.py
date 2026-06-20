"""
C — metric-semantics guard (§4.6 of 2026-06-20-pre-gpu-code-review-and-report-validation-design.md).

Pure stdlib module.  Default-OFF: OPENRESEARCH_METRIC_SEMANTICS_GUARD.

Fires AFTER the zero-metrics guard in run_experiment.  A clear out-of-range
rate metric (accuracy, success_rate, f1, precision, recall > 1.0+eps) or a
non-finite loss/reward is degraded to failure_class="fabrication_suspected"
so the root re-implements.

Conservative:
  - 0.0 is in range (a legitimately-zero rate is fine).
  - Only a CLEAR violation triggers (values in [0.0, 1.0+eps] never fire).
  - Non-numeric / missing keys never fire.
  - Fail-soft everywhere.
"""

from __future__ import annotations

import math
import os
import re

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def metric_semantics_guard_enabled() -> bool:
    """True iff OPENRESEARCH_METRIC_SEMANTICS_GUARD opts this guard ON."""
    return os.environ.get(
        "OPENRESEARCH_METRIC_SEMANTICS_GUARD", ""
    ).strip().lower() in _ENABLED_VALUES


# ---------------------------------------------------------------------------
# Rate-named keys that must be in [0, 1+eps]
# ---------------------------------------------------------------------------

# A key whose lowercased form matches this pattern is treated as a rate.
_RATE_KEY_RE = re.compile(
    r"\b(?:accuracy|success_rate|success|f1|precision|recall)\b",
    re.IGNORECASE,
)

# Tolerance above 1.0 for rounding errors (e.g. 1.0000001).
_RATE_EPS = 1e-4

# Loss / reward keys that must be finite.
_FINITE_KEY_RE = re.compile(
    r"\b(?:loss|reward|mean_reward|return|mean_return)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Recursive leaf visitor
# ---------------------------------------------------------------------------

def _collect_violations(obj: object, key_path: str, out: list[str]) -> None:
    """Recursively walk a metrics dict; append violation strings to `out`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_violations(v, k, out)
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_violations(item, key_path, out)
        return
    # Leaf
    if isinstance(obj, bool):
        return
    try:
        fv = float(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return

    lk = key_path.lower() if key_path else ""

    # Rate check
    if _RATE_KEY_RE.search(lk):
        if not (0.0 <= fv <= 1.0 + _RATE_EPS):
            # A value in (1+eps, 100] is plausibly a percentage (e.g. 84.0 = 84%),
            # which the root should have normalised — flag it as out-of-range.
            out.append(
                f"rate key '{key_path}' = {fv} is outside [0, 1] "
                f"(expected a fraction; if this is a percentage, "
                f"normalise to [0, 1] before writing metrics.json)"
            )
        return

    # Finite check for loss/reward
    if _FINITE_KEY_RE.search(lk):
        if not math.isfinite(fv):
            out.append(
                f"loss/reward key '{key_path}' = {fv} is non-finite "
                f"(NaN or Inf — training diverged or the metric was never computed)"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def metric_semantics_violation(metrics: object) -> str | None:
    """Return a detail string if a clear semantic violation is found, else None.

    Checks rate-named keys (accuracy|success_rate|f1|precision|recall) must
    be in [0, 1+eps]; loss/reward must be finite.

    Conservative: 0.0 is in range; only a CLEAR out-of-range fires.
    Fail-soft: any exception returns None.
    """
    try:
        if not isinstance(metrics, dict):
            return None
        violations: list[str] = []
        _collect_violations(metrics, "", violations)
        if violations:
            return "; ".join(violations)
        return None
    except Exception:  # noqa: BLE001
        return None
