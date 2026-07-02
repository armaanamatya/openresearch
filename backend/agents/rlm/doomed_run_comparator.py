"""doomed_run_comparator.py — doomed-attempt early-kill comparator (spec
§10.3, ``OPENRESEARCH_DOOMED_KILL``, default OFF).

During WATCH, compares the live training curve against the best COMPLETED
attempt's curve at the same step, with the same two-signal conservative
discipline as the execution-reliability stall guard: it never fires on the
first attempt (no baseline), never on an unknown metric direction, never
below a progress floor, and only after several CONSECUTIVE worse-by-margin
observations (any non-worse observation resets the streak).

Pure module (stdlib only). NOT wired into any live path in this unit -- the
campaign's WATCH integration is a later, explicitly operator-validated step
(spec keeps the flag OFF by default); this module ships the decision logic
and its test coverage only.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

DOOMED_KILL_ENV = "OPENRESEARCH_DOOMED_KILL"
DOOMED_MARGIN_ENV = "OPENRESEARCH_DOOMED_MARGIN"
DOOMED_POLLS_ENV = "OPENRESEARCH_DOOMED_POLLS"
DOOMED_MIN_PROGRESS_ENV = "OPENRESEARCH_DOOMED_MIN_PROGRESS"

_DEFAULT_MARGIN = 0.30
_DEFAULT_POLLS_REQUIRED = 6
_DEFAULT_MIN_PROGRESS = 0.2
_TRUTHY = ("1", "true", "yes", "on")


def enabled() -> bool:
    """Whether the doomed-attempt early-kill comparator is enabled.

    Reads the env on every call (no import-time capture, so tests can
    monkeypatch per-case); default OFF.
    """
    return os.environ.get(DOOMED_KILL_ENV, "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class CurvePoint:
    step: int
    value: float


@dataclass
class DoomedComparator:
    """Stateful per-WATCH comparator: call :meth:`observe` once per poll.
    A ``True`` return means ``polls_required`` CONSECUTIVE worse-by-margin
    observations have now fired (abort advised, spec §10.3). Any
    non-worse observation resets the streak.
    """

    best_curve: Sequence[CurvePoint] | None  # from the best COMPLETED attempt
    higher_is_better: bool | None  # from the metric contract; None = unknown
    planned_steps: int | None
    margin: float = _DEFAULT_MARGIN
    polls_required: int = _DEFAULT_POLLS_REQUIRED
    min_progress: float = _DEFAULT_MIN_PROGRESS
    _streak: int = field(default=0, init=False, repr=False)

    def observe(self, point: CurvePoint) -> bool:
        if not self._worse_by_margin(point):
            self._streak = 0
            return False
        self._streak += 1
        return self._streak >= self.polls_required

    def _worse_by_margin(self, point: CurvePoint) -> bool:
        if not self.best_curve:  # None or empty: no baseline (e.g. attempt 1)
            return False
        if self.higher_is_better is None:  # unknown metric direction
            return False
        if not self.planned_steps or self.planned_steps <= 0:
            return False
        if point.step < self.min_progress * self.planned_steps:
            return False
        best_value = _value_at_or_before(self.best_curve, point.step)
        if best_value is None:
            return False
        gap = _relative_gap(point.value, best_value, higher_is_better=self.higher_is_better)
        return gap >= self.margin


def _value_at_or_before(curve: Sequence[CurvePoint], step: int) -> float | None:
    """Last-value-carried-forward lookup: the value of the latest curve
    point whose ``step <= step``, or ``None`` when no such point exists."""
    candidates = [p for p in curve if p.step <= step]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.step).value


def _relative_gap(live_value: float, best_value: float, *, higher_is_better: bool) -> float:
    """Relative margin by which ``live_value`` trails ``best_value`` in the
    metric's favorable direction; <= 0 when live is at or ahead of best. A
    zero baseline has no computable relative gap (treated as no gap, never
    a spurious division by zero)."""
    if best_value == 0:
        return 0.0
    gap = (best_value - live_value) if higher_is_better else (live_value - best_value)
    return gap / abs(best_value)


def from_env(
    best_curve: Sequence[CurvePoint] | None,
    higher_is_better: bool | None,
    planned_steps: int | None,
) -> DoomedComparator:
    """Build a :class:`DoomedComparator` reading the three tunable env
    knobs (spec §10.3). Callers still gate WHETHER to construct/consult one
    on :func:`enabled` -- this function itself does not read
    ``DOOMED_KILL_ENV``.
    """
    return DoomedComparator(
        best_curve=best_curve,
        higher_is_better=higher_is_better,
        planned_steps=planned_steps,
        margin=_env_float(DOOMED_MARGIN_ENV, _DEFAULT_MARGIN),
        polls_required=_env_int(DOOMED_POLLS_ENV, _DEFAULT_POLLS_REQUIRED),
        min_progress=_env_float(DOOMED_MIN_PROGRESS_ENV, _DEFAULT_MIN_PROGRESS),
    )


def _env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


__all__ = [
    "CurvePoint",
    "DOOMED_KILL_ENV",
    "DOOMED_MARGIN_ENV",
    "DOOMED_MIN_PROGRESS_ENV",
    "DOOMED_POLLS_ENV",
    "DoomedComparator",
    "enabled",
    "from_env",
]
