"""doomed_run_comparator.py — doomed-attempt early-kill comparator (spec
§10.3, ``OPENRESEARCH_DOOMED_KILL``, default OFF).

During the campaign's AWAIT stage, compares the live attempt's training
curves against the best COMPLETED attempt's curves at the same step, with
the same two-signal conservative discipline as the execution-reliability
stall guard: it never fires on the first attempt (no baseline), never on an
unknown metric direction, never below a progress floor, and only after
several CONSECUTIVE worse-by-margin observations (any non-worse observation
resets the streak).

Three layers, deliberately separated:

* :class:`DoomedComparator` — the PURE per-cell decision. No I/O, no clock,
  no env. Its ``observe`` semantics are frozen (``tests/rlm/
  test_doomed_run_comparator.py``); the wiring below never reaches around it.
* :func:`read_cell_curves` — the deterministic EVIDENCE reader: a run's
  ``code/`` tree -> per-cell measured curves. Pure function of on-disk
  artifacts (stdlib only, fail-soft). This is the north-star invariant in
  force: a kill keys on MEASURED on-disk evidence, never on an LLM grade,
  never on a grader justification, never on agent prose.
* :class:`DoomedWatch` — the pure multi-cell aggregator the campaign feeds
  once per poll. Owns the two false-positive backstops that a single
  comparator cannot express: STALENESS (a repeated reading is not a new
  observation) and CONSENSUS (one healthy matched cell vetoes the kill).

The campaign state machine (``reproduction_campaign.py``) owns everything
impure: the poll thread, the process-group kill, the ledger/SSE record.

FALSE-POSITIVE SAFETY (the whole point — killing a healthy-but-slow run is
the one error a needle-in-haystack triage product cannot afford; cf.
``learn.md`` 2026-07-07, where a guard false-blocked faithful SDAR work for
three cycles). Every condition below must hold before a single kill fires:

1. **A baseline exists.** Attempt 1 is structurally unkillable — there is no
   completed attempt to compare against.
2. **The baseline is trusted.** A hard-quarantined (fabrication-flagged) or
   itself-doomed-killed attempt is never a kill baseline; ranking is by
   EVIDENCE count, never by score (the LLM grade is not a fitness signal).
3. **A shared, explicit step axis.** Both curves must carry an explicit,
   length-matched ``step``/``epoch`` series. No step axis -> no comparison ->
   no kill. (Index-position fallback is deliberately refused: a live run that
   logs every step against a baseline that logged every 10 would read as
   10x "behind" purely from cadence.)
4. **Same metric, same direction.** Cell keys AND metric AND direction must
   match, or the pair is skipped.
5. **Past the progress floor.** Nothing below ``min_progress`` (default 0.2)
   of the reference run's own total steps is ever judged — a shared warmup
   cancels out anyway, because the baseline at that same step was warming up
   too.
6. **A large relative margin.** Default 0.30 — the live cell must trail the
   reference by >=30% in the metric's favourable direction.
7. **Sustained, and on FRESH evidence.** Default 6 CONSECUTIVE worse-by-margin
   observations, where an "observation" only counts when the curve has
   ADVANCED to a new step. A quiet eval phase (no new curve points) produces
   zero observations, so it can neither build nor break a streak. Any single
   non-worse reading resets the streak to zero.
8. **Consensus.** Every matched cell with a confident reading must be firing.
   ONE healthy matched cell vetoes the kill outright.

DIVERGED CURVES. A live reading that is non-finite (NaN/±Inf) is scored as
infinitely worse than any finite baseline (:func:`_relative_gap`), mirroring
the 2026-07-13 IEEE-754 fixes elsewhere in the harness: under self-inequality
a NaN silently satisfied no predicate at all, so the single most degenerate
training outcome was invisible. It still takes ``polls_required`` CONSECUTIVE
non-finite readings to kill — a transient fp16 spike the grad-scaler recovers
from resets the streak, exactly like ``dead_training_guard.NONFINITE_WINDOW``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DOOMED_KILL_ENV = "OPENRESEARCH_DOOMED_KILL"
DOOMED_MARGIN_ENV = "OPENRESEARCH_DOOMED_MARGIN"
DOOMED_POLLS_ENV = "OPENRESEARCH_DOOMED_POLLS"
DOOMED_MIN_PROGRESS_ENV = "OPENRESEARCH_DOOMED_MIN_PROGRESS"

#: The honest failure class of an attempt the campaign STOPPED PAYING for.
#: Deliberately distinct from every science failure class (``report_missing``,
#: ``fabrication_suspected``, ``all_models_failed``, ...) so the campaign
#: report and any downstream triage consumer can tell "we cut our losses"
#: apart from "the science failed". ``campaign_policy`` keys its
#: science-counter exclusions on this exact string.
DOOMED_KILLED_CLASS = "doomed_killed"

_DEFAULT_MARGIN = 0.30
_DEFAULT_POLLS_REQUIRED = 6
_DEFAULT_MIN_PROGRESS = 0.2
_TRUTHY = ("1", "true", "yes", "on")

#: A cell needs at least this many persisted curve points before it is
#: judgeable at all -- on EITHER side of the comparison.
_MIN_CURVE_POINTS = 3
#: Bounds on the evidence reader (a curve file is agent-written; never trust
#: it to be small).
_MAX_CURVE_POINTS = 5000
_MAX_CURVE_FILES = 128

# Curve key vocabulary -- deliberately IDENTICAL to figure_sidecars.py /
# no_learning_signal.py (the harness's existing curve contract, which
# baseline_implementation.py already instructs the implementer to emit).
_CURVE_REWARD_KEYS = ("reward_history", "reward", "rewards", "mean_reward")
_CURVE_LOSS_KEYS = ("loss_history", "loss", "train_loss")
_CURVE_STEP_KEYS = ("step", "steps", "epoch", "epochs")


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
    metric's favorable direction; <= 0 when live is at or ahead of best.

    A NON-FINITE live reading (NaN/±Inf) is ``+inf`` worse than any finite
    baseline: training blew up, so this reading is not a slow result, it is
    no result. (Without this the IEEE-754 self-inequality rules make every
    NaN comparison False, and the single most degenerate training outcome
    would be the one thing this guard could never see. Sustaining still
    requires ``polls_required`` CONSECUTIVE non-finite readings, so a
    grad-scaler-recovered spike is not a kill.)

    A zero or non-finite BASELINE has no computable relative gap (treated as
    no gap, never a spurious division by zero and never a kill).
    """
    if not math.isfinite(live_value):
        return math.inf
    if best_value == 0 or not math.isfinite(best_value):
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


# ===========================================================================
# The deterministic evidence layer: run dir -> measured per-cell curves.
# Pure functions of on-disk artifacts. Stdlib only. Fail-soft everywhere --
# an unreadable/absent/ill-shaped artifact yields NO curve, and no curve can
# never produce a kill.
# ===========================================================================


@dataclass(frozen=True)
class CellCurve:
    """One matched-comparison unit: a single training cell's measured curve.

    ``key`` is the cell's structural coordinate (``model/env/baseline``, or
    the ``outputs/`` relpath) derived IDENTICALLY on both sides of the
    comparison, so the live attempt is only ever compared against the SAME
    cell of the reference attempt -- never against a different (easier)
    cell, which would be a first-class false-positive source.
    """

    key: str
    metric: str  # "reward" | "loss" -- the only two directions we can know
    higher_is_better: bool
    points: tuple[CurvePoint, ...]  # step-ascending, deduped

    @property
    def latest(self) -> CurvePoint:
        return self.points[-1]

    @property
    def total_steps(self) -> int:
        return self.points[-1].step


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_series(raw: Any) -> list[float] | None:
    """A list of floats iff ``raw`` is a non-empty, ALL-numeric sequence.

    Non-finite entries (NaN/±Inf) are PRESERVED -- a diverged curve is
    evidence, not missing data (see the module docstring).
    """
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    if not all(_is_number(v) for v in raw):
        return None
    return [float(v) for v in raw]


def _first_series(src: Mapping[str, Any], keys: Sequence[str]) -> list[float] | None:
    for key in keys:
        series = _numeric_series(src.get(key))
        if series is not None:
            return series
    return None


def _curve_from_mapping(src: Any) -> tuple[str, bool, list[CurvePoint]] | None:
    """``(metric, higher_is_better, points)`` from ONE mapping, or ``None``.

    Requires an EXPLICIT, exactly length-matched step/epoch series. A curve
    with no step axis (or a mismatched one) is refused rather than fabricated
    from list indices: two attempts logging at different cadences would then
    be compared off-by-a-factor, which is precisely how a healthy run gets
    killed. No axis -> no comparison -> no kill.

    Metric precedence is reward-then-loss, so the direction is always known
    (an unknown direction is what ``DoomedComparator`` refuses to fire on).
    """
    if not isinstance(src, Mapping):
        return None
    steps = _first_series(src, _CURVE_STEP_KEYS)
    if steps is None:
        return None

    values = _first_series(src, _CURVE_REWARD_KEYS)
    metric, higher_is_better = "reward", True
    if values is None:
        values = _first_series(src, _CURVE_LOSS_KEYS)
        metric, higher_is_better = "loss", False
    if values is None:
        return None

    if len(steps) != len(values) or len(steps) < _MIN_CURVE_POINTS:
        return None
    if not all(math.isfinite(s) for s in steps):
        return None  # a non-finite STEP is a broken axis, not evidence

    # Dedupe by step (last write wins), then sort ascending -- the axis the
    # comparator's last-value-carried-forward lookup assumes.
    by_step: dict[int, float] = {}
    for step, value in zip(steps, values):
        by_step[int(step)] = float(value)
    points = [CurvePoint(step, by_step[step]) for step in sorted(by_step)]
    if len(points) < _MIN_CURVE_POINTS:
        return None
    return metric, higher_is_better, points[-_MAX_CURVE_POINTS:]


def _leaf_curve(leaf: Any) -> tuple[str, bool, list[CurvePoint]] | None:
    """A curve from a per_model leaf: the nested ``training_curves`` dict
    first (the aggregated ``metrics.json`` shape), then the leaf's own flat
    ``reward_history``/``loss_history``/``step`` keys (the per-cell
    ``training_curves.json`` shape)."""
    if not isinstance(leaf, Mapping):
        return None
    nested = leaf.get("training_curves")
    if isinstance(nested, Mapping):
        curve = _curve_from_mapping(nested)
        if curve is not None:
            return curve
    return _curve_from_mapping(leaf)


def _walk_curves(node: Any, prefix: tuple[str, ...], out: dict[str, CellCurve], depth: int = 0) -> None:
    """Collect ``key -> CellCurve`` from a per_model-shaped subtree (flat OR
    nested model/env/baseline). First key wins -- callers feed sources in
    precedence order."""
    if not isinstance(node, Mapping) or depth > 4:
        return
    curve = _leaf_curve(node)
    if curve is not None and prefix:
        key = "/".join(prefix)
        if key not in out:
            metric, higher_is_better, points = curve
            out[key] = CellCurve(
                key=key, metric=metric, higher_is_better=higher_is_better, points=tuple(points)
            )
        return
    for name in sorted(node):
        child = node.get(name)
        if isinstance(child, Mapping):
            _walk_curves(child, (*prefix, str(name)), out, depth + 1)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- absent/torn/unparseable artifact -> no evidence
        return None


def _collect_from_metrics(path: Path, out: dict[str, CellCurve]) -> None:
    data = _read_json(path)
    if not isinstance(data, Mapping):
        return
    per_model = data.get("per_model")
    _walk_curves(per_model if isinstance(per_model, Mapping) else data, (), out)


def read_cell_curves(code_dir: Any) -> dict[str, CellCurve]:
    """Every measured per-cell training curve persisted under a run's
    ``code/`` tree, keyed by structural cell coordinate.

    Sources, in precedence order (a key claimed by an earlier source is never
    overwritten by a later one, so the reading is deterministic):

    1. ``code/metrics.json`` (the aggregated ``per_model`` tree)
    2. ``code/training_curves.json`` (the sidecar ``figure_sidecars`` writes)
    3. ``code/outputs/<cell>/**/{metrics,training_curves}.json`` -- the
       PER-CELL artifacts, which are the only curve evidence that exists
       MID-FLIGHT (the aggregated files above are written at the end of a
       ``run_experiment`` call). Keyed by the cell's ``outputs/``-relative
       directory path, derived identically on both sides.

    Never raises. An empty dict means "no comparable evidence", which every
    caller must treat as "do not kill".
    """
    out: dict[str, CellCurve] = {}
    try:
        code_path = Path(code_dir)
        if not code_path.is_dir():
            return {}

        _collect_from_metrics(code_path / "metrics.json", out)

        sidecar = _read_json(code_path / "training_curves.json")
        if isinstance(sidecar, Mapping):
            _walk_curves(sidecar, (), out)

        outputs = code_path / "outputs"
        if outputs.is_dir():
            paths: list[Path] = []
            for name in ("metrics.json", "training_curves.json"):
                paths.extend(p for p in outputs.rglob(name) if p.is_file())
            for path in sorted(paths)[:_MAX_CURVE_FILES]:
                cell_key = path.parent.relative_to(outputs).as_posix()
                if not cell_key or cell_key == ".":
                    continue
                data = _read_json(path)
                if not isinstance(data, Mapping):
                    continue
                per_model = data.get("per_model")
                if isinstance(per_model, Mapping):
                    _walk_curves(per_model, (cell_key,), out)
                    continue
                curve = _leaf_curve(data)
                if curve is not None and cell_key not in out:
                    metric, higher_is_better, points = curve
                    out[cell_key] = CellCurve(
                        key=cell_key,
                        metric=metric,
                        higher_is_better=higher_is_better,
                        points=tuple(points),
                    )
    except Exception:  # noqa: BLE001 -- the evidence reader NEVER breaks a run
        return {}
    return out


# ===========================================================================
# Baseline selection: which COMPLETED attempt's curves are the reference?
# Pure function of the campaign ledger rows (plain dicts) + an injected
# code-dir resolver. EVIDENCE-ranked, never score-ranked.
# ===========================================================================


def _evidence_count(assessment: Mapping[str, Any]) -> int:
    """True evidence predicates, excluding the aggregate ``run_level_clean``
    (derived from the others -- counting it too would double-weight). Mirrors
    ``campaign_policy.evidence_count`` over the raw ledger dict."""
    predicates = assessment.get("evidence_predicates")
    if not isinstance(predicates, Mapping):
        return 0
    return sum(1 for key, value in predicates.items() if key != "run_level_clean" and value)


def select_baseline(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude_attempt_n: int,
    resolve_code_dir: Callable[[int], str | None],
) -> tuple[int, dict[str, CellCurve]] | None:
    """The reference attempt for the doomed comparison: ``(attempt_n,
    curves)``, or ``None`` when nothing qualifies (in which case NO kill is
    possible -- attempt 1 always lands here).

    Eligibility is a TRUST filter, not a quality one:
      * assessed (a launched-but-unassessed attempt has no verdict),
      * never the attempt currently in flight,
      * NOT hard-quarantined -- a fabrication-flagged attempt must never
        become the yardstick a real run is killed against,
      * NOT itself doomed-killed -- an attempt we cut short is not a
        completed reference curve,
      * and it must actually have readable curves on disk.

    Ranking is by EVIDENCE count (desc), then attempt_n (asc) -- deliberately
    NOT by ``final_report.score``: the score is an LLM grade, and the red-line
    invariant says no money/trust decision may key on it.
    """
    latest_assessed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "assessed":
            continue
        attempt_n = row.get("attempt_n")
        assessment = row.get("assessment")
        if isinstance(attempt_n, int) and isinstance(assessment, Mapping):
            latest_assessed[attempt_n] = assessment

    candidates = [
        (attempt_n, assessment)
        for attempt_n, assessment in latest_assessed.items()
        if attempt_n != exclude_attempt_n
        and not assessment.get("hard_quarantined")
        and assessment.get("failure_class") != DOOMED_KILLED_CLASS
    ]
    candidates.sort(key=lambda item: (-_evidence_count(item[1]), item[0]))

    for attempt_n, _assessment in candidates:
        try:
            code_dir = resolve_code_dir(attempt_n)
        except Exception:  # noqa: BLE001 -- a resolver failure is simply "no baseline"
            continue
        if not code_dir:
            continue
        curves = read_cell_curves(code_dir)
        curves = {k: c for k, c in curves.items() if len(c.points) >= _MIN_CURVE_POINTS}
        if curves:
            return attempt_n, curves
    return None


# ===========================================================================
# DoomedWatch: the pure multi-cell aggregator. One instance per in-flight
# attempt; fed the live curve set once per poll.
# ===========================================================================


@dataclass(frozen=True)
class KillVerdict:
    """Why the campaign is about to stop paying. Every field is machine-
    checkable and traceable back to a measured artifact -- there is nowhere
    in this record for an LLM judgement to hide."""

    reason: str
    cells: tuple[str, ...]
    observations: int
    margin: float
    polls_required: int
    min_progress: float
    detail: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "cells": list(self.cells),
            "observations": self.observations,
            "margin": self.margin,
            "polls_required": self.polls_required,
            "min_progress": self.min_progress,
            "detail": list(self.detail),
        }


class DoomedWatch:
    """Per-attempt aggregator over one :class:`DoomedComparator` per MATCHED
    cell. Call :meth:`observe` once per poll with the live curve set.

    Adds the two backstops a single comparator cannot express:

    * **Staleness.** A cell whose latest point has not ADVANCED past the last
      one fed is not re-fed. A quiet eval phase (or any pause in curve
      writing) therefore produces no observations at all: it can neither
      build a streak nor break one. Without this, six identical polls of one
      stale bad reading would kill a run that is merely busy.
    * **Consensus.** The kill needs EVERY matched cell that has a confident
      reading to be firing simultaneously. One healthy matched cell vetoes
      it -- the same "any good leaf means the run is not degenerate" shape as
      ``no_learning_signal`` and ``dead_training_guard``.

    Plus a global floor: no kill before ``polls_required`` ADVANCING polls
    have been seen at all, so a single bad reading can never kill regardless
    of how the per-cell streaks line up.
    """

    def __init__(
        self,
        baseline: Mapping[str, CellCurve],
        *,
        margin: float = _DEFAULT_MARGIN,
        polls_required: int = _DEFAULT_POLLS_REQUIRED,
        min_progress: float = _DEFAULT_MIN_PROGRESS,
    ) -> None:
        self.baseline = dict(baseline)
        self.margin = margin
        self.polls_required = max(1, int(polls_required))
        self.min_progress = min_progress
        self._comparators: dict[str, DoomedComparator] = {}
        self._fired: dict[str, bool] = {}
        self._detail: dict[str, str] = {}
        self._last_step: dict[str, int] = {}
        self._observations = 0

    @property
    def observations(self) -> int:
        """Polls in which at least one matched cell advanced to a new step."""
        return self._observations

    @property
    def matched_cells(self) -> tuple[str, ...]:
        return tuple(sorted(self._fired))

    def observe(self, live: Mapping[str, CellCurve]) -> KillVerdict | None:
        """Feed one poll's worth of measured evidence. ``None`` unless every
        conservative condition in the module docstring is now satisfied."""
        advanced = False

        for key in sorted(live):
            live_curve = live[key]
            base = self.baseline.get(key)
            if base is None:
                continue  # unmatched cell -- never compared, never killed on
            if base.metric != live_curve.metric or base.higher_is_better != live_curve.higher_is_better:
                continue  # incomparable pair (different metric/direction)
            if len(live_curve.points) < _MIN_CURVE_POINTS or len(base.points) < _MIN_CURVE_POINTS:
                continue

            point = live_curve.latest
            last = self._last_step.get(key)
            if last is not None and point.step <= last:
                continue  # STALE: not a new observation (quiet eval / paused writer)
            self._last_step[key] = point.step
            advanced = True

            comparator = self._comparators.get(key)
            if comparator is None:
                comparator = DoomedComparator(
                    best_curve=base.points,
                    higher_is_better=base.higher_is_better,
                    # The reference run COMPLETED, so its own final step is the
                    # honest "planned steps" for this cell -- no hyperparameter
                    # parsing, no guessing, and the progress floor below is
                    # therefore expressed against a real measured total.
                    planned_steps=base.total_steps,
                    margin=self.margin,
                    polls_required=self.polls_required,
                    min_progress=self.min_progress,
                )
                self._comparators[key] = comparator

            fired = comparator.observe(point)
            self._fired[key] = fired
            if fired:
                self._detail[key] = self._explain(base, point)

        if advanced:
            self._observations += 1

        matched = self.matched_cells
        if not matched:
            return None
        if self._observations < self.polls_required:
            return None  # global floor: never on a single (or few) readings
        if not all(self._fired[key] for key in matched):
            return None  # consensus: one healthy matched cell vetoes the kill

        return KillVerdict(
            reason="curve_below_best_completed_attempt",
            cells=matched,
            observations=self._observations,
            margin=self.margin,
            polls_required=self.polls_required,
            min_progress=self.min_progress,
            detail=tuple(self._detail[key] for key in matched if key in self._detail),
        )

    def _explain(self, base: CellCurve, point: CurvePoint) -> str:
        best_value = _value_at_or_before(base.points, point.step)
        if best_value is None:
            return f"{base.key}: {base.metric}={point.value:.4g} at step {point.step} (no baseline point)"
        gap = _relative_gap(point.value, best_value, higher_is_better=base.higher_is_better)
        gap_str = "non-finite (diverged)" if math.isinf(gap) else f"{gap:.2f}"
        return (
            f"{base.key}: measured {base.metric}={point.value:.4g} at step {point.step} vs "
            f"baseline {best_value:.4g} at the same step -- relative gap {gap_str} "
            f">= margin {self.margin:.2f}, sustained {self.polls_required} consecutive "
            f"advancing polls"
        )


def watch_from_env(baseline: Mapping[str, CellCurve]) -> DoomedWatch:
    """Build a :class:`DoomedWatch` reading the three tunable env knobs
    (spec §10.3). Callers still gate WHETHER to construct one on
    :func:`enabled` -- this function does not read ``DOOMED_KILL_ENV``."""
    return DoomedWatch(
        baseline,
        margin=_env_float(DOOMED_MARGIN_ENV, _DEFAULT_MARGIN),
        polls_required=_env_int(DOOMED_POLLS_ENV, _DEFAULT_POLLS_REQUIRED),
        min_progress=_env_float(DOOMED_MIN_PROGRESS_ENV, _DEFAULT_MIN_PROGRESS),
    )


__all__ = [
    "CellCurve",
    "CurvePoint",
    "DOOMED_KILLED_CLASS",
    "DOOMED_KILL_ENV",
    "DOOMED_MARGIN_ENV",
    "DOOMED_MIN_PROGRESS_ENV",
    "DOOMED_POLLS_ENV",
    "DoomedComparator",
    "DoomedWatch",
    "KillVerdict",
    "enabled",
    "from_env",
    "read_cell_curves",
    "select_baseline",
    "watch_from_env",
]
