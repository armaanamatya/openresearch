"""BES competing-candidates data model + SELECT (spec 2026-06-07, default OFF).

A :class:`Candidate` is one of N isolated attempts at a single cluster's
sub-goal. The pool is scored STATICALLY — the leaf scorer reads each candidate's
scratch dir, no GPU; the expensive experiment runs ONCE on the winner-merged
``code/``. The best candidate is selected by ``cluster_score`` (tie-break: fewest
failed leaves, then earliest). Evolve/splice is deferred to v2 (see
``docs/superpowers/specs/2026-06-07-bes-integration/phase-3-bes-on-rdr.md`` §4).

This module is pure (no I/O, no LLM) and is import-inert until ``bes_enabled``.

Smoke-gated SELECT (spec 2026-06-16 §D2, default OFF — gated on
``OPENRESEARCH_BES_SMOKE_SELECT``): the plain code-only static grade above is blind
to the *runtime* axis (a torch-repin or a VAE device-side-assert) — exactly
where this repo's failures live. An all-statically-faithful-but-non-runnable
pool still "selects" a winner, and tiny score spreads (< ~σ_grader) are
coin-flips that bank grader noise. :func:`select_best_gated` adds, all
flag-gated and pure (the only I/O is a read-only AST scan of the snapshot):

1. a **construct/import smoke** per candidate (:func:`smoke_check_candidate`,
   reusing ``preflight_ast.scan_code_dir`` — the existing AST completeness gate)
   so a candidate whose code carries a guaranteed runtime crash (a missing
   method / undefined name / broken teacher-student env interface — it cannot
   even construct/import) cannot outrank a runnable one;
2. a **deterministic sub-σ tie-break** (import-smoke pass → AST-completeness →
   lowest candidate index) when the top-2 score spread is below
   ``OPENRESEARCH_BES_SELECT_MIN_SPREAD`` (default 0.05, a σ_grader proxy) — instead
   of banking the luckier grader draw;
3. a **degenerate-pool** verdict when no candidate is selectable (every one
   failed or smoke-failed) so the caller emits ``degenerate_pool`` and falls
   through to single-shot repair instead of "selecting" a doomed winner.

``OPENRESEARCH_BES_SMOKE_SELECT`` unset/off → :func:`select_best_gated` is a thin
pass-through to :func:`select_best`, byte-for-byte the prior behaviour.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from backend.agents.rdr.models import Artifacts

logger = logging.getLogger(__name__)

# Flag gate for the whole §D2 smoke-gated SELECT layer. Default OFF: when unset
# (or 0/false/no/off) select_best_gated == select_best, byte-for-byte.
ENV_SMOKE_SELECT = "OPENRESEARCH_BES_SMOKE_SELECT"
# Top-2 score spread below this (a σ_grader proxy) is treated as a tie and
# resolved on a deterministic signal instead of banking grader noise.
ENV_SELECT_MIN_SPREAD = "OPENRESEARCH_BES_SELECT_MIN_SPREAD"
_DEFAULT_MIN_SPREAD = 0.05


def _flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Candidate:
    """One isolated attempt at a cluster's sub-goal, scored before merge."""

    candidate_id: str            # f"{cluster.id}#{n}"
    cluster_id: str
    scratch_dir: Path            # project_dir/"candidates"/candidate_id (its code/ lives here)
    artifacts: Artifacts
    parent_id: str | None = None  # provenance for splice (v2)
    score: float | None = None
    failed_leaves: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SmokeResult:
    """The runtime-axis smoke verdict for one candidate's snapshot.

    ``runnable`` is the gate: a candidate whose code carries a *hard* AST
    violation (a guaranteed ``AttributeError`` / ``NameError`` / broken
    teacher-student env interface — i.e. it cannot even construct/import) is
    ``runnable=False`` and ranks below any runnable peer in the gated SELECT.
    ``ast_complete`` (no violations at all, hard OR soft) is the finer
    tie-break signal. ``checked=False`` means the scan could not run (no code
    dir / scan error) — fail-soft, treated as "not disqualifying".
    """

    candidate_id: str
    checked: bool
    runnable: bool
    ast_complete: bool
    hard_violations: int
    soft_violations: int
    detail: str = ""


def smoke_check_candidate(candidate_id: str, code_dir: Path) -> SmokeResult:
    """Run the construct/import smoke over one candidate's snapshot ``code/``.

    Pure + read-only: this reuses the existing ``preflight_ast.scan_code_dir``
    AST gate (the same one that catches the 2026-05-31 SDAR env-interface
    ``AttributeError`` and the missing-method/undefined-name class before any
    GPU dispatch). No code is executed and nothing is written — it is safe to
    run on the host against a candidate snapshot. Any scan error degrades to
    ``checked=False`` (never disqualifying), so an un-scannable candidate is
    judged on its static grade exactly as today.
    """
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return SmokeResult(
            candidate_id=candidate_id, checked=False, runnable=True,
            ast_complete=False, hard_violations=0, soft_violations=0,
            detail="no code dir",
        )
    try:
        from backend.agents.rlm.preflight_ast import scan_code_dir

        violations = scan_code_dir(code_dir)
    except Exception:  # noqa: BLE001 — smoke must never cost the run its SELECT
        logger.debug("candidates: smoke scan failed for %s", candidate_id, exc_info=True)
        return SmokeResult(
            candidate_id=candidate_id, checked=False, runnable=True,
            ast_complete=False, hard_violations=0, soft_violations=0,
            detail="scan error",
        )
    hard = sum(1 for v in violations if getattr(v, "severity", "hard") == "hard")
    soft = len(violations) - hard
    detail = ""
    if violations:
        first = violations[0]
        detail = (
            f"{getattr(first, 'file', '?')}:{getattr(first, 'line', 0)} "
            f"{getattr(first, 'detail', '')}"
        )[:200]
    return SmokeResult(
        candidate_id=candidate_id,
        checked=True,
        runnable=hard == 0,
        ast_complete=not violations,
        hard_violations=hard,
        soft_violations=soft,
        detail=detail,
    )


def select_best(
    candidates: list[Candidate],
    *,
    select_metric: str = "cluster_score",
) -> Candidate | None:
    """Return the winning candidate, or ``None`` for an empty pool.

    A failed candidate (``artifacts.failed`` or ``score is None``) always ranks
    below any scored one. ``cluster_score`` (default): highest score first, then
    fewest failed leaves. ``failed_leaves``: fewest failed leaves first, then
    highest score. Ties resolve to the EARLIEST candidate (``max`` returns the
    first element achieving the maximal key), so selection is deterministic.
    """
    if not candidates:
        return None

    def _key(c: Candidate) -> tuple:
        ok = 0 if (c.artifacts.failed or c.score is None) else 1
        score = c.score if c.score is not None else -1.0
        neg_failed = -len(c.failed_leaves)
        if select_metric == "failed_leaves":
            return (ok, neg_failed, score)
        return (ok, score, neg_failed)

    return max(candidates, key=_key)


def _scored_ok(c: Candidate) -> bool:
    return not (c.artifacts.failed or c.score is None)


def _candidate_index(c: Candidate, fallback: int) -> int:
    """The canonical integer index for the 'lowest candidate index' tie-break.

    Both BES paths build ids as ``<prefix>#<i>`` (``rlm_impl#0`` /
    ``<cluster>#0``), so the trailing integer IS the candidate's position in the
    pool. Parsing it (rather than the input-list position) makes the tie-break
    ORDER-INDEPENDENT — the same pool resolves to the same winner regardless of
    how the list is presented. ``fallback`` (the input position) is used when an
    id carries no parseable ``#<int>`` suffix.
    """
    cid = c.candidate_id or ""
    if "#" in cid:
        tail = cid.rsplit("#", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return fallback


def select_best_gated(
    candidates: list[Candidate],
    *,
    select_metric: str = "cluster_score",
    smokes: dict[str, SmokeResult] | None = None,
) -> tuple[Candidate | None, dict]:
    """Smoke-gated SELECT with a sub-σ tie-break and a degenerate-pool verdict.

    Returns ``(winner, decision)`` where ``decision`` is a structured, JSON-able
    record of HOW the winner was chosen (for the run_warning + the persisted
    pool state). Behaviour:

    * ``OPENRESEARCH_BES_SMOKE_SELECT`` off → ``(select_best(...), {"path": "legacy"})``
      — byte-for-byte the prior SELECT (``smokes`` ignored).
    * On: a candidate marked ``runnable=False`` by ``smokes`` is dropped from
      contention (a statically-faithful but non-runnable candidate can't win
      over a runnable one). If that empties the runnable set, the run is treated
      as **degenerate** — ``(None, {"degenerate": True, ...})`` — so the caller
      falls through to single-shot repair.
    * When ≥2 runnable candidates survive and the **top-2 score spread is below
      the σ_grader proxy** (``OPENRESEARCH_BES_SELECT_MIN_SPREAD``, default 0.05),
      the tie is broken DETERMINISTICALLY — import/construct-smoke pass first,
      then AST-completeness, then lowest candidate index — instead of banking
      the luckier grader draw.

    Pure: it mutates nothing and reads only the supplied ``smokes`` map; the
    caller owns running :func:`smoke_check_candidate` (one read-only AST scan
    per candidate) and emitting the ``degenerate_pool`` warning.
    """
    if not _flag_on(ENV_SMOKE_SELECT):
        return select_best(candidates, select_metric=select_metric), {"path": "legacy"}

    smokes = smokes or {}

    def _smoke(c: Candidate) -> SmokeResult:
        return smokes.get(c.candidate_id) or SmokeResult(
            candidate_id=c.candidate_id, checked=False, runnable=True,
            ast_complete=False, hard_violations=0, soft_violations=0,
        )

    # Canonical index map for the deterministic "lowest candidate index"
    # tie-break. Keyed on the parsed ``#<i>`` suffix (order-independent) with the
    # input position as the fallback, so the same pool resolves identically
    # regardless of presentation order.
    order = {id(c): _candidate_index(c, i) for i, c in enumerate(candidates)}

    # 1. Runnable, scored candidates are the only ones eligible to "win". A
    #    non-runnable candidate (hard AST violation) is disqualified even if its
    #    static code-grade is high — the runtime-axis blind spot D2 closes.
    eligible = [c for c in candidates if _scored_ok(c) and _smoke(c).runnable]
    smoke_dropped = [
        c.candidate_id for c in candidates if _scored_ok(c) and not _smoke(c).runnable
    ]

    if not eligible:
        # Either every candidate failed to implement/grade, or every survivor
        # is non-runnable. Degenerate pool → no winner; caller repairs.
        decision = {
            "path": "smoke_gated",
            "degenerate": True,
            "smoke_dropped": smoke_dropped,
            "n_total": len(candidates),
            "n_runnable": 0,
        }
        return None, decision

    ranked = sorted(
        eligible,
        key=lambda c: (float(c.score or 0.0), -order.get(id(c), 1_000_000)),
        reverse=True,
    )
    top = ranked[0]
    min_spread = _env_float(ENV_SELECT_MIN_SPREAD, _DEFAULT_MIN_SPREAD)

    if len(ranked) >= 2:
        spread = float(top.score or 0.0) - float(ranked[1].score or 0.0)
        if spread < min_spread:
            # Sub-σ spread: the top scores are inside grader noise. The tie set
            # is every eligible candidate within min_spread of the leader. Break
            # it on a DETERMINISTIC signal, not the noisy score: AST-completeness
            # → fewest violations → lowest candidate index. (All are already
            # runnable — they passed the eligibility filter — so import/construct
            # -smoke is the completeness layer here.)
            tie_set = [
                c for c in ranked
                if (float(top.score or 0.0) - float(c.score or 0.0)) < min_spread
            ]

            def _tie_key(c: Candidate) -> tuple:
                sm = _smoke(c)
                return (
                    0 if sm.ast_complete else 1,
                    sm.hard_violations,
                    sm.soft_violations,
                    order.get(id(c), 1_000_000),
                )

            winner = min(tie_set, key=_tie_key)
            decision = {
                "path": "smoke_gated",
                "tie_break": "sub_sigma",
                "spread": round(spread, 6),
                "min_spread": min_spread,
                "tie_set": [c.candidate_id for c in tie_set],
                "smoke_dropped": smoke_dropped,
                "winner_score": winner.score,
                "top_score": top.score,
            }
            return winner, decision

    decision = {
        "path": "smoke_gated",
        "tie_break": "score",
        "smoke_dropped": smoke_dropped,
        "winner_score": top.score,
        "n_runnable": len(eligible),
    }
    return top, decision


__all__ = [
    "Candidate",
    "ENV_SELECT_MIN_SPREAD",
    "ENV_SMOKE_SELECT",
    "SmokeResult",
    "select_best",
    "select_best_gated",
    "smoke_check_candidate",
]
