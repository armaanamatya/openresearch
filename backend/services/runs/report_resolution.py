"""report_resolution.py — canonical report-selection and score-extraction logic.

Single abstraction for choosing the *best* final_report.json across a run dir
(top-level + all archived attempts under ``attempts/*/final_report.json``) and
for normalising scores across schema generations:

* **Nested-rubric schema** (current): ``rubric.overall_score`` /
  ``rubric.compute_adjusted_score``.
* **Flat-score schema** (legacy ``rlm_oauth_smoke_*`` runs): top-level
  ``rubric_score`` float; ``rubric`` key absent or a list.

Public API (stable — import from here, not from leaderboard.py):
    extract_scores(report)            → (overall, adjusted)
    normalized_score(overall, adjusted) → float | None
    resolve_best_report(run_dir)      → ResolvedReport
    ResolvedReport                    dataclass

Design notes:
- ``resolve_best_report`` keeps the top-level ``final_report.json`` hot in the
  leaderboard_cache (mtime-based invalidation); archived attempts are loaded
  with plain ``json`` since they are immutable once archived.
- All I/O is fail-soft: parse errors → skip candidate; missing attempt dirs →
  zero candidates; empty run dir → empty ResolvedReport.
- O(A) extra work per project per request where A = number of attempts (small).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.services.events.leaderboard_cache import get_or_load

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def extract_scores(report: dict) -> tuple[float | None, float | None]:
    """Extract (overall_score, compute_adjusted_score) from a report dict.

    Handles three schema variants:
    1. ``rubric`` is a dict with ``overall_score`` / ``compute_adjusted_score``.
    2. ``rubric`` absent or not a dict (list, None) → fall back to top-level
       ``rubric_score`` for *both* overall and adjusted.
    3. ``overall_score`` present but ``compute_adjusted_score`` absent → adjusted
       = overall (legacy runs without the field).

    Coerces to ``float`` where possible; propagates ``None`` when the value is
    absent or non-numeric.
    """
    rubric = report.get("rubric")
    if not isinstance(rubric, dict):
        rubric = {}

    def _coerce(v) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    overall = _coerce(rubric.get("overall_score"))
    adjusted = _coerce(rubric.get("compute_adjusted_score"))

    # Flat-schema fallback: no usable rubric dict → try top-level rubric_score.
    if overall is None and adjusted is None:
        flat = _coerce(report.get("rubric_score"))
        overall = flat
        adjusted = flat

    # If adjusted is absent but overall is set → use overall as the adjusted
    # value (matches the leaderboard's existing fallback for pre-β3 reports).
    if adjusted is None and overall is not None:
        adjusted = overall

    return overall, adjusted


def normalized_score(
    overall: float | None,
    adjusted: float | None,
) -> float | None:
    """Return the single comparable score used for leaderboard ranking.

    Prefers ``adjusted`` (compute_adjusted_score) so efficient-mode and
    max-mode runs are comparable; falls back to ``overall``; returns None
    when both are absent (sort-key treats None as lowest rank).
    """
    if adjusted is not None:
        return adjusted
    return overall


_IMPL_RANK_ORDER: dict[str, int] = {"faithful": 2, "partial": 1, "broken": 0}


def two_axis_fidelity_key(report: dict) -> tuple[int, float] | None:
    """Fidelity-first ranking key for a two-axis (schema>=2) report, else None.

    Ranks by the implementation (fidelity) axis so a faithful reproduction that
    REFUTES the paper outranks a broken-but-high-rubric-score attempt — the
    replication outcome is a badge, never a rank penalty (A5 / decision 7).
    Returns ``None`` for legacy reports so their score-based ranking is
    byte-for-byte unchanged.
    """
    impl = report.get("implementation_verdict")
    schema = report.get("schema_version", 1)
    if impl is None and (not isinstance(schema, int) or schema < 2):
        return None
    rank = _IMPL_RANK_ORDER.get(impl, 0)
    repro = report.get("reproducibility")
    fid = repro.get("fidelity_score") if isinstance(repro, dict) else None
    if fid is None:
        fid = report.get("overall_score")
    try:
        fid_f = float(fid) if fid is not None else 0.0
    except (TypeError, ValueError):
        fid_f = 0.0
    return (rank, fid_f)


# ---------------------------------------------------------------------------
# ResolvedReport
# ---------------------------------------------------------------------------


@dataclass
class ResolvedReport:
    """Result of :func:`resolve_best_report`."""

    #: Winning report dict, or None when no readable report was found.
    report: dict | None
    #: Path to the winning report file, or None.
    report_path: Path | None
    #: Total number of archived attempt ``final_report.json`` files found
    #: (readable or not).  Used for the ``attempts`` column.
    attempts_total: int
    #: True when the winning report lives under ``attempts/``.
    picked_from_attempt: bool = field(default=False)

    @classmethod
    def empty(cls) -> "ResolvedReport":
        return cls(report=None, report_path=None, attempts_total=0, picked_from_attempt=False)


# ---------------------------------------------------------------------------
# resolve_best_report
# ---------------------------------------------------------------------------


def resolve_best_report(run_dir: Path) -> ResolvedReport:
    """Select the *best* final_report.json across the run dir and its attempts.

    Candidates:
    - ``run_dir/final_report.json`` (loaded via the leaderboard mtime cache).
    - ``run_dir/attempts/*/final_report.json`` (loaded with plain ``json`` —
      immutable once archived, no cache needed).

    Ranking: highest ``normalized_score(*extract_scores(report))``.
    Tie-break: ``report.get("completed_at")`` (lexicographic ISO-8601 string),
    then file mtime.

    Fail-soft: unreadable or unparseable files are skipped with a WARNING log.
    Returns :meth:`ResolvedReport.empty` when no readable report exists at all.
    """
    run_dir = Path(run_dir)

    # --- count + load attempt candidates ---
    attempts_dir = run_dir / "attempts"
    attempt_paths: list[Path] = []
    if attempts_dir.is_dir():
        for attempt_subdir in attempts_dir.iterdir():
            if not attempt_subdir.is_dir():
                continue
            ap = attempt_subdir / "final_report.json"
            if ap.exists():
                attempt_paths.append(ap)

    attempts_total = len(attempt_paths)

    # --- build candidate list ---
    # Each element: (report_dict, path, is_attempt)
    candidates: list[tuple[dict, Path, bool]] = []

    # Top-level report via the mtime cache (keeps it warm for subsequent requests).
    top_level_path = run_dir / "final_report.json"
    if top_level_path.is_file():
        top_data = get_or_load(run_dir.name, top_level_path)
        if top_data is not None:
            candidates.append((top_data, top_level_path, False))
        else:
            logger.warning(
                "report_resolution: unreadable top-level final_report.json in %s",
                run_dir.name,
            )

    # Attempt reports — plain json (immutable).
    for ap in attempt_paths:
        try:
            data = json.loads(ap.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                candidates.append((data, ap, True))
            else:
                logger.warning(
                    "report_resolution: non-dict JSON in attempt %s — skipping",
                    ap,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "report_resolution: failed to parse attempt %s — %s — skipping",
                ap,
                exc,
            )

    if not candidates:
        return ResolvedReport.empty()

    # --- pick the best candidate ---
    def _rank(item: tuple[dict, Path, bool]):
        report, path, _ = item
        ns = normalized_score(*extract_scores(report))
        # None sorts below every real score.  We're using max(), so higher is
        # better.  Represent None as (-inf, -1) and a real score as (score, 0)
        # so real scores always beat None regardless of value.
        if ns is None:
            score_key: tuple = (float("-inf"), -1)
        else:
            score_key = (ns, 0)
        # A5 — fidelity-first for two-axis reports: a faithful attempt (even one
        # that refutes the paper) outranks a broken-but-high-score attempt, so
        # the honest negative result is the one surfaced.  Legacy reports get a
        # neutral (-1, score) prefix → their score ordering is unchanged.
        ta = two_axis_fidelity_key(report)
        fidelity_key = ta if ta is not None else (-1, score_key[0])
        # Tie-break 1: completed_at (lexicographic; None sorts last).
        completed_at = report.get("completed_at") or ""
        # Tie-break 2: file mtime (newer = better).
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (fidelity_key, score_key, completed_at, mtime)

    best_report, best_path, best_is_attempt = max(candidates, key=_rank)

    return ResolvedReport(
        report=best_report,
        report_path=best_path,
        attempts_total=attempts_total,
        picked_from_attempt=best_is_attempt,
    )


__all__ = [
    "ResolvedReport",
    "extract_scores",
    "normalized_score",
    "resolve_best_report",
    "two_axis_fidelity_key",
]
