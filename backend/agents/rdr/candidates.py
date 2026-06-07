"""BES competing-candidates data model + SELECT (spec 2026-06-07, default OFF).

A :class:`Candidate` is one of N isolated attempts at a single cluster's
sub-goal. The pool is scored STATICALLY — the leaf scorer reads each candidate's
scratch dir, no GPU; the expensive experiment runs ONCE on the winner-merged
``code/``. The best candidate is selected by ``cluster_score`` (tie-break: fewest
failed leaves, then earliest). Evolve/splice is deferred to v2 (see
``docs/superpowers/specs/2026-06-07-bes-integration/phase-3-bes-on-rdr.md`` §4).

This module is pure (no I/O, no LLM) and is import-inert until ``bes_enabled``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.agents.rdr.models import Artifacts


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


__all__ = ["Candidate", "select_best"]
