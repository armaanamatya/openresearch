"""cell_scheduler — shared pure helpers for gpu_cell_runner and k8s_job_cell_runner.

Contains the placement-independent logic that both cell runners need identically:

* ``CELL_MANIFEST_NAME`` — authoritative filename for per-cell resume records.
* ``STATUS_*`` — status-vocabulary string constants (ok / oom_failed / skipped /
  error / timeout) so callers never hard-code string literals.
* ``CellResult`` — result record shared by both runners.
* ``headline_metric`` — extract a single headline scalar from a flat metrics dict.
* ``load_cell_manifest`` — load ``cell_manifest.json`` from an output directory.
* ``should_skip_cell`` — resume skip predicate (Track B): status=ok +
  fingerprint-match + not force-listed.
* ``write_cell_manifest`` — write the authoritative per-cell resume manifest
  (fail-soft).
* ``is_resume_armed`` — read the ``REPROLAB_RESUME_CELLS`` env var once.
* ``deadline_from_timeout`` — compute a monotonic deadline from an optional
  timeout in seconds.
* ``clamp_cell_timeout`` — clamp a per-cell timeout to the remaining matrix
  budget.

Stdlib-only — no third-party imports, no LLM calls, no I/O beyond the local
filesystem.  Mirror of ``cell_matrix.py``'s purity guarantee.

Both ``gpu_cell_runner`` and ``k8s_job_cell_runner`` import from here.  Do NOT
add provider-specific or placement-specific logic here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    # Constants
    "CELL_MANIFEST_NAME",
    "STATUS_OK",
    "STATUS_OOM_FAILED",
    "STATUS_SKIPPED",
    "STATUS_ERROR",
    "STATUS_TIMEOUT",
    "STATUS_TRAINING_DIVERGED",
    # Data type
    "CellResult",
    # Pure helpers
    "headline_metric",
    "load_cell_manifest",
    "should_skip_cell",
    "write_cell_manifest",
    "is_resume_armed",
    "cap_overall_budget",
    "deadline_from_timeout",
    "clamp_cell_timeout",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File name of the per-cell resume manifest, written into each cell's output_dir.
CELL_MANIFEST_NAME = "cell_manifest.json"

# Status-vocabulary string constants — use these everywhere instead of literals.
STATUS_OK = "ok"
STATUS_OOM_FAILED = "oom_failed"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
STATUS_TRAINING_DIVERGED = "training_diverged"  # dead-training early-stop (repairable)


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

class CellResult:
    """Result record for a single training cell.

    Attributes:
        cell_id:  Identifier matching the input ``cells`` entry.
        status:   One of the ``STATUS_*`` constants:
                  ``"ok"`` | ``"oom_failed"`` | ``"timeout"`` | ``"error"`` |
                  ``"skipped"`` (resume: prior ok cell reused without launching).
        metrics:  Dict loaded from the cell's ``metrics.json``, or ``None``.
        gpu:      Physical GPU id the cell ran on (last attempt).
        retries:  Number of OOM retries attempted (0 = first attempt succeeded
                  or failed with a non-OOM error).
        error:    Stderr snippet / exception message, or ``None`` on success.
    """

    __slots__ = ("cell_id", "status", "metrics", "gpu", "retries", "error")

    def __init__(
        self,
        *,
        cell_id: str,
        status: str,
        metrics: dict[str, Any] | None,
        gpu: str,
        retries: int,
        error: str | None,
    ) -> None:
        self.cell_id = cell_id
        self.status = status
        self.metrics = metrics
        self.gpu = gpu
        self.retries = retries
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "gpu": self.gpu,
            "retries": self.retries,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def headline_metric(metrics: dict[str, Any] | None) -> Any:
    """Extract a single headline scalar from a cell's flat ``metrics.json``.

    Prefers an explicit ``"metric"`` key; falls back to ``"reward_mean"`` then
    ``"accuracy"``.  Returns the value only when it is a real number (``int`` /
    ``float`` but not ``bool``); otherwise ``None``.  Never raises.
    """
    if not isinstance(metrics, dict):
        return None
    for key in ("metric", "reward_mean", "accuracy"):
        val = metrics.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            return val
    return None


def load_cell_manifest(output_dir: Path) -> dict[str, Any] | None:
    """Load ``cell_manifest.json`` from ``output_dir``, None on any failure."""
    mf = output_dir / CELL_MANIFEST_NAME
    if not mf.is_file():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def should_skip_cell(
    cell_id: str,
    output_dir: Path,
    fingerprints: dict[str, str],
    force_cells: set[str],
) -> bool:
    """Return True iff a prior run's manifest authorises skipping this cell.

    Skip iff the prior ``cell_manifest.json`` exists with ``status == "ok"`` AND
    its stored ``fingerprint`` equals the current ``fingerprints[cell_id]`` AND
    the cell is not force-listed.  A cell with no current fingerprint, a missing
    or non-ok manifest, a fingerprint mismatch, or a force-list membership is
    NOT skipped (it re-runs).  Pure / fail-soft — never raises.
    """
    if cell_id in force_cells:
        return False
    current_fp = fingerprints.get(cell_id)
    if not current_fp:
        # No fingerprint to compare → cannot safely assert the cell is unchanged.
        return False
    manifest = load_cell_manifest(output_dir)
    if not manifest or manifest.get("status") != STATUS_OK:
        return False
    return manifest.get("fingerprint") == current_fp


def write_cell_manifest(
    output_dir: Path,
    *,
    caller: str = "cell_scheduler",
    cell_id: str,
    status: str,
    fingerprint: str | None,
    metrics: dict[str, Any] | None,
    retries: int,
    now_iso: str | None,
) -> None:
    """Write the authoritative per-cell ``cell_manifest.json`` (fail-soft).

    The manifest is the harness-owned record of a cell's terminal outcome — the
    sole input to the resume skip predicate on a later run.  ``completed_at`` is
    omitted entirely when ``now_iso`` is None (this module supplies no clock of
    its own).  A write failure is logged and swallowed: a missing manifest just
    means the cell re-runs next time, which is the safe default.

    Args:
        output_dir:   The cell's output directory (``output_root/<cell_id>/``).
        caller:       Module name string for log messages (e.g. ``"gpu_cell_runner"``).
        cell_id:      Cell identifier.
        status:       Terminal status (one of the ``STATUS_*`` constants).
        fingerprint:  Cell fingerprint recorded from the current run, or None.
        metrics:      Parsed metrics dict, or None.
        retries:      Number of OOM retries attempted.
        now_iso:      ISO-8601 timestamp string for ``completed_at``, or None.
    """
    manifest: dict[str, Any] = {
        "cell_id": cell_id,
        "status": status,
        "fingerprint": fingerprint,
        "metric": headline_metric(metrics),
        "retries": retries,
    }
    if now_iso is not None:
        manifest["completed_at"] = now_iso
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CELL_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning(
            "%s: could not write %s for cell=%s: %s",
            caller, CELL_MANIFEST_NAME, cell_id, exc,
        )


def is_resume_armed() -> bool:
    """Return True when ``REPROLAB_RESUME_CELLS`` env var is truthy.

    Centralises the single env-var read so both runners stay in sync if the
    variable name ever changes.
    """
    return bool(os.environ.get("REPROLAB_RESUME_CELLS", "").strip())


def cap_overall_budget(
    overall_timeout_s: float | None,
    remaining_run_s: float | None,
    *,
    reserve_s: float = 2700.0,
    floor_s: float = 900.0,
) -> float | None:
    """Cap a matrix's overall budget by the RUN's remaining wall clock.

    2026-06-10 Adam v6: the cell grid got ``per_cell_timeout × waves`` of
    budget with no knowledge of the run-level deadline, ran straight into the
    14h watchdog, and the process was hard-killed mid-cell (salvaged at 0.151)
    — when the runner's own deadline machinery could have TRIMMED the tail
    cells to honest ``timeout`` leaves at the boundary and returned a
    scoreable partial.

    ``reserve_s`` is held back for verify + report after the grid. The result
    never drops below ``floor_s`` (or half the remaining time, if smaller) so
    a late-run matrix still gets a real chance rather than an instant trim.
    Returns ``overall_timeout_s`` unchanged when the remaining time is unknown
    or non-positive (no-limit semantics preserved).
    """
    if remaining_run_s is None or remaining_run_s <= 0:
        return overall_timeout_s
    budget = max(remaining_run_s - reserve_s, min(remaining_run_s * 0.5, floor_s))
    if overall_timeout_s is None or overall_timeout_s <= 0:
        return budget
    return min(overall_timeout_s, budget)


def deadline_from_timeout(timeout_s: float | None) -> float | None:
    """Return a monotonic deadline for ``timeout_s`` seconds from now, or None.

    Returns None when ``timeout_s`` is None or <= 0, preserving the "no limit"
    semantics callers already use.
    """
    if timeout_s is not None and timeout_s > 0:
        return time.monotonic() + timeout_s
    return None


def clamp_cell_timeout(
    per_cell_timeout_s: float | None,
    overall_deadline: float | None,
) -> float | None:
    """Clamp a per-cell timeout to the remaining matrix budget.

    Returns the effective per-cell timeout in seconds (or None for no limit),
    guaranteeing that a single in-flight cell cannot run past the overall matrix
    deadline.

    Args:
        per_cell_timeout_s:  Per-cell wall-clock limit in seconds, or None.
        overall_deadline:    Monotonic deadline for the whole matrix, or None.

    Returns:
        Effective per-cell timeout in seconds (at least 1.0 when clamped),
        or None when both inputs are None.
    """
    if overall_deadline is None:
        return per_cell_timeout_s
    remaining = overall_deadline - time.monotonic()
    if per_cell_timeout_s is None:
        return max(1.0, remaining)
    return max(1.0, min(per_cell_timeout_s, remaining))
