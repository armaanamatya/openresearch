"""Run-directory retention: the GC the ``.preserved`` contract promised.

Core logic moved here from ``scripts/prune_runs.py`` (audit 2026-06-10) so the
server can run it periodically; the script remains the manual CLI and imports
from this module.

A run directory is pruned only when ALL of:
  - its ``demo_status.json`` status is terminal
    (completed | failed | stopped | killed | interrupted);
  - its last activity (max mtime of demo_status.json / final_report.json /
    dashboard_events.jsonl / the dir itself) is older than the cutoff;
  - it does NOT contain a ``.preserved`` marker;
  - its name is not in ``keep``.

Unreadable/missing demo_status.json counts as NOT terminal (skip — a run that
never wrote status may still be starting up; the liveness sweep, not this
module, is responsible for classifying it).

Periodic mode is **opt-in**: ``OPENRESEARCH_RUNS_RETENTION_DAYS`` unset, empty,
or ``0`` disables it entirely (the shipped default — deleting run artifacts is
an operator decision). When set to a positive float, the app lifespan starts a
daemon thread that prunes terminal runs older than that many days once per
``interval_s`` (default hourly), honoring every guard above.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_TERMINAL = {"completed", "failed", "stopped", "killed", "interrupted"}

_RETENTION_ENV = "OPENRESEARCH_RUNS_RETENTION_DAYS"


def _last_activity_s(run_dir: Path) -> float:
    """Newest mtime among the dir and its cheap status artifacts."""
    candidates = [run_dir]
    for name in ("demo_status.json", "final_report.json", "dashboard_events.jsonl"):
        p = run_dir / name
        if p.exists():
            candidates.append(p)
    return max(p.stat().st_mtime for p in candidates)


def _status_of(run_dir: Path) -> str | None:
    try:
        payload = json.loads((run_dir / "demo_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(payload.get("status") or "") or None


def prune(
    runs_root: Path,
    *,
    older_than_days: float,
    delete: bool,
    keep: frozenset[str] = frozenset(),
    log=print,
) -> list[Path]:
    """Return the run dirs selected for pruning (deleted when delete=True).

    ``log`` defaults to ``print`` for the CLI; the periodic sweep passes
    ``logger.info`` so server output goes through logging.
    """
    if not runs_root.is_dir():
        log(f"[prune_runs] runs root does not exist: {runs_root}")
        return []
    cutoff = time.time() - older_than_days * 86400
    selected: list[Path] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        why_kept: str | None = None
        if run_dir.name in keep:
            why_kept = "--keep"
        elif (run_dir / ".preserved").exists():
            why_kept = ".preserved marker"
        else:
            status = _status_of(run_dir)
            if status is None:
                why_kept = "no readable demo_status.json"
            elif status not in _TERMINAL:
                why_kept = f"status={status} (not terminal)"
            elif _last_activity_s(run_dir) > cutoff:
                why_kept = "younger than cutoff"
        if why_kept is not None:
            log(f"[prune_runs] keep   {run_dir.name}  ({why_kept})")
            continue
        selected.append(run_dir)
        if delete:
            shutil.rmtree(run_dir)
            log(f"[prune_runs] DELETED {run_dir.name}")
        else:
            log(f"[prune_runs] would delete {run_dir.name}  (re-run with --delete)")
    return selected


def retention_days_from_env() -> float | None:
    """Parse the opt-in env knob. None = retention disabled (the default)."""
    raw = (os.environ.get(_RETENTION_ENV) or "").strip()
    if not raw:
        return None
    try:
        days = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number — retention disabled", _RETENTION_ENV, raw)
        return None
    return days if days > 0 else None


def sweep_once(runs_root: Path, *, older_than_days: float) -> list[Path]:
    """One destructive retention pass (the unit the periodic thread runs)."""
    return prune(
        runs_root,
        older_than_days=older_than_days,
        delete=True,
        log=logger.info,
    )


def periodic_retention_sweep(
    runs_root: Path,
    *,
    stop_event: threading.Event,
    interval_s: float = 3600.0,
) -> threading.Thread | None:
    """Start the hourly retention daemon iff the env knob opts in.

    Returns the thread, or None when retention is disabled. Mirrors
    ``run_liveness.periodic_liveness_sweep``: daemon thread, stop via the
    event, every pass fail-soft.
    """
    days = retention_days_from_env()
    if days is None:
        return None

    def _loop() -> None:
        logger.info(
            "runs retention: pruning terminal runs older than %.1f day(s) every %ss",
            days, int(interval_s),
        )
        while not stop_event.wait(timeout=interval_s):
            try:
                sweep_once(runs_root, older_than_days=days)
            except Exception:  # noqa: BLE001 — retention must never crash the server
                logger.exception("runs retention sweep failed (will retry)")

    thread = threading.Thread(target=_loop, name="runs-retention", daemon=True)
    thread.start()
    return thread
