#!/usr/bin/env python3
"""Prune old run directories from runs/ — the GC the .preserved contract promised.

``report.py`` stamps a ``.preserved`` marker into run dirs that must survive
cleanup, and its docstring mandates that "any cleanup script that prunes
runs/ MUST skip directories carrying this file" — but no such script existed,
so runs/ grew without bound (audit 2026-06-09). This is that script.

A run directory is pruned only when ALL of:
  - its demo_status.json status is terminal
    (completed | failed | stopped | killed | interrupted);
  - its last activity (max mtime of demo_status.json / final_report.json /
    the dir itself) is older than --older-than-days;
  - it does NOT contain a .preserved marker;
  - its name is not in --keep.

Unreadable/missing demo_status.json counts as NOT terminal (skip — a run that
never wrote status may still be starting up; the liveness sweep, not this
script, is responsible for classifying it).

DRY-RUN BY DEFAULT: prints what would be deleted; pass --delete to act.

Usage:
    python scripts/prune_runs.py                       # dry-run, 14-day cutoff
    python scripts/prune_runs.py --older-than-days 30 --delete
    python scripts/prune_runs.py --runs-root /elsewhere/runs --keep prj_x
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_TERMINAL = {"completed", "failed", "stopped", "killed", "interrupted"}


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
) -> list[Path]:
    """Return the run dirs selected for pruning (deleted when delete=True)."""
    if not runs_root.is_dir():
        print(f"[prune_runs] runs root does not exist: {runs_root}", file=sys.stderr)
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
            print(f"[prune_runs] keep   {run_dir.name}  ({why_kept})")
            continue
        selected.append(run_dir)
        if delete:
            shutil.rmtree(run_dir)
            print(f"[prune_runs] DELETED {run_dir.name}")
        else:
            print(f"[prune_runs] would delete {run_dir.name}  (re-run with --delete)")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--older-than-days", type=float, default=14.0)
    parser.add_argument(
        "--delete", action="store_true",
        help="actually delete (default is a dry run that only prints)",
    )
    parser.add_argument(
        "--keep", action="append", default=[],
        help="project id to always keep (repeatable)",
    )
    args = parser.parse_args(argv)
    selected = prune(
        args.runs_root.resolve(),
        older_than_days=args.older_than_days,
        delete=args.delete,
        keep=frozenset(args.keep),
    )
    verb = "deleted" if args.delete else "would delete"
    print(f"[prune_runs] {verb} {len(selected)} run dir(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
