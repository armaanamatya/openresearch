#!/usr/bin/env python3
"""Garbage-collect heavy reproduction artifacts under ``runs/`` — safely.

The 2026-06-08 Adam attempt died ``disk_exhausted`` (10.9 GB free) because
every training cell keeps its own ``datasets/`` copy and ``model.pt``
checkpoint, per attempt, forever (one All-CNN run alone: 15 GB of
``code/outputs``). This tool reclaims exactly those *recomputable* artifacts
while never touching the run's RECORD (reports, metrics, logs, events).

Safety model
------------
* DRY-RUN by default — prints what would be deleted; ``--apply`` deletes.
* Never touches a project dir carrying a ``.preserved`` marker unless
  ``--include-preserved`` is passed (the marker is the showcase/GC-skip signal
  written by ``write_final_report_rlm``).
* Only deletes content matching the HEAVY_ARTIFACT patterns below — weights,
  per-cell dataset copies, caches. All ``*.json`` / ``*.jsonl`` / ``*.md`` /
  ``*.log`` / ``*.txt`` / source files survive, so a GC'd run still scores,
  replays, and renders identically.
* ``--min-age-days N`` (default 2) skips anything modified recently — never
  races a live run.

Usage
-----
    python scripts/gc_runs.py                 # dry-run report
    python scripts/gc_runs.py --apply         # reclaim
    python scripts/gc_runs.py --runs-root runs --min-age-days 0 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Directory names whose ENTIRE subtree is recomputable training bulk.
HEAVY_DIR_NAMES = ("datasets", "wandb", "__pycache__", ".pip-cache")

# File suffixes that are recomputable training bulk (weights/checkpoints).
HEAVY_FILE_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npz")

# Never descend into these top-level entries of runs/ (shared caches are
# cross-run state with their own lifecycle; _archive is operator-managed).
SKIP_TOP_LEVEL = (".cache", "_archive", "_lessons")


def _age_days(path: Path) -> float:
    try:
        return (time.time() - path.stat().st_mtime) / 86400.0
    except OSError:
        return 0.0


def _du(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _candidates(project_dir: Path, min_age_days: float):
    """Yield heavy artifact paths within one project/attempt tree."""
    for path in project_dir.rglob("*"):
        try:
            if path.is_dir() and path.name in HEAVY_DIR_NAMES:
                if _age_days(path) >= min_age_days:
                    yield path
            elif path.is_file() and path.suffix in HEAVY_FILE_SUFFIXES:
                if _age_days(path) >= min_age_days:
                    yield path
        except OSError:
            continue


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default="runs", type=Path)
    ap.add_argument("--min-age-days", default=2.0, type=float,
                    help="skip artifacts modified in the last N days (default 2)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--include-preserved", action="store_true",
                    help="also GC projects carrying a .preserved marker")
    args = ap.parse_args(argv)

    root: Path = args.runs_root
    if not root.is_dir():
        print(f"runs root {root} does not exist", file=sys.stderr)
        return 2

    grand_total = 0
    n_deleted = 0
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        if project.name in SKIP_TOP_LEVEL:
            continue
        if (project / ".preserved").exists() and not args.include_preserved:
            print(f"SKIP (preserved): {project.name}")
            continue
        project_total = 0
        for victim in sorted(set(_candidates(project, args.min_age_days)),
                             key=lambda p: len(p.parts), reverse=True):
            if not victim.exists():
                continue  # parent already removed this pass
            size = _du(victim)
            project_total += size
            if args.apply:
                try:
                    if victim.is_dir():
                        shutil.rmtree(victim, ignore_errors=True)
                    else:
                        victim.unlink(missing_ok=True)
                    n_deleted += 1
                except OSError as exc:
                    print(f"  ! could not delete {victim}: {exc}", file=sys.stderr)
            else:
                rel = victim.relative_to(root)
                print(f"  would delete {size / 1e9:6.2f} GB  {rel}")
        if project_total:
            verb = "reclaimed" if args.apply else "reclaimable"
            print(f"{project.name}: {project_total / 1e9:.2f} GB {verb}")
        grand_total += project_total

    mode = "RECLAIMED" if args.apply else "RECLAIMABLE (dry-run; pass --apply)"
    print(f"\nTOTAL {mode}: {grand_total / 1e9:.2f} GB"
          + (f" across {n_deleted} path(s)" if args.apply else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
