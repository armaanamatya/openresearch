#!/usr/bin/env python3
"""Prune old run directories from runs/ — the GC the .preserved contract promised.

Thin CLI over :mod:`backend.services.runs.retention` (the logic moved there
2026-06-10 so the server can also run it periodically via the opt-in
``OPENRESEARCH_RUNS_RETENTION_DAYS`` knob; see that module for the full
keep/prune contract).

DRY-RUN BY DEFAULT: prints what would be deleted; pass --delete to act.

Usage:
    python scripts/prune_runs.py                       # dry-run, 14-day cutoff
    python scripts/prune_runs.py --older-than-days 30 --delete
    python scripts/prune_runs.py --runs-root /elsewhere/runs --keep prj_x
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/prune_runs.py` from the repo root without installs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.runs.retention import prune  # noqa: E402


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
