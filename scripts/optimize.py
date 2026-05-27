#!/usr/bin/env python3
"""Unified optimization CLI — single entry point for every lane.

Subcommands wrap the existing per-lane scripts so users (and cron) have
one surface to learn:

    python scripts/optimize.py mine                    # Lane D — trace miner (Phase 1, $0)
    python scripts/optimize.py gepa run --help         # Lane B — GEPA prompt mutation
    python scripts/optimize.py gepa validate --help    # Lane B — §4.6 validation
    python scripts/optimize.py gepa promote --help     # Lane B — promote to source
    python scripts/optimize.py skillopt train --help   # Lane E — SkillOpt (Phase 5, NOT YET)
    python scripts/optimize.py bake-off --help         # Phase 3 — bake-off (NOT YET)

The script does not implement any lane itself — it dispatches to the
per-lane scripts under ``scripts/``. Each subcommand prints "not yet
available" with a phase pointer when its target is deferred.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent

_LANES_AVAILABLE = {
    "mine": ("scripts/mine_traces.py", "Lane D — programmatic trace miner (Phase 1)"),
    "gepa": ("scripts/optimize_prompts_gepa.py", "Lane B — GEPA prompt mutation (shipped)"),
    "promote-gepa": ("scripts/promote_gepa_mutation.py", "Lane B — promote GEPA mutation to source"),
}

_LANES_DEFERRED = {
    "skillopt": "Phase 5 (awaits Phase 3 bake-off evidence). See 2026-05-27-optimization-platform-design.md §3 Phase 5.",
    "bake-off": "Phase 3 (awaits LLM budget approval). See 2026-05-27-optimization-platform-design.md §3 Phase 3.",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optimize",
        description="Unified optimization CLI for ReproLab.",
    )
    sub = p.add_subparsers(dest="lane", required=True)
    for name, (path, descr) in _LANES_AVAILABLE.items():
        s = sub.add_parser(name, help=descr, add_help=False)
        s.set_defaults(_target=path)
    for name, reason in _LANES_DEFERRED.items():
        s = sub.add_parser(name, help=f"[NOT YET] {reason}", add_help=False)
        s.set_defaults(_target=None, _deferred_reason=reason)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _build_parser().print_help()
        return 1
    lane = argv[0]
    rest = argv[1:]
    if lane in _LANES_DEFERRED:
        sys.stderr.write(
            f"optimize: lane '{lane}' is not yet available.\n  {_LANES_DEFERRED[lane]}\n"
        )
        return 2
    if lane not in _LANES_AVAILABLE:
        _build_parser().print_help()
        return 1
    target = _REPO_ROOT / _LANES_AVAILABLE[lane][0]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO_ROOT))
    return subprocess.call([sys.executable, str(target), *rest], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
