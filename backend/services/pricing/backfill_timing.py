"""CLI script to backfill timing.json for all existing preserved runs.

Usage:
    python -m backend.services.pricing.backfill_timing [--runs-root PATH]

Walks every run under `runs_root` that has a `.preserved` marker but lacks
`timing.json`, and computes + writes the timing data.

Safe to re-run: runs that already have `timing.json` are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_timing")


def _default_runs_root() -> Path:
    try:
        from backend.config import get_settings
        s = get_settings()
        if s.runs_root:
            return Path(s.runs_root)
    except Exception:
        pass
    return Path("runs")


def backfill(runs_root: Path) -> tuple[int, int, int]:
    """Walk preserved runs and write missing timing.json files.

    Returns (skipped_already_exist, written, errors).
    """
    from backend.services.pricing.timing import write_timing_json

    runs_root = Path(runs_root)
    skipped = 0
    written = 0
    errors = 0

    if not runs_root.exists():
        logger.error("runs_root does not exist: %s", runs_root)
        return 0, 0, 0

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / ".preserved").exists():
            continue
        timing_path = run_dir / "timing.json"
        if timing_path.exists():
            skipped += 1
            continue
        try:
            result = write_timing_json(run_dir)
            if result is not None:
                written += 1
                logger.info("wrote %s", result)
            else:
                logger.warning("nothing written for %s (no timing data available)", run_dir.name)
                errors += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("error writing timing.json for %s: %s", run_dir.name, exc)
            errors += 1

    return skipped, written, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Path to the runs/ directory (default: from settings or ./runs)",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root) if args.runs_root else _default_runs_root()
    logger.info("Backfilling timing.json under: %s", runs_root)

    skipped, written, errors = backfill(runs_root)

    logger.info(
        "Done. skipped=%d (already had timing.json), written=%d, errors=%d",
        skipped, written, errors,
    )
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
