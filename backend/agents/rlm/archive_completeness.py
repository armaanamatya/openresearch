"""Pure gate: a BES efficacy claim requires a complete run archive (the Adam
lesson — without it the most important number becomes folklore). Stdlib only.

NOTE: this checks an ARCHIVED A/B arm directory (e.g. best_runs/<paper>_ab/<arm>/),
where metrics.json / dashboard_events.jsonl / etc. sit at the top level. A LIVE run
keeps metrics.json under code/ — this checker is NOT for live run dirs."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "bes_candidates.json", "dashboard_events.jsonl", "experiment_runs.jsonl",
    "rubric_evaluation.json", "final_report.json", "metrics.json",
    "generated_rubric.json",
)
REQUIRED_DIRS = ("candidates",)


@dataclass
class ArchiveCheck:
    complete: bool
    missing: list[str] = field(default_factory=list)


def check_bes_archive(run_dir: Path) -> ArchiveCheck:
    run_dir = Path(run_dir)
    missing = [n for n in REQUIRED_ARTIFACTS if not (run_dir / n).is_file()]
    missing += [d + "/" for d in REQUIRED_DIRS if not (run_dir / d).is_dir()]
    return ArchiveCheck(complete=not missing, missing=missing)
