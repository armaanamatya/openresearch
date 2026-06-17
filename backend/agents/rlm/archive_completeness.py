"""Pure gate: a BES efficacy claim requires a complete run archive (the Adam
lesson — without it the most important number becomes folklore). Stdlib only.

Supports TWO layouts transparently:

* **Curated archive** (``best_runs/<paper>_ab/<arm>/``): all artifacts are
  flattened to the top level by the archiver.
* **Live run dir** (``runs/<project_id>/``): artifacts live at their natural
  production locations (``code/metrics.json``, ``rlm_state/bes_candidates.json``,
  etc.).

``check_bes_archive`` resolves each logical artifact against its ordered list of
candidate relative paths — the first path that exists satisfies the requirement.
A logical artifact is reported missing only when NONE of its candidates exist.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# Logical artifact name → tuple of candidate relative paths.
# Any one path satisfying the check is sufficient (ordered: live location first,
# then curated-archive / legacy top-level location).
REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "final_report.json": ("final_report.json",),
    "dashboard_events.jsonl": ("dashboard_events.jsonl",),
    "experiment_runs.jsonl": ("experiment_runs.jsonl",),
    "rubric_evaluation.json": ("rubric_evaluation.json", "rlm_state/rubric_evaluation.json"),
    "generated_rubric.json": ("generated_rubric.json",),
    "metrics.json": ("code/metrics.json", "metrics.json"),
    "bes_candidates.json": ("rlm_state/bes_candidates.json", "bes_candidates.json"),
}

# Logical directory name → tuple of candidate relative paths.
REQUIRED_DIRS: dict[str, tuple[str, ...]] = {
    "candidates": ("candidates",),
}


@dataclass
class ArchiveCheck:
    complete: bool
    missing: list[str] = field(default_factory=list)


def check_bes_archive(run_dir: Path) -> ArchiveCheck:
    run_dir = Path(run_dir)
    missing: list[str] = []

    for logical_name, candidates in REQUIRED_ARTIFACTS.items():
        if not any((run_dir / p).is_file() for p in candidates):
            missing.append(logical_name)

    for logical_name, candidates in REQUIRED_DIRS.items():
        if not any((run_dir / p).is_dir() for p in candidates):
            missing.append(logical_name + "/")

    return ArchiveCheck(complete=not missing, missing=missing)
