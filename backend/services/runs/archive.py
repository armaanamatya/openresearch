"""Archive prior-attempt artifacts under runs/<id>/attempts/<ts>/.

Re-running `reproduce` against an existing project_id used to mix new and
prior attempts in the UI and the final report. This module moves run-derived
artifacts into a timestamped subdir so the next attempt starts with a clean
per-run surface, while preserving the ingested paper (paper.pdf,
parsed_full_text.txt, raw_paper.pdf, SQLite event store) so the paper is
not re-ingested.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Files moved if they exist (top-level under runs/<id>/).
_TOP_LEVEL_FILES: tuple[str, ...] = (
    "final_report.json",
    "final_report.md",
    "experiment_runs.jsonl",
    "cost_ledger.jsonl",
    "demo_status.json",
    "dashboard_events.jsonl",
    "runner.stdout.log",
    "runner.stderr.log",
    "agent_telemetry.jsonl",
    "generated_rubric.json",
    "pipeline_state.json",
)

# Whole directories moved if they exist.
_TOP_LEVEL_DIRS: tuple[str, ...] = ("rlm_state", "outputs")

# At least one of these must exist for archiving to fire — otherwise the run
# dir has only ingestion artifacts and there is nothing worth archiving.
_TRIGGERS: tuple[str, ...] = (
    "final_report.json",
    "experiment_runs.jsonl",
    "dashboard_events.jsonl",
    "rlm_state",
)

# Files inside code/ preserved (ingested paper). All other code/ contents
# are moved under attempts/<ts>/code/.
_CODE_PRESERVE: frozenset[str] = frozenset({"paper.pdf"})


def _now_iso() -> str:
    """ISO-8601 UTC stamp safe as a directory name (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive_run_artifacts(project_id: str, runs_root: Path) -> dict | None:
    """Move prior-attempt artifacts under ``runs/<id>/attempts/<ts>/``.

    Returns ``None`` when the run dir is absent or carries no triggering
    artifacts (a clean run dir — nothing to archive). Returns
    ``{"attempt_dir": str, "moved": list[str]}`` on success.

    Idempotent: missing items are silently skipped. Preserves ingestion
    artifacts (paper.pdf inside code/, parsed_full_text.txt, raw_paper.pdf,
    paper_html.html, the SQLite event store) so the next attempt skips
    re-ingestion.
    """
    runs_root = Path(runs_root)
    run_dir = runs_root / project_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    if not any((run_dir / name).exists() for name in _TRIGGERS):
        return None

    attempt_dir = run_dir / "attempts" / _now_iso()
    attempt_dir.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []

    for name in _TOP_LEVEL_FILES:
        src = run_dir / name
        if src.exists() and src.is_file():
            shutil.move(str(src), str(attempt_dir / name))
            moved.append(name)

    for name in _TOP_LEVEL_DIRS:
        src = run_dir / name
        if src.exists() and src.is_dir():
            shutil.move(str(src), str(attempt_dir / name))
            moved.append(name + "/")

    code_dir = run_dir / "code"
    if code_dir.exists() and code_dir.is_dir():
        code_attempt = attempt_dir / "code"
        for child in code_dir.iterdir():
            if child.name in _CODE_PRESERVE:
                continue
            code_attempt.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(code_attempt / child.name))
            moved.append(f"code/{child.name}")

    logger.info(
        "archive_run_artifacts: moved %d item(s) for %s into %s",
        len(moved), project_id, attempt_dir,
    )
    return {"attempt_dir": str(attempt_dir), "moved": moved}
