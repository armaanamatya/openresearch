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
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# POSIX file-locking (fcntl) — POSIX-only (Linux/macOS). Windows falls back
# to a no-op so the code stays portable.
# ---------------------------------------------------------------------------
try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows
    _fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

# Files moved if they exist (top-level under runs/<id>/).
#
# Paper-level artifacts — generated_rubric.json, raw_paper.pdf,
# paper_html.html, raw_paper.html, parsed_full_text.txt, paperMeta.json —
# are intentionally EXCLUDED here.  They are paper-level data that must
# persist across attempts so the next run does not re-ingest or re-generate
# them.  (Codex finding B1: generated_rubric.json was previously in this list
# by mistake.)
_TOP_LEVEL_FILES: tuple[str, ...] = (
    "final_report.json",
    "final_report.md",
    "experiment_runs.jsonl",
    "cost_ledger.jsonl",
    "demo_status.json",
    "dashboard_events.jsonl",
    "worker_reports.jsonl",   # 2026-05-30: was NOT archived → old+new rows
                              # commingled across attempts, corrupting per-run
                              # worker-failure analysis (e.g. stale "failed"
                              # baseline rows read as the current attempt's).
    "runner.stdout.log",
    "runner.stderr.log",
    "agent_telemetry.jsonl",
    "pipeline_state.json",
    # NOTE: generated_rubric.json is NOT listed here — paper-level artifact,
    # must persist across attempts (Codex finding B1).
    # NOTE: raw_paper.pdf, paper_html.html, raw_paper.html,
    # parsed_full_text.txt, paperMeta.json are likewise excluded.
)

# Whole directories moved if they exist.
_TOP_LEVEL_DIRS: tuple[str, ...] = ("rlm_state", "outputs", "reports")

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


def _make_attempt_id() -> str:
    """Return a collision-proof attempt directory name.

    Format: <YYYYmmddTHHMMSS>-<microseconds>-<uuid6>

    Microsecond resolution + a 6-hex uuid suffix makes same-second collisions
    astronomically unlikely, and same-microsecond collisions essentially
    impossible.  (Codex finding B2: second-only resolution could collide.)
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
    suffix = uuid4().hex[:6]
    return f"{ts}-{suffix}"


def archive_run_artifacts(project_id: str, runs_root: Path) -> dict | None:
    """Move prior-attempt artifacts under ``runs/<id>/attempts/<ts>/``.

    Returns ``None`` when the run dir is absent, carries no triggering
    artifacts (a clean run dir — nothing to archive), or the per-project lock
    is already held by another process (concurrent archive — skip safely).
    Returns ``{"attempt_dir": str, "moved": list[str]}`` on success.

    Idempotent: missing items are silently skipped. Preserves ingestion
    artifacts (paper.pdf inside code/, parsed_full_text.txt, raw_paper.pdf,
    paper_html.html, generated_rubric.json, the SQLite event store) so the
    next attempt skips re-ingestion.

    A per-project POSIX flock on ``runs/<id>/.archive.lock`` serialises
    concurrent archive calls.  If the lock cannot be acquired immediately
    (another process is archiving) the call logs a warning and returns None
    rather than racing.  (Codex finding B2.)
    """
    runs_root = Path(runs_root)
    run_dir = runs_root / project_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    if not any((run_dir / name).exists() for name in _TRIGGERS):
        return None

    # --- per-project lock (POSIX only) ---
    lock_path = run_dir / ".archive.lock"
    lock_fh = None
    if _HAS_FCNTL:
        try:
            lock_fh = lock_path.open("w")
            _fcntl.flock(lock_fh, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except (IOError, BlockingIOError):
            if lock_fh is not None:
                lock_fh.close()
            logger.warning(
                "archive_run_artifacts: lock held for %s — skipping archive "
                "(another process is archiving concurrently)",
                project_id,
            )
            return None

    try:
        attempt_id = _make_attempt_id()
        attempt_dir = run_dir / "attempts" / attempt_id
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

    finally:
        if lock_fh is not None:
            if _HAS_FCNTL:
                _fcntl.flock(lock_fh, _fcntl.LOCK_UN)
            lock_fh.close()
