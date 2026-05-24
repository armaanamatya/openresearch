"""attempt_isolation.py — archive prior-attempt artifacts at run-start.

When ``run_pipeline_rlm`` is invoked on a project that already has a
``final_report.json`` (i.e. a completed prior attempt), this module moves
all run-derived artifacts into a timestamped subdirectory so the new attempt
starts with a clean surface.  The UI therefore shows only the current
attempt's data — never a mixture of runs.

Archiving fires only when ``final_report.json`` exists.  A failed prior
attempt that did not produce ``final_report.json`` is partially archived (any
listed artifacts that do exist are moved) without crashing.

Paper-level artifacts that are stable across attempts — ``paperMeta.json``,
``raw_paper.pdf``, ``raw_paper.html`` (or ``paper_html.html``),
``parsed_full_text.txt``, ``generated_rubric.json`` — are intentionally left
in place so the paper is not re-ingested on the next run.

Design contract: task #42 / ``docs/runbooks/2026-05-23-sdar-baseline-handoff.md``.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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

# ---------------------------------------------------------------------------
# Archive manifest
# ---------------------------------------------------------------------------

# Top-level files moved if present.
_ARCHIVE_FILES: tuple[str, ...] = (
    "final_report.json",
    "final_report.md",
    "experiment_runs.jsonl",
    "cost_ledger.jsonl",
    "dashboard_events.jsonl",
    "user_messages.jsonl",
    "_user_message_cursor.json",
)

# Sub-paths inside rlm_state/ that are moved per-attempt.
# gpu_escalation_state.json is intentionally included (A2): the escalation
# counter must reset to 0 on a new attempt so the cap is not exhausted before
# the fresh run even starts.  gpu_plan.json is in _PAPER_ARTIFACTS (stable
# across attempts) and must NOT appear here.
_RLM_STATE_ITER = "iterations.jsonl"
_RLM_STATE_PER_ATTEMPT: tuple[str, ...] = (
    "iterations.jsonl",
    "gpu_escalation_state.json",
)

# Pickle snapshot at the top level.
_REPL_PICKLE = "repl_state.pickle"

# The whole ``code/`` directory is moved to keep the next attempt's rebuild clean.
_CODE_DIR = "code"

# Stable paper-level artifacts — NEVER archived.
_PAPER_ARTIFACTS: frozenset[str] = frozenset({
    "paperMeta.json",
    "raw_paper.pdf",
    "raw_paper.html",
    "paper_html.html",
    "parsed_full_text.txt",
    "generated_rubric.json",
    "rlm_state/gpu_plan.json",
})

# Archiving fires only when this file exists — signals a completed prior run.
_TRIGGER_FILE = "final_report.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fs_safe_ts() -> str:
    """Return a collision-proof, filesystem-safe attempt-directory name.

    Format: <YYYYmmddTHHMMSS>-<microseconds>-<uuid6>

    Microsecond resolution + a 6-hex uuid suffix makes same-second collisions
    astronomically unlikely, and same-microsecond collisions essentially
    impossible.  (Codex finding B2: second-only resolution could collide.)
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
    suffix = uuid4().hex[:6]
    return f"{ts}-{suffix}"


def _reset_demo_status(project_dir: Path, project_id: str) -> None:
    """Write a fresh 'queued' demo_status.json for the incoming attempt."""
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "projectId": project_id,
        "outputDir": str(project_dir),
        "runMode": "rlm",
        "status": "queued",
        "startedAt": now,
        "updatedAt": now,
    }
    path = project_dir / "demo_status.json"
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(path)
    except Exception:  # noqa: BLE001 — status reset is best-effort
        logger.exception(
            "attempt_isolation: could not write fresh demo_status.json for %s",
            project_id,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_archive_prior_attempt(project_id: str, runs_root: Path) -> dict | None:
    """Archive prior-attempt artifacts if a completed run is present.

    Returns ``None`` when archiving was not needed (first-ever run, the run
    dir does not exist, or the per-project lock is already held by another
    process).  Returns ``{"attempt_dir": str, "moved": list[str]}`` on success.

    Idempotent: a missing file in the archive list is silently skipped —
    a failed prior run that did not produce every listed file is handled
    without a crash.

    A per-project POSIX flock on ``runs/<id>/.archive.lock`` serialises
    concurrent archive calls.  If the lock cannot be acquired immediately
    (another process is archiving) the call logs a warning and returns None
    rather than racing.  (Codex finding B2.)
    """
    runs_root = Path(runs_root)
    project_dir = runs_root / project_id

    if not project_dir.is_dir():
        return None

    # Guard: only archive when a completed run is present.
    if not (project_dir / _TRIGGER_FILE).exists():
        return None

    # --- per-project lock (POSIX only) ---
    lock_path = project_dir / ".archive.lock"
    lock_fh = None
    if _HAS_FCNTL:
        try:
            lock_fh = lock_path.open("w")
            _fcntl.flock(lock_fh, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except (IOError, BlockingIOError):
            if lock_fh is not None:
                lock_fh.close()
            logger.warning(
                "attempt_isolation: lock held for %s — skipping archive "
                "(another process is archiving concurrently)",
                project_id,
            )
            return None

    try:
        ts = _fs_safe_ts()
        attempt_dir = project_dir / "attempts" / ts
        attempt_dir.mkdir(parents=True, exist_ok=True)

        moved: list[str] = []

        # 1. Top-level files.
        for name in _ARCHIVE_FILES:
            src = project_dir / name
            if src.exists() and src.is_file():
                shutil.move(str(src), str(attempt_dir / name))
                moved.append(name)

        # 2. Per-attempt rlm_state/ files (iterations.jsonl + gpu_escalation_state.json).
        # _RLM_STATE_PER_ATTEMPT supersedes the legacy _RLM_STATE_ITER single-file
        # block: any file in this tuple is archived together so the next attempt
        # starts with a fresh rlm_state/ surface AND a reset escalation count.
        for _rlm_name in _RLM_STATE_PER_ATTEMPT:
            _rlm_src = project_dir / "rlm_state" / _rlm_name
            if _rlm_src.exists() and _rlm_src.is_file():
                (attempt_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
                shutil.move(str(_rlm_src), str(attempt_dir / "rlm_state" / _rlm_name))
                moved.append(f"rlm_state/{_rlm_name}")

        # 3. repl_state.pickle.
        pickle_src = project_dir / _REPL_PICKLE
        if pickle_src.exists() and pickle_src.is_file():
            shutil.move(str(pickle_src), str(attempt_dir / _REPL_PICKLE))
            moved.append(_REPL_PICKLE)

        # 4. code/ directory — rebuild from scratch on the new attempt.
        code_src = project_dir / _CODE_DIR
        if code_src.exists() and code_src.is_dir():
            shutil.move(str(code_src), str(attempt_dir / _CODE_DIR))
            moved.append(_CODE_DIR + "/")

        msg = (
            f"attempt_isolation: archiving prior attempt to "
            f"runs/{project_id}/attempts/{ts}/ "
            f"({len(moved)} item(s) moved)"
        )
        logger.info(msg)
        print(msg, file=sys.stderr)

        # 5. Reset demo_status.json so the UI shows the new attempt from the start.
        _reset_demo_status(project_dir, project_id)

        return {"attempt_dir": str(attempt_dir), "moved": moved}

    finally:
        if lock_fh is not None:
            if _HAS_FCNTL:
                _fcntl.flock(lock_fh, _fcntl.LOCK_UN)
            lock_fh.close()


__all__ = ["maybe_archive_prior_attempt"]
