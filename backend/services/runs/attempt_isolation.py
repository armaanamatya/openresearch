"""attempt_isolation.py — archive prior-attempt artifacts at run-start.

When ``run_pipeline_rlm`` is invoked on a project that already has a
``final_report.json`` (i.e. a completed prior attempt), this module moves
all run-derived artifacts into a timestamped subdirectory so the new attempt
starts with a clean surface.  The UI therefore shows only the current
attempt's data — never a mixture of runs.

Archiving fires only when ``final_report.json`` exists.  A failed prior
attempt that did not produce ``final_report.json`` is partially archived (any
listed artifacts that do exist are moved) without crashing.

**Warm retry (Lane A)**: when ``final_report.json`` is absent BUT ``code/``
already holds an agent-written artifact (e.g. ``code/commands.json`` or
``code/train.py``), this is a kill-and-relaunch of a still-in-progress run.
The function logs a "warm retry detected" message and returns ``None`` — the
prior ``code/`` is left in place so the cached ``implement_baseline`` result
can short-circuit the ~5-min sub-agent call on the next iteration.  The agent
still operates in fix-existing-code mode via ``repair_context`` if the cache
hit is invalidated.

Paper-level artifacts that are stable across attempts — ``paperMeta.json``,
``raw_paper.pdf``, ``raw_paper.html`` (or ``paper_html.html``),
``parsed_full_text.txt``, ``generated_rubric.json`` — are intentionally left
in place so the paper is not re-ingested on the next run.

Design contract: task #42 / ``docs/runbooks/2026-05-23-sdar-baseline-handoff.md``.
Lane A warm-retry: 2026-05-24 spec.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
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

# Per-attempt sidecar FILES — the SINGLE SOURCE OF TRUTH shared by BOTH
# archivers: this module (the run.py path) and backend/services/runs/archive.py
# (the CLI path, `archive_run_artifacts`). 2026-06-09: rubric_evaluation.json /
# rubric_tree.json were missing from the archive manifests — a fresh attempt
# inherited the PREVIOUS attempt's graded leaves (every 0-iteration attempt in
# the 06-08/06-09 cluster reported weak_leaves it never produced, and
# write_final_report_rlm's merge + leaf_scorer.finalize_rescore both read these
# files assuming they belong to the current attempt). The telemetry sidecars
# (timing/tokens/worker_reports) leak the same way into the next attempt's
# report rendering and fidelity evidence. The same evening's live re-run then
# showed the TWO archivers had drifted independently — the CLI one still leaked
# everything this module had just been taught to move. Hence one shared tuple;
# extend it HERE, never in archive.py. All entries are per-attempt run
# products; their absence at fresh-attempt start is handled everywhere.
PER_ATTEMPT_SIDECARS: tuple[str, ...] = (
    "rubric_evaluation.json",
    "rubric_tree.json",
    "timing.json",
    "tokens_total.json",
    "worker_reports.jsonl",
    "environment_spec.json",
)

# Top-level files moved if present.
_ARCHIVE_FILES: tuple[str, ...] = (
    "final_report.json",
    "final_report.md",
    "experiment_runs.jsonl",
    "cost_ledger.jsonl",
    "dashboard_events.jsonl",
    "user_messages.jsonl",
    "_user_message_cursor.json",
) + PER_ATTEMPT_SIDECARS

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

# Files inside ``code/`` whose presence indicates a prior attempt's
# code-generation phase ran.  Used to detect the "warm retry" case (no
# final_report.json, but code/ already holds an agent-written artifact).
_WARM_RETRY_MARKERS: tuple[str, ...] = ("commands.json", "train.py")


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


def _is_warm_retry(project_dir: Path) -> bool:
    """Detect a kill-and-relaunch of a still-in-progress run.

    Returns True when:
      * ``final_report.json`` is absent (no completed prior run), AND
      * ``code/`` exists and contains at least one agent-written marker
        (``commands.json`` or ``train.py``).

    The agent's first ``implement_baseline`` call will then see the existing
    code on disk and either reuse it (cache hit, Lane A) or operate in
    fix-existing-code mode via ``repair_context``.
    """
    if (project_dir / _TRIGGER_FILE).exists():
        return False
    code_dir = project_dir / _CODE_DIR
    if not code_dir.is_dir():
        return False
    return any((code_dir / m).exists() for m in _WARM_RETRY_MARKERS)


def _chown_root_owned_code(code_dir: Path) -> None:
    """Reclaim ownership of root-owned files inside ``code_dir`` before the move.

    Docker containers that run as root inside (the default) write artifacts to
    the bind-mounted host directory as root.  The host process (running as the
    user) then cannot ``shutil.move`` those files into ``attempts/`` — every
    archive attempt aborts with PermissionError until the operator manually
    chowns the directory.

    The elegant fix: run ``docker run --rm -v <code_dir>:/work alpine
    chown -R <uid>:<gid> /work`` BEFORE the move.  Fail-soft: if Docker is
    unavailable or the chown fails, log a warning and let the subsequent move
    fail with the original PermissionError — so the cause is visible in logs
    instead of silently working some of the time.

    Skipped on Windows (no ``os.getuid``) and when the directory is missing.
    """
    if not hasattr(os, "getuid"):  # Windows — nothing to chown
        return
    if not code_dir.exists() or not code_dir.is_dir():
        return
    try:
        uid = os.getuid()  # type: ignore[attr-defined]
        gid = os.getgid()  # type: ignore[attr-defined]
        # 30 s cap: a hung docker daemon must NOT block archive forever.
        subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{code_dir}:/work",
                "alpine", "chown", "-R", f"{uid}:{gid}", "/work",
            ],
            check=True,
            timeout=30,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError) as exc:
        logger.warning(
            "attempt_isolation: docker chown -R failed for %s (%s) — "
            "the subsequent shutil.move may hit PermissionError on "
            "root-owned files; manually `docker run --rm -v %s:/work alpine "
            "chown -R %s:%s /work` if so",
            code_dir, exc, code_dir,
            getattr(os, "getuid", lambda: "?")(),
            getattr(os, "getgid", lambda: "?")(),
        )


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
    dir does not exist, the per-project lock is already held by another
    process, or a warm-retry was detected).  Returns
    ``{"attempt_dir": str, "moved": list[str]}`` on success.

    **Warm retry (Lane A)**: when ``final_report.json`` is absent BUT
    ``code/commands.json`` or ``code/train.py`` exists (kill-and-relaunch of
    a still-in-progress run), the function logs a warm-retry notice and
    returns ``None`` — the prior ``code/`` is preserved so the
    ``implement_baseline`` cache can short-circuit the ~5-min sub-agent call.

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

    # Warm-retry detection (Lane A): kill-and-relaunch of a still-in-progress
    # run.  No final_report.json, but code/ already holds an agent-written
    # artifact — leave everything in place so the implement_baseline cache can
    # short-circuit the ~5-min sub-agent call on the next iteration.
    if _is_warm_retry(project_dir):
        msg = (
            f"attempt_isolation: warm retry detected for {project_id} "
            f"(prior code/ present, no final_report.json) — skipping archive"
        )
        logger.info(msg)
        print(msg, file=sys.stderr)
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
        # Reclaim ownership FIRST: docker containers run as root inside and
        # bind-mounted writes land as root on the host; without the chown the
        # subsequent shutil.move trips PermissionError on every root-owned file.
        # (Fail-soft — see _chown_root_owned_code for the failure narrative.)
        code_src = project_dir / _CODE_DIR
        if code_src.exists() and code_src.is_dir():
            _chown_root_owned_code(code_src)
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
