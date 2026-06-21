"""PR-π Module B orphan-run liveness sweeper.

Scans file-backed run directories for stale ``demo_status.json`` records that
still say ``running`` after their host process disappeared, then atomically
marks them terminal and emits dashboard diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "interrupted", "stopped"}
_ORPHAN_REASON = "run process disappeared (host suspend / SIGKILL / OOM)"


@dataclass(frozen=True)
class OrphanReport:
    """Summary for one run converted from stale running to terminal state."""

    project_id: str
    last_status: str
    last_updated_at: datetime
    pid: int | None
    reason: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _last_status_update(status_path: Path, status: dict[str, Any]) -> datetime:
    parsed = (
        _parse_datetime(status.get("updatedAt"))
        or _parse_datetime(status.get("updated_at"))
        or _parse_datetime(status.get("startedAt"))
    )
    if parsed is not None:
        return parsed
    try:
        return datetime.fromtimestamp(status_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _coerce_pid(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` appears alive on this host.

    Pre: ``pid`` is a positive integer.
    Post: returns ``True`` only when the process appears to exist.  On Windows,
    returns a conservative ``True`` when optional ``psutil`` is unavailable.
    Side effects: sends signal ``0`` on POSIX, which does not terminate the
    process.
    Exceptions raised: none; OS probing failures are converted to ``False`` on
    POSIX and conservative ``True`` on Windows without psutil.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import psutil  # type: ignore[import-not-found]
        except Exception:
            return True
        return bool(psutil.pid_exists(pid))
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # EPERM: the process EXISTS but belongs to another user (e.g. the
        # backend server and the CLI run launched under different OS users on
        # one host). Treating it as dead would falsely sweep a live run.
        return True
    except OSError:
        return False


def _count_iterations(run_dir: Path) -> int:
    path = run_dir / "rlm_state" / "iterations.jsonl"
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _last_rubric_score(run_dir: Path) -> float:
    candidates = [
        run_dir / "rlm_state" / "iterations.jsonl",
        run_dir / "dashboard_events.jsonl",
    ]
    score = 0.0
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    score = _extract_score(payload, default=score)
        except OSError:
            continue
    return float(score)


def _extract_score(payload: Any, *, default: float) -> float:
    if not isinstance(payload, dict):
        return default
    rubric = payload.get("rubric")
    if isinstance(rubric, dict):
        raw = rubric.get("overall_score") or rubric.get("score")
        if isinstance(raw, int | float):
            return float(raw)
    for key in ("rubric_score", "overall_score", "score"):
        raw = payload.get(key)
        if isinstance(raw, int | float):
            return float(raw)
    return default


def _cost_usd(run_dir: Path) -> float:
    path = run_dir / "cost_ledger.jsonl"
    total = 0.0
    if not path.exists():
        return total
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                for key in ("cost_usd", "usd", "total_cost_usd"):
                    raw = payload.get(key)
                    if isinstance(raw, int | float):
                        total += float(raw)
                        break
    except OSError:
        return total
    return total


def _write_final_report_if_missing(run_dir: Path, status: dict[str, Any], now_iso: str) -> None:
    json_path = run_dir / "final_report.json"
    if json_path.exists():
        return
    score = _last_rubric_score(run_dir)
    verdict = "partial" if score > 0.0 else "failed"
    project_id = status.get("projectId") or run_dir.name
    paper_id = status.get("paperId")
    payload = {
        "status": "interrupted",
        "verdict": verdict,
        "mode": status.get("runMode") or status.get("mode"),
        "paperId": paper_id,
        "projectId": project_id,
        "startedAt": status.get("startedAt"),
        "completed_at": now_iso,
        "iterations": _count_iterations(run_dir),
        "rubric_score": score,
        "cost_usd": _cost_usd(run_dir),
        "reason": "orphaned",
        "stop_reason": {"kind": "orphaned", "detail": _ORPHAN_REASON},
    }
    _atomic_write_json(json_path, payload)
    _write_salvage_md(run_dir / "final_report.md", project_id, paper_id, score, verdict, now_iso)


def _write_salvage_md(
    md_path: Path,
    project_id: str,
    paper_id: str | None,
    score: float,
    verdict: str,
    completed_at: str,
) -> None:
    if md_path.exists():
        return
    lines = [
        f"# Salvaged report — {project_id}",
        "",
        f"**Verdict:** {verdict}  ",
        f"**Score:** {score:.4f}  ",
        f"**Paper:** {paper_id or 'unknown'}  ",
        f"**Completed at:** {completed_at}  ",
        "",
        "> This report was salvaged by the orphan-run sweeper after the run "
        "process disappeared without writing a final report. The score is the "
        "best rubric score observed on disk before the process was lost.",
    ]
    try:
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("run_liveness: could not write salvage md %s: %s", md_path, exc)


def _append_dashboard_events(run_dir: Path, report: OrphanReport, now_iso: str) -> None:
    path = run_dir / "dashboard_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "timestamp": now_iso,
        "project_id": report.project_id,
        "level": "warn",
        "code": "orphaned_stale_run",
        "message": report.reason,
        "pid": report.pid,
    }
    events = [
        {"event": "run_interrupted", **base},
        {"event": "run_warning", **base},
    ]
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")


def sweep_orphaned_runs(
    runs_root: Path,
    *,
    stale_after_s: float = 120.0,
    stale_threshold_s: float | None = None,
    emit_event: bool = True,
) -> list[OrphanReport]:
    """Mark stale file-backed running runs whose process disappeared.

    Pre: ``runs_root`` points at a directory containing run subdirectories; it
    may be missing.
    Post: each stale orphan is atomically updated to ``status=interrupted`` and
    returned as an ``OrphanReport``.  Already-terminal runs are skipped.
    Side effects: writes ``demo_status.json``, appends dashboard events when
    ``emit_event`` is true, and writes ``final_report.json`` if it is missing.
    Exceptions raised: none for corrupt or unreadable run directories; failures
    are logged and the sweep continues.
    """
    threshold = stale_after_s if stale_threshold_s is None else stale_threshold_s
    root = Path(runs_root)
    if not root.exists():
        return []
    now = _utc_now()
    reports: list[OrphanReport] = []
    for status_path in sorted(root.glob("*/demo_status.json")):
        try:
            status = _read_json(status_path)
            if not status:
                continue
            last_status = str(status.get("status") or "")
            if last_status in _TERMINAL_STATUSES or last_status != "running":
                continue
            last_updated = _last_status_update(status_path, status)
            if (now - last_updated).total_seconds() <= threshold:
                continue
            pid = _coerce_pid(status.get("pid"))
            if pid is None:
                # No pid recorded — we cannot verify liveness, so we must NOT
                # mark this as orphan. PID instrumentation is a prereq for the
                # sweeper; until every writer of status=running stamps a pid,
                # absent-pid means "unknown", not "dead".
                continue
            pid_host = str(status.get("pidHost") or "").strip()
            if pid_host and pid_host != socket.gethostname():
                # The pid was minted on a different host / pid namespace — e.g.
                # a host-launched CLI run observed by the containerized backend
                # through the bind-mounted runs/ (compose mounts ./runs). An
                # os.kill probe is meaningless across that boundary (the host
                # pid simply doesn't exist in the container), so liveness is
                # UNKNOWN, not dead — same conservative posture as absent-pid.
                # A missing pidHost (legacy snapshots) keeps single-host
                # behavior unchanged.
                continue
            if _pid_alive(pid):
                continue

            run_dir = status_path.parent
            now_iso = _iso(now)
            merged = {
                **status,
                "projectId": status.get("projectId") or run_dir.name,
                "status": "interrupted",
                "degraded": True,
                "degraded_reason": _ORPHAN_REASON,
                "error": status.get("error") or "orphaned_stale_run",
                "updatedAt": now_iso,
                "completedAt": now_iso,
                "completed_at": now_iso,
            }
            _atomic_write_json(status_path, merged)
            _write_final_report_if_missing(run_dir, merged, now_iso)

            report = OrphanReport(
                project_id=run_dir.name,
                last_status=last_status,
                last_updated_at=last_updated,
                pid=pid,
                reason="orphaned_stale_run",
            )
            if emit_event:
                _append_dashboard_events(run_dir, report, now_iso)
            reports.append(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_liveness: failed to sweep %s: %s", status_path, exc)
    return reports


def periodic_liveness_sweep(
    runs_root: Path,
    *,
    interval_s: float = 120.0,
    stale_after_s: float = 120.0,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Start a daemon thread that periodically sweeps orphaned runs.

    Pre: ``runs_root`` is the file-backed runs directory to scan.  ``interval_s``
    must be positive for regular sweeps.
    Post: returns the started daemon thread; the first sweep runs after one
    interval unless the optional ``stop_event`` is set.
    Side effects: creates a daemon thread and calls ``sweep_orphaned_runs`` on
    each interval.
    Exceptions raised: none from sweep failures; they are logged and the thread
    keeps running.
    """
    stopper = stop_event or threading.Event()

    def _loop() -> None:
        while not stopper.wait(max(0.1, interval_s)):
            try:
                sweep_orphaned_runs(runs_root, stale_after_s=stale_after_s)
            except Exception as exc:  # noqa: BLE001
                logger.warning("run_liveness: periodic sweep failed: %s", exc)

    thread = threading.Thread(target=_loop, name="run-liveness-sweep", daemon=True)
    thread.start()
    return thread


__all__ = [
    "OrphanReport",
    "_pid_alive",
    "periodic_liveness_sweep",
    "sweep_orphaned_runs",
]
