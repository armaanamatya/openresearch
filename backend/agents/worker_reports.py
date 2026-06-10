"""Structured worker report capture for agent invocations.

Extended (2026-05-24) with first-class per-worker JSON persistence,
structured blockers, enriched commands, artifacts, tests, and a
run-level summary. Backwards-compatible: old flat reports still load.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

WORKER_REPORTS_FILENAME = "worker_reports.jsonl"

# ---------------------------------------------------------------------------
# Worker types
# ---------------------------------------------------------------------------

WORKER_TYPE_RDR_CLUSTER = "rdr_cluster"
WORKER_TYPE_RLM_PRIMITIVE = "rlm_primitive"
WORKER_TYPE_SDK_AGENT = "sdk_agent"
WORKER_TYPE_HYBRID_ITERATION = "hybrid_iteration"

WORKER_REPORT_PROMPT_SUFFIX = """

Before you finish, include a final section exactly titled "Worker report" with these headings:
- What was implemented
- What was left undone
- Commands run + exit codes
- Issues discovered
- Whether procedures were followed

Use concise bullets. For every command you ran, include the command and its exit code. If a section has nothing to report, write "None".
"""

_HEADING_ALIASES = {
    "implemented": {
        "what was implemented",
        "implemented",
    },
    "left_undone": {
        "what was left undone",
        "left undone",
        "undone",
    },
    "commands": {
        "commands run + exit codes",
        "commands run and exit codes",
        "commands",
    },
    "issues": {
        "issues discovered",
        "issues",
    },
    "procedures": {
        "whether procedures were followed",
        "procedures followed",
        "procedures",
    },
}

_ALIAS_TO_KEY = {
    alias: key
    for key, aliases in _HEADING_ALIASES.items()
    for alias in aliases
}


def append_worker_report_instruction(prompt: str) -> str:
    """Append the required worker-report contract unless it is already present."""

    if "Worker report" in prompt and "Commands run + exit codes" in prompt:
        return prompt
    return f"{prompt.rstrip()}{WORKER_REPORT_PROMPT_SUFFIX}"


def worker_reports_run_dir(project_dir: Path) -> Path:
    """Resolve the run artifact root from an agent working directory."""

    project_dir = project_dir.resolve()
    if (project_dir / "demo_status.json").exists():
        return project_dir
    if project_dir.name == "code" and (project_dir.parent / "demo_status.json").exists():
        return project_dir.parent
    return project_dir


def worker_reports_path(project_dir: Path) -> Path:
    return worker_reports_run_dir(project_dir) / WORKER_REPORTS_FILENAME


def parse_worker_report_sections(text: str) -> dict[str, Any]:
    """Parse the final worker-report block from free-form agent text."""

    sections: dict[str, list[str]] = {
        "implemented": [],
        "left_undone": [],
        "commands": [],
        "issues": [],
        "procedures": [],
    }
    current: str | None = None
    in_report = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = _normalize_heading(line)
        if normalized == "worker report":
            in_report = True
            current = None
            continue
        key = _ALIAS_TO_KEY.get(normalized)
        if key is not None:
            in_report = True
            current = key
            continue
        if current and in_report:
            item = re.sub(r"^[-*•]\s*", "", line).strip()
            if item and item.lower() not in {"none", "n/a", "na"}:
                sections[current].append(item)

    commands = [_parse_command_item(item) for item in sections["commands"]]
    procedure_text = "\n".join(sections["procedures"]).strip()
    return {
        "implemented": sections["implemented"],
        "left_undone": sections["left_undone"],
        "commands": commands,
        "issues": sections["issues"],
        "procedures_followed": _parse_procedures_followed(procedure_text),
        "procedure_notes": procedure_text,
    }


def build_worker_report(
    *,
    agent_id: str,
    project_dir: Path,
    model: str | None,
    provider: str | None,
    status: str,
    started_at: str,
    finished_at: str,
    raw_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    parsed = parse_worker_report_sections(raw_text)
    commands = parsed["commands"]
    commands.extend(_commands_from_tool_calls(tool_calls or []))
    return {
        "report_id": str(uuid4()),
        "agent_id": agent_id,
        "project_dir": str(project_dir.resolve()),
        "model": model,
        "provider": provider,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "implemented": parsed["implemented"],
        "left_undone": parsed["left_undone"],
        "commands": commands,
        "issues": parsed["issues"] + ([error] if error else []),
        "procedures_followed": parsed["procedures_followed"],
        "procedure_notes": parsed["procedure_notes"],
        "raw_text": raw_text,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_worker_report(project_dir: Path, report: dict[str, Any]) -> Path:
    path = worker_reports_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
    return path


def _normalize_heading(line: str) -> str:
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^\*\*(.+)\*\*$", r"\1", line)
    line = line.rstrip(":").strip().lower()
    line = line.replace("&", "and")
    return re.sub(r"\s+", " ", line)


def _parse_command_item(item: str) -> dict[str, Any]:
    exit_code: int | None = None
    match = re.search(r"(?:exit(?:\s+code)?|code)\s*[:=]?\s*(-?\d+)", item, re.IGNORECASE)
    if match:
        try:
            exit_code = int(match.group(1))
        except ValueError:
            exit_code = None
    command = re.sub(r"\s*(?:[-–—>]+)?\s*(?:exit(?:\s+code)?|code)\s*[:=]?\s*-?\d+\s*$", "", item, flags=re.IGNORECASE)
    command = command.strip("` ")
    return {"command": command or item, "exit_code": exit_code, "source": "worker_report"}


def _commands_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for call in tool_calls:
        if call.get("tool_name") != "Bash":
            continue
        tool_input = call.get("tool_input")
        if not isinstance(tool_input, dict):
            continue
        command = tool_input.get("command") or tool_input.get("cmd")
        if isinstance(command, str) and command.strip():
            commands.append({
                "command": command.strip(),
                "exit_code": None,
                "source": "tool_call",
            })
    return commands


def _parse_procedures_followed(text: str) -> bool | None:
    lowered = text.lower()
    if not lowered:
        return None
    if any(token in lowered for token in ("not followed", "did not follow", "deviated", "skipped required")):
        return False
    if any(token in lowered for token in ("followed", "yes", "complied", "completed")):
        return True
    return None


# ---------------------------------------------------------------------------
# Extended report builder (2026-05-24)
# ---------------------------------------------------------------------------

_SDK_SUCCESS_NO_TEXT_PATTERN = re.compile(
    r"Claude Code returned an error result:\s*success",
    re.IGNORECASE,
)


def classify_sdk_success_blocker(error: str | None) -> dict[str, Any] | None:
    """If *error* matches the SDK 'success-with-no-text' pattern, return a structured blocker."""
    if not error:
        return None
    if _SDK_SUCCESS_NO_TEXT_PATTERN.search(error):
        return {
            "title": "SDK success-with-no-text",
            "description": (
                "Claude Code returned exit status 'success' but produced no text output. "
                "The SDK treats a zero-text success as an error, which blocks cluster completion."
            ),
            "severity": "critical",
            "source": "claude_agent_sdk",
            "suggested_fix": (
                "Check the agent prompt contract — ensure the agent always produces "
                "text output. May also be a transient SDK issue."
            ),
        }
    return None


def build_extended_worker_report(
    *,
    run_id: str | None = None,
    worker_id: str | None = None,
    worker_type: str = WORKER_TYPE_SDK_AGENT,
    agent_id: str,
    project_dir: Path,
    model: str | None = None,
    provider: str | None = None,
    status: str = "running",
    started_at: str | None = None,
    finished_at: str | None = None,
    raw_text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    error: str | None = None,
    cluster_id: str | None = None,
    task_id: str | None = None,
    parent_task_id: str | None = None,
    duration_ms: int | None = None,
    assignment: dict[str, Any] | None = None,
    execution_summary: dict[str, Any] | None = None,
    blockers: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    tests: list[dict[str, Any]] | None = None,
    next_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an extended worker report with all new fields.

    Backwards-compatible: all new fields are optional and default to None/[].
    """
    parsed = parse_worker_report_sections(raw_text) if raw_text else {
        "implemented": [], "left_undone": [], "commands": [],
        "issues": [], "procedures_followed": None, "procedure_notes": "",
    }
    commands = list(parsed["commands"])
    commands.extend(_commands_from_tool_calls(tool_calls or []))

    now = datetime.now(timezone.utc).isoformat()
    wid = worker_id or str(uuid4())

    # Auto-classify SDK success-with-no-text as a blocker
    auto_blockers = list(blockers or [])
    sdk_blocker = classify_sdk_success_blocker(error)
    if sdk_blocker:
        auto_blockers.append(sdk_blocker)

    # Auto-populate errors list
    auto_errors = list(errors or [])
    if error and not any(e.get("message") == error for e in auto_errors):
        auto_errors.append({
            "message": error,
            "stack_or_trace": None,
            "source_file": None,
            "recoverable": False,
        })

    report: dict[str, Any] = {
        # Original fields (backward compat)
        "report_id": str(uuid4()),
        "agent_id": agent_id,
        "project_dir": str(project_dir.resolve()) if project_dir else None,
        "model": model,
        "provider": provider,
        "status": status,
        "started_at": started_at or now,
        "finished_at": finished_at,
        "implemented": parsed["implemented"],
        "left_undone": parsed["left_undone"],
        "commands": commands,
        "issues": parsed["issues"] + ([error] if error else []),
        "procedures_followed": parsed["procedures_followed"],
        "procedure_notes": parsed["procedure_notes"],
        "raw_text": raw_text,
        "error": error,
        "created_at": now,
        # New fields (2026-05-24)
        "run_id": run_id,
        "worker_id": wid,
        "worker_type": worker_type,
        "cluster_id": cluster_id,
        "task_id": task_id or wid,
        "parent_task_id": parent_task_id,
        "duration_ms": duration_ms,
        "assignment": assignment,
        "execution_summary": execution_summary,
        "blockers": auto_blockers if auto_blockers else [],
        "errors": auto_errors if auto_errors else [],
        "artifacts": artifacts or [],
        "tests": tests or [],
        "next_actions": next_actions or [],
    }
    return report


# ---------------------------------------------------------------------------
# Structured persistence: runs/<id>/reports/
# ---------------------------------------------------------------------------

def reports_dir(run_dir: Path) -> Path:
    """Return the reports directory for a run."""
    return run_dir / "reports"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via .tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def open_worker_report(run_dir: Path, report: dict[str, Any]) -> Path:
    """Write a 'running' report BEFORE invoking the agent.

    Returns the path to the per-worker JSON file. On SIGKILL this file
    will still exist with status='running', providing forensic value.
    """
    rd = reports_dir(run_dir)
    worker_id = report["worker_id"]

    # Per-worker JSON
    worker_path = rd / "worker_reports" / f"{worker_id}.json"
    _atomic_write_json(worker_path, report)

    # Also append to the flat JSONL log
    jsonl_path = rd / "worker_reports.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")

    # Legacy flat JSONL for backward compat
    legacy_path = run_dir / WORKER_REPORTS_FILENAME
    with legacy_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")

    return worker_path


def finalize_worker_report(
    run_dir: Path,
    report: dict[str, Any],
    *,
    status: str = "completed",
    finished_at: str | None = None,
    duration_ms: int | None = None,
    execution_summary: dict[str, Any] | None = None,
    blockers: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    raw_text: str | None = None,
    error: str | None = None,
) -> Path:
    """Finalize a previously-opened worker report. Updates the per-worker JSON."""
    now = datetime.now(timezone.utc).isoformat()
    report["status"] = status
    report["finished_at"] = finished_at or now
    if duration_ms is not None:
        report["duration_ms"] = duration_ms
    if execution_summary is not None:
        report["execution_summary"] = execution_summary
    if blockers:
        report.setdefault("blockers", []).extend(blockers)
    if errors:
        report.setdefault("errors", []).extend(errors)
    if artifacts:
        report.setdefault("artifacts", []).extend(artifacts)
    if raw_text is not None:
        report["raw_text"] = raw_text
        parsed = parse_worker_report_sections(raw_text)
        report["implemented"] = parsed["implemented"]
        report["left_undone"] = parsed["left_undone"]
        report["procedures_followed"] = parsed["procedures_followed"]
        report["procedure_notes"] = parsed["procedure_notes"]
    if error is not None:
        report["error"] = error
        sdk_blocker = classify_sdk_success_blocker(error)
        if sdk_blocker and sdk_blocker not in report.get("blockers", []):
            report.setdefault("blockers", []).append(sdk_blocker)

    worker_id = report["worker_id"]
    worker_path = reports_dir(run_dir) / "worker_reports" / f"{worker_id}.json"
    _atomic_write_json(worker_path, report)

    # Append finalized version to JSONL
    jsonl_path = reports_dir(run_dir) / "worker_reports.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")

    return worker_path


# ---------------------------------------------------------------------------
# Read all worker reports (from per-worker JSON files)
# ---------------------------------------------------------------------------

def read_worker_reports(run_dir: Path) -> list[dict[str, Any]]:
    """Read all per-worker JSON reports from the reports directory.

    Falls back to the legacy flat JSONL if the reports/ dir doesn't exist.
    Returns latest version of each worker (deduped by worker_id).
    """
    worker_dir = reports_dir(run_dir) / "worker_reports"
    if worker_dir.is_dir():
        reports = []
        for p in sorted(worker_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    reports.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        if reports:
            return reports

    # Fallback: legacy flat JSONL
    legacy = run_dir / WORKER_REPORTS_FILENAME
    if not legacy.exists():
        return []
    seen: dict[str, dict[str, Any]] = {}
    try:
        for line in legacy.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    key = data.get("worker_id") or data.get("report_id") or str(uuid4())
                    seen[key] = data
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return list(seen.values())


# ---------------------------------------------------------------------------
# Run-level summary
# ---------------------------------------------------------------------------

def build_summary_report(run_dir: Path) -> dict[str, Any]:
    """Build a run-level summary from all worker reports.

    Safe to call at any time — returns partial summaries during the run.
    """
    workers = read_worker_reports(run_dir)
    by_status: dict[str, int] = {}
    critical_blockers: list[dict[str, Any]] = []
    files_changed: list[str] = []
    commands_run = 0
    failed_commands = 0
    total_tests_passed = 0
    total_tests_failed = 0

    for w in workers:
        st = w.get("status", "unknown")
        by_status[st] = by_status.get(st, 0) + 1

        for b in w.get("blockers", []):
            if b.get("severity") == "critical":
                critical_blockers.append(b)

        es = w.get("execution_summary") or {}
        files_changed.extend(es.get("changed_files", []))
        files_changed.extend(es.get("created_files", []))

        for cmd in w.get("commands", []):
            commands_run += 1
            ec = cmd.get("exit_code")
            if ec is not None and ec != 0:
                failed_commands += 1

        for t in w.get("tests", []):
            total_tests_passed += t.get("passed_count", 0)
            total_tests_failed += t.get("failed_count", 0)

    # Collect all next_actions across workers
    top_next_actions: list[dict[str, Any]] = []
    for w in workers:
        top_next_actions.extend(w.get("next_actions", []))

    # Determine final_run_status from demo_status.json
    final_run_status = "unknown"
    status_path = run_dir / "demo_status.json"
    if status_path.exists():
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            final_run_status = status_data.get("status", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

    # Dedup files_changed
    files_changed = sorted(set(files_changed))

    summary = {
        "total_workers": len(workers),
        "by_status": by_status,
        "critical_blockers": critical_blockers[:10],
        "files_changed": files_changed[:50],
        "commands_run": commands_run,
        "failed_commands": failed_commands,
        "tests_summary": {
            "passed": total_tests_passed,
            "failed": total_tests_failed,
        },
        "final_run_status": final_run_status,
        "top_next_actions": top_next_actions[:10],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def write_summary_report(run_dir: Path) -> Path:
    """Build and persist summary_report.json."""
    summary = build_summary_report(run_dir)
    path = reports_dir(run_dir) / "summary_report.json"
    _atomic_write_json(path, summary)
    return path


def get_or_build_summary(run_dir: Path) -> dict[str, Any]:
    """Return summary_report.json, building lazily if missing."""
    path = reports_dir(run_dir) / "summary_report.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return build_summary_report(run_dir)


# ---------------------------------------------------------------------------
# SSE event builders for report lifecycle
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_worker_report_started_event(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "worker_report_started",
        "timestamp": _now_iso(),
        "worker_id": report.get("worker_id"),
        "worker_type": report.get("worker_type"),
        "agent_id": report.get("agent_id"),
        "cluster_id": report.get("cluster_id"),
        "task_id": report.get("task_id"),
    }


def build_worker_report_completed_event(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "worker_report_completed",
        "timestamp": _now_iso(),
        "worker_id": report.get("worker_id"),
        "worker_type": report.get("worker_type"),
        "agent_id": report.get("agent_id"),
        "status": report.get("status"),
        "cluster_id": report.get("cluster_id"),
        "duration_ms": report.get("duration_ms"),
        "blockers_count": len(report.get("blockers", [])),
        "error": report.get("error"),
    }


def build_worker_report_failed_event(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "worker_report_failed",
        "timestamp": _now_iso(),
        "worker_id": report.get("worker_id"),
        "worker_type": report.get("worker_type"),
        "agent_id": report.get("agent_id"),
        "error": report.get("error"),
        "failure_class": report.get("failure_class"),
        "contract_violations": report.get("contract_violations"),
        "repairable": report.get("repairable"),
        "source": report.get("source"),
        "blockers": report.get("blockers", [])[:5],
    }


def build_worker_blocker_recorded_event(
    worker_id: str, blocker: dict[str, Any]
) -> dict[str, Any]:
    return {
        "event": "worker_blocker_recorded",
        "timestamp": _now_iso(),
        "worker_id": worker_id,
        "blocker": blocker,
    }
