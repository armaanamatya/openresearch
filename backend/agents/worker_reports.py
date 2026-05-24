"""Structured worker report capture for agent invocations."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


WORKER_REPORTS_FILENAME = "worker_reports.jsonl"

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
