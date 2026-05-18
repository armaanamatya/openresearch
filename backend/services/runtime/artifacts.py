"""Artifact writers for sandboxed experiment runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class CommandLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: str
    phase: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    exit_code: int | None = None
    stdout_path: str = ""
    stderr_path: str = ""
    cause_kind: str = ""


def initialize_run_artifacts(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    _write_json_if_missing(run_dir / "metrics.json", {})
    _write_json_if_missing(run_dir / "provenance.json", {})
    for name in ("commands.log", "report.md"):
        path = run_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    return run_dir


def append_command_log(run_dir: Path, entry: CommandLogEntry) -> Path:
    path = run_dir / "commands.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.model_dump(), sort_keys=True) + "\n")
    return path


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    return write_json(run_dir / "metrics.json", metrics)


def write_provenance(run_dir: Path, provenance: dict[str, Any]) -> Path:
    return write_json(run_dir / "provenance.json", provenance)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit utf-8 — defense in depth. json.dumps default ensure_ascii=True
    # produces ASCII, so this works today even on Windows cp1252, but adding
    # encoding= ensures correctness if anyone flips ensure_ascii=False.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_if_missing(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        write_json(path, payload)


__all__ = [
    "CommandLogEntry",
    "append_command_log",
    "initialize_run_artifacts",
    "utc_now_iso",
    "write_json",
    "write_metrics",
    "write_provenance",
]
