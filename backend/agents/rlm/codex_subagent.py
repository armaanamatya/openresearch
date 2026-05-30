"""Bounded Codex CLI subagent for repo-editing repair tasks.

This module is intentionally not part of the normal RLM provider path. It only
invokes the installed ``codex`` CLI through ``codex exec`` after checking
``codex login status``. It never reads or parses ``~/.codex/auth.json``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_MAX_OUTPUT_CHARS = 12000

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{12,}"),
    re.compile(r"glpat-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{12,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"(?i)(access_token|refresh_token|id_token|api_key|secret)"
        r"([\"'\s:=]+)([A-Za-z0-9_\-\.]{12,})"
    ),
)

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".codex",
}
_SKIP_FILES = {
    "dashboard_events.jsonl",
    "cost_ledger.jsonl",
    "worker_reports.jsonl",
}


@dataclass
class CodexSubagentResult:
    ok: bool
    timed_out: bool
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    changed_files: list[str]
    duration_s: float
    error_type: str | None
    stdout_tail_truncated: bool = False
    stderr_tail_truncated: bool = False
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "timed_out": self.timed_out,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "changed_files": self.changed_files,
            "duration_s": self.duration_s,
            "error_type": self.error_type,
            "stdout_tail_truncated": self.stdout_tail_truncated,
            "stderr_tail_truncated": self.stderr_tail_truncated,
            "message": self.message,
        }


@dataclass(frozen=True)
class _FileSig:
    size: int
    mtime_ns: int
    mode: int


@dataclass
class _Tail:
    text: str
    truncated: bool = False


EventSink = Callable[[str, dict], None]


def _resolve_codex_cli() -> str | None:
    configured = os.environ.get("REPROLAB_CODEX_CLI_PATH", "").strip()
    if configured:
        return configured
    return shutil.which("codex")


def check_codex_available() -> tuple[bool, str | None]:
    """Return availability via ``codex login status`` only."""
    cli = _resolve_codex_cli()
    if not cli:
        return False, "codex_cli_missing"
    try:
        proc = subprocess.run(
            [cli, "login", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "codex_login_status_timeout"
    except OSError:
        return False, "codex_login_status_failed"
    if proc.returncode != 0:
        return False, "codex_unavailable"
    return True, None


def _redact_secrets(text: str) -> str:
    redacted = text or ""

    def _replace_group(match: re.Match[str]) -> str:
        if match.lastindex and match.lastindex >= 3:
            return f"{match.group(1)}{match.group(2)}[REDACTED]"
        return "[REDACTED]"

    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_replace_group, redacted)
    return redacted


def _tail(text: str, max_chars: int) -> _Tail:
    redacted = _redact_secrets(text or "")
    if len(redacted) <= max_chars:
        return _Tail(redacted, False)
    return _Tail(redacted[-max_chars:], True)


def _snapshot(workspace: Path) -> dict[str, _FileSig]:
    root = workspace.resolve()
    out: dict[str, _FileSig] = {}
    if not root.exists():
        return out
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        current_path = Path(current)
        for name in files:
            if name in _SKIP_FILES:
                continue
            path = current_path / name
            try:
                st = path.lstat()
                rel = path.relative_to(root).as_posix()
            except OSError:
                continue
            out[rel] = _FileSig(size=st.st_size, mtime_ns=st.st_mtime_ns, mode=st.st_mode)
    return out


def _changed_files(before: dict[str, _FileSig], after: dict[str, _FileSig]) -> list[str]:
    changed = {
        path
        for path, sig in after.items()
        if before.get(path) != sig
    }
    changed.update(path for path in before if path not in after)
    return sorted(changed)


def _emit(event_sink: EventSink | None, event_type: str, payload: dict) -> None:
    if event_sink is None:
        return
    try:
        event_sink(event_type, payload)
    except Exception:
        return


def _event_payload(
    *,
    task_type: str,
    timeout_s: int,
    duration_s: float,
    exit_code: int | None,
    changed_file_count: int,
    stdout_tail_truncated: bool,
    stderr_tail_truncated: bool,
) -> dict:
    return {
        "task_type": task_type,
        "timeout_s": timeout_s,
        "duration_s": round(duration_s, 3),
        "exit_code": exit_code,
        "changed_file_count": changed_file_count,
        "stdout_tail_truncated": stdout_tail_truncated,
        "stderr_tail_truncated": stderr_tail_truncated,
    }


def run_codex_subagent(
    prompt: str,
    workspace: Path,
    timeout_s: int,
    profile: str | None,
    readonly: bool,
    *,
    task_type: str = "implementation_repair",
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    event_sink: EventSink | None = None,
) -> CodexSubagentResult:
    """Run ``codex exec --cd <workspace> <prompt>`` with a hard timeout."""
    start = time.monotonic()
    workspace = Path(workspace)
    available, unavailable_reason = check_codex_available()
    if not available:
        duration = time.monotonic() - start
        result = CodexSubagentResult(
            ok=False,
            timed_out=False,
            exit_code=None,
            stdout_tail="",
            stderr_tail="",
            changed_files=[],
            duration_s=duration,
            error_type="unavailable",
            message=unavailable_reason or "codex unavailable",
        )
        _emit(
            event_sink,
            "codex_subagent_failed",
            _event_payload(
                task_type=task_type,
                timeout_s=timeout_s,
                duration_s=duration,
                exit_code=None,
                changed_file_count=0,
                stdout_tail_truncated=False,
                stderr_tail_truncated=False,
            ),
        )
        return result

    cli = _resolve_codex_cli()
    if not cli:
        duration = time.monotonic() - start
        return CodexSubagentResult(
            ok=False,
            timed_out=False,
            exit_code=None,
            stdout_tail="",
            stderr_tail="",
            changed_files=[],
            duration_s=duration,
            error_type="unavailable",
            message="codex_cli_missing",
        )

    before = _snapshot(workspace)
    _emit(
        event_sink,
        "codex_subagent_started",
        _event_payload(
            task_type=task_type,
            timeout_s=timeout_s,
            duration_s=0.0,
            exit_code=None,
            changed_file_count=0,
            stdout_tail_truncated=False,
            stderr_tail_truncated=False,
        ),
    )

    cmd = [cli, "exec"]
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(["--sandbox", "read-only" if readonly else "workspace-write"])
    cmd.extend(["--cd", str(workspace), prompt])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            cwd=str(workspace),
        )
        duration = time.monotonic() - start
        after = _snapshot(workspace)
        stdout = _tail(proc.stdout or "", max_output_chars)
        stderr = _tail(proc.stderr or "", max_output_chars)
        changed = _changed_files(before, after)
        needs_change = task_type != "traceback_explanation"
        ok = proc.returncode == 0 and (not readonly or not changed) and (not needs_change or bool(changed))
        error_type = None
        if proc.returncode != 0:
            error_type = "subprocess_failed"
        elif readonly and changed:
            error_type = "readonly_changed_files"
        elif needs_change and not changed:
            error_type = "no_changed_files"
        result = CodexSubagentResult(
            ok=ok,
            timed_out=False,
            exit_code=proc.returncode,
            stdout_tail=stdout.text,
            stderr_tail=stderr.text,
            changed_files=changed,
            duration_s=duration,
            error_type=error_type,
            stdout_tail_truncated=stdout.truncated,
            stderr_tail_truncated=stderr.truncated,
        )
        _emit(
            event_sink,
            "codex_subagent_completed" if ok else "codex_subagent_failed",
            _event_payload(
                task_type=task_type,
                timeout_s=timeout_s,
                duration_s=duration,
                exit_code=proc.returncode,
                changed_file_count=len(changed),
                stdout_tail_truncated=stdout.truncated,
                stderr_tail_truncated=stderr.truncated,
            ),
        )
        return result
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        after = _snapshot(workspace)
        stdout = _tail(exc.stdout or "", max_output_chars)
        stderr = _tail(exc.stderr or "", max_output_chars)
        changed = _changed_files(before, after)
        result = CodexSubagentResult(
            ok=False,
            timed_out=True,
            exit_code=None,
            stdout_tail=stdout.text,
            stderr_tail=stderr.text,
            changed_files=changed,
            duration_s=duration,
            error_type="timeout",
            stdout_tail_truncated=stdout.truncated,
            stderr_tail_truncated=stderr.truncated,
        )
        _emit(
            event_sink,
            "codex_subagent_timeout",
            _event_payload(
                task_type=task_type,
                timeout_s=timeout_s,
                duration_s=duration,
                exit_code=None,
                changed_file_count=len(changed),
                stdout_tail_truncated=stdout.truncated,
                stderr_tail_truncated=stderr.truncated,
            ),
        )
        return result
    except OSError as exc:
        duration = time.monotonic() - start
        stderr = _tail(str(exc), max_output_chars)
        result = CodexSubagentResult(
            ok=False,
            timed_out=False,
            exit_code=None,
            stdout_tail="",
            stderr_tail=stderr.text,
            changed_files=_changed_files(before, _snapshot(workspace)),
            duration_s=duration,
            error_type="subprocess_failed",
        )
        _emit(
            event_sink,
            "codex_subagent_failed",
            _event_payload(
                task_type=task_type,
                timeout_s=timeout_s,
                duration_s=duration,
                exit_code=None,
                changed_file_count=len(result.changed_files),
                stdout_tail_truncated=False,
                stderr_tail_truncated=stderr.truncated,
            ),
        )
        return result


__all__ = [
    "CodexSubagentResult",
    "check_codex_available",
    "run_codex_subagent",
]
