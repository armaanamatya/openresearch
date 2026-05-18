"""Run-scoped root-logger configuration (Tier 2a).

When REPROLAB_LOG_DIR (or REPROLAB_RUNS_ROOT, used as the fallback for the
common case where the dev launcher sets only the latter) is defined,
installs two FileHandlers on the root logger:

    <run_dir>/pipeline.log    — human-readable text log
    <run_dir>/pipeline.jsonl  — one JSON record per line, for verify_run.py

Every existing ``logging.getLogger(__name__).info(...)`` call across the
agents lands in both. No call-site changes required.

Idempotent + env-gated: calling ``configure_root_logger()`` repeatedly is
safe, and it is a no-op when neither env var is set (tests, production,
ad-hoc CLI invocations are unaffected).

See docs/design/tier2-observability-plan.md for the broader plan.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_LOGGING_CONFIGURED = False
_LOCK = threading.Lock()


class _JsonlFormatter(logging.Formatter):
    """One JSON object per line — line-delimited so external tools can stream
    the file with ``for line in open(...): json.loads(line)`` without having
    to parse a full JSON document."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_run_dir() -> Optional[Path]:
    """REPROLAB_LOG_DIR wins when set; falls back to REPROLAB_RUNS_ROOT.

    The dev launchers (``scripts/dev.ps1`` / ``scripts/dev.sh``) currently
    only export ``REPROLAB_RUNS_ROOT`` — letting that double as the log
    directory keeps the launcher contract simple. Operators who want the
    pipeline workspaces and the consolidated log to live in *different*
    directories can set both env vars explicitly.
    """
    value = os.environ.get("REPROLAB_LOG_DIR") or os.environ.get(
        "REPROLAB_RUNS_ROOT"
    )
    if not value:
        return None
    return Path(value)


def configure_root_logger() -> Optional[Path]:
    """Install FileHandlers on the root logger.

    Returns the run directory that was configured, or ``None`` if neither
    env var is set (a true no-op). Idempotent — repeat calls during the
    same process lifetime are skipped, so it's safe to call from multiple
    entry points (FastAPI startup, CLI main, the in-line subprocess that
    backend/services/events/live_runs.py spawns).
    """
    global _LOGGING_CONFIGURED
    with _LOCK:
        if _LOGGING_CONFIGURED:
            return _resolve_run_dir()

        run_dir = _resolve_run_dir()
        if run_dir is None:
            # Mark as configured so we don't keep re-checking env on every
            # entry-point startup; if env var arrives later, the caller can
            # explicitly reset via _LOGGING_CONFIGURED if needed.
            _LOGGING_CONFIGURED = True
            return None

        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            _LOGGING_CONFIGURED = True
            return None

        root = logging.getLogger()

        # delay=True means the file is opened lazily on the first emit. Without
        # it, FileHandler pre-creates a 0-byte file at construction — verifier
        # checks then flag it as a zero-byte text file even when no agent run
        # ever happened.
        text_handler = logging.FileHandler(
            run_dir / "pipeline.log", encoding="utf-8", delay=True
        )
        text_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        text_handler.setLevel(logging.INFO)

        jsonl_handler = logging.FileHandler(
            run_dir / "pipeline.jsonl", encoding="utf-8", delay=True
        )
        jsonl_handler.setFormatter(_JsonlFormatter())
        jsonl_handler.setLevel(logging.INFO)

        # Tag so verify_run.py and follow-up Tier 2b code can find these
        # handlers without re-matching on path strings.
        text_handler.set_name("reprolab.pipeline.log")
        jsonl_handler.set_name("reprolab.pipeline.jsonl")

        root.addHandler(text_handler)
        root.addHandler(jsonl_handler)

        # Without an explicit setLevel, the default of WARNING swallows all
        # the agent INFO lines the handlers were installed to capture.
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

        _LOGGING_CONFIGURED = True
        return run_dir


# ---------------------------------------------------------------------------
# Tier 2b — per-agent-invocation transcript recorder
#
# Captures the StreamText / StreamToolCall / StreamUsage events that
# resilience/engine.py already produces during `runtime.run_agent(...)`,
# and fans them into
#
#   <project_dir>/agents/<NN>-<agent_id>/
#       prompt.md          # exact user_input passed to the runtime
#       trace.log          # concatenated StreamText events
#       tool_calls.jsonl   # one StreamToolCall per line
#       usage.json         # latest StreamUsage frame
#       result.txt         # final consolidated agent output
#       meta.json          # agent_id, seq, started_at, ended_at, status, retries
#
# Threaded through the orchestrator + resilience engine via a contextvar so
# we don't have to alter any public signatures. All recorder calls are
# best-effort: an I/O failure inside the recorder must never bubble up
# into the agent execution path.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _safe_jsonable(x: Any) -> Any:
    try:
        json.dumps(x, ensure_ascii=False)
        return x
    except (TypeError, ValueError):
        return repr(x)


class AgentTranscriptRecorder:
    """Owns one ``agents/<NN>-<agent_id>/`` directory for one invocation.

    Use as a context manager so retries / cancellations / exceptions all
    close the trace and tool_calls handles deterministically::

        with AgentTranscriptRecorder(project_dir, "paper-understanding", 1, prompt) as rec:
            set_recorder(rec)
            ...                       # runtime.run_agent streams here
            rec.finalize(output, "ok")
    """

    def __init__(
        self,
        project_dir: Path,
        agent_id: str,
        seq: int,
        prompt: str,
    ) -> None:
        self.agent_id = agent_id
        self.seq = seq
        self.dir = project_dir / "agents" / f"{seq:02d}-{agent_id}"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "prompt.md").write_text(prompt, encoding="utf-8")
            self._trace = (self.dir / "trace.log").open("a", encoding="utf-8")
            self._tools = (self.dir / "tool_calls.jsonl").open(
                "a", encoding="utf-8"
            )
            self._enabled = True
        except OSError:
            # Disk full, perms — degrade silently. The agent path keeps
            # running; we just don't capture this invocation.
            self._enabled = False
            self._trace = None  # type: ignore[assignment]
            self._tools = None  # type: ignore[assignment]
        self.started_at = _now_iso()
        self.msg_count = 0
        self.tool_call_count = 0
        self.text_chars = 0
        self.retries = 0
        self._finalized = False

    # ---- context manager ----
    def __enter__(self) -> "AgentTranscriptRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If finalize() wasn't called (e.g. exception escaped), record a
        # best-effort "interrupted" outcome so the directory isn't an
        # ambiguous half-state on disk.
        if not self._finalized:
            error = f"{exc_type.__name__}: {exc}" if exc_type else None
            self.finalize(result="", status="interrupted", error=error)

    # ---- streaming hooks (called from resilience/engine.py event loop) ----
    def record_text(self, text: str) -> None:
        if not self._enabled or not text:
            return
        try:
            self._trace.write(text)
            self._trace.flush()
            self.text_chars += len(text)
            self.msg_count += 1
        except OSError:
            pass

    def record_tool(
        self,
        *,
        tool_id: str,
        tool_name: str,
        tool_input: Any,
    ) -> None:
        if not self._enabled:
            return
        try:
            line = json.dumps(
                {
                    "ts": _now_iso(),
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "tool_input": _safe_jsonable(tool_input),
                },
                ensure_ascii=False,
            )
            self._tools.write(line + "\n")
            self._tools.flush()
            self.tool_call_count += 1
            self.msg_count += 1
        except OSError:
            pass

    def record_usage(self, usage: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            (self.dir / "usage.json").write_text(
                json.dumps(usage, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def record_retry(self, attempt: int, reason: str) -> None:
        if not self._enabled:
            return
        try:
            self._trace.write(f"\n--- retry {attempt} ({reason}) ---\n")
            self._trace.flush()
            self.retries += 1
        except OSError:
            pass

    # ---- finalize (called from orchestrator._invoke_agent finally block) ----
    def finalize(
        self,
        result: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        if self._finalized:
            return
        self._finalized = True
        if not self._enabled:
            return
        try:
            (self.dir / "result.txt").write_text(result or "", encoding="utf-8")
            (self.dir / "meta.json").write_text(
                json.dumps(
                    {
                        "agent_id": self.agent_id,
                        "seq": self.seq,
                        "started_at": self.started_at,
                        "ended_at": _now_iso(),
                        "status": status,
                        "error": error,
                        "msg_count": self.msg_count,
                        "tool_call_count": self.tool_call_count,
                        "text_chars": self.text_chars,
                        "retries": self.retries,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        # Always close handles, even if writes failed.
        for handle in (self._trace, self._tools):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass


# ----- contextvar plumbing ---------------------------------------------------

_RECORDER_VAR: contextvars.ContextVar[Optional[AgentTranscriptRecorder]] = (
    contextvars.ContextVar("reprolab_agent_recorder", default=None)
)


def get_recorder() -> Optional[AgentTranscriptRecorder]:
    """Return the recorder for the current asyncio task, if any.

    Returns ``None`` outside an active ``_invoke_agent`` scope OR when
    REPROLAB_LOG_DIR / REPROLAB_RUNS_ROOT is unset. Callers in
    resilience/engine.py should treat ``None`` as "don't record" and
    branch silently.
    """
    return _RECORDER_VAR.get()


def set_recorder(rec: Optional[AgentTranscriptRecorder]) -> contextvars.Token:
    """Install ``rec`` as the active recorder. Returns a token; pass it to
    ``reset_recorder`` from the orchestrator's finally block to restore."""
    return _RECORDER_VAR.set(rec)


def reset_recorder(token: contextvars.Token) -> None:
    _RECORDER_VAR.reset(token)


def recording_enabled() -> bool:
    """Whether per-agent transcripts should be captured.

    Mirrors the configure_root_logger() gate — when no run_dir is set, we
    skip all the recorder bookkeeping in the orchestrator hot path.
    """
    return _resolve_run_dir() is not None
