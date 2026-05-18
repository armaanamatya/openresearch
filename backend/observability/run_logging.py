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

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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

        text_handler = logging.FileHandler(
            run_dir / "pipeline.log", encoding="utf-8"
        )
        text_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        text_handler.setLevel(logging.INFO)

        jsonl_handler = logging.FileHandler(
            run_dir / "pipeline.jsonl", encoding="utf-8"
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
