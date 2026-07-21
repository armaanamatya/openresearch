"""Central structured-logging config for OpenResearch — opt-in, no default change.

The codebase had no central logging setup (scattered ``logging.basicConfig`` in a
couple of scripts; everything else a bare ``getLogger(__name__)``), so log level
and format were whatever the root logger happened to default to. This module
provides the building blocks:

- :func:`resolve_level` — env-driven level (``OPENRESEARCH_LOG_LEVEL``) so an
  operator can quiet or deepen logs without code changes ("less noise / levels").
- :class:`JsonFormatter` — one-line JSON per record for queryable/grep-able logs,
  carrying any structured extra fields (e.g. ``project_id``, ``gate``).
- :class:`ProjectIdFilter` — stamps a run/project correlation id onto every record.
- :func:`configure_logging` — idempotent root setup, opt-in.

Nothing here runs unless an entrypoint calls ``configure_logging()`` (or sets
``OPENRESEARCH_LOG_FORMAT``/``OPENRESEARCH_LOG_LEVEL`` and calls it), so importing
this module changes no behavior.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import traceback
from typing import Any

__all__ = ["resolve_level", "JsonFormatter", "ProjectIdFilter", "configure_logging"]

_DEFAULT_LEVEL = logging.INFO

# LogRecord attributes that are framework internals, not structured payload.
_STD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
})


def resolve_level(raw: Any) -> int:
    """Resolve a log level from a name (``"DEBUG"``) or number (``"10"``).

    Junk / empty / ``None`` → ``INFO`` (never raises), so a mis-set env var can
    never crash startup or silence logging entirely.
    """
    if raw is None:
        return _DEFAULT_LEVEL
    s = str(raw).strip()
    if not s:
        return _DEFAULT_LEVEL
    if s.isdigit():
        return int(s)
    level = logging.getLevelName(s.upper())
    return level if isinstance(level, int) else _DEFAULT_LEVEL


class JsonFormatter(logging.Formatter):
    """Format a record as a single JSON line with core fields + structured extras."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, _dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Structured extras (anything set via logger.info(..., extra={...})).
        for key, val in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_") and key not in payload:
                payload[key] = val
        if record.exc_info:
            payload["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).strip()
        return json.dumps(payload, default=str)


class ProjectIdFilter(logging.Filter):
    """Stamp a run/project correlation id onto every record (never clobbers one
    already set via ``extra={"project_id": ...}``)."""

    def __init__(self, project_id: str) -> None:
        super().__init__()
        self.project_id = project_id

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "project_id"):
            record.project_id = self.project_id
        return True


def configure_logging(
    *, level: str | int | None = None, fmt: str | None = None, force: bool = False
) -> logging.Handler:
    """Idempotently configure the root logger. Opt-in — call from an entrypoint.

    Level: explicit ``level`` arg → ``OPENRESEARCH_LOG_LEVEL`` → ``INFO``.
    Format: explicit ``fmt`` (``"json"``/``"text"``) → ``OPENRESEARCH_LOG_FORMAT`` → ``"text"``.

    Idempotent: a second call replaces this module's own handler rather than
    stacking a duplicate (unless ``force`` installs a fresh one). Returns the handler.
    """
    root = logging.getLogger()
    resolved_level = resolve_level(level if level is not None else os.environ.get("OPENRESEARCH_LOG_LEVEL"))
    resolved_fmt = (fmt or os.environ.get("OPENRESEARCH_LOG_FORMAT") or "text").strip().lower()

    # Remove any handler we previously installed (marked) so we stay idempotent.
    for h in list(root.handlers):
        if getattr(h, "_openresearch_managed", False) and not force:
            root.removeHandler(h)

    handler = logging.StreamHandler()
    handler._openresearch_managed = True  # type: ignore[attr-defined]
    if resolved_fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(resolved_level)
    return handler
