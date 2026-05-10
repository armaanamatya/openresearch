"""Load .env into os.environ once at package import time.

pydantic-settings already reads .env for ``REPROLAB_*`` Settings fields,
but it never mutates ``os.environ``. Several call-sites read raw API
keys directly from the process environment:

* ``backend/hermes_audit/providers.py`` —
  ``ClaudeAuditProvider.is_available`` checks ``ANTHROPIC_API_KEY``;
  ``OpenAIAuditProvider.is_available`` checks ``OPENAI_API_KEY``;
  the SDK constructors then read the same vars internally.
* ``backend/services/runtime/runpod_backend.py`` — falls back to
  ``REPROLAB_RUNPOD_API_KEY`` / ``RUNPOD_API_KEY`` from os.environ.

Without this bootstrap, those reads return ``None`` whenever the
process was started by something other than a shell that had already
``source``-d the .env (Lab UI's child python spawn, ``pytest``,
``python -m backend.cli``, the docker entrypoint, …) and the whole
provider chain skips with ``is_available() == False``.

Contract:

* Runs exactly once per interpreter (idempotent).
* ``os.environ.setdefault`` semantics — never clobbers values the
  caller already exported. The shell / docker ``-e`` flag wins.
* Walks up from this file looking for a ``pyproject.toml`` so it works
  regardless of cwd (cwd-only lookup matches pydantic, but breaks for
  callers who run from a subdirectory).
* Silent on missing files — production deployments may inject env
  vars directly and never carry a .env on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []

    override = os.environ.get("REPROLAB_DOTENV_PATH")
    if override:
        candidates.append(Path(override))

    candidates.append(Path.cwd() / ".env")

    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        candidates.append(ancestor / ".env")
        if (ancestor / "pyproject.toml").exists():
            break

    seen: set[Path] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _apply(path: Path) -> int:
    applied = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            applied += 1
    return applied


def load_dotenv_once() -> Path | None:
    """Load the first available .env into os.environ.

    Returns the path that was loaded, or ``None`` if no .env was found
    or this function has already run in this interpreter.
    """
    global _LOADED
    if _LOADED:
        return None
    _LOADED = True
    for candidate in _candidate_paths():
        if candidate.is_file():
            _apply(candidate)
            return candidate
    return None


__all__ = ["load_dotenv_once"]
