"""Canonical call-time readers for ``OPENRESEARCH_*`` boolean feature flags.

The RLM layer reads most flags directly from ``os.environ`` at call time (not
through cached ``config.py`` Settings) on purpose: a run can toggle behaviour
mid-flight and tests monkeypatch ``os.environ`` after import (see the naming
note in ``CLAUDE.md`` — Settings monkeypatching does not bridge post-import).

This module keeps that call-time, env-first semantics but removes the truthy
parsing that was duplicated across ~9 sites, so "what counts as on" lives in one
place. No caching — every call re-reads ``os.environ``.
"""
from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes", "on")


def env_truthy(name: str, default: bool = False) -> bool:
    """True iff ``os.environ[name]`` is a truthy token, read at call time.

    Empty/unset falls back to ``default``. Truthy tokens: 1/true/yes/on
    (case-insensitive, surrounding whitespace stripped).
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def use_author_repo() -> bool:
    """``OPENRESEARCH_USE_AUTHOR_REPO`` — the #62 repo-first master gate."""
    return env_truthy("OPENRESEARCH_USE_AUTHOR_REPO")
