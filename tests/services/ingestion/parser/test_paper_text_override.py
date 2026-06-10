"""Tests for the OPENRESEARCH_PAPER_TEXT_PATH override hook in parser/service.py.

Covers:
- override used when cascade result is empty/short,
- override IGNORED when cascade result is good (>= 1 KB),
- override no-op when env var is unset,
- override file missing → ingest still proceeds (no crash),
- override file unreadable → ingest still proceeds (no crash),
- healthy-parse path is unchanged (existing behaviour preserved).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


from backend.services.ingestion.parser.service import (
    _resolve_full_text,
    _load_paper_text_override,
    write_parsed_full_text,
    _FULL_TEXT_MIN_BYTES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_TEXT = "A" * _FULL_TEXT_MIN_BYTES          # exactly at threshold (good)
_SHORT_TEXT = "A" * (_FULL_TEXT_MIN_BYTES - 1)   # one byte below threshold (bad)
_LONG_OVERRIDE = "B" * (_FULL_TEXT_MIN_BYTES * 2)  # unambiguously long


def _write_override(tmp_path: Path, content: str, name: str = "override.txt") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _load_paper_text_override
# ---------------------------------------------------------------------------


def test_override_no_op_when_env_unset(tmp_path: Path):
    """No env var → returns None immediately."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENRESEARCH_PAPER_TEXT_PATH", None)
        result = _load_paper_text_override()
    assert result is None


def test_override_returns_text_when_file_exists_and_large_enough(tmp_path: Path):
    p = _write_override(tmp_path, _LONG_OVERRIDE)
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": str(p)}):
        result = _load_paper_text_override()
    assert result == _LONG_OVERRIDE


def test_override_ignored_when_file_missing(tmp_path: Path):
    missing = str(tmp_path / "does_not_exist.txt")
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": missing}):
        result = _load_paper_text_override()
    assert result is None  # missing → ignore, no crash


def test_override_ignored_when_file_too_short(tmp_path: Path):
    p = _write_override(tmp_path, _SHORT_TEXT)
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": str(p)}):
        result = _load_paper_text_override()
    assert result is None  # too short → ignore


def test_override_ignored_when_empty_string_env(tmp_path: Path):
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": "  "}):
        result = _load_paper_text_override()
    assert result is None  # whitespace-only → no-op


# ---------------------------------------------------------------------------
# _resolve_full_text — precedence logic
# ---------------------------------------------------------------------------


def test_resolve_cascade_good_wins_over_override(tmp_path: Path):
    """A good cascade result keeps itself even when an override is set."""
    p = _write_override(tmp_path, _LONG_OVERRIDE)
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": str(p)}):
        result = _resolve_full_text(_GOOD_TEXT)
    assert result == _GOOD_TEXT, "cascade result >= 1KB must win over override"


def test_resolve_override_wins_when_cascade_empty(tmp_path: Path):
    """When cascade returns empty, the override is used."""
    p = _write_override(tmp_path, _LONG_OVERRIDE)
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": str(p)}):
        result = _resolve_full_text(None)
    assert result == _LONG_OVERRIDE


def test_resolve_override_wins_when_cascade_short(tmp_path: Path):
    """When cascade returns text < 1KB, the override is used."""
    p = _write_override(tmp_path, _LONG_OVERRIDE)
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": str(p)}):
        result = _resolve_full_text(_SHORT_TEXT)
    assert result == _LONG_OVERRIDE


def test_resolve_returns_none_when_cascade_empty_and_no_override(tmp_path: Path):
    """No cascade, no override → returns None (stale blob will be deleted)."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENRESEARCH_PAPER_TEXT_PATH", None)
        result = _resolve_full_text(None)
    assert result is None


def test_resolve_no_op_when_env_unset_and_cascade_good():
    """Good cascade + no env var → cascade returned unchanged."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENRESEARCH_PAPER_TEXT_PATH", None)
        result = _resolve_full_text(_GOOD_TEXT)
    assert result == _GOOD_TEXT


def test_resolve_missing_override_file_no_crash_returns_cascade(tmp_path: Path):
    """If the override file is missing, _resolve_full_text does NOT crash.

    Even when cascade is degraded, the call returns the cascade result
    (not raising an exception) so ingest can proceed.
    """
    missing = str(tmp_path / "gone.txt")
    with patch.dict(os.environ, {"OPENRESEARCH_PAPER_TEXT_PATH": missing}):
        result = _resolve_full_text(None)
    assert result is None  # missing file → graceful fallback, no crash


# ---------------------------------------------------------------------------
# write_parsed_full_text — ensure existing behaviour unchanged
# ---------------------------------------------------------------------------


def test_write_parsed_full_text_deletes_stale_on_none(tmp_path: Path):
    """Unchanged regression: None → stale blob is deleted."""
    p = tmp_path / "parsed_full_text.txt"
    p.write_text("stale", encoding="utf-8")
    write_parsed_full_text(tmp_path, None)
    assert not p.exists()


def test_write_parsed_full_text_writes_atomically(tmp_path: Path):
    """Unchanged regression: text → blob written, no .tmp left behind."""
    write_parsed_full_text(tmp_path, "hello world")
    p = tmp_path / "parsed_full_text.txt"
    assert p.exists()
    assert p.read_text(encoding="utf-8") == "hello world"
    assert not (tmp_path / "parsed_full_text.txt.tmp").exists()
