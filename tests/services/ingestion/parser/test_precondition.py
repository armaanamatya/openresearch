"""Tests for PR-π Module E — parsed_full_text.txt precondition gate.

Tests the _assert_paper_text_precondition helper from backend.agents.rlm.run:
  - When allow_lossy=False and file missing → RuntimeError.
  - When allow_lossy=True and file missing → warning logged, no exception.
  - When file present with ≥1KB content → no exception, no warning.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from backend.agents.rlm.run import _assert_paper_text_precondition


# ---------------------------------------------------------------------------
# Test 1: strict mode + missing file → RuntimeError
# ---------------------------------------------------------------------------

def test_run_fails_fast_when_parsed_full_text_missing_and_strict(tmp_path: Path) -> None:
    """When allow_lossy=False and parsed_full_text.txt does not exist,
    _assert_paper_text_precondition must raise RuntimeError with a helpful message.
    """
    project_dir = tmp_path / "prj_strict"
    project_dir.mkdir()
    # No parsed_full_text.txt created.

    with pytest.raises(RuntimeError) as exc_info:
        _assert_paper_text_precondition(project_dir, allow_lossy=False)

    msg = str(exc_info.value)
    assert "parsed_full_text.txt" in msg
    assert "missing" in msg.lower() or "1kb" in msg.lower() or "1 kb" in msg.lower() or "<1" in msg


def test_run_fails_fast_when_parsed_full_text_too_small_and_strict(tmp_path: Path) -> None:
    """When allow_lossy=False and parsed_full_text.txt is <1KB, raise RuntimeError."""
    project_dir = tmp_path / "prj_small"
    project_dir.mkdir()
    (project_dir / "parsed_full_text.txt").write_text("short", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        _assert_paper_text_precondition(project_dir, allow_lossy=False)

    assert "parsed_full_text.txt" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2: lossy allowed + missing file → warning, no exception
# ---------------------------------------------------------------------------

def test_run_proceeds_with_warning_when_lossy_allowed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When allow_lossy=True and parsed_full_text.txt is missing, no exception is
    raised, but a WARNING must be emitted mentioning the degraded state.
    """
    project_dir = tmp_path / "prj_lossy"
    project_dir.mkdir()
    # No parsed_full_text.txt created.

    with caplog.at_level(logging.WARNING, logger="backend.agents.rlm.run"):
        _assert_paper_text_precondition(project_dir, allow_lossy=True)  # must not raise

    assert any("degraded" in record.message.lower() or "lossy" in record.message.lower()
               for record in caplog.records), (
        f"Expected a warning about degraded/lossy state; got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Test 3: file present with ≥1KB content → no exception, no warning
# ---------------------------------------------------------------------------

def test_run_proceeds_when_parsed_full_text_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When parsed_full_text.txt exists and is ≥1KB, no exception and no WARNING
    about degraded state.
    """
    project_dir = tmp_path / "prj_ok"
    project_dir.mkdir()
    # Write slightly more than 1 KB of content.
    (project_dir / "parsed_full_text.txt").write_text("x" * 2048, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="backend.agents.rlm.run"):
        _assert_paper_text_precondition(project_dir, allow_lossy=False)  # must not raise

    # No degraded/lossy warning should appear.
    degraded_records = [
        r for r in caplog.records
        if "degraded" in r.message.lower() or "lossy" in r.message.lower()
    ]
    assert not degraded_records, f"Unexpected warning(s): {[r.message for r in degraded_records]}"
