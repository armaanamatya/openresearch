"""Tests for refresh_harness_helpers — the $0 vendored-helper refresh (Track B).

The cell-level resume path re-copies the stdlib-only harness helpers into a
prior run's ``code/`` WITHOUT a codegen pass, so a bug-fixed helper (e.g.
``alfworld_env.py``) actually reaches disk and the fingerprint reflects it.
This verifies the public wrapper copies the expected files, is idempotent +
refreshes stale content, and creates a missing target dir.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.baseline_implementation import (
    _HARNESS_CODE_HELPERS,
    refresh_harness_helpers,
)


def test_copies_all_harness_helpers(tmp_path):
    code = tmp_path / "code"
    refresh_harness_helpers(code)
    for helper in _HARNESS_CODE_HELPERS:
        assert (code / helper).is_file(), f"{helper} not copied"


def test_creates_missing_code_dir_and_returns_path(tmp_path):
    code = tmp_path / "does_not_exist_yet" / "code"
    assert not code.exists()
    returned = refresh_harness_helpers(code)
    assert returned == code
    assert code.is_dir()


def test_overwrites_stale_helper_content(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    # A stale copy from a prior warm-retry.
    stale = code / "gpu_cell_runner.py"
    stale.write_text("# STALE\n", encoding="utf-8")

    refresh_harness_helpers(code)

    refreshed = stale.read_text(encoding="utf-8")
    assert refreshed != "# STALE\n"
    # The real module's content is back.
    assert "run_matrix" in refreshed


def test_idempotent(tmp_path):
    code = tmp_path / "code"
    refresh_harness_helpers(code)
    first = {h: (code / h).read_bytes() for h in _HARNESS_CODE_HELPERS}
    refresh_harness_helpers(code)
    second = {h: (code / h).read_bytes() for h in _HARNESS_CODE_HELPERS}
    assert first == second


def test_accepts_str_path(tmp_path):
    code = tmp_path / "code"
    returned = refresh_harness_helpers(str(code))
    assert isinstance(returned, Path)
    assert (returned / "sdar_env_base.py").is_file()
