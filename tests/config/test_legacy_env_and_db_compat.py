"""Characterization tests for the two REPROLAB_->OPENRESEARCH_ rename-safety
mechanisms (research 2026-06-21 flagged both as UNTESTED). These lock in the
NON-BREAKING contract before any canonicalize work touches it.

- `_apply_legacy_env_aliases` (config.py): bidirectional env bridge, never overwrites.
- `_fall_back_to_legacy_sqlite_db` (Settings model_validator): open an existing
  legacy `reprolab.db` when the new default `openresearch.db` is absent.
"""
import os
from pathlib import Path

import pytest

from backend.config import Settings, _apply_legacy_env_aliases


# ---------------------------------------------------------------------------
# Bridge: REPROLAB_* <-> OPENRESEARCH_* (config.py::_apply_legacy_env_aliases)
# ---------------------------------------------------------------------------
def test_bridge_legacy_to_new(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_FOO_TEST", raising=False)
    monkeypatch.setenv("REPROLAB_FOO_TEST", "legacy-value")
    _apply_legacy_env_aliases()
    assert os.environ.get("OPENRESEARCH_FOO_TEST") == "legacy-value"


def test_bridge_new_to_legacy(monkeypatch):
    monkeypatch.delenv("REPROLAB_BAR_TEST", raising=False)
    monkeypatch.setenv("OPENRESEARCH_BAR_TEST", "new-value")
    _apply_legacy_env_aliases()
    assert os.environ.get("REPROLAB_BAR_TEST") == "new-value"


def test_bridge_never_overwrites_explicit(monkeypatch):
    # Both spellings explicitly set to different values -> neither is clobbered.
    monkeypatch.setenv("REPROLAB_BAZ_TEST", "legacy")
    monkeypatch.setenv("OPENRESEARCH_BAZ_TEST", "new")
    _apply_legacy_env_aliases()
    assert os.environ["REPROLAB_BAZ_TEST"] == "legacy"
    assert os.environ["OPENRESEARCH_BAZ_TEST"] == "new"


# ---------------------------------------------------------------------------
# DB fallback (Settings::_fall_back_to_legacy_sqlite_db)
# ---------------------------------------------------------------------------
@pytest.fixture
def _isolated_cwd(tmp_path, monkeypatch):
    """Run in an empty cwd with no DB-url env override, so the validator sees
    only the files we create."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENRESEARCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("REPROLAB_DATABASE_URL", raising=False)
    return tmp_path


def test_db_default_when_neither_file_present(_isolated_cwd):
    s = Settings(database_url="sqlite:///openresearch.db")
    assert s.database_url == "sqlite:///openresearch.db"


def test_db_falls_back_to_legacy_when_only_reprolab_present(_isolated_cwd):
    Path("reprolab.db").write_text("")  # legacy file exists, new does not
    s = Settings(database_url="sqlite:///openresearch.db")
    assert s.database_url == "sqlite:///reprolab.db"


def test_db_keeps_new_when_both_present(_isolated_cwd):
    Path("reprolab.db").write_text("")
    Path("openresearch.db").write_text("")  # new file exists -> no fallback
    s = Settings(database_url="sqlite:///openresearch.db")
    assert s.database_url == "sqlite:///openresearch.db"


def test_db_explicit_override_untouched(_isolated_cwd):
    Path("reprolab.db").write_text("")  # legacy present, but URL is explicit/non-default
    s = Settings(database_url="sqlite:///custom.db")
    assert s.database_url == "sqlite:///custom.db"
