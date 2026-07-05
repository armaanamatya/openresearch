"""Tests for the cells.json operator pre-seed (Task #7).

An operator can pre-author a training-grid manifest and point
``OPENRESEARCH_CELLS_SEED_PATH`` at it; ``_seed_cells_manifest`` copies it into
``code/cells.json`` on the FIRST ``implement_baseline`` call (``code/cells.json``
absent) so the harness guarantees the cells-route manifest exists regardless
of the executor sub-agent's own code-generation quality.

Default-OFF / byte-identical: unset -> no cells.json created; an EXISTING
cells.json (a prior implement_baseline call, or a repair pass) is never
overwritten.
"""
from __future__ import annotations

import inspect
import json

from backend.agents.rlm import primitives
from backend.agents.rlm.primitives import _seed_cells_manifest


# ---------------------------------------------------------------------------
# ON: seeds when the env var is set and code/cells.json is absent
# ---------------------------------------------------------------------------

class TestSeedCellsManifestOn:
    def test_seeds_when_path_set_and_cells_json_absent(self, tmp_path, monkeypatch):
        seed_src = tmp_path / "operator_cells.json"
        seed_src.write_text(json.dumps([{"id": "c0"}]), encoding="utf-8")
        monkeypatch.setenv("OPENRESEARCH_CELLS_SEED_PATH", str(seed_src))

        code_dir = tmp_path / "code"
        code_dir.mkdir()

        result = _seed_cells_manifest(code_dir)

        assert result is True
        dest = code_dir / "cells.json"
        assert dest.exists()
        assert json.loads(dest.read_text(encoding="utf-8")) == [{"id": "c0"}]

    def test_creates_code_dir_if_missing(self, tmp_path, monkeypatch):
        seed_src = tmp_path / "operator_cells.json"
        seed_src.write_text("[]", encoding="utf-8")
        monkeypatch.setenv("OPENRESEARCH_CELLS_SEED_PATH", str(seed_src))

        code_dir = tmp_path / "code"  # deliberately not pre-created

        result = _seed_cells_manifest(code_dir)

        assert result is True
        assert (code_dir / "cells.json").exists()


# ---------------------------------------------------------------------------
# OFF: unset -> no-op; existing manifest is never clobbered; fail-soft
# ---------------------------------------------------------------------------

class TestSeedCellsManifestOff:
    def test_unset_env_var_seeds_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_CELLS_SEED_PATH", raising=False)

        code_dir = tmp_path / "code"
        code_dir.mkdir()

        result = _seed_cells_manifest(code_dir)

        assert result is False
        assert not (code_dir / "cells.json").exists()

    def test_existing_cells_json_is_never_overwritten(self, tmp_path, monkeypatch):
        seed_src = tmp_path / "operator_cells.json"
        seed_src.write_text(json.dumps([{"id": "operator"}]), encoding="utf-8")
        monkeypatch.setenv("OPENRESEARCH_CELLS_SEED_PATH", str(seed_src))

        code_dir = tmp_path / "code"
        code_dir.mkdir()
        existing = code_dir / "cells.json"
        existing.write_text(json.dumps([{"id": "already-there"}]), encoding="utf-8")

        result = _seed_cells_manifest(code_dir)

        assert result is False
        assert json.loads(existing.read_text(encoding="utf-8")) == [{"id": "already-there"}]

    def test_missing_seed_source_file_is_fail_soft(self, tmp_path, monkeypatch):
        """A nonexistent OPENRESEARCH_CELLS_SEED_PATH must never raise — the
        seed is best-effort and must not block the run."""
        monkeypatch.setenv("OPENRESEARCH_CELLS_SEED_PATH", str(tmp_path / "does-not-exist.json"))

        code_dir = tmp_path / "code"
        code_dir.mkdir()

        result = _seed_cells_manifest(code_dir)

        assert result is False
        assert not (code_dir / "cells.json").exists()

    def test_blank_env_var_is_treated_as_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_CELLS_SEED_PATH", "   ")

        code_dir = tmp_path / "code"
        code_dir.mkdir()

        result = _seed_cells_manifest(code_dir)

        assert result is False
        assert not (code_dir / "cells.json").exists()


# ---------------------------------------------------------------------------
# Wiring: implement_baseline must call the helper (source-level check — a full
# implement_baseline invocation needs a RunContext + SDK subprocess and is out
# of scope for a unit test).
# ---------------------------------------------------------------------------

class TestImplementBaselineWiring:
    def test_implement_baseline_calls_seed_cells_manifest(self):
        src = inspect.getsource(primitives.implement_baseline)
        assert "_seed_cells_manifest(code_dir)" in src

    def test_seed_call_is_after_repo_seed_block_and_before_route_retention(self):
        """Ordering per the task spec: immediately after the repo-seed block,
        before the route-retention _had_cells_manifest stash."""
        src = inspect.getsource(primitives.implement_baseline)
        repo_seed_idx = src.index("repo_code_seeded")
        seed_call_idx = src.index("_seed_cells_manifest(code_dir)")
        retention_idx = src.index("_had_cells_manifest")
        assert repo_seed_idx < seed_call_idx < retention_idx
