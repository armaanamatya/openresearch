"""Tests for the cell-level resume CLI selection logic (Track B).

These cover the PURE, testable core of the --resume-cells short-circuit in
isolation (no live run, no GPU, no codegen):

  * argparse: the four new flags parse with the right defaults + types.
  * _build_resume_force_cells: --rerun-env expands to matching cell ids,
    --rerun-cell adds explicit ids, both compose, unknown ids are no-ops.
  * _load_any_cell_manifest: finds the newest manifest under outputs/*/<cell_id>/.
  * _select_cells_needing_rerun: the re-run predicate (forced / no-prior-ok /
    fingerprint-changed re-run; unchanged-ok is skipped) using REAL fingerprints.

The full run wiring (RunContext build → run_experiment) is exercised by the
integration seam described in the report, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm import cell_fingerprint
from backend.agents.rlm.gpu_cell_runner import CELL_MANIFEST_NAME
from backend.cli import (
    _build_parser,
    _build_resume_force_cells,
    _load_any_cell_manifest,
    _select_cells_needing_rerun,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cell(cid: str, env: str, **over) -> dict:
    cell = {
        "id": cid, "model_key": cid.split("__")[0], "baseline": "sdar",
        "env": env, "seed": 42, "steps": 150, "model_id": "Qwen/Qwen3-1.7B",
    }
    cell.update(over)
    return cell


_MATRIX = [
    _cell("qwen3_1_7b__sdar__alfworld__s42", "alfworld"),
    _cell("qwen3_1_7b__sdar__search_qa__s42", "search_qa"),
    _cell("qwen2_5_3b__sdar__webshop__s42", "webshop"),
]


def _seed_helpers(code_dir: Path) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("alfworld_env.py", "search_qa_env.py", "webshop_env.py",
                 "sdar_env_base.py", "agentic_rollout.py"):
        (code_dir / name).write_text(f"# {name} v1\n", encoding="utf-8")


def _write_manifest(code_dir: Path, run_id: str, cell: dict, *, status: str, fp: str) -> None:
    d = code_dir / "outputs" / run_id / cell["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / CELL_MANIFEST_NAME).write_text(
        json.dumps({"cell_id": cell["id"], "status": status, "fingerprint": fp}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

class TestArgparse:
    def test_flags_parse(self):
        ns = _build_parser().parse_args([
            "reproduce", "2605.15155",
            "--resume-cells",
            "--rerun-env", "alfworld", "--rerun-env", "webshop",
            "--rerun-cell", "c1", "--rerun-cell", "c2",
            "--force-regen",
        ])
        assert ns.resume_cells is True
        assert ns.rerun_env == ["alfworld", "webshop"]
        assert ns.rerun_cell == ["c1", "c2"]
        assert ns.force_regen is True

    def test_defaults(self):
        ns = _build_parser().parse_args(["reproduce", "2605.15155"])
        assert ns.resume_cells is False
        assert ns.rerun_env is None
        assert ns.rerun_cell is None
        assert ns.force_regen is False

    def test_rerun_env_choices_enforced(self):
        import pytest
        with pytest.raises(SystemExit):
            _build_parser().parse_args([
                "reproduce", "x", "--resume-cells", "--rerun-env", "not_an_env",
            ])


# ---------------------------------------------------------------------------
# _build_resume_force_cells
# ---------------------------------------------------------------------------

class TestBuildForceCells:
    def test_rerun_env_expands_to_matching_ids(self):
        forced = _build_resume_force_cells(_MATRIX, ["alfworld"], None)
        assert forced == {"qwen3_1_7b__sdar__alfworld__s42"}

    def test_multiple_envs(self):
        forced = _build_resume_force_cells(_MATRIX, ["alfworld", "webshop"], None)
        assert forced == {
            "qwen3_1_7b__sdar__alfworld__s42",
            "qwen2_5_3b__sdar__webshop__s42",
        }

    def test_rerun_cell_adds_explicit_ids(self):
        forced = _build_resume_force_cells(_MATRIX, None, ["qwen3_1_7b__sdar__search_qa__s42"])
        assert forced == {"qwen3_1_7b__sdar__search_qa__s42"}

    def test_env_and_cell_compose(self):
        forced = _build_resume_force_cells(
            _MATRIX, ["webshop"], ["qwen3_1_7b__sdar__search_qa__s42"]
        )
        assert forced == {
            "qwen2_5_3b__sdar__webshop__s42",
            "qwen3_1_7b__sdar__search_qa__s42",
        }

    def test_unknown_cell_id_is_kept_as_noop(self):
        # A typo'd id is kept verbatim — harmless (matches no cell) rather than dropped.
        forced = _build_resume_force_cells(_MATRIX, None, ["does_not_exist"])
        assert forced == {"does_not_exist"}

    def test_unknown_env_matches_nothing(self):
        forced = _build_resume_force_cells(_MATRIX, ["nonexistent_env"], None)
        assert forced == set()

    def test_none_none_is_empty(self):
        assert _build_resume_force_cells(_MATRIX, None, None) == set()

    def test_case_insensitive_env(self):
        forced = _build_resume_force_cells(_MATRIX, ["ALFWorld"], None)
        assert forced == {"qwen3_1_7b__sdar__alfworld__s42"}


# ---------------------------------------------------------------------------
# _load_any_cell_manifest
# ---------------------------------------------------------------------------

class TestLoadAnyCellManifest:
    def test_finds_manifest(self, tmp_path):
        _write_manifest(tmp_path, "run_a", _MATRIX[0], status="ok", fp="FP")
        got = _load_any_cell_manifest(tmp_path, _MATRIX[0]["id"])
        assert got is not None and got["status"] == "ok" and got["fingerprint"] == "FP"

    def test_missing_returns_none(self, tmp_path):
        assert _load_any_cell_manifest(tmp_path, "nope") is None

    def test_no_outputs_dir_returns_none(self, tmp_path):
        # code dir exists but no outputs/ at all.
        (tmp_path / "code").mkdir()
        assert _load_any_cell_manifest(tmp_path / "code", "c0") is None

    def test_newest_run_wins(self, tmp_path):
        import os
        import time
        _write_manifest(tmp_path, "run_old", _MATRIX[0], status="oom_failed", fp="OLD")
        # Force the old manifest's mtime into the past so ordering is deterministic.
        old_mf = tmp_path / "outputs" / "run_old" / _MATRIX[0]["id"] / CELL_MANIFEST_NAME
        os.utime(old_mf, (time.time() - 100, time.time() - 100))
        _write_manifest(tmp_path, "run_new", _MATRIX[0], status="ok", fp="NEW")
        got = _load_any_cell_manifest(tmp_path, _MATRIX[0]["id"])
        assert got["fingerprint"] == "NEW"  # newest mtime wins


# ---------------------------------------------------------------------------
# _select_cells_needing_rerun — the re-run predicate end-to-end
# ---------------------------------------------------------------------------

class TestSelectCellsNeedingRerun:
    def test_unchanged_ok_is_skipped(self, tmp_path):
        code = tmp_path / "code"
        _seed_helpers(code)
        # Seed an ok manifest with the CURRENT fingerprint for every cell.
        for c in _MATRIX:
            fp = cell_fingerprint.compute_fingerprint(c, str(code))
            _write_manifest(code, "run0", c, status="ok", fp=fp)
        selected = _select_cells_needing_rerun(_MATRIX, code, set())
        assert selected == {}  # all unchanged → nothing to re-run

    def test_no_prior_manifest_selects_all(self, tmp_path):
        code = tmp_path / "code"
        _seed_helpers(code)
        selected = _select_cells_needing_rerun(_MATRIX, code, set())
        assert set(selected) == {c["id"] for c in _MATRIX}
        assert all(reason == "no_prior_ok" for reason in selected.values())

    def test_fingerprint_change_selects_only_changed(self, tmp_path):
        code = tmp_path / "code"
        _seed_helpers(code)
        for c in _MATRIX:
            fp = cell_fingerprint.compute_fingerprint(c, str(code))
            _write_manifest(code, "run0", c, status="ok", fp=fp)
        # Edit ONLY the alfworld helper → only the alfworld cell's fp changes.
        (code / "alfworld_env.py").write_text("# alfworld_env.py v2\n", encoding="utf-8")
        selected = _select_cells_needing_rerun(_MATRIX, code, set())
        assert set(selected) == {"qwen3_1_7b__sdar__alfworld__s42"}
        assert selected["qwen3_1_7b__sdar__alfworld__s42"] == "fingerprint_changed"

    def test_forced_cell_selected_even_when_unchanged(self, tmp_path):
        code = tmp_path / "code"
        _seed_helpers(code)
        for c in _MATRIX:
            fp = cell_fingerprint.compute_fingerprint(c, str(code))
            _write_manifest(code, "run0", c, status="ok", fp=fp)
        forced = {"qwen2_5_3b__sdar__webshop__s42"}
        selected = _select_cells_needing_rerun(_MATRIX, code, forced)
        assert selected == {"qwen2_5_3b__sdar__webshop__s42": "forced"}

    def test_prior_failed_is_selected(self, tmp_path):
        code = tmp_path / "code"
        _seed_helpers(code)
        c = _MATRIX[0]
        fp = cell_fingerprint.compute_fingerprint(c, str(code))
        _write_manifest(code, "run0", c, status="oom_failed", fp=fp)  # matched fp but failed
        selected = _select_cells_needing_rerun([c], code, set())
        assert selected == {c["id"]: "no_prior_ok"}

    def test_force_plus_changed_compose(self, tmp_path):
        code = tmp_path / "code"
        _seed_helpers(code)
        for c in _MATRIX:
            fp = cell_fingerprint.compute_fingerprint(c, str(code))
            _write_manifest(code, "run0", c, status="ok", fp=fp)
        (code / "search_qa_env.py").write_text("# search_qa_env.py v2\n", encoding="utf-8")
        forced = {"qwen2_5_3b__sdar__webshop__s42"}
        selected = _select_cells_needing_rerun(_MATRIX, code, forced)
        assert set(selected) == {
            "qwen2_5_3b__sdar__webshop__s42",       # forced
            "qwen3_1_7b__sdar__search_qa__s42",     # fingerprint changed
        }
        assert selected["qwen2_5_3b__sdar__webshop__s42"] == "forced"
        assert selected["qwen3_1_7b__sdar__search_qa__s42"] == "fingerprint_changed"
