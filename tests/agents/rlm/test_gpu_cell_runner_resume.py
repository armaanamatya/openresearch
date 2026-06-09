"""Tests for gpu_cell_runner's cell-level resume (Track B).

Resume is gated on the env var ``OPENRESEARCH_RESUME_CELLS``.  When armed, a cell
whose prior ``cell_manifest.json`` is status=ok + fingerprint-matched + not in
``force_cells`` is recorded ``status="skipped"`` WITHOUT launching a subprocess.

The skip-path tests use a REAL cell_script that writes a SENTINEL file on every
launch, so "the cell was skipped" is proven by the sentinel's ABSENCE — there is
no way to fake not-running.  The re-run cases (mismatch / force / flag-unset)
assert the subprocess DID run.  No GPUs / torch / CUDA required — the script is
trivial Python pinned to one fake gpu id.

The suite verifies:
  * armed + prior ok + fingerprint match → SKIPPED, subprocess NOT launched.
  * fingerprint MISMATCH → re-runs.
  * cell in force_cells → re-runs even on a match.
  * flag unset → always runs (default behaviour unchanged).
  * a terminal cell writes an authoritative cell_manifest.json (ok + failed).
  * the manifest carries the fingerprint and (when supplied) completed_at.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import backend.agents.rlm.gpu_cell_runner as gcr
from backend.agents.rlm.gpu_cell_runner import CELL_MANIFEST_NAME, run_matrix


# ---------------------------------------------------------------------------
# A real, trivial single-cell trainer.  Writes a per-launch sentinel + metrics.
# It NEVER touches CUDA, so a fake gpu id ("0") is fine.
# ---------------------------------------------------------------------------

_CELL_SCRIPT_SRC = '''\
import json
import os
import sys

out = os.environ["OPENRESEARCH_CELL_OUTPUT_DIR"]
os.makedirs(out, exist_ok=True)

# Sentinel proving THIS process actually launched.  Append so multiple launches
# of the same cell are visible (one char per launch).
with open(os.path.join(out, "launched.sentinel"), "a", encoding="utf-8") as fh:
    fh.write("x")

with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
    json.dump({"status": "ok", "metric": 0.5, "reward_mean": 1.2}, fh)

sys.exit(0)
'''


@pytest.fixture()
def cell_script(tmp_path: Path) -> Path:
    p = tmp_path / "train_cell.py"
    p.write_text(_CELL_SCRIPT_SRC, encoding="utf-8")
    return p


def _cell(cell_id: str = "c0", env: str = "alfworld") -> dict:
    return {"id": cell_id, "model_key": "qwen3_1_7b", "baseline": "sdar",
            "env": env, "seed": 42}


def _seed_prior_ok_cell(output_root: Path, cell_id: str, fingerprint: str) -> Path:
    """Pre-seed a cell output_dir as if a prior run finished it ok."""
    output_dir = output_root / cell_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"status": "ok", "metric": 0.42}), encoding="utf-8"
    )
    (output_dir / CELL_MANIFEST_NAME).write_text(
        json.dumps({"cell_id": cell_id, "status": "ok", "fingerprint": fingerprint,
                    "metric": 0.42, "retries": 0}),
        encoding="utf-8",
    )
    return output_dir


def _sentinel(output_root: Path, cell_id: str) -> Path:
    return output_root / cell_id / "launched.sentinel"


# ---------------------------------------------------------------------------
# Skip path — armed + ok + fingerprint match → SKIPPED, subprocess NOT launched
# ---------------------------------------------------------------------------

class TestResumeSkip:
    def test_matching_fingerprint_is_skipped_without_launch(
        self, tmp_path, cell_script, monkeypatch
    ):
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP_MATCH")

        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP_MATCH"},
        )

        assert results["c0"]["status"] == "skipped"
        # The prior metrics are reused.
        assert results["c0"]["metrics"]["metric"] == 0.42
        # The REAL trainer never ran → no sentinel was written.
        assert not _sentinel(out, "c0").exists()

    def test_skipped_cell_keeps_prior_metrics(self, tmp_path, cell_script, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP")
        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},
        )
        assert results["c0"]["metrics"] == {"status": "ok", "metric": 0.42}

    def test_mixed_skip_and_run(self, tmp_path, cell_script, monkeypatch):
        """One matched (skip) + one unmatched (run) in the same matrix."""
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP0")
        _seed_prior_ok_cell(out, "c1", "OLD")  # stale fingerprint → must run

        results = run_matrix(
            [_cell("c0"), _cell("c1")], cell_script, output_root=out, gpus=["0", "1"],
            fingerprints={"c0": "FP0", "c1": "NEW"},
        )

        assert results["c0"]["status"] == "skipped"
        assert not _sentinel(out, "c0").exists()
        assert results["c1"]["status"] == "ok"
        assert _sentinel(out, "c1").exists()


# ---------------------------------------------------------------------------
# Re-run paths — mismatch / force / flag-unset
# ---------------------------------------------------------------------------

class TestResumeRerun:
    def test_fingerprint_mismatch_reruns(self, tmp_path, cell_script, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "STORED_OLD")

        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "CURRENT_NEW"},  # differs from stored → re-run
        )

        assert results["c0"]["status"] == "ok"
        assert _sentinel(out, "c0").exists()  # the trainer ran

    def test_force_cells_reruns_on_match(self, tmp_path, cell_script, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP")

        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},  # matches…
            force_cells={"c0"},          # …but force overrides → re-run
        )

        assert results["c0"]["status"] == "ok"
        assert _sentinel(out, "c0").exists()

    def test_flag_unset_always_runs(self, tmp_path, cell_script, monkeypatch):
        # Resume NOT armed → a perfectly-matching prior cell STILL re-runs.
        monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP")

        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},
        )

        assert results["c0"]["status"] == "ok"
        assert _sentinel(out, "c0").exists()

    def test_no_fingerprint_for_cell_runs(self, tmp_path, cell_script, monkeypatch):
        # Armed but no current fingerprint to compare → cannot assert unchanged → run.
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP")
        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={},  # empty → no skip
        )
        assert results["c0"]["status"] == "ok"
        assert _sentinel(out, "c0").exists()

    def test_no_prior_manifest_runs(self, tmp_path, cell_script, monkeypatch):
        # Armed + fingerprint supplied, but the cell has never run (no manifest).
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},
        )
        assert results["c0"]["status"] == "ok"
        assert _sentinel(out, "c0").exists()

    def test_prior_failed_manifest_reruns(self, tmp_path, cell_script, monkeypatch):
        """A prior NON-ok manifest never authorises a skip, even if fp matches."""
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        output_dir = out / "c0"
        output_dir.mkdir(parents=True)
        (output_dir / CELL_MANIFEST_NAME).write_text(
            json.dumps({"cell_id": "c0", "status": "oom_failed", "fingerprint": "FP"}),
            encoding="utf-8",
        )
        results = run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},
        )
        assert results["c0"]["status"] == "ok"
        assert _sentinel(out, "c0").exists()


# ---------------------------------------------------------------------------
# Manifest writing — every terminal cell writes an authoritative manifest
# ---------------------------------------------------------------------------

class TestManifestWriting:
    def test_ok_cell_writes_manifest_with_fingerprint(
        self, tmp_path, cell_script, monkeypatch
    ):
        monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)
        out = tmp_path / "outputs"
        run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP_ABC"}, now_iso="2026-06-02T12:00:00Z",
        )
        manifest = json.loads((out / "c0" / CELL_MANIFEST_NAME).read_text())
        assert manifest["cell_id"] == "c0"
        assert manifest["status"] == "ok"
        assert manifest["fingerprint"] == "FP_ABC"
        assert manifest["metric"] == 0.5  # headline pulled from metrics.json
        assert manifest["completed_at"] == "2026-06-02T12:00:00Z"

    def test_manifest_omits_completed_at_when_now_iso_none(
        self, tmp_path, cell_script, monkeypatch
    ):
        monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)
        out = tmp_path / "outputs"
        run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},  # now_iso defaults to None
        )
        manifest = json.loads((out / "c0" / CELL_MANIFEST_NAME).read_text())
        assert "completed_at" not in manifest

    def test_manifest_fingerprint_null_when_not_supplied(
        self, tmp_path, cell_script, monkeypatch
    ):
        monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)
        out = tmp_path / "outputs"
        run_matrix([_cell("c0")], cell_script, output_root=out, gpus=["0"])
        manifest = json.loads((out / "c0" / CELL_MANIFEST_NAME).read_text())
        assert manifest["fingerprint"] is None

    def test_failed_cell_writes_manifest(self, tmp_path, monkeypatch):
        """A non-OOM error is terminal → it must also write a manifest."""
        monkeypatch.delenv("OPENRESEARCH_RESUME_CELLS", raising=False)
        out = tmp_path / "outputs"
        # A script that always exits non-zero with a NON-OOM message → status=error.
        bad = tmp_path / "bad_cell.py"
        bad.write_text("import sys; sys.exit(3)", encoding="utf-8")

        results = run_matrix(
            [_cell("c0")], bad, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"},
        )
        assert results["c0"]["status"] == "error"
        manifest = json.loads((out / "c0" / CELL_MANIFEST_NAME).read_text())
        assert manifest["status"] == "error"
        assert manifest["fingerprint"] == "FP"
        assert manifest["metric"] is None

    def test_skipped_cell_does_not_overwrite_prior_manifest(
        self, tmp_path, cell_script, monkeypatch
    ):
        """A skip reuses the prior manifest verbatim — no rewrite needed."""
        monkeypatch.setenv("OPENRESEARCH_RESUME_CELLS", "1")
        out = tmp_path / "outputs"
        _seed_prior_ok_cell(out, "c0", "FP")
        before = (out / "c0" / CELL_MANIFEST_NAME).read_text()
        run_matrix(
            [_cell("c0")], cell_script, output_root=out, gpus=["0"],
            fingerprints={"c0": "FP"}, now_iso="2026-06-02T00:00:00Z",
        )
        after = (out / "c0" / CELL_MANIFEST_NAME).read_text()
        assert before == after  # untouched
