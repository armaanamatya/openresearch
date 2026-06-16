"""Unit tests for backend.agents.rlm.champion_artifact (Workstream A4).

Properties under test:
  * snapshot_code excludes heavy artifacts (*.pt, datasets/) but keeps *.py
  * snapshot_code removes a stale dest before copying
  * restore_snapshot round-trips source, and does NOT delete heavy artifacts
    (datasets/, outputs/, *.pt) that live in code_dir
  * record_champion / best_champion: highest median wins; tie -> most recent
  * registry is fail-soft (missing/corrupt -> empty)
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm import champion_artifact as ca


def _make_code_dir(root, *, with_artifacts=True):
    code = root / "code"
    code.mkdir()
    (code / "train.py").write_text("print('train')\n", encoding="utf-8")
    (code / "model.py").write_text("class M: pass\n", encoding="utf-8")
    sub = code / "pkg"
    sub.mkdir()
    (sub / "util.py").write_text("X = 1\n", encoding="utf-8")
    if with_artifacts:
        (code / "weights.pt").write_bytes(b"\x00" * 1024)
        ds = code / "datasets"
        ds.mkdir()
        (ds / "cifar.bin").write_bytes(b"\x01" * 4096)
        outs = code / "outputs"
        outs.mkdir()
        (outs / "metrics.json").write_text('{"per_model": {}}', encoding="utf-8")
        venv = code / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /x\n", encoding="utf-8")
    return code


# --- snapshot_code ---------------------------------------------------------

def test_snapshot_keeps_source_excludes_heavy(tmp_path):
    code = _make_code_dir(tmp_path)
    snap = ca.snapshot_code(code, tmp_path / "snap")

    assert (snap / "train.py").is_file()
    assert (snap / "model.py").is_file()
    assert (snap / "pkg" / "util.py").is_file()
    # Heavy artifacts excluded.
    assert not (snap / "weights.pt").exists()
    assert not (snap / "datasets").exists()
    assert not (snap / "outputs").exists()
    assert not (snap / ".venv").exists()


def test_snapshot_returns_dest(tmp_path):
    code = _make_code_dir(tmp_path, with_artifacts=False)
    dest = tmp_path / "snap"
    out = ca.snapshot_code(code, dest)
    assert out == dest


def test_snapshot_removes_stale_dest(tmp_path):
    code = _make_code_dir(tmp_path, with_artifacts=False)
    dest = tmp_path / "snap"
    dest.mkdir()
    (dest / "stale.py").write_text("# old\n", encoding="utf-8")
    ca.snapshot_code(code, dest)
    # Stale file from the prior write must be gone.
    assert not (dest / "stale.py").exists()
    assert (dest / "train.py").is_file()


# --- restore_snapshot ------------------------------------------------------

def test_restore_round_trips_source(tmp_path):
    code = _make_code_dir(tmp_path, with_artifacts=False)
    snap = ca.snapshot_code(code, tmp_path / "snap")
    # Mutate the live code (simulate a regressing repair).
    (code / "train.py").write_text("BROKEN\n", encoding="utf-8")
    (code / "model.py").write_text("BROKEN\n", encoding="utf-8")

    ca.restore_snapshot(snap, code)

    assert (code / "train.py").read_text(encoding="utf-8") == "print('train')\n"
    assert (code / "model.py").read_text(encoding="utf-8") == "class M: pass\n"
    assert (code / "pkg" / "util.py").read_text(encoding="utf-8") == "X = 1\n"


def test_restore_does_not_delete_heavy_artifacts(tmp_path):
    code = _make_code_dir(tmp_path, with_artifacts=True)
    # Snapshot is source-only.
    snap = ca.snapshot_code(code, tmp_path / "snap")
    # Regress source, then restore.
    (code / "train.py").write_text("BROKEN\n", encoding="utf-8")
    ca.restore_snapshot(snap, code)

    # Source restored ...
    assert (code / "train.py").read_text(encoding="utf-8") == "print('train')\n"
    # ... and the heavy artifacts that live in code/ are untouched.
    assert (code / "weights.pt").is_file()
    assert (code / "weights.pt").stat().st_size == 1024
    assert (code / "datasets" / "cifar.bin").is_file()
    assert (code / "outputs" / "metrics.json").is_file()


def test_restore_creates_missing_code_dir(tmp_path):
    code = _make_code_dir(tmp_path, with_artifacts=False)
    snap = ca.snapshot_code(code, tmp_path / "snap")
    fresh = tmp_path / "fresh_code"
    ca.restore_snapshot(snap, fresh)
    assert (fresh / "train.py").is_file()


# --- registry --------------------------------------------------------------

def test_record_and_best_highest_median(tmp_path):
    reg = tmp_path / "champions.json"
    ca.record_champion(reg, evidence_key="k1", snapshot_dir=tmp_path / "s1", median_score=0.60)
    ca.record_champion(reg, evidence_key="k2", snapshot_dir=tmp_path / "s2", median_score=0.81)
    ca.record_champion(reg, evidence_key="k3", snapshot_dir=tmp_path / "s3", median_score=0.55)

    best = ca.best_champion(reg)
    assert best is not None
    assert best["evidence_key"] == "k2"
    assert best["median_score"] == pytest.approx(0.81)


def test_tie_breaks_to_most_recent(tmp_path):
    reg = tmp_path / "champions.json"
    ca.record_champion(reg, evidence_key="early", snapshot_dir=tmp_path / "s1", median_score=0.70)
    ca.record_champion(reg, evidence_key="late", snapshot_dir=tmp_path / "s2", median_score=0.70)
    best = ca.best_champion(reg)
    assert best["evidence_key"] == "late"


def test_record_persists_json_list(tmp_path):
    reg = tmp_path / "champions.json"
    ca.record_champion(reg, evidence_key="k1", snapshot_dir="s1", median_score=0.5)
    data = json.loads(reg.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["evidence_key"] == "k1"
    assert data[0]["seq"] == 1


def test_seq_is_monotonic(tmp_path):
    reg = tmp_path / "champions.json"
    e1 = ca.record_champion(reg, evidence_key="a", snapshot_dir="s", median_score=0.1)
    e2 = ca.record_champion(reg, evidence_key="b", snapshot_dir="s", median_score=0.2)
    assert e1["seq"] == 1 and e2["seq"] == 2


def test_best_champion_empty_returns_none(tmp_path):
    assert ca.best_champion(tmp_path / "does_not_exist.json") is None


def test_registry_corrupt_reads_empty(tmp_path):
    reg = tmp_path / "champions.json"
    reg.write_text("{ not valid json", encoding="utf-8")
    assert ca.best_champion(reg) is None
    # And a record over a corrupt file still succeeds (overwrites).
    ca.record_champion(reg, evidence_key="k", snapshot_dir="s", median_score=0.9)
    best = ca.best_champion(reg)
    assert best is not None and best["evidence_key"] == "k"


def test_record_non_numeric_median_coerces(tmp_path):
    reg = tmp_path / "champions.json"
    entry = ca.record_champion(reg, evidence_key="k", snapshot_dir="s", median_score=None)  # type: ignore[arg-type]
    assert entry["median_score"] == 0.0


def test_record_returns_in_memory_entry_when_write_fails(tmp_path):
    # Point the registry at a path whose parent is a FILE -> mkdir/write fails;
    # record_champion must still return the entry (fail-soft).
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    reg = blocker / "champions.json"
    entry = ca.record_champion(reg, evidence_key="k", snapshot_dir="s", median_score=0.4)
    assert entry["evidence_key"] == "k"
    assert entry["median_score"] == pytest.approx(0.4)
