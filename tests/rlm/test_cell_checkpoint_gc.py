"""Tests for the post-cell checkpoint GC added to gpu_cell_runner.

Hermetic (tmp_path, monkeypatch) — no subprocess, no GPU, no network.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.gpu_cell_runner import (
    _gc_cell_weights,
    cell_checkpoint_gc_enabled,
)


# ---------------------------------------------------------------------------
# 1. Default OFF — nothing deleted even when a .pt file is present
# ---------------------------------------------------------------------------

def test_gc_disabled_by_default_no_deletion(tmp_path, monkeypatch):
    """With the flag unset (default), _gc_cell_weights is a no-op."""
    monkeypatch.delenv("OPENRESEARCH_CELL_CHECKPOINT_GC", raising=False)

    weight = tmp_path / "model_checkpoint.pt"
    weight.write_bytes(b"\x00" * 1024)
    evidence = tmp_path / "metrics.json"
    evidence.write_text('{"loss": 0.5}')

    removed = _gc_cell_weights(tmp_path)

    assert removed == 0
    assert weight.exists(), "weight file must NOT be deleted when flag is off"
    assert evidence.exists()


def test_gc_flag_false_values(tmp_path, monkeypatch):
    """Explicitly-false env values are all no-ops."""
    for val in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", val)
        weight = tmp_path / f"model_{val}.pt"
        weight.write_bytes(b"\x00" * 16)
        removed = _gc_cell_weights(tmp_path)
        assert removed == 0
        assert weight.exists()


# ---------------------------------------------------------------------------
# 2. Flag ON — weight file deleted, evidence files preserved
# ---------------------------------------------------------------------------

def test_gc_removes_pt_keeps_evidence(tmp_path, monkeypatch):
    """Flag on: .pt removed; metrics.json, provenance.json, training_curves.json survive."""
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", "1")

    weight = tmp_path / "model_checkpoint.pt"
    weight.write_bytes(b"\x00" * 2048)

    evidence_files = {
        "metrics.json": '{"accuracy": 0.9}',
        "provenance.json": '{"run_id": "abc"}',
        "training_curves.json": "{}",
    }
    for name, content in evidence_files.items():
        (tmp_path / name).write_text(content)

    removed = _gc_cell_weights(tmp_path)

    assert removed == 1
    assert not weight.exists(), "model_checkpoint.pt must be deleted"
    for name in evidence_files:
        assert (tmp_path / name).exists(), f"{name} must NOT be deleted"


def test_gc_removes_multiple_weight_types(tmp_path, monkeypatch):
    """Flag on: .safetensors, optimizer.pt, .bin, .ckpt all deleted; others kept."""
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", "true")

    weights = {
        "model.safetensors": b"\x01" * 32,
        "optimizer.pt": b"\x02" * 32,
        "model.bin": b"\x03" * 32,
        "pytorch_model.ckpt": b"\x04" * 32,
    }
    for name, data in weights.items():
        (tmp_path / name).write_bytes(data)

    # Evidence file that must survive
    env_health = tmp_path / "env_health.jsonl"
    env_health.write_text('{"ok":true}\n')
    cell_manifest = tmp_path / "cell_manifest.json"
    cell_manifest.write_text('{"status":"ok"}')

    removed = _gc_cell_weights(tmp_path)

    assert removed == len(weights)
    for name in weights:
        assert not (tmp_path / name).exists(), f"{name} must be deleted"
    assert env_health.exists()
    assert cell_manifest.exists()


def test_gc_removes_checkpoints_subdir(tmp_path, monkeypatch):
    """Flag on: .pt inside checkpoints/ subdir is deleted."""
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", "yes")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_weight = ckpt_dir / "epoch_5.pt"
    ckpt_weight.write_bytes(b"\x00" * 64)

    # A weight in the top-level dir too
    top_weight = tmp_path / "final_model.pt"
    top_weight.write_bytes(b"\x00" * 64)

    # Evidence that must survive
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    commands_log = tmp_path / "commands.log"
    commands_log.write_text("echo done\n")

    removed = _gc_cell_weights(tmp_path)

    assert removed == 2
    assert not ckpt_weight.exists()
    assert not top_weight.exists()
    assert metrics.exists()
    assert commands_log.exists()


# ---------------------------------------------------------------------------
# 4. Fail-soft: non-existent dir → returns 0, no raise
# ---------------------------------------------------------------------------

def test_gc_nonexistent_dir_no_raise(monkeypatch):
    """Non-existent directory: returns 0 without raising."""
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", "1")
    result = _gc_cell_weights("/nonexistent/path/that/does/not/exist/42")
    assert result == 0


def test_gc_flag_on_but_dir_missing(tmp_path, monkeypatch):
    """When the dir itself is missing, return 0 cleanly."""
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", "on")
    missing = tmp_path / "no_such_cell"
    result = _gc_cell_weights(missing)
    assert result == 0


# ---------------------------------------------------------------------------
# 5. Never deletes a non-weight file (e.g. stray notes.txt)
# ---------------------------------------------------------------------------

def test_gc_does_not_delete_non_weight_files(tmp_path, monkeypatch):
    """Only weight-blob suffixes are eligible; other extensions are untouched."""
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", "1")

    # Weight to be cleaned
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"\x00" * 16)

    # Non-weight files that must all survive
    non_weights = [
        "notes.txt",
        "run_log.csv",
        "config.yaml",
        "train.py",
        "metrics.json",     # evidence file AND wrong suffix
        "eval_provenance.json",
        "debug.log",
    ]
    for name in non_weights:
        (tmp_path / name).write_text("content")

    removed = _gc_cell_weights(tmp_path)

    assert removed == 1
    assert not weight.exists()
    for name in non_weights:
        assert (tmp_path / name).exists(), f"{name} must NOT be deleted"


# ---------------------------------------------------------------------------
# Helper: cell_checkpoint_gc_enabled() sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    ("1", True),
    ("true", True),
    ("yes", True),
    ("on", True),
    ("TRUE", True),
    ("YES", True),
    ("ON", True),
    ("0", False),
    ("false", False),
    ("off", False),
    ("no", False),
    ("", False),
    ("  ", False),
])
def test_gc_enabled_flag_parsing(val, expected, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CELL_CHECKPOINT_GC", val)
    assert cell_checkpoint_gc_enabled() is expected


def test_gc_enabled_unset(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CELL_CHECKPOINT_GC", raising=False)
    assert cell_checkpoint_gc_enabled() is False
