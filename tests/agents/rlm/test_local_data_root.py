"""Tests for run._ensure_local_data_root — the 2026-05-29 SDAR local env_load_failed fix.

Local sandboxes have no /workspace volume; the helper repoints the volume-mount data
root (OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH) at a writable shared cache so dataset/env
setup does not die at os.makedirs. RunPod/Docker keep /workspace; explicit overrides win.
"""
from __future__ import annotations

import os
from pathlib import Path

from backend.agents.rlm.run import _ensure_local_data_root


def test_local_sandbox_sets_writable_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", raising=False)
    _ensure_local_data_root("local", tmp_path)
    root = os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH")
    assert root, "local sandbox must set a data root"
    assert Path(root).is_dir(), "the data root must be created (writable)"
    assert str(tmp_path) in root, "data root must live under runs_root, not /workspace"


def test_local_sandbox_overrides_workspace_default(tmp_path, monkeypatch):
    """The bare /workspace default is treated as 'unset' and replaced for local."""
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", "/workspace")
    _ensure_local_data_root("local", tmp_path)
    assert os.environ["OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH"] != "/workspace"


def test_local_sandbox_respects_explicit_override(tmp_path, monkeypatch):
    """An operator-pinned writable root is never clobbered."""
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", "/mnt/custom")
    _ensure_local_data_root("local", tmp_path)
    assert os.environ["OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH"] == "/mnt/custom"


def test_runpod_keeps_workspace_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", raising=False)
    _ensure_local_data_root("runpod", tmp_path)
    # untouched → callers fall back to /workspace
    assert not os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH")


def test_enum_like_sandbox_mode(tmp_path, monkeypatch):
    """sandbox_mode may be an enum exposing .value (e.g. SandboxMode.local)."""
    monkeypatch.delenv("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH", raising=False)

    class _Mode:
        value = "local"

    _ensure_local_data_root(_Mode(), tmp_path)
    assert os.environ.get("OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH")
