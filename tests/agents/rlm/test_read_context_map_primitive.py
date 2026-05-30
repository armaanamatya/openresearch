from __future__ import annotations

import pytest

from backend.agents.rlm import context_map as cm
from backend.agents.rlm.primitives import read_context_map


class _Ctx:
    def __init__(self, project_dir):
        self.project_dir = project_dir


def test_read_context_map_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    out = read_context_map(ctx=_Ctx(tmp_path))
    assert out == {"version": "v1", "bytes": 0, "entries": []}


def test_read_context_map_returns_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    out = read_context_map(ctx=_Ctx(tmp_path))
    assert out["entries"][0]["key"] == "extract_hyperparameters:batch_size"


def test_read_context_map_failsoft(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "context_map.json").write_text("{bad")
    out = read_context_map(ctx=_Ctx(tmp_path))
    assert out["entries"] == []


def test_read_context_map_registered():
    from backend.agents.rlm.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_DESCRIPTIONS
    assert "read_context_map" in PRIMITIVE_REGISTRY
    assert "read_context_map" in PRIMITIVE_DESCRIPTIONS
