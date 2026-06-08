"""PEEK-lite intra-run context map (FLAG-1, OPENRESEARCH_CONTEXT_MAP)."""
import json

import pytest

from backend.agents.rlm import context_map as cm


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CONTEXT_MAP", "on")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CONTEXT_MAP", raising=False)


# --- off-state contract -----------------------------------------------------

def test_off_state_update_is_noop_and_read_empty(off, tmp_path):
    cm.update_context_map(tmp_path, "understand_section", {"datasets": ["alfworld"]})
    assert not (tmp_path / "rlm_state" / "context_map.json").exists()
    assert cm.read_context_map(tmp_path) == {}


# --- on-state: union + sourcing --------------------------------------------

def test_union_across_orientation_primitives(on, tmp_path):
    cm.update_context_map(tmp_path, "understand_section", {"datasets": ["alfworld"]})
    cm.update_context_map(tmp_path, "extract_hyperparameters", {"lr": 1e-4, "beta": 10})
    cm.update_context_map(tmp_path, "detect_environment", {"datasets": ["webshop"]})
    m = cm.read_context_map(tmp_path)
    assert set(m["datasets"]) == {"alfworld", "webshop"}  # unioned, deduped
    assert m["lr"] == ["0.0001"]
    assert m["beta"] == ["10"]


def test_non_orientation_primitive_is_ignored(on, tmp_path):
    cm.update_context_map(tmp_path, "run_experiment", {"success": True})
    assert cm.read_context_map(tmp_path) == {}


def test_dedupes_and_caps_values(on, tmp_path):
    for i in range(20):
        cm.update_context_map(tmp_path, "understand_section", {"metric": [f"m{i}"]})
    cm.update_context_map(tmp_path, "understand_section", {"metric": ["m0"]})  # dup
    m = cm.read_context_map(tmp_path)
    assert len(m["metric"]) == cm.MAX_VALUES


def test_field_budget_capped(on, tmp_path):
    big = {f"f{i}": f"v{i}" for i in range(cm.MAX_FIELDS + 10)}
    cm.update_context_map(tmp_path, "detect_environment", big)
    assert len(cm.read_context_map(tmp_path)) <= cm.MAX_FIELDS


def test_byte_ceiling_enforced(on, tmp_path):
    payload = {f"field{i}": "x" * cm.MAX_VALUE_LEN for i in range(cm.MAX_FIELDS)}
    cm.update_context_map(tmp_path, "understand_section", payload)
    blob = (tmp_path / "rlm_state" / "context_map.json").read_text()
    assert len(blob.encode("utf-8")) <= cm.MAX_BYTES


def test_nested_and_empty_values_skipped(on, tmp_path):
    cm.update_context_map(
        tmp_path,
        "understand_section",
        {"nested": {"a": 1}, "empty": "", "none": None, "good": "keep"},
    )
    m = cm.read_context_map(tmp_path)
    assert m == {"good": ["keep"]}


def test_fail_soft_on_corrupt_file(on, tmp_path):
    p = tmp_path / "rlm_state" / "context_map.json"
    p.parent.mkdir(parents=True)
    p.write_text("{ not json", encoding="utf-8")
    # update must not raise; it overwrites with a valid map
    cm.update_context_map(tmp_path, "understand_section", {"datasets": ["x"]})
    assert cm.read_context_map(tmp_path)["datasets"] == ["x"]


# --- primitive + registry wiring -------------------------------------------

def test_read_context_map_primitive_registered():
    from backend.agents.rlm.primitives import (
        PRIMITIVE_REGISTRY,
        PRIMITIVE_DESCRIPTIONS,
    )

    assert "read_context_map" in PRIMITIVE_REGISTRY
    assert "read_context_map" in PRIMITIVE_DESCRIPTIONS


def test_read_primitive_delegates(on, tmp_path):
    from types import SimpleNamespace
    from backend.agents.rlm.primitives import read_context_map as prim

    cm.update_context_map(tmp_path, "understand_section", {"datasets": ["alfworld"]})
    out = prim(ctx=SimpleNamespace(project_dir=tmp_path))
    assert out["datasets"] == ["alfworld"]
