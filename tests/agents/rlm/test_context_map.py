from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from backend.agents.rlm import context_map as cm


def _on(monkeypatch):
    monkeypatch.setenv("REPROLAB_CONTEXT_MAP", "1")


def _key(out, entries):
    return next((e for e in entries if e["key"] == out), None)


# --- Task 1: read side + flag ------------------------------------------------


def test_read_missing_file_returns_empty(tmp_path, monkeypatch):
    _on(monkeypatch)
    out = cm.read(tmp_path)
    assert out == {"version": "v1", "bytes": 0, "entries": []}


def test_read_disabled_returns_empty_even_if_file_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "context_map.json").write_text(
        json.dumps({"version": "v1", "bytes": 5, "entries": [{"key": "x"}]})
    )
    assert cm.read(tmp_path)["entries"] == []


def test_read_corrupt_file_returns_empty(tmp_path, monkeypatch):
    _on(monkeypatch)
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "context_map.json").write_text("{not json")
    assert cm.read(tmp_path)["entries"] == []


# --- Task 2: record() — union, dedup, caps, atomic persist -------------------


def test_union_accumulates_distinct_scalars_no_clobber(tmp_path, monkeypatch):
    """The SDAR regression: batch_size from two sections must both survive."""
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="sec-1.7B")
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 16}, slice_hint="sec-7B")
    entry = _key("extract_hyperparameters:batch_size", cm.read(tmp_path)["entries"])
    assert sorted(v["value"] for v in entry["values"]) == [8, 16]


def test_union_flattens_list_fields_per_element(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "understand_section", {"datasets": [{"name": "ALFWorld"}]}, slice_hint="a")
    cm.record(tmp_path, "understand_section", {"datasets": [{"name": "WebShop"}]}, slice_hint="b")
    entry = _key("understand_section:datasets", cm.read(tmp_path)["entries"])
    names = sorted(v["value"]["name"] for v in entry["values"])
    assert names == ["ALFWorld", "WebShop"]


def test_dedup_identical_value_is_noop(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="y")
    entry = _key("extract_hyperparameters:batch_size", cm.read(tmp_path)["entries"])
    assert len(entry["values"]) == 1


def test_empty_and_null_values_skipped(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters",
              {"batch_size": None, "optimizer": "", "learning_rate": 0.1}, slice_hint="x")
    entries = cm.read(tmp_path)["entries"]
    keys = {e["key"] for e in entries}
    assert keys == {"extract_hyperparameters:learning_rate"}


def test_non_orientation_primitive_writes_nothing(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "run_experiment", {"success": True, "metrics": {"acc": 1.0}}, slice_hint="x")
    assert not (tmp_path / "rlm_state" / "context_map.json").exists()


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CONTEXT_MAP", raising=False)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    assert not (tmp_path / "rlm_state" / "context_map.json").exists()


def test_value_cap_refuses_new_keeps_existing(tmp_path, monkeypatch):
    _on(monkeypatch)
    for i in range(12):  # > _MAX_VALUES_PER_ENTRY (8)
        cm.record(tmp_path, "extract_hyperparameters", {"batch_size": i}, slice_hint=f"s{i}")
    entry = _key("extract_hyperparameters:batch_size", cm.read(tmp_path)["entries"])
    assert len(entry["values"]) == 8
    assert [v["value"] for v in entry["values"]] == list(range(8))  # earliest kept


def test_byte_ceiling_drops_overflow_keeps_prior(tmp_path, monkeypatch):
    _on(monkeypatch)
    cm.record(tmp_path, "extract_hyperparameters", {"batch_size": 8}, slice_hint="x")
    before = (tmp_path / "rlm_state" / "context_map.json").read_text()
    # A single value larger than the whole budget — must be dropped, prior stands.
    huge = {"name": "x" * (cm._MAX_BYTES + 1000)}
    cm.record(tmp_path, "understand_section", {"datasets": [huge]}, slice_hint="big")
    after = (tmp_path / "rlm_state" / "context_map.json").read_text()
    assert before == after  # overflow value dropped; the small prior entry stays


def test_byte_ceiling_keeps_early_values_drops_late_overflow(tmp_path, monkeypatch):
    """Incremental ceiling: a small valuable value lands even when a later value
    in the SAME call would overflow — not an all-or-nothing rollback."""
    _on(monkeypatch)
    big = {"name": "y" * (cm._MAX_BYTES + 1000)}
    # datasets is recorded before hardware_clues (valuable-first order); put the
    # small dataset first and an oversized hardware clue second.
    cm.record(tmp_path, "understand_section",
              {"datasets": [{"name": "ALFWorld"}], "hardware_clues": [big]}, slice_hint="s")
    entries = cm.read(tmp_path)["entries"]
    keys = {e["key"] for e in entries}
    assert "understand_section:datasets" in keys           # small early value landed
    assert "understand_section:hardware_clues" not in keys  # oversized late value dropped


def test_concurrent_writes_do_not_lose_values(tmp_path, monkeypatch):
    _on(monkeypatch)

    def worker(n):
        cm.record(tmp_path, "extract_hyperparameters", {"learning_rate": n}, slice_hint=f"s{n}")

    threads = [threading.Thread(target=worker, args=(i / 100.0,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entry = _key("extract_hyperparameters:learning_rate", cm.read(tmp_path)["entries"])
    assert len(entry["values"]) == 8  # no lost updates under the lock
