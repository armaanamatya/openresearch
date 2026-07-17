"""Track G S1 observed-DAG recorder (OPENRESEARCH_DAG_BACKBONE) — OFF+ON pair."""

from __future__ import annotations

from backend.agents.rlm.dag_nodes import append_dag_node, dag_backbone_enabled, read_dag_nodes


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_DAG_BACKBONE", raising=False)
    assert dag_backbone_enabled() is False


def test_append_and_read(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DAG_BACKBONE", "1")
    assert append_dag_node(tmp_path, node_id="e1", kind="run_experiment", ts="t1") is True
    append_dag_node(tmp_path, node_id="e2", kind="run_experiment", ts="t2", deps=["e1"])
    rows = read_dag_nodes(tmp_path)
    assert [r["node_id"] for r in rows] == ["e1", "e2"]
    assert rows[1]["deps"] == ["e1"]
    assert rows[0]["kind"] == "run_experiment" and rows[0]["status"] == "done"


def test_off_state_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_DAG_BACKBONE", raising=False)
    assert append_dag_node(tmp_path, node_id="e1", kind="run_experiment", ts="t") is False
    assert not (tmp_path / "rlm_state" / "dag_nodes.jsonl").exists()
    assert read_dag_nodes(tmp_path) == []


def test_deps_deduped_and_sorted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DAG_BACKBONE", "1")
    append_dag_node(tmp_path, node_id="e3", kind="k", ts="t", deps=["b", "a", "b", ""])
    assert read_dag_nodes(tmp_path)[0]["deps"] == ["a", "b"]
