"""Tests for schema v2, neighbors() traversal, and graph metrics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.services.knowledge.corpus.graph_metrics import (
    compute_graph_metrics,
    graph_boost,
)
from backend.services.knowledge.corpus.store import CorpusStore


def _store_with_edges(tmp_path: Path) -> CorpusStore:
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    # target T cites A and B; C cites T and A; D cites T; B cites A.
    for src, dst in (("T", "A"), ("T", "B"), ("C", "T"), ("C", "A"), ("D", "T"), ("B", "A")):
        store.add_relation(src, dst)
    return store


def test_v1_database_is_migrated_in_place(tmp_path: Path):
    """A corpus.db created by the v1 schema gains the v2 columns on initialize()."""
    root = tmp_path / "_corpus"
    root.mkdir(parents=True)
    conn = sqlite3.connect(root / "corpus.db")
    conn.executescript(
        """
        CREATE TABLE lit_papers (
            id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '', year INTEGER,
            venue TEXT, arxiv_id TEXT, doi TEXT, s2_id TEXT, abstract TEXT,
            fetched_level TEXT NOT NULL DEFAULT 'metadata');
        CREATE TABLE lit_relations (
            src_id TEXT NOT NULL, dst_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'cites', PRIMARY KEY (src_id, dst_id, kind));
        CREATE TABLE lit_entities (
            paper_id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
            aliases TEXT NOT NULL DEFAULT '[]', PRIMARY KEY (paper_id, kind, name));
        """
    )
    conn.execute("INSERT INTO lit_papers (id, title) VALUES ('arxiv:1', 'Old Row')")
    conn.commit()
    conn.close()

    store = CorpusStore(root)
    store.initialize()  # must migrate, not raise
    row = store.get_paper("arxiv:1")
    assert row is not None and row["title"] == "Old Row"
    assert "openalex_id" in row and "corpus_id" in row
    cols = {r[1] for r in store.connection.execute("PRAGMA table_info(lit_entities)")}
    assert "source" in cols
    store.close()


def test_neighbors_one_hop_both_directions(tmp_path: Path):
    store = _store_with_edges(tmp_path)
    ids = {n["id"]: n for n in store.neighbors("T", depth=1)}
    assert set(ids) == {"A", "B", "C", "D"}
    assert ids["A"]["direction"] == "out"   # T cites A
    assert ids["C"]["direction"] == "in"    # C cites T
    store.close()


def test_neighbors_depth_capped_and_two_hop(tmp_path: Path):
    store = _store_with_edges(tmp_path)
    one_hop = {n["id"] for n in store.neighbors("D", depth=1)}
    assert one_hop == {"T"}
    two_hop = {n["id"]: n["hop"] for n in store.neighbors("D", depth=2)}
    assert two_hop["T"] == 1
    assert {k for k, v in two_hop.items() if v == 2} == {"A", "B", "C"}
    # depth is clamped to 2 — a huge value must not walk further/explode.
    assert {n["id"] for n in store.neighbors("D", depth=99)} == set(two_hop)
    store.close()


def test_reverse_traversal_uses_dst_index(tmp_path: Path):
    """EXPLAIN QUERY PLAN sanity: 'who cites X' must hit idx_lit_relations_dst."""
    store = _store_with_edges(tmp_path)
    plan = " ".join(
        str(row[3])
        for row in store.connection.execute(
            "EXPLAIN QUERY PLAN SELECT src_id FROM lit_relations"
            " WHERE dst_id = ? AND kind = 'cites'",
            ("T",),
        ).fetchall()
    )
    assert "idx_lit_relations_dst" in plan
    store.close()


def test_graph_metrics_pure_subset_and_determinism(tmp_path: Path):
    store = _store_with_edges(tmp_path)
    edges = store.edges()
    m1 = compute_graph_metrics(edges, "T")
    m2 = compute_graph_metrics(list(edges), "T")
    # Determinism: identical input ⇒ identical output.
    assert m1 == m2
    assert "T" not in m1
    # A is cited by T, C, B → in_degree 3; A and B are both cited by T…
    assert m1["A"]["in_degree"] == 3.0
    # co_citation(A) vs T: citers(A)={T,C,B}, citers(T)={C,D} → shared {C} = 1.
    assert m1["A"]["co_citation"] == 1.0
    # coupling(C) vs T: refs(C)={T,A}, refs(T)={A,B} → shared {A} = 1.
    assert m1["C"]["coupling"] == 1.0
    store.close()


def test_graph_metrics_networkx_fields_when_available(tmp_path: Path):
    store = _store_with_edges(tmp_path)
    metrics = compute_graph_metrics(store.edges(), "T")
    try:
        import networkx  # noqa: F401
    except ImportError:
        assert all("pagerank" not in m for m in metrics.values())
    else:
        assert all("pagerank" in m for m in metrics.values())
        assert all("ppr" in m for m in metrics.values())
        # PPR is target-seeded: direct neighbours get more mass than A gets
        # from two hops... at minimum, all values are finite and positive.
        assert all(m["ppr"] >= 0 for m in metrics.values())
    store.close()


def test_graph_boost_is_bounded():
    assert graph_boost({}) == 0.0
    huge = {"co_citation": 100.0, "coupling": 100.0, "ppr": 100.0}
    # Strictly under the 0.5 adjacency-tier gap (direct_ref 3.0 vs s2_ref 2.5):
    # even a maximally connected paper can never jump a tier on graph evidence.
    assert graph_boost(huge) == 0.45
    assert graph_boost(huge) < 0.5


def test_put_and_get_graph_metrics_roundtrip(tmp_path: Path):
    store = _store_with_edges(tmp_path)
    store.put_graph_metrics("T", {"A": {"in_degree": 3.0, "co_citation": 1.0}})
    assert store.get_graph_metrics("T") == {"A": {"in_degree": 3.0, "co_citation": 1.0}}
    # Replacement semantics: a rebuild replaces, never accumulates.
    store.put_graph_metrics("T", {"B": {"in_degree": 1.0}})
    assert store.get_graph_metrics("T") == {"B": {"in_degree": 1.0}}
    store.close()
