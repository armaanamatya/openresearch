"""Tests for chunk indexing + hybrid retrieval (hermetic — no network, no models)."""

from __future__ import annotations

import json
from pathlib import Path

from backend.services.knowledge.corpus.index import (
    MAX_CHUNKS_PER_PAPER,
    build_chunk_index,
    chunk_text,
)
from backend.services.knowledge.corpus.retrieval import retrieve
from backend.services.knowledge.corpus.store import CorpusStore

_METHOD_TEXT = (
    "Introduction\n\nWe study agentic reinforcement learning for tool use.\n\n"
    "Methods\n\nOur training uses a cosine learning rate schedule with warmup "
    "over the ALFWorld benchmark suite and self-distillation.\n\n"
    "Results\n\nWe achieve 61.2 success rate on ALFWorld after distillation."
)
_OTHER_TEXT = (
    "Introduction\n\nA general survey of optimization tricks.\n\n"
    "Discussion\n\nExponential moving averages stabilize training dynamics."
)


def _corpus(tmp_path: Path) -> CorpusStore:
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    store.upsert_paper("arxiv:1", title="Agentic RL Paper", fetched_level="fulltext")
    store.upsert_paper("arxiv:2", title="Optimization Survey", fetched_level="fulltext")
    store.upsert_paper("arxiv:3", title="Cited But Textless", abstract="A cited paper about ALFWorld baselines.")
    (store.paper_dir("arxiv:1", create=True) / "parsed_full_text.txt").write_text(
        _METHOD_TEXT, encoding="utf-8"
    )
    (store.paper_dir("arxiv:2", create=True) / "parsed_full_text.txt").write_text(
        _OTHER_TEXT, encoding="utf-8"
    )
    # Citation edge: paper 1 cites paper 3 (the abstract-only paper).
    store.add_relation("arxiv:1", "arxiv:3")
    return store


# --- chunker -----------------------------------------------------------------


def test_chunk_text_is_section_aware_and_deterministic():
    chunks = chunk_text(_METHOD_TEXT)
    assert chunks == chunk_text(_METHOD_TEXT)  # deterministic
    sections = [s for s, _ in chunks]
    assert "methods" in sections
    assert "results" in sections
    method_chunk = next(t for s, t in chunks if s == "methods")
    assert "cosine learning rate" in method_chunk


def test_chunk_text_bounds():
    huge = "word " * 100_000
    chunks = chunk_text(huge)
    assert len(chunks) <= MAX_CHUNKS_PER_PAPER


# --- index build -------------------------------------------------------------


def test_build_chunk_index_covers_fulltext_and_abstract_papers(tmp_path: Path):
    store = _corpus(tmp_path)
    stats = build_chunk_index(store)
    assert stats["papers"] == 3
    assert stats["chunks"] >= 3
    rows = store.connection.execute(
        "SELECT paper_id, COUNT(*) AS n FROM lit_chunks GROUP BY paper_id"
    ).fetchall()
    by_paper = {r["paper_id"]: r["n"] for r in rows}
    assert by_paper["arxiv:3"] == 1  # abstract-only fallback chunk
    store.close()


def test_build_chunk_index_is_idempotent(tmp_path: Path):
    store = _corpus(tmp_path)
    build_chunk_index(store)
    first = store.connection.execute(
        "SELECT chunk_id, text FROM lit_chunks ORDER BY chunk_id"
    ).fetchall()
    build_chunk_index(store)
    second = store.connection.execute(
        "SELECT chunk_id, text FROM lit_chunks ORDER BY chunk_id"
    ).fetchall()
    assert [tuple(r) for r in first] == [tuple(r) for r in second]
    store.close()


def test_reindex_prunes_stale_dense_vectors(tmp_path: Path, monkeypatch):
    """Shrinking a paper's chunk count must not orphan rows in lit_chunks_vec —
    stale vectors would still win MATCH slots and waste the dense seed budget."""
    from backend.services.knowledge.corpus import index as index_mod

    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    store.upsert_paper("arxiv:9", title="Shrinking Paper", fetched_level="fulltext")
    text_path = store.paper_dir("arxiv:9", create=True) / "parsed_full_text.txt"
    text_path.write_text("long paragraph " * 400, encoding="utf-8")  # several chunks

    # Stand-in for the vec0 virtual table (same columns, plain SQL semantics) —
    # the sqlite-vec extension is unavailable in the hermetic suite, and
    # CREATE VIRTUAL TABLE IF NOT EXISTS no-ops on the existing name.
    store.connection.execute("CREATE TABLE lit_chunks_vec (embedding TEXT, chunk_id TEXT)")
    monkeypatch.setattr(index_mod, "try_load_sqlite_vec", lambda conn: True)

    def _embed(texts):
        return [[0.0, 0.0] for _ in texts]

    build_chunk_index(store, embed_fn=_embed)
    n_first = store.connection.execute("SELECT COUNT(*) FROM lit_chunks_vec").fetchone()[0]
    assert n_first > 1

    text_path.write_text("now just one short paragraph", encoding="utf-8")
    build_chunk_index(store, embed_fn=_embed)

    vec_ids = [
        r[1] for r in store.connection.execute("SELECT rowid, chunk_id FROM lit_chunks_vec")
    ]
    chunk_ids = {
        r[0]
        for r in store.connection.execute(
            "SELECT chunk_id FROM lit_chunks WHERE paper_id = 'arxiv:9'"
        )
    }
    assert set(vec_ids) == chunk_ids            # no orphaned vectors
    assert len(vec_ids) == len(set(vec_ids))    # and no duplicated rows
    store.close()


# --- retrieval ---------------------------------------------------------------


def test_retrieve_lexical_finds_the_right_section(tmp_path: Path):
    store = _corpus(tmp_path)
    build_chunk_index(store)
    result = retrieve(store, "cosine learning rate warmup schedule", top_n=3)
    assert result["hits"], "lexical channel must produce hits"
    top = result["hits"][0]
    assert top["paper_id"] == "arxiv:1"
    assert top["lane"] == "A"
    assert "bm25" in top["channels"]
    assert len(top["quote"]) <= 300
    store.close()


def test_retrieve_expansion_pulls_cited_paper_missed_by_lexical(tmp_path: Path):
    """arxiv:3 shares no query terms but is 1-hop from the seed paper."""
    store = _corpus(tmp_path)
    build_chunk_index(store)
    result = retrieve(store, "self-distillation training", top_n=10)
    papers = {h["paper_id"] for h in result["hits"]}
    assert "arxiv:1" in papers
    assert "arxiv:3" in papers, "citation expansion must add the cited abstract-only paper"
    expanded = next(h for h in result["hits"] if h["paper_id"] == "arxiv:3")
    assert "graph_expand" in expanded["channels"]
    store.close()


def test_retrieve_is_deterministic_and_traced(tmp_path: Path):
    store = _corpus(tmp_path)
    build_chunk_index(store)
    trace_path = tmp_path / "trace.json"
    r1 = retrieve(store, "ALFWorld success rate", top_n=5, trace_path=trace_path)
    r2 = retrieve(store, "ALFWorld success rate", top_n=5)
    assert r1["hits"] == r2["hits"]  # byte-identical across calls
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["query"] == "ALFWorld success rate"
    assert "bm25" in trace["channels"]
    assert "vec" in trace["channels"]           # recorded even when unused
    assert trace["channels"]["vec"]["used"] is False
    store.close()


def test_retrieve_top_n_knob(tmp_path: Path):
    store = _corpus(tmp_path)
    build_chunk_index(store)
    assert len(retrieve(store, "training", top_n=1)["hits"]) <= 1
    store.close()


def test_retrieve_empty_index_returns_empty(tmp_path: Path):
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    result = retrieve(store, "anything at all")
    assert result["hits"] == []
    store.close()


def test_retrieve_entity_linking_is_deterministic_lane(tmp_path: Path):
    store = _corpus(tmp_path)
    # arxiv:2 carries a deterministic dataset entity matching a query token;
    # an llm-sourced row must NOT link (Lane-B may never drive Lane-A hits).
    conn = store.connection
    conn.execute(
        "INSERT INTO lit_entities (paper_id, kind, name, aliases, source)"
        " VALUES ('arxiv:2', 'dataset', 'webshop', '[]', 'deterministic')"
    )
    conn.execute(
        "INSERT INTO lit_entities (paper_id, kind, name, aliases, source)"
        " VALUES ('arxiv:3', 'dataset', 'webshop', '[]', 'llm')"
    )
    conn.commit()
    build_chunk_index(store)
    result = retrieve(store, "webshop", top_n=10)
    papers = {h["paper_id"] for h in result["hits"]}
    assert "arxiv:2" in papers
    store.close()
