"""Tests for backend.services.knowledge.corpus.store — the lit_* corpus store."""

from __future__ import annotations

from pathlib import Path

from backend.services.knowledge.corpus.store import (
    CorpusStore,
    corpus_root,
    normalize_paper_id,
)


def test_corpus_root_is_runs_underscore_corpus(tmp_path: Path):
    assert corpus_root(tmp_path) == tmp_path / "_corpus"


def test_normalize_paper_id_prefers_arxiv_and_strips_version():
    assert normalize_paper_id(arxiv_id="2101.00001v3") == "arxiv:2101.00001"
    assert normalize_paper_id(arxiv_id="2101.00001", doi="10.1/x") == "arxiv:2101.00001"


def test_normalize_paper_id_doi_lowercased_then_s2_fallback():
    assert normalize_paper_id(doi="10.1000/XYZ") == "doi:10.1000/xyz"
    assert normalize_paper_id(s2_id="Abc123") == "s2:Abc123"
    assert normalize_paper_id() is None
    assert normalize_paper_id(arxiv_id="  ", doi="") is None


def test_initialize_idempotent_and_counts(tmp_path: Path):
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    store.initialize()  # must not raise
    assert store.paper_count() == 0
    assert store.relation_count() == 0
    store.close()


def test_upsert_insert_then_enrich_never_degrades(tmp_path: Path):
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    store.upsert_paper("arxiv:2101.00001", title="A Paper", year=2021, fetched_level="fulltext")
    # A later, sparser upsert must not blank the title nor degrade fulltext.
    store.upsert_paper("arxiv:2101.00001", title="", year=None, fetched_level="metadata")
    row = store.get_paper("arxiv:2101.00001")
    assert row is not None
    assert row["title"] == "A Paper"
    assert row["year"] == 2021
    assert row["fetched_level"] == "fulltext"
    # But an enriching upsert fills previously-empty columns.
    store.upsert_paper("arxiv:2101.00001", venue="NeurIPS", doi="10.1/x")
    row = store.get_paper("arxiv:2101.00001")
    assert row["venue"] == "NeurIPS"
    assert row["doi"] == "10.1/x"
    store.close()


def test_relations_are_deduped(tmp_path: Path):
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    store.add_relation("arxiv:1", "arxiv:2")
    store.add_relation("arxiv:1", "arxiv:2")
    assert store.relation_count() == 1
    store.close()


def test_paper_dir_deterministic_and_collision_free(tmp_path: Path):
    store = CorpusStore(tmp_path / "_corpus")
    # Two ids that sanitize identically must land in distinct directories.
    a = store.paper_dir("doi:10.1000/a:b")
    b = store.paper_dir("doi:10.1000/a_b")
    assert a != b
    assert a == store.paper_dir("doi:10.1000/a:b")  # deterministic
    assert a.parent == store.root / "papers"


def test_write_paper_meta_is_fail_soft_and_atomic(tmp_path: Path):
    store = CorpusStore(tmp_path / "_corpus")
    store.write_paper_meta("arxiv:2101.00001", {"title": "A Paper"})
    meta = store.paper_dir("arxiv:2101.00001") / "meta.json"
    assert meta.exists()
    assert not meta.with_suffix(".json.tmp").exists()
