"""Tests for the grounded extraction slice (Lane A dictionary + Lane B LLM)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.services.knowledge.corpus.extraction import (
    extract_paper_results,
    grounded,
    match_datasets,
    run_deterministic_extraction,
    run_llm_extraction,
)
from backend.services.knowledge.corpus.store import CorpusStore

_PAPER_TEXT = (
    "Methods\n\nWe fine-tune on ALFWorld and evaluate on WebShop.\n\n"
    "Results\n\nOur method reaches 58.1% success rate on ALFWorld, "
    "outperforming the ILSVRC-2012 pretrained baseline."
)


class _FakeExtractClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.response


def _store_with_fulltext(tmp_path: Path, text: str = _PAPER_TEXT) -> CorpusStore:
    store = CorpusStore(tmp_path / "_corpus")
    store.initialize()
    store.upsert_paper("arxiv:1", title="Agentic RL", fetched_level="fulltext")
    (store.paper_dir("arxiv:1", create=True) / "parsed_full_text.txt").write_text(
        text * 3, encoding="utf-8"  # ×3 to clear the 500-char floor
    )
    return store


# --- Lane A: dictionary ------------------------------------------------------


def test_match_datasets_hits_canonical_and_alias():
    names = {name for name, _ in match_datasets(_PAPER_TEXT)}
    assert "alfworld" in names
    assert "webshop" in names
    assert "imagenet" in names  # via the ILSVRC-2012 alias


def test_match_datasets_respects_word_boundaries():
    assert not any(n == "c4" for n, _ in match_datasets("the c40 engine block"))
    assert any(n == "c4" for n, _ in match_datasets("pretrained on C4 corpus"))


def test_run_deterministic_extraction_populates_entities(tmp_path: Path):
    store = _store_with_fulltext(tmp_path)
    stats = run_deterministic_extraction(store)
    assert stats.entities_added >= 3
    rows = store.connection.execute(
        "SELECT name, source FROM lit_entities WHERE paper_id = 'arxiv:1'"
    ).fetchall()
    assert all(r["source"] == "deterministic" for r in rows)
    assert {r["name"] for r in rows} >= {"alfworld", "webshop", "imagenet"}
    store.close()


# --- Lane B: grounding gates -------------------------------------------------


def test_grounded_requires_quote_in_text_and_value_in_quote():
    assert grounded("reaches 58.1% success rate on ALFWorld", 58.1, _PAPER_TEXT)
    # Quote not in paper → rejected.
    assert not grounded("we report 58.1 accuracy", 58.1, _PAPER_TEXT)
    # Quote real but value absent from it → rejected.
    assert not grounded("We fine-tune on ALFWorld", 58.1, _PAPER_TEXT)
    # Whitespace/case normalization still grounds.
    assert grounded("REACHES  58.1%   success rate on alfworld", 58.1, _PAPER_TEXT)
    # Tiny quotes are never grounding evidence.
    assert not grounded("58.1", 58.1, _PAPER_TEXT)


def test_extract_paper_results_filters_ungrounded_rows():
    response = """{
      "results": [
        {"method": "SDAR", "dataset": "alfworld", "metric": "success rate",
         "value": 58.1, "quote": "reaches 58.1% success rate on ALFWorld"},
        {"method": "SDAR", "dataset": "webshop", "metric": "success rate",
         "value": 99.9, "quote": "totally fabricated quote with 99.9"},
        {"method": "SDAR", "dataset": "alfworld", "metric": "reward",
         "value": 12.3, "quote": "We fine-tune on ALFWorld and evaluate on WebShop."}
      ],
      "entities": [
        {"kind": "dataset", "name": "alfworld"},
        {"kind": "dataset", "name": "made-up-benchmark"}
      ]
    }"""
    results, entities, rejected = extract_paper_results(
        _FakeExtractClient(response), "arxiv:1", _PAPER_TEXT
    )
    assert len(results) == 1  # only the verbatim-grounded row survives
    assert results[0]["dataset"] == "alfworld"
    assert results[0]["value"] == 58.1
    assert rejected == 2
    assert entities == [{"kind": "dataset", "name": "alfworld"}]  # literal-only


def test_extract_paper_results_malformed_json_degrades():
    results, entities, rejected = extract_paper_results(
        _FakeExtractClient("sorry, I cannot"), "arxiv:1", _PAPER_TEXT
    )
    assert (results, entities, rejected) == ([], [], 0)


# --- Lane B: run over the store ----------------------------------------------

_GOOD_RESPONSE = """{
  "results": [{"method": "sdar", "dataset": "alfworld", "metric": "success rate",
   "value": 58.1, "quote": "reaches 58.1% success rate on ALFWorld"}],
  "entities": []
}"""


def test_run_llm_extraction_stores_grounded_rows_with_llm_source(tmp_path: Path):
    store = _store_with_fulltext(tmp_path)
    client = _FakeExtractClient(_GOOD_RESPONSE)
    stats = run_llm_extraction(store, client)
    assert stats.results_added == 1
    row = store.connection.execute("SELECT * FROM lit_results").fetchone()
    assert row["source"] == "llm"
    assert row["dataset"] == "alfworld"
    assert "58.1" in row["span_quote"]
    # Idempotent: papers with rows are skipped unless force=True.
    stats2 = run_llm_extraction(store, client)
    assert stats2.papers_seen == 0
    stats3 = run_llm_extraction(store, client, force=True)
    assert stats3.papers_seen == 1
    store.close()


def test_run_llm_extraction_caps_and_fails_soft(tmp_path: Path):
    store = _store_with_fulltext(tmp_path)
    store.upsert_paper("arxiv:2", title="Second", fetched_level="fulltext")
    (store.paper_dir("arxiv:2", create=True) / "parsed_full_text.txt").write_text(
        _PAPER_TEXT * 3, encoding="utf-8"
    )

    class _Boom:
        def complete(self, *, system: str, user: str) -> str:
            raise RuntimeError("endpoint down")

    stats = run_llm_extraction(store, _Boom(), max_papers=1)
    assert stats.papers_seen == 1        # cap respected
    assert stats.results_added == 0
    assert stats.errors                   # recorded, not raised
    store.close()


# --- migration ----------------------------------------------------------------


def test_v1_lit_results_gains_source_column(tmp_path: Path):
    root = tmp_path / "_corpus"
    root.mkdir(parents=True)
    conn = sqlite3.connect(root / "corpus.db")
    conn.executescript(
        "CREATE TABLE lit_results (paper_id TEXT NOT NULL, method TEXT NOT NULL,"
        " dataset TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL,"
        " span_quote TEXT NOT NULL DEFAULT '',"
        " PRIMARY KEY (paper_id, method, dataset, metric));"
    )
    conn.close()
    store = CorpusStore(root)
    store.initialize()
    cols = {r[1] for r in store.connection.execute("PRAGMA table_info(lit_results)")}
    assert "source" in cols
    store.close()
