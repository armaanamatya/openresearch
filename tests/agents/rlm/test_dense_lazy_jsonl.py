"""Dense E5 + lazy-jsonl retrieval path (2026-06-02 full-scope dense retrieval).

Validates the in-process dense tier rebuilt for the wiki-18 / 21M-passage scale:
  * ``_LazyJsonlStore`` builds + caches a byte-offset index and seeks per line so a
    huge corpus never loads into RAM, and never raises on an out-of-range/bad line.
  * ``Retriever`` loads a FAISS index (mmap, with full-read fallback) + the lazy
    jsonl corpus and returns the right passage for an injected query embedding.
  * the query encoder is configurable via ``SEARCH_QA_ENCODER`` (must match the
    encoder the index was built with).

Real e5 weights are never loaded — the encoder is injected or a fake
``sentence_transformers`` module is substituted.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

# Mirror the runtime flat-import (env modules are copied flat into code/).
_RLM_DIR = "/home/sww35/openresearch-fullscope/backend/agents/rlm"
sys.path.insert(0, _RLM_DIR)
try:
    import search_qa_env as sq  # noqa: E402
finally:
    while _RLM_DIR in sys.path:
        sys.path.remove(_RLM_DIR)


def _write_corpus(d: Path) -> None:
    rows = [
        ("Apple", "apple is a common red or green fruit"),
        ("Banana", "banana is a long yellow tropical fruit"),
        ("Cat", "cat is a small domesticated pet animal"),
    ]
    with open(d / "wiki_dump.jsonl", "w", encoding="utf-8") as f:
        for i, (t, c) in enumerate(rows):
            f.write(json.dumps({"id": str(i), "contents": f'"{t}"\n{c}'}) + "\n")


def _write_index(d: Path) -> None:
    vecs = np.eye(3, 4, dtype="float32")  # 3 one-hot vectors, dim 4
    ix = faiss.IndexFlatIP(4)
    ix.add(vecs)
    faiss.write_index(ix, str(d / "e5_HNSW64.index"))


class _FakeEncoder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = np.asarray([vec], dtype="float32")

    def encode(self, queries, **kw):  # noqa: ANN001
        return self._vec


class TestLazyJsonlStore:
    def test_builds_offsets_seeks_and_caches(self, tmp_path):
        _write_corpus(tmp_path)
        store = sq._LazyJsonlStore(str(tmp_path / "wiki_dump.jsonl"))
        assert len(store) == 3
        assert "banana" in store[1]
        assert "cat" in store[2]
        assert (tmp_path / "wiki_dump.jsonl.offsets.npy").exists()  # cached
        assert store[99] == ""   # out of range — no raise
        assert store[-1] == ""   # negative — no raise

    def test_reuses_cached_offsets(self, tmp_path):
        _write_corpus(tmp_path)
        sq._LazyJsonlStore(str(tmp_path / "wiki_dump.jsonl"))      # build cache
        store2 = sq._LazyJsonlStore(str(tmp_path / "wiki_dump.jsonl"))  # load cache
        assert "apple" in store2[0]


class TestDenseRetrieveLazyJsonl:
    def test_dense_tier_loads_index_and_lazy_corpus(self, tmp_path):
        _write_corpus(tmp_path)
        _write_index(tmp_path)
        # Encoder returns the one-hot selecting passage 1 (banana).
        r = sq.Retriever(index_dir=str(tmp_path), _encoder=_FakeEncoder([0, 1, 0, 0]))
        hits = r.retrieve("a long yellow fruit", k=1)
        assert r.backend == "e5"
        assert len(hits) == 1
        assert "banana" in hits[0].text

    def test_configurable_encoder_via_env(self, tmp_path, monkeypatch):
        _write_corpus(tmp_path)
        _write_index(tmp_path)
        seen: dict[str, str] = {}

        class _FakeST:
            def __init__(self, name: str) -> None:
                seen["name"] = name

            def encode(self, queries, **kw):  # noqa: ANN001
                return np.asarray([[0, 0, 1, 0]], dtype="float32")  # selects passage 2 (cat)

        # No injected encoder → _ensure_dense imports SentenceTransformer; substitute
        # a fake module so no real model downloads, and assert the env name is used.
        monkeypatch.setitem(
            sys.modules, "sentence_transformers",
            types.SimpleNamespace(SentenceTransformer=_FakeST),
        )
        monkeypatch.setenv("SEARCH_QA_ENCODER", "intfloat/e5-large-v2")
        r = sq.Retriever(index_dir=str(tmp_path))
        hits = r.retrieve("a small pet", k=1)
        assert seen["name"] == "intfloat/e5-large-v2"
        assert r.backend == "e5"
        assert "cat" in hits[0].text

    def test_dense_unavailable_falls_back_to_bm25(self, tmp_path):
        # No index dir → dense tier skipped → BM25 over the per-question pool.
        r = sq.Retriever(index_dir=None)
        hits = r.retrieve(
            "yellow fruit", pool=["banana is yellow", "cat is a pet"], k=1
        )
        assert r.backend in ("bm25", "overlap")
        assert "banana" in hits[0].text

    def test_ntotal_mismatch_falls_back_to_bm25(self, tmp_path):
        # Index built from a DIFFERENT snapshot than the corpus (2 vectors vs 3
        # passages) -> alignment guard trips -> dense disabled -> BM25, never silent
        # wrong passages (adversarial-review H1).
        _write_corpus(tmp_path)  # 3-line corpus
        vecs = np.eye(2, 4, dtype="float32")  # only 2 vectors
        ix = faiss.IndexFlatIP(4)
        ix.add(vecs)
        faiss.write_index(ix, str(tmp_path / "e5.index"))
        r = sq.Retriever(index_dir=str(tmp_path), _encoder=_FakeEncoder([0, 1, 0, 0]))
        hits = r.retrieve("yellow fruit", pool=["banana is yellow", "cat is a pet"], k=1)
        assert r.backend in ("bm25", "overlap")  # NOT "e5"
        assert "banana" in hits[0].text

    def test_stale_offsets_cache_rebuilt_on_corpus_change(self, tmp_path):
        # A cached .offsets.npy must NOT be trusted when the corpus byte-size changed
        # (adversarial-review H1) — else seeks land on wrong lines.
        p = tmp_path / "wiki_dump.jsonl"
        _write_corpus(tmp_path)  # 3 rows -> builds offsets + .offsets.size
        assert len(sq._LazyJsonlStore(str(p))) == 3
        assert (tmp_path / "wiki_dump.jsonl.offsets.size").exists()
        # Regenerate the corpus with a different size (4 longer rows).
        with open(p, "w", encoding="utf-8") as f:
            for i in range(4):
                f.write(json.dumps({"id": str(i), "contents": f'"T{i}"\nrow number {i} body text here'}) + "\n")
        store2 = sq._LazyJsonlStore(str(p))  # size differs -> must rebuild, not reuse stale
        assert len(store2) == 4
        assert "row number 3" in store2[3]
