"""Dense search-qa provisioning: pre-staged index dir + configurable encoder (2026-06-02).

The dense E5 retriever needs the query encoder to MATCH the encoder the index was
built with, and a roomy host pre-stages the multi-GB index out-of-band. These pin:
  * ``_search_qa_encoder`` default (e5-base-v2) + override.
  * ``_default_search_qa_index_builder`` honours a pre-staged ``OPENRESEARCH_SEARCH_QA_INDEX_DIR``
    (no download) and falls through cleanly when it holds no index.
  * ``ensure_search_qa_index`` emits ``SEARCH_QA_ENCODER`` alongside the e5 selection.
"""
from __future__ import annotations

from backend.services.runtime import env_cache as EC
from backend.services.runtime.env_cache import EnvCacheManager


class TestSearchQaEncoder:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_SEARCH_QA_ENCODER", raising=False)
        assert EC._search_qa_encoder() == "intfloat/e5-base-v2"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_ENCODER", "intfloat/e5-large-v2")
        assert EC._search_qa_encoder() == "intfloat/e5-large-v2"


class TestPreStagedIndexBuilder:
    def test_direct_dir_with_index_is_returned(self, tmp_path, monkeypatch):
        (tmp_path / "e5_HNSW64.index").write_bytes(b"fake-index")
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_DENSE", "1")
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_INDEX_DIR", str(tmp_path))
        out = EC._default_search_qa_index_builder(tmp_path / "cache")
        assert out == tmp_path  # used directly, no download

    def test_direct_dir_without_index_falls_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_DENSE", "1")
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_INDEX_DIR", str(tmp_path))  # empty dir
        monkeypatch.delenv("OPENRESEARCH_SEARCH_QA_INDEX_REPO", raising=False)
        assert EC._default_search_qa_index_builder(tmp_path / "cache") is None

    def test_dense_disabled_returns_none(self, tmp_path, monkeypatch):
        (tmp_path / "x.index").write_bytes(b"fake")
        monkeypatch.delenv("OPENRESEARCH_SEARCH_QA_DENSE", raising=False)
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_INDEX_DIR", str(tmp_path))
        assert EC._default_search_qa_index_builder(tmp_path / "cache") is None


class TestEnsureSearchQaEmitsEncoder:
    def test_e5_env_vars_include_encoder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_ENCODER", "intfloat/e5-base-v2")
        idx = tmp_path / "idx"
        idx.mkdir()
        (idx / "e5.index").write_bytes(b"fake")
        mgr = EnvCacheManager(cache_dir=tmp_path / "cache", index_builder=lambda cd: idx)
        res = mgr.ensure_search_qa_index()
        assert res.ok
        assert res.env_vars.get("SEARCH_QA_RETRIEVER") == "e5"
        assert res.env_vars.get("SEARCH_QA_INDEX_DIR") == str(idx)
        assert res.env_vars.get("SEARCH_QA_ENCODER") == "intfloat/e5-base-v2"

    def test_bm25_fallback_has_no_index_or_encoder(self, tmp_path):
        mgr = EnvCacheManager(cache_dir=tmp_path / "cache", index_builder=lambda cd: None)
        res = mgr.ensure_search_qa_index()
        assert res.ok
        assert res.env_vars.get("SEARCH_QA_RETRIEVER") == "bm25"
        assert "SEARCH_QA_INDEX_DIR" not in res.env_vars
