from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.search_qa import SearchQaAdapter
from backend.services.runtime.env_adapters.base import ProvisionCtx


def test_bm25_when_no_index(tmp_path):
    a = SearchQaAdapter(AssetCache(tmp_path), index_builder=lambda c: None)
    r = a.provision(ProvisionCtx(display_name="Search-QA"))
    assert r.ok and r.as_env_vars() == {"SEARCH_QA_RETRIEVER": "bm25"}


def test_dense_when_index_built(tmp_path):
    idx = tmp_path / "idx"; idx.mkdir(); (idx / "x.faiss").write_text("")
    a = SearchQaAdapter(AssetCache(tmp_path), index_builder=lambda c: idx)
    r = a.provision(ProvisionCtx(display_name="Search-QA"))
    assert r.ok and r.env_vars["SEARCH_QA_RETRIEVER"] == "e5"
    assert r.env_vars["SEARCH_QA_INDEX_DIR"] == str(idx)


def test_never_excludes_on_builder_raise(tmp_path):
    def _boom(c): raise RuntimeError("x")
    a = SearchQaAdapter(AssetCache(tmp_path), index_builder=_boom)
    r = a.provision(ProvisionCtx(display_name="Search-QA"))
    assert r.ok and r.env_vars == {"SEARCH_QA_RETRIEVER": "bm25"}   # degrade, never exclude
