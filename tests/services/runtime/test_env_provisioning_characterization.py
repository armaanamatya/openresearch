"""Refactor-robust behavior contract for env provisioning (Phase 1a).

Injection is via the public constructor ONLY (no module monkeypatching), so these
tests pin BEHAVIOR independent of where the provisioning logic physically lives.
They must stay green byte-for-byte across the env_adapters refactor.
"""
from __future__ import annotations
from pathlib import Path

from backend.agents.rlm import exclusion as X
from backend.services.runtime.env_cache import EnvCacheManager, provision_scope


def _dl_ok(cache_dir: Path) -> None:
    d = Path(cache_dir) / "json_2.1.1" / "train" / "g0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "traj_data.json").write_text("{}", encoding="utf-8")


def _dl_fail(cache_dir: Path) -> None:
    raise RuntimeError("alfworld-download exit 1")


def test_alfworld_env_var_shape(tmp_path: Path):
    m = EnvCacheManager(tmp_path, downloader=_dl_ok)
    r = m.ensure_alfworld()
    assert r.ok and r.as_env_vars() == {"ALFWORLD_DATA": str((tmp_path / "alfworld").resolve())}


def test_alfworld_failure_is_verified_exclusion_only(tmp_path: Path):
    m = EnvCacheManager(tmp_path, downloader=_dl_fail)
    r = m.ensure_alfworld()
    assert not r.ok and r.as_env_vars() == {}
    assert r.exclusion and r.exclusion.verified and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED


def test_search_qa_bm25_env_var_shape(tmp_path: Path):
    # index_builder returns None → BM25, never an exclusion.
    m = EnvCacheManager(tmp_path, index_builder=lambda c: None)
    r = m.ensure_search_qa_index()
    assert r.ok and r.as_env_vars() == {"SEARCH_QA_RETRIEVER": "bm25"}


def test_provision_scope_env_vars_and_exclusions_contract(tmp_path: Path):
    # ALFWorld fails (→ exclusion), Search-QA runs BM25 (→ env var, no exclusion).
    m = EnvCacheManager(tmp_path, downloader=_dl_fail, index_builder=lambda c: None)
    res = provision_scope(["ALFWorld", "Search-QA"], m)
    assert res.env_vars == {"SEARCH_QA_RETRIEVER": "bm25"}
    assert [e.item for e in res.exclusions] == ["ALFWorld"]
    assert X.build_scope_block(res.exclusions)["environments_skipped"] == ["ALFWorld"]
    res.release()  # no webshop lease → safe no-op


def test_webshop_inprocess_env_var_shape(tmp_path: Path, monkeypatch):
    data = tmp_path / "ws"; data.mkdir()
    monkeypatch.setenv("WEBSHOP_DATA_DIR", str(data))
    monkeypatch.delenv("WEBSHOP_PACKAGE_DIR", raising=False)
    m = EnvCacheManager(tmp_path, inprocess_smoke=lambda d: True)
    r = m.acquire_webshop()
    assert r.ok and r.as_env_vars() == {"WEBSHOP_DATA_DIR": str(data)}
    assert r.base_url is None
