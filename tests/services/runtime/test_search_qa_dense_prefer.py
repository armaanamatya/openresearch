"""Regression tests for the staged-dense-index-prefer fix in SearchQaAdapter.

BUG (2026-07-02): ``_default_search_qa_index_builder`` returned None unless
``OPENRESEARCH_SEARCH_QA_DENSE`` was truthy — so a physically-present FAISS index
staged at ``SEARCH_QA_INDEX_DIR`` (the raw env var the env itself reads, set by the
SDAR preflight) was ignored, and ``SEARCH_QA_RETRIEVER=bm25`` was emitted instead of
``e5``.  Flat-zero reward on NQ-open tasks resulted.

FIX INVARIANT: a physically-present built dense index is ALWAYS used regardless of the
opt-in flag.  ``OPENRESEARCH_SEARCH_QA_DENSE`` gates only a network BUILD/download,
never USE of an already-staged index.  Both ``OPENRESEARCH_SEARCH_QA_INDEX_DIR``
(harness-prefixed) and ``SEARCH_QA_INDEX_DIR`` (the raw env var the env reads) are
probed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.search_qa import (
    SearchQaAdapter,
    _default_search_qa_index_builder,
)
from backend.services.runtime.env_adapters.base import ProvisionCtx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_SEARCH_QA_VARS = (
    "OPENRESEARCH_SEARCH_QA_DENSE",
    "OPENRESEARCH_SEARCH_QA_INDEX_DIR",
    "OPENRESEARCH_SEARCH_QA_INDEX_REPO",
    "OPENRESEARCH_SEARCH_QA_INDEX_REPO_TYPE",
    "OPENRESEARCH_SEARCH_QA_ENCODER",
    "SEARCH_QA_INDEX_DIR",
)


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all search-QA env vars so host env can't leak in."""
    for var in _ALL_SEARCH_QA_VARS:
        monkeypatch.delenv(var, raising=False)


def _staged_dir(tmp_path: Path, fname: str = "wiki.index") -> Path:
    """A fake pre-staged index directory containing one index file."""
    d = tmp_path / "staged_index"
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_bytes(b"fake-faiss")
    return d


# ---------------------------------------------------------------------------
# Test 1: THE REGRESSION TEST — SEARCH_QA_INDEX_DIR (raw) + no opt-in flag
# ---------------------------------------------------------------------------

def test_staged_dir_via_raw_var_no_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: pre-staged index at SEARCH_QA_INDEX_DIR is used even when
    OPENRESEARCH_SEARCH_QA_DENSE is UNSET.  This is the bug that caused flat-zero reward
    on NQ-open tasks (the SDAR preflight sets SEARCH_QA_INDEX_DIR; the harness opt-in
    flag was never exported, so the builder fell through to BM25)."""
    _clean_env(monkeypatch)
    staged = _staged_dir(tmp_path)
    monkeypatch.setenv("SEARCH_QA_INDEX_DIR", str(staged))
    # OPENRESEARCH_SEARCH_QA_DENSE is explicitly NOT set — the core regression.
    result = _default_search_qa_index_builder(tmp_path / "cache")
    assert result == staged, (
        f"Expected builder to return pre-staged dir {staged}, got {result!r}. "
        "The fix must use a staged index WITHOUT requiring the opt-in flag."
    )


# ---------------------------------------------------------------------------
# Test 2: OPENRESEARCH_SEARCH_QA_INDEX_DIR (prefixed) + no opt-in flag
# ---------------------------------------------------------------------------

def test_staged_dir_via_prefixed_var_no_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-staged index at OPENRESEARCH_SEARCH_QA_INDEX_DIR is used even when the
    opt-in flag is unset."""
    _clean_env(monkeypatch)
    staged = _staged_dir(tmp_path, fname="e5_Flat.index")
    monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_INDEX_DIR", str(staged))
    result = _default_search_qa_index_builder(tmp_path / "cache")
    assert result == staged, (
        f"Expected builder to return pre-staged dir {staged}, got {result!r}."
    )


# ---------------------------------------------------------------------------
# Test 3: no index file anywhere, no flag, no repo → None (BM25)
# ---------------------------------------------------------------------------

def test_returns_none_when_nothing_staged_and_no_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns None (→ BM25) when no staged index exists and the opt-in flag is unset."""
    _clean_env(monkeypatch)
    # Point to an existing EMPTY directory so the probe doesn't skip on missing dir.
    empty = tmp_path / "empty_index"
    empty.mkdir()
    monkeypatch.setenv("SEARCH_QA_INDEX_DIR", str(empty))
    result = _default_search_qa_index_builder(tmp_path / "cache")
    assert result is None, (
        f"Expected None (BM25 fallback) when dir has no .index/.faiss, got {result!r}."
    )


# ---------------------------------------------------------------------------
# Test 4a: SearchQaAdapter.provision emits e5 when staged index exists
# ---------------------------------------------------------------------------

def test_adapter_provision_e5_when_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SearchQaAdapter.provision emits SEARCH_QA_RETRIEVER=e5 when index_builder returns
    a valid staged directory."""
    _clean_env(monkeypatch)
    staged = _staged_dir(tmp_path)
    adapter = SearchQaAdapter(
        AssetCache(tmp_path / "cache"),
        index_builder=lambda _c: staged,
    )
    result = adapter.provision(ProvisionCtx(display_name="Search-QA"))
    assert result.ok, f"provision() returned ok=False: {result!r}"
    assert result.env_vars.get("SEARCH_QA_RETRIEVER") == "e5", (
        f"Expected SEARCH_QA_RETRIEVER=e5, got {result.env_vars!r}"
    )
    assert result.env_vars.get("SEARCH_QA_INDEX_DIR") == str(staged), (
        f"Expected SEARCH_QA_INDEX_DIR={staged}, got {result.env_vars!r}"
    )


# ---------------------------------------------------------------------------
# Test 4b: SearchQaAdapter.provision emits bm25 when builder returns None
# ---------------------------------------------------------------------------

def test_adapter_provision_bm25_when_no_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SearchQaAdapter.provision emits SEARCH_QA_RETRIEVER=bm25 when index_builder
    returns None (no staged index, no opt-in flag)."""
    _clean_env(monkeypatch)
    adapter = SearchQaAdapter(
        AssetCache(tmp_path / "cache"),
        index_builder=lambda _c: None,
    )
    result = adapter.provision(ProvisionCtx(display_name="Search-QA"))
    assert result.ok, f"provision() returned ok=False (Search-QA must never exclude): {result!r}"
    assert result.env_vars.get("SEARCH_QA_RETRIEVER") == "bm25", (
        f"Expected SEARCH_QA_RETRIEVER=bm25, got {result.env_vars!r}"
    )
    assert "SEARCH_QA_INDEX_DIR" not in result.env_vars, (
        f"SEARCH_QA_INDEX_DIR should be absent on bm25, got {result.env_vars!r}"
    )


# ---------------------------------------------------------------------------
# Test 5: opt-in flag present, prefixed var points to valid dir → still works
# ---------------------------------------------------------------------------

def test_staged_dir_with_flag_also_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the opt-in flag alongside a staged index still returns the staged dir
    (not a download attempt)."""
    _clean_env(monkeypatch)
    staged = _staged_dir(tmp_path, fname="data.faiss")
    monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_DENSE", "1")
    monkeypatch.setenv("OPENRESEARCH_SEARCH_QA_INDEX_DIR", str(staged))
    result = _default_search_qa_index_builder(tmp_path / "cache")
    assert result == staged
