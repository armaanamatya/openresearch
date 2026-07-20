"""SearchQaAdapter — Search-QA dense/BM25 retriever provisioning (Phase 1a).

Part of the provisioning-seam refactor (see
``docs/history/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``):
lifts ``EnvCacheManager.ensure_search_qa_index`` and its helper functions
(``_search_qa_encoder`` / ``_default_search_qa_index_builder``) out of
``env_cache.py`` verbatim, behind the
:class:`~backend.services.runtime.env_adapters.base.EnvironmentAdapter`
contract. ``env_cache.py`` itself is untouched by this unit; a later task
rewrites it as a facade delegating to this adapter.

Unlike ALFWorld/WebShop, Search-QA NEVER returns an exclusion — a cold or
unavailable dense index degrades to ``SEARCH_QA_RETRIEVER=bm25`` and the env's
BM25/overlap retriever runs; the environment always runs. The on-disk state key
stays ``"search_qa"`` (== :attr:`SearchQaAdapter.key`) so a warm SDAR cache disk
is byte-compatible across the refactor.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.base import (
    EnvironmentAdapter,
    EnvSetupResult,
    ProvisionCtx,
)

logger = logging.getLogger(__name__)

__all__ = ["SearchQaAdapter"]

_ALIASES = {
    "search-qa", "searchqa", "search_qa", "search qa",
    "nq", "nq-open", "nq_open", "hotpotqa", "hotpot_qa",
}


def _search_qa_encoder() -> str:
    """The e5 encoder the dense index was built with — the query encoder MUST match
    it (same dimension + semantics) or FAISS search errors. The prebuilt wiki-18
    indexes are e5-base-v2; override with ``OPENRESEARCH_SEARCH_QA_ENCODER`` for an index
    built with a different e5 variant."""
    return os.environ.get("OPENRESEARCH_SEARCH_QA_ENCODER", "").strip() or "intfloat/e5-base-v2"


def _default_search_qa_index_builder(cache_dir: Path) -> "Path | None":
    """Build/download a dense E5 wiki-18 retrieval index (real path; injected in tests).

    Returns the index dir on success, ``None`` to fall back to BM25 — NEVER raises.
    Opt-in + configurable so a cold/offline host degrades gracefully:

      * ``OPENRESEARCH_SEARCH_QA_DENSE`` gates a network BUILD/download only — it is
        NOT required to USE a pre-staged index that already exists on disk.
      * ``OPENRESEARCH_SEARCH_QA_INDEX_REPO`` — a HF repo holding a prebuilt FAISS index
        + passage store; snapshot-downloaded into ``cache_dir`` when set (fastest,
        no local embedding). ``OPENRESEARCH_SEARCH_QA_INDEX_REPO_TYPE`` selects the HF
        repo type (default ``dataset``).

    The downloaded artifact is cached under ``cache_dir`` and reused by
    :meth:`SearchQaAdapter.provision`. A local-embed path (download the corpus +
    embed with e5 on GPU) is intentionally left to a follow-up — the repo download
    keeps the common case fast and the BM25 fallback keeps every host live.
    """
    # (1) USE a pre-staged on-disk dense index if one physically exists — NO opt-in
    # flag required. OPENRESEARCH_SEARCH_QA_DENSE gates a network BUILD/download
    # (below), never USE of an index already on disk. Probe BOTH the prefixed
    # override and the raw SEARCH_QA_INDEX_DIR the env itself reads (set by the SDAR
    # preflight) so a staged e5_Flat.index is never shadowed by a bm25 fallback.
    for _var in ("OPENRESEARCH_SEARCH_QA_INDEX_DIR", "SEARCH_QA_INDEX_DIR"):
        direct = os.environ.get(_var, "").strip()
        if not direct:
            continue
        ddir = Path(direct)
        if ddir.is_dir() and (any(ddir.rglob("*.index")) or any(ddir.rglob("*.faiss"))):
            return ddir
        logger.warning(
            "env_adapters.search_qa: %s=%s has no .index/.faiss file; "
            "falling through to repo download / BM25.", _var, direct,
        )
    # (2) A network BUILD/download requires the opt-in flag (avoids a surprise
    # multi-GB fetch on a host with no staged index).
    flag = os.environ.get("OPENRESEARCH_SEARCH_QA_DENSE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    repo = os.environ.get("OPENRESEARCH_SEARCH_QA_INDEX_REPO", "").strip()
    if not repo:
        logger.info(
            "env_adapters.search_qa: OPENRESEARCH_SEARCH_QA_DENSE set but no repo and "
            "no pre-staged index found — using BM25 (set the repo or stage an index)."
        )
        return None
    try:
        from huggingface_hub import snapshot_download

        repo_type = os.environ.get("OPENRESEARCH_SEARCH_QA_INDEX_REPO_TYPE", "dataset").strip() or "dataset"
        dest = cache_dir / "search_qa_index"
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo, repo_type=repo_type, local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        if any(dest.rglob("*.index")) or any(dest.rglob("*.faiss")):
            return dest
        logger.warning(
            "env_adapters.search_qa: index repo %s downloaded but no .index/.faiss "
            "file found — BM25 fallback.", repo,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — dense is best-effort; BM25 always works
        logger.warning(
            "env_adapters.search_qa: dense index build failed (%s: %s); BM25 fallback.",
            type(exc).__name__, str(exc)[:160],
        )
        return None


class SearchQaAdapter(EnvironmentAdapter):
    """Idempotent Search-QA retriever provisioning: dense E5 when buildable, else BM25.

    The dense build is idempotent + shared (cached under
    ``<cache>/search_qa_index``). A failure at any stage degrades to
    ``SEARCH_QA_RETRIEVER=bm25`` rather than excluding the environment —
    Search-QA always runs.
    """

    key = "search_qa"

    def __init__(
        self,
        cache: AssetCache,
        *,
        index_builder: "Callable[[Path], Path | None] | None" = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = cache
        self._index_builder = index_builder or _default_search_qa_index_builder
        self._clock = clock

    def applies(self, env_name: str) -> bool:
        return (env_name or "").strip().lower() in _ALIASES

    def provision(self, ctx: ProvisionCtx) -> EnvSetupResult:
        """Provide a Search-QA retriever: dense E5 index when buildable + cached,
        else BM25 (always works).

        Unlike ALFWorld/WebShop, Search-QA NEVER returns an exclusion — a cold or
        unavailable dense index degrades to ``SEARCH_QA_RETRIEVER=bm25`` and the
        env's BM25/overlap retriever runs. The dense build is idempotent + shared
        (cached under ``<cache>/search_qa_index``).
        """
        display_name = ctx.display_name or "Search-QA"
        try:
            with self._cache.locked_state() as state:
                rec = state.get(self.key) or {}
                if (rec.get("ready") and rec.get("retriever") == "e5"
                        and Path(rec.get("index_dir", "")).exists()):
                    return EnvSetupResult(
                        env=display_name, ok=True, detail="cache hit (e5)",
                        env_vars={"SEARCH_QA_INDEX_DIR": rec["index_dir"],
                                  "SEARCH_QA_RETRIEVER": "e5",
                                  "SEARCH_QA_ENCODER": _search_qa_encoder()},
                    )
                built = self._index_builder(self._cache.cache_dir)  # injected; None → BM25
                if built is not None and Path(built).exists():
                    state[self.key] = {"ready": True, "retriever": "e5",
                                       "index_dir": str(built), "built_at": self._clock()}
                    return EnvSetupResult(
                        env=display_name, ok=True, detail="dense index ready",
                        env_vars={"SEARCH_QA_INDEX_DIR": str(built),
                                  "SEARCH_QA_RETRIEVER": "e5",
                                  "SEARCH_QA_ENCODER": _search_qa_encoder()},
                    )
                state[self.key] = {"ready": True, "retriever": "bm25",
                                   "built_at": self._clock()}
                return EnvSetupResult(
                    env=display_name, ok=True, detail="bm25 (no dense index)",
                    env_vars={"SEARCH_QA_RETRIEVER": "bm25"},
                )
        except Exception as exc:  # noqa: BLE001 — Search-QA must always run
            logger.warning("env_adapters.search_qa: provisioning issue (%s); BM25",
                           type(exc).__name__)
            return EnvSetupResult(
                env=display_name, ok=True, detail="bm25 (provisioning fell back)",
                env_vars={"SEARCH_QA_RETRIEVER": "bm25"},
            )
