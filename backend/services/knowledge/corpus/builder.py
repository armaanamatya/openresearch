"""Corpus builder — seed → rank → cap → persist → fetch, all fail-soft.

Given a reproduction target, assembles the *field-specific* related-paper set:

  1. **Seed** (specificity comes from the target's own graph, not topic search):
     - the target paper's parsed reference list (arXiv/DOI-bearing rows),
     - the Semantic Scholar citation graph, 1 hop both directions
       (``fetch()`` neighbours from the Phase-0 connector extension),
     - a small connector title search (recall backstop, lowest-ranked).
  2. **Rank** deterministically: citation adjacency (direct reference >
     S2-reference > citer > search hit) + a title token-overlap bonus.
  3. **Cap** at ``MAX_PAPERS``; persist metadata + citation edges to the
     global ``CorpusStore``; fetch + parse full text for the top ``FETCH_K``
     arXiv-hosted papers.
  4. Write the per-target ranking to
     ``runs/<project_id>/rlm_state/literature_spec.json`` (the store itself
     holds facts only — relevance is target-relative).

Flags: ``OPENRESEARCH_LITERATURE_CORPUS`` gates the builder entirely;
network operations (S2 graph, search, PDF fetch) additionally require
``OPENRESEARCH_LITERATURE_GROUNDING`` — without it the corpus builds
metadata-only from the already-parsed local reference list. Everything is
LLM-free and CPU-only, so the campaign UNDERSTAND phase can host it without
violating its $0 contract. Structured entity/result extraction (the
``lit_entities``/``lit_results`` tables) is a later, separately-gated slice.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.services.knowledge.connectors.base import (
    Connector,
    literature_grounding_enabled,
)
from backend.services.knowledge.corpus.store import (
    RELATION_CITES,
    CorpusStore,
    corpus_root,
    normalize_paper_id,
)

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")

# Bounded-cache constants (context-map convention: explicit, deterministic).
MAX_PAPERS = 25          # ranked candidates persisted per build
FETCH_K = 8              # top candidates fetched as full text
MAX_SEED_REFS = 50       # target-paper reference rows considered
SEARCH_LIMIT = 5         # connector title-search hits considered
MAX_ERRORS = 10          # bounded error log on the build report

# Ranking: citation adjacency dominates; token overlap only tiebreaks.
_SCORE_DIRECT_REF = 3.0  # row in the target's own parsed reference list
_SCORE_S2_REF = 2.5      # S2 says the target cites it (catches regex misses)
_SCORE_CITER = 2.0       # S2 says it cites the target
_SCORE_SEARCH = 1.0      # title-search hit
_OVERLAP_BONUS_MAX = 0.5
_MIN_TOKEN_LEN = 4
_SPEC_ABSTRACT_MAX = 600   # per-paper abstract cap in literature_spec.json

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def literature_corpus_enabled() -> bool:
    """True iff ``OPENRESEARCH_LITERATURE_CORPUS`` is truthy (default OFF)."""
    return os.environ.get("OPENRESEARCH_LITERATURE_CORPUS", "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class TargetPaper:
    """The reproduction target the corpus is built around."""

    project_id: str
    arxiv_id: str | None = None
    title: str = ""
    # Parsed Reference rows as dicts: {"arxiv_id", "doi", "raw_text", "title"}.
    references: tuple[dict, ...] = ()


@dataclass
class BuildReport:
    """What a build did — surfaced by the CLI and the campaign artifact."""

    target_id: str
    papers_ranked: int = 0
    papers_persisted: int = 0
    relations_added: int = 0
    fetched_fulltext: int = 0
    network_enabled: bool = False
    spec_path: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "papers_ranked": self.papers_ranked,
            "papers_persisted": self.papers_persisted,
            "relations_added": self.relations_added,
            "fetched_fulltext": self.fetched_fulltext,
            "network_enabled": self.network_enabled,
            "spec_path": self.spec_path,
            "errors": list(self.errors),
        }


@dataclass
class _Candidate:
    norm_id: str
    title: str = ""
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    abstract: str | None = None
    relation: str = "direct_ref"   # direct_ref | s2_ref | citer | search
    score: float = 0.0


def _title_tokens(title: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(title or "") if len(t) >= _MIN_TOKEN_LEN}


def _overlap_bonus(target_tokens: set[str], title: str) -> float:
    if not target_tokens:
        return 0.0
    tokens = _title_tokens(title)
    if not tokens:
        return 0.0
    return _OVERLAP_BONUS_MAX * len(tokens & target_tokens) / len(tokens | target_tokens)


def _note_error(report: BuildReport, message: str) -> None:
    if len(report.errors) < MAX_ERRORS:
        report.errors.append(message[:200])


def build_corpus(
    *,
    runs_root: Path,
    target: TargetPaper,
    s2_connector: Connector | None = None,
    search_connector: Connector | None = None,
    download_pdf_fn: Callable[..., Any] | None = None,
    parser: Any | None = None,
    store: CorpusStore | None = None,
) -> BuildReport | None:
    """Build/refresh the corpus around ``target``. ``None`` when the flag is off.

    Every external step (each connector call, each PDF fetch/parse) is
    individually fail-soft: an error is recorded on the bounded report error
    list and the build continues. The function itself only raises on
    programmer error (bad arguments), never on environment trouble.
    """
    if not literature_corpus_enabled():
        return None

    target_norm = normalize_paper_id(arxiv_id=target.arxiv_id) or f"project:{target.project_id}"
    report = BuildReport(target_id=target_norm, network_enabled=literature_grounding_enabled())
    target_tokens = _title_tokens(target.title)

    candidates: dict[str, _Candidate] = {}

    def _admit(cand: _Candidate) -> None:
        existing = candidates.get(cand.norm_id)
        if existing is None:
            candidates[cand.norm_id] = cand
            return
        # Keep the strongest relation/score; enrich empty metadata in place.
        if cand.score > existing.score:
            existing.relation, existing.score = cand.relation, cand.score
        existing.title = existing.title or cand.title
        existing.year = existing.year if existing.year is not None else cand.year
        existing.venue = existing.venue or cand.venue
        existing.arxiv_id = existing.arxiv_id or cand.arxiv_id
        existing.doi = existing.doi or cand.doi
        existing.s2_id = existing.s2_id or cand.s2_id
        existing.abstract = existing.abstract or cand.abstract

    # --- 1a. seed: the target's own parsed reference list --------------------
    for ref in target.references[:MAX_SEED_REFS]:
        if not isinstance(ref, dict):
            continue
        norm = normalize_paper_id(arxiv_id=ref.get("arxiv_id"), doi=ref.get("doi"))
        if norm is None or norm == target_norm:
            continue
        title = (ref.get("title") or "").strip()
        _admit(
            _Candidate(
                norm_id=norm,
                title=title,
                arxiv_id=ref.get("arxiv_id"),
                doi=ref.get("doi"),
                relation="direct_ref",
                score=_SCORE_DIRECT_REF + _overlap_bonus(target_tokens, title),
            )
        )

    # --- 1b. seed: S2 citation graph, 1 hop both directions ------------------
    if report.network_enabled and target.arxiv_id:
        s2 = s2_connector
        if s2 is None:
            from backend.services.knowledge.connectors import SemanticScholarConnector

            s2 = SemanticScholarConnector()
        try:
            record = s2.fetch(f"ARXIV:{_strip_version(target.arxiv_id)}")
        except Exception as exc:  # noqa: BLE001 — connector trouble degrades
            record = None
            _note_error(report, f"s2 fetch failed: {exc}")
        if record is not None:
            for neighbour, relation, base in (
                *((n, "s2_ref", _SCORE_S2_REF) for n in record.references),
                *((n, "citer", _SCORE_CITER) for n in record.citations),
            ):
                norm = normalize_paper_id(
                    arxiv_id=neighbour.get("arxiv_id"),
                    doi=neighbour.get("doi"),
                    s2_id=neighbour.get("id"),
                )
                if norm is None or norm == target_norm:
                    continue
                title = neighbour.get("title") or ""
                _admit(
                    _Candidate(
                        norm_id=norm,
                        title=title,
                        arxiv_id=neighbour.get("arxiv_id"),
                        doi=neighbour.get("doi"),
                        s2_id=neighbour.get("id") or None,
                        relation=relation,
                        score=base + _overlap_bonus(target_tokens, title),
                    )
                )

    # --- 1c. seed: connector title search (recall backstop) ------------------
    if report.network_enabled and target.title:
        search = search_connector
        if search is None:
            from backend.services.knowledge.connectors import ArxivConnector

            search = ArxivConnector()
        try:
            hits = search.search(target.title, limit=SEARCH_LIMIT)
        except Exception as exc:  # noqa: BLE001
            hits = []
            _note_error(report, f"search failed: {exc}")
        for hit in hits:
            arxiv_hint = hit.id if hit.source == "arxiv" else None
            norm = normalize_paper_id(arxiv_id=arxiv_hint)
            if norm is None or norm == target_norm:
                continue
            _admit(
                _Candidate(
                    norm_id=norm,
                    title=hit.title,
                    year=hit.year,
                    venue=hit.venue,
                    arxiv_id=arxiv_hint,
                    abstract=hit.abstract_snippet,
                    relation="search",
                    score=_SCORE_SEARCH + _overlap_bonus(target_tokens, hit.title),
                )
            )

    # --- 2/3. rank, cap, persist ---------------------------------------------
    ranked = sorted(candidates.values(), key=lambda c: (-c.score, c.norm_id))[:MAX_PAPERS]
    report.papers_ranked = len(ranked)

    corpus = store if store is not None else CorpusStore(corpus_root(runs_root))
    try:
        corpus.initialize()
        corpus.upsert_paper(
            target_norm,
            title=target.title,
            arxiv_id=target.arxiv_id,
        )
        for cand in ranked:
            corpus.upsert_paper(
                cand.norm_id,
                title=cand.title,
                year=cand.year,
                venue=cand.venue,
                arxiv_id=cand.arxiv_id,
                doi=cand.doi,
                s2_id=cand.s2_id,
                abstract=cand.abstract,
            )
            report.papers_persisted += 1
            if cand.relation in ("direct_ref", "s2_ref"):
                corpus.add_relation(target_norm, cand.norm_id, kind=RELATION_CITES)
                report.relations_added += 1
            elif cand.relation == "citer":
                corpus.add_relation(cand.norm_id, target_norm, kind=RELATION_CITES)
                report.relations_added += 1
            # "search" hits carry no citation relation — nothing persisted.
    except Exception as exc:  # noqa: BLE001 — a broken store must not break the caller
        _note_error(report, f"store persist failed: {exc}")
        logger.debug("corpus: persist failed", exc_info=True)

    # --- 3b. deterministic graph metrics + bounded ranking blend ---------------
    # Computed BEFORE the full-text fetch so the FETCH_K budget follows the
    # same final_score ordering the spec/prior_work_refs present to the agent.
    graph_metrics_map: dict[str, dict[str, float]] = {}
    try:
        from backend.services.knowledge.corpus.graph_metrics import compute_graph_metrics

        graph_metrics_map = compute_graph_metrics(corpus.edges(), target_norm)
        corpus.put_graph_metrics(target_norm, graph_metrics_map)
    except Exception as exc:  # noqa: BLE001 — analytics must never break the build
        _note_error(report, f"graph metrics failed: {exc}")
        logger.debug("corpus: graph metrics failed", exc_info=True)

    # --- 3c. fetch + parse full text for the top FETCH_K by final_score --------
    if report.network_enabled:
        _fetch_fulltexts(
            _final_ranked(ranked, graph_metrics_map)[:FETCH_K],
            corpus,
            report,
            download_pdf_fn,
            parser,
        )

    # --- 3d. chunk + lexical index over fetched texts --------------------------
    try:
        from backend.services.knowledge.corpus.index import build_chunk_index

        build_chunk_index(corpus)
    except Exception as exc:  # noqa: BLE001
        _note_error(report, f"chunk index failed: {exc}")
        logger.debug("corpus: chunk index failed", exc_info=True)

    # --- 3e. deterministic dictionary entities (Lane A, free, $0) --------------
    # LLM result extraction is deliberately NOT here — it costs money and is
    # operator-invoked via `corpus extract` (keeps UNDERSTAND's $0 contract).
    try:
        from backend.services.knowledge.corpus.extraction import (
            run_deterministic_extraction,
        )

        run_deterministic_extraction(corpus)
    except Exception as exc:  # noqa: BLE001
        _note_error(report, f"entity extraction failed: {exc}")
        logger.debug("corpus: entity extraction failed", exc_info=True)

    # --- 4. per-target spec ----------------------------------------------------
    try:
        spec_path = _write_spec(
            runs_root, target, target_norm, ranked, corpus, graph_metrics_map
        )
        report.spec_path = str(spec_path)
    except Exception as exc:  # noqa: BLE001
        _note_error(report, f"spec write failed: {exc}")
        logger.debug("corpus: spec write failed", exc_info=True)

    return report


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _final_ranked(
    ranked: list["_Candidate"],
    graph_metrics_map: dict[str, dict[str, float]],
) -> list["_Candidate"]:
    """Candidates ordered by final_score (adjacency + bounded graph boost)."""
    from backend.services.knowledge.corpus.graph_metrics import graph_boost

    def _final(c: "_Candidate") -> float:
        return c.score + graph_boost(graph_metrics_map.get(c.norm_id, {}))

    return sorted(ranked, key=lambda c: (-_final(c), c.norm_id))


def _fetch_fulltexts(
    top: list[_Candidate],
    corpus: CorpusStore,
    report: BuildReport,
    download_pdf_fn: Callable[..., Any] | None,
    parser: Any | None,
) -> None:
    """Fetch+parse arXiv-hosted candidates into their corpus paper dirs.

    Per-paper fail-soft: one bad PDF never stops the rest. ``download_pdf``'s
    validated-cache reuse makes re-builds free (arXiv versions are immutable).
    """
    dl = download_pdf_fn
    if dl is None:
        from backend.services.ingestion.intake.fetchers.remote_pdf import download_pdf

        dl = download_pdf

    for cand in top:
        if not cand.arxiv_id:
            continue
        try:
            paper_dir = corpus.paper_dir(cand.norm_id, create=True)
            dl(
                url=f"https://arxiv.org/pdf/{_strip_version(cand.arxiv_id)}",
                runs_root=paper_dir.parent,
                project_id=paper_dir.name,
                fetched_via="literature_corpus",
            )
            text = _parse_pdf(paper_dir, cand.norm_id, parser)
            if text:
                tmp = paper_dir / "parsed_full_text.txt.tmp"
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(paper_dir / "parsed_full_text.txt")
                corpus.upsert_paper(cand.norm_id, fetched_level="fulltext")
                corpus.write_paper_meta(
                    cand.norm_id,
                    {"id": cand.norm_id, "title": cand.title, "arxiv_id": cand.arxiv_id},
                )
                report.fetched_fulltext += 1
        except Exception as exc:  # noqa: BLE001 — per-paper fail-soft
            _note_error(report, f"fetch {cand.norm_id}: {exc}")
            logger.debug("corpus: fulltext fetch failed for %s", cand.norm_id, exc_info=True)


def _parse_pdf(paper_dir: Path, norm_id: str, parser: Any | None) -> str:
    pdf_path = paper_dir / "raw_paper.pdf"
    if not pdf_path.exists():
        return ""
    p = parser
    if p is None:
        from backend.services.ingestion.parser.resolving_parser import ResolvingParser

        p = ResolvingParser()
    result = p.parse(project_id=f"corpus:{norm_id}", paper_path=pdf_path)
    return result.full_text or ""


def _write_spec(
    runs_root: Path,
    target: TargetPaper,
    target_norm: str,
    ranked: list[_Candidate],
    corpus: CorpusStore,
    graph_metrics_map: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Atomic per-target ranking spec (mirrors ``rlm_state/repo_spec.json``).

    Papers are ordered by ``final_score`` = adjacency score + the bounded
    graph-metric boost (≤0.45 — reorders within a tier, never jumps one).
    """
    from backend.services.knowledge.corpus.graph_metrics import graph_boost

    gm = graph_metrics_map or {}

    def _paper_entry(c: _Candidate) -> dict:
        per_paper = gm.get(c.norm_id, {})
        boost = graph_boost(per_paper) if per_paper else 0.0
        entry = {
            "id": c.norm_id,
            "title": c.title,
            "year": c.year,
            "relation": c.relation,
            "score": round(c.score, 4),
            "final_score": round(c.score + boost, 4),
            "arxiv_id": c.arxiv_id,
            "doi": c.doi,
        }
        abstract = c.abstract
        if not abstract:
            # A prior build may already hold the abstract in the global store.
            try:
                row = corpus.get_paper(c.norm_id)
                abstract = (row or {}).get("abstract")
            except Exception:  # noqa: BLE001 — the spec must survive a broken store
                abstract = None
        if isinstance(abstract, str) and abstract.strip():
            entry["abstract"] = abstract.strip()[:_SPEC_ABSTRACT_MAX]
        if per_paper:
            entry["graph"] = {k: round(v, 6) for k, v in sorted(per_paper.items())}
        return entry

    papers = sorted(
        (_paper_entry(c) for c in ranked),
        key=lambda e: (-e["final_score"], e["id"]),
    )
    spec = {
        "target": {
            "project_id": target.project_id,
            "id": target_norm,
            "arxiv_id": target.arxiv_id,
            "title": target.title,
        },
        "corpus_root": str(corpus.root),
        "papers": papers,
    }
    state_dir = Path(runs_root) / target.project_id / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    spec_path = state_dir / "literature_spec.json"
    tmp = spec_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    tmp.replace(spec_path)
    return spec_path


__all__ = [
    "FETCH_K",
    "MAX_PAPERS",
    "BuildReport",
    "TargetPaper",
    "build_corpus",
    "literature_corpus_enabled",
]
