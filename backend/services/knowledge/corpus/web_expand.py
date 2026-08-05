"""Runtime web expansion of the literature corpus — Phase 3 of the
literature-corpus plan (``OPENRESEARCH_LITERATURE_WEB``, default-OFF).

Lets the ROOT ask for corpus growth mid-run — ``search_literature(
web_query=...)`` to discover candidate papers, ``search_literature(
fetch_id=...)`` to pull one into the corpus — while every byte of network
I/O stays ORCHESTRATOR-SIDE, behind the existing connectors and fetchers
with their size caps, parsers, and fail-soft conventions. The deliberate
non-goal (pivot-brief §7) is unchanged: sub-agents get zero web tools, and
nothing here hands the model a browser — it hands the model a bounded,
typed record of what the server fetched.

Bounds:
  - ``MAX_RUNTIME_FETCHES`` (5) full-text fetches per run, enforced via a
    persisted ledger (``rlm_state/literature_web_fetches.json``) so a warm
    retry cannot reset the budget;
  - a re-fetch of a paper the corpus already holds at ``fulltext`` is FREE
    (no network, not counted) — arXiv versions are immutable;
  - discovery returns bounded metadata only (id/title/year/snippet), never
    page content.

Network calls additionally require ``OPENRESEARCH_LITERATURE_GROUNDING``
(the connectors' call-site gate, same as the corpus builder) — without it
this module answers ``network_disabled`` instead of dialing out.

Everything fetched is corpus content: ADVISORY context only, never citable
report evidence, same as the rest of the corpus.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from backend.services.knowledge.connectors import literature_grounding_enabled
from backend.services.knowledge.corpus.store import CorpusStore, normalize_paper_id

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")

#: Full-text fetch budget per run — an advisory expansion valve, not a crawler.
MAX_RUNTIME_FETCHES = 5
#: Discovery hits returned per query (bounded metadata only).
MAX_DISCOVER_HITS = 5

_LEDGER_FILENAME = "literature_web_fetches.json"
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def literature_web_enabled() -> bool:
    """True iff ``OPENRESEARCH_LITERATURE_WEB`` is truthy."""
    return os.environ.get("OPENRESEARCH_LITERATURE_WEB", "").strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Persisted per-run fetch ledger
# ---------------------------------------------------------------------------


def _ledger_path(project_dir: Path) -> Path:
    return Path(project_dir) / "rlm_state" / _LEDGER_FILENAME


def _load_ledger(project_dir: Path) -> dict[str, Any]:
    try:
        path = _ledger_path(project_dir)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("fetched"), list):
                return data
    except (OSError, ValueError):
        pass
    return {"fetched": []}


def _record_fetch(project_dir: Path, fetch_id: str) -> None:
    ledger = _load_ledger(project_dir)
    if fetch_id not in ledger["fetched"]:
        ledger["fetched"].append(fetch_id)
    path = _ledger_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tmp.replace(path)


def fetches_remaining(project_dir: Path) -> int:
    return max(0, MAX_RUNTIME_FETCHES - len(_load_ledger(project_dir)["fetched"]))


# ---------------------------------------------------------------------------
# Discovery — server-side search, bounded metadata out
# ---------------------------------------------------------------------------


def discover_papers(
    query: str,
    *,
    search_connector: Any | None = None,
    limit: int = MAX_DISCOVER_HITS,
) -> dict[str, Any]:
    """Search for candidate papers via the literature connectors.

    Returns bounded metadata records the root can pass back as ``fetch_id``.
    Uses the arXiv connector by default — its hits carry directly-fetchable
    ids (unlike a generic web search, which returns URLs the fetch path
    could not consume). Fail-soft: trouble -> ``{"status": "error"}``.
    """
    if not literature_web_enabled():
        return {"status": "disabled"}
    if not literature_grounding_enabled():
        return {"status": "network_disabled", "note": "set OPENRESEARCH_LITERATURE_GROUNDING=1"}
    query = (query or "").strip()
    if not query:
        return {"status": "error", "error": "web_query must be non-empty"}
    try:
        connector = search_connector
        if connector is None:
            from backend.services.knowledge.connectors import ArxivConnector

            connector = ArxivConnector()
        hits = connector.search(query, limit=max(1, min(int(limit), MAX_DISCOVER_HITS)))
        records = []
        for hit in hits[:MAX_DISCOVER_HITS]:
            arxiv_hint = hit.id if hit.source == "arxiv" else None
            records.append(
                {
                    "fetch_id": normalize_paper_id(arxiv_id=arxiv_hint),
                    "title": (hit.title or "")[:200],
                    "year": hit.year,
                    "snippet": (hit.abstract_snippet or "")[:300],
                }
            )
        return {
            "status": "ok",
            "kind": "web_discover",
            "hits": records,
            "note": (
                "Pass a non-null fetch_id back as search_literature(fetch_id=...) "
                "to pull that paper into the corpus (ADVISORY context only)."
            ),
        }
    except Exception as exc:  # noqa: BLE001 — discovery must never break the run
        logger.debug("web_expand: discover failed", exc_info=True)
        return {"status": "error", "error": str(exc)[:300]}


# ---------------------------------------------------------------------------
# Fetch-one — on-demand corpus growth
# ---------------------------------------------------------------------------


def _normalize_fetch_id(fetch_id: str) -> tuple[str | None, str | None]:
    """(corpus_id, arxiv_id) for a root-supplied fetch id; (None, None) if unusable.

    Accepts ``arxiv:2101.00001``, a bare ``2101.00001``(v-suffixed ok), or a
    ``doi:...`` corpus id (metadata-only — no PDF source for bare DOIs).
    """
    raw = (fetch_id or "").strip()
    if not raw:
        return None, None
    if raw.lower().startswith("arxiv:"):
        raw = raw[len("arxiv:"):]
    if _ARXIV_ID_RE.match(raw):
        return normalize_paper_id(arxiv_id=raw), re.sub(r"v\d+$", "", raw)
    if raw.lower().startswith("doi:"):
        return normalize_paper_id(doi=raw[len("doi:"):]), None
    return None, None


def fetch_paper(
    fetch_id: str,
    *,
    store: CorpusStore,
    project_dir: Path,
    s2_connector: Any | None = None,
    download_pdf_fn: Callable[..., Any] | None = None,
    parser: Any | None = None,
) -> dict[str, Any]:
    """Fetch ONE paper into the corpus at runtime; bounded + budgeted.

    Order of gates (each answers with a distinct status, never raises):
    flag -> id shape -> already-present (free) -> network gate -> budget ->
    fetch. On success the paper lands exactly like a build-time fetch:
    ``lit_papers`` row, ``parsed_full_text.txt``, deterministic dictionary
    entities, refreshed chunk index.
    """
    if not literature_web_enabled():
        return {"status": "disabled"}
    try:
        corpus_id, arxiv_id = _normalize_fetch_id(fetch_id)
        if corpus_id is None:
            return {
                "status": "error",
                "error": f"unusable fetch_id {fetch_id[:80]!r} (want an arXiv id)",
            }

        existing = store.get_paper(corpus_id)
        if existing is not None and existing.get("fetched_level") == "fulltext":
            return {
                "status": "ok",
                "kind": "web_fetch",
                "id": corpus_id,
                "title": (existing.get("title") or "")[:200],
                "already_present": True,
                "fetches_remaining": fetches_remaining(project_dir),
            }

        if not literature_grounding_enabled():
            return {"status": "network_disabled", "note": "set OPENRESEARCH_LITERATURE_GROUNDING=1"}
        if fetches_remaining(project_dir) <= 0:
            return {
                "status": "budget_exhausted",
                "note": f"the {MAX_RUNTIME_FETCHES}-fetch runtime budget is spent",
            }
        if arxiv_id is None:
            return {
                "status": "error",
                "error": "only arXiv-hosted papers can be fetched at runtime",
            }

        # The budget is charged for the ATTEMPT (network was dialed), not the
        # outcome — a failing id cannot be retried into a free crawl loop.
        _record_fetch(project_dir, corpus_id)

        title, year, abstract = "", None, None
        s2 = s2_connector
        if s2 is None:
            from backend.services.knowledge.connectors import SemanticScholarConnector

            s2 = SemanticScholarConnector()
        try:
            record = s2.fetch(f"ARXIV:{arxiv_id}")
            if record is not None:
                title = record.title or ""
                year = record.year
                abstract = record.abstract_snippet
        except Exception:  # noqa: BLE001 — metadata is a nicety, the PDF is the point
            logger.debug("web_expand: s2 metadata fetch failed", exc_info=True)

        store.upsert_paper(
            corpus_id, title=title, year=year, arxiv_id=arxiv_id, abstract=abstract
        )

        from backend.services.knowledge.corpus.builder import _fetch_fulltexts, _Candidate

        report = _WebFetchReport()
        _fetch_fulltexts(
            [
                _Candidate(
                    norm_id=corpus_id,
                    title=title,
                    arxiv_id=arxiv_id,
                    relation="web_fetch",
                    score=0.0,
                )
            ],
            store,
            report,
            download_pdf_fn,
            parser,
        )
        if report.fetched_fulltext < 1:
            return {
                "status": "error",
                "error": (report.errors[0] if report.errors else "fetch/parse produced no text"),
                "fetches_remaining": fetches_remaining(project_dir),
            }

        from backend.services.knowledge.corpus.extraction import run_deterministic_extraction
        from backend.services.knowledge.corpus.index import build_chunk_index

        try:
            run_deterministic_extraction(store)
            build_chunk_index(store)
        except Exception:  # noqa: BLE001 — the text is in; indexing trouble degrades
            logger.debug("web_expand: post-fetch indexing failed", exc_info=True)

        fetched = store.get_paper(corpus_id) or {}
        return {
            "status": "ok",
            "kind": "web_fetch",
            "id": corpus_id,
            "title": (fetched.get("title") or title or "")[:200],
            "already_present": False,
            "fetches_remaining": fetches_remaining(project_dir),
            "note": "ADVISORY corpus content — never cite as report evidence",
        }
    except Exception as exc:  # noqa: BLE001 — an expansion valve must never break the run
        logger.debug("web_expand: fetch_paper failed", exc_info=True)
        return {"status": "error", "error": str(exc)[:300]}


class _WebFetchReport:
    """The minimal duck-typed surface ``_fetch_fulltexts`` writes to."""

    def __init__(self) -> None:
        self.fetched_fulltext = 0
        self.errors: list[str] = []


# _fetch_fulltexts appends via builder._note_error(report, ...), which does
# report.errors.append(...) with a bound — mirror the attribute it needs.


__all__ = [
    "MAX_DISCOVER_HITS",
    "MAX_RUNTIME_FETCHES",
    "discover_papers",
    "fetch_paper",
    "fetches_remaining",
    "literature_web_enabled",
]
