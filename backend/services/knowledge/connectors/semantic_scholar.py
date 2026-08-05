"""Semantic Scholar connector — the Academic Graph API (``graph/v1/paper``).

Adapted from OpenScience's ``science/connectors/literature/semantic-scholar.ts``
(Apache-2.0). Key-free tier (rate-limited); ids may be a Semantic Scholar
``paperId`` or an external id such as ``"DOI:10.…"``, ``"ARXIV:2101.00001"``,
``"PMID:12345678"``, ``"CorpusId:12345"``. An OPTIONAL
``SEMANTIC_SCHOLAR_API_KEY`` lifts the shared key-free rate limit (read at
call time so a key set mid-session applies without a restart — same
convention as OpenAlex's polite-pool params).

``fetch()`` additionally carries the paper's citation-graph neighbours
(``references``/``citations``, capped at ``_MAX_RELATED`` per direction, with
arXiv/DOI external ids) — the seed material for literature-corpus expansion.
``search()`` hits stay neighbour-free (the search endpoint is not asked for
graph fields; volume × neighbours would be unbounded).
"""

from __future__ import annotations

import os

from backend.services.knowledge.connectors.base import (
    Connector,
    LiteratureRecord,
    join_authors,
    snippet,
)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
_FIELDS = "title,abstract,url,year,venue,citationCount,externalIds,authors.name"
_FETCH_FIELDS = (
    f"{_FIELDS},references.title,references.externalIds,citations.title,citations.externalIds"
)

# Citation-graph neighbours kept per direction. S2 returns the FULL reference/
# citation lists (hundreds for well-cited papers); the record must stay a
# bounded, constant-ish size like every other corpus-adjacent structure.
_MAX_RELATED = 100


def _api_headers() -> dict[str, str] | None:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    return {"x-api-key": key} if key else None


def _authors(entries: list) -> str | None:
    names = [a.get("name") for a in (entries or ()) if isinstance(a, dict) and a.get("name")]
    return join_authors(names)


def _related(entries: list | None) -> tuple[dict, ...]:
    """Normalize S2 ``references``/``citations`` entries to bounded neighbour dicts.

    Each kept entry becomes ``{"id", "title", "arxiv_id", "doi"}`` (plain dicts —
    lossless through the JSONL cache). Entries with neither an id nor a title
    are dropped; output is capped at ``_MAX_RELATED``. Defensive: any odd shape
    degrades to skipping the entry, never raising.
    """
    out: list[dict] = []
    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        ext = entry.get("externalIds")
        if not isinstance(ext, dict):
            ext = {}
        ref = {
            "id": entry.get("paperId") or "",
            "title": snippet(entry.get("title"), 300),
            "arxiv_id": ext.get("ArXiv"),
            "doi": ext.get("DOI"),
        }
        if not (ref["id"] or ref["title"]):
            continue
        out.append(ref)
        if len(out) >= _MAX_RELATED:
            break
    return tuple(out)


def _to_record(paper: dict) -> LiteratureRecord:
    paper_id = paper.get("paperId") or ""
    who = _authors(paper.get("authors") or [])
    venue = paper.get("venue")
    year = paper.get("year")
    meta = ". ".join(p for p in (who, venue, str(year) if year else None) if p)
    title = paper.get("title")
    return LiteratureRecord(
        source="semantic_scholar",
        id=paper_id,
        title=snippet(title, 300) or paper_id or "Untitled",
        authors=who,
        year=year if isinstance(year, int) else None,
        abstract_snippet=snippet(paper.get("abstract")) or (meta or None),
        url=paper.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None),
        venue=venue,
        references=_related(paper.get("references")),
        citations=_related(paper.get("citations")),
    )


class SemanticScholarConnector(Connector):
    """Semantic Scholar — AI-powered academic graph (abstracts/citations/influence)."""

    source = "semantic_scholar"

    def search(self, query: str, *, limit: int = 10) -> list[LiteratureRecord]:
        n = max(1, min(int(limit or 10), 50))
        data = self._fetch_json(
            f"{_BASE_URL}/search",
            params={"query": query, "limit": n, "fields": _FIELDS},
            headers=_api_headers(),
        )
        if not data:
            return []
        results = data.get("data")
        if not isinstance(results, list):
            return []
        try:
            return [_to_record(p) for p in results if isinstance(p, dict)]
        except Exception:  # noqa: BLE001 — malformed record degrades to no results
            return []

    def fetch(self, record_id: str) -> LiteratureRecord | None:
        # Ids ("DOI:10.x/y", "ARXIV:…", a bare paperId hex) are a raw path
        # segment — colon/slash in an external id must not be percent-encoded
        # (mirrors upstream: `${BASE}/${id.trim()}`, no encodeURIComponent).
        rid = (record_id or "").strip()
        data = self._fetch_json(f"{_BASE_URL}/{rid}", params={"fields": _FETCH_FIELDS}, headers=_api_headers())
        if not isinstance(data, dict) or not data:
            return None
        try:
            return _to_record(data)
        except Exception:  # noqa: BLE001
            return None


__all__ = ["SemanticScholarConnector"]
