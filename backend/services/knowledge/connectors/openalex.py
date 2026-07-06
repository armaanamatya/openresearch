"""OpenAlex connector — open catalog of works/authors/venues/concepts.

Adapted from OpenScience's ``science/connectors/literature/openalex.ts``
(Apache-2.0). Abstracts arrive as an ``abstract_inverted_index`` (word ->
positions) and are reconstructed to plain text via ``base.from_inverted``.
``mailto`` opts into OpenAlex's polite pool (higher/faster limits); an
optional ``OPENALEX_API_KEY`` enables the premium pool. Both are read at call
time so a credential set mid-session applies without a restart. Key-free:
works fine with neither set (falls into OpenAlex's shared "common pool") —
unlike upstream, we never embed a borrowed third-party contact address as a
hardcoded fallback; the ``mailto``/``api_key`` params are simply omitted when
unset.
"""

from __future__ import annotations

import os
import re
import urllib.parse

from backend.services.knowledge.connectors.base import (
    Connector,
    LiteratureRecord,
    from_inverted,
    join_authors,
    snippet,
)

_BASE_URL = "https://api.openalex.org/works"
_OPENALEX_ID_RE = re.compile(r"^https?://openalex\.org/", re.IGNORECASE)
_DOI_RE = re.compile(r"^10\.\d")


def _polite_params() -> dict[str, str]:
    """OpenAlex's polite-pool params, read at call time (env may change mid-session)."""
    params: dict[str, str] = {}
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def _short_id(work_id: str | None) -> str:
    return _OPENALEX_ID_RE.sub("", work_id or "")


def _authors(authorships: list) -> str | None:
    names = []
    for a in authorships or ():
        if isinstance(a, dict):
            name = (a.get("author") or {}).get("display_name")
            if name:
                names.append(name)
    return join_authors(names)


def _to_record(work: dict) -> LiteratureRecord:
    work_id = work.get("id")
    doi = work.get("doi")
    short_id = _short_id(work_id) or (doi or "")
    location = work.get("primary_location") or {}
    source_meta = (location.get("source") or {}).get("display_name")
    year = work.get("publication_year")
    who = _authors(work.get("authorships") or [])
    meta = ". ".join(p for p in (who, source_meta, str(year) if year else None) if p)
    title = work.get("display_name") or work.get("title")
    return LiteratureRecord(
        source="openalex",
        id=short_id or "",
        title=snippet(title, 300) or short_id or "Untitled",
        authors=who,
        year=year if isinstance(year, int) else None,
        abstract_snippet=snippet(from_inverted(work.get("abstract_inverted_index"))) or (meta or None),
        url=work_id or location.get("landing_page_url") or doi,
        venue=source_meta,
    )


class OpenAlexConnector(Connector):
    """OpenAlex — open scholarly graph of works, authors, venues, concepts."""

    source = "openalex"

    def search(self, query: str, *, limit: int = 10) -> list[LiteratureRecord]:
        per_page = max(1, min(int(limit or 10), 50))
        params = {"search": query, "per-page": per_page, **_polite_params()}
        data = self._fetch_json(_BASE_URL, params=params)
        if not data:
            return []
        results = data.get("results")
        if not isinstance(results, list):
            return []
        try:
            return [_to_record(w) for w in results if isinstance(w, dict)]
        except Exception:  # noqa: BLE001 — malformed record degrades to no results
            return []

    def fetch(self, record_id: str) -> LiteratureRecord | None:
        # OpenAlex accepts a bare work id (W…), or a DOI as a raw path segment
        # ("works/doi:10.x/y"); the DOI's slash/colon must stay unencoded
        # (mirrors upstream — only the non-DOI branch is percent-encoded).
        rid = (record_id or "").strip()
        if _DOI_RE.match(rid):
            path = f"doi:{rid}"
        else:
            path = urllib.parse.quote(_short_id(rid) or rid, safe="")
        data = self._fetch_json(f"{_BASE_URL}/{path}", params=_polite_params())
        if not isinstance(data, dict) or not data:
            return None
        try:
            return _to_record(data)
        except Exception:  # noqa: BLE001
            return None


__all__ = ["OpenAlexConnector"]
