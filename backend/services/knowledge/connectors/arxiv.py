"""arXiv connector — the Atom Query API (``export.arxiv.org/api``).

Adapted from OpenScience's ``science/connectors/literature/arxiv.ts``
(Apache-2.0). The API returns Atom XML, not JSON, so entries are parsed with
the small regex helpers in :mod:`base` (``xml_text`` / ``xml_blocks`` /
``xml_attr``) — mirroring upstream's own ``getText``/``getJSON`` split
(``arxiv.ts`` is the one connector that uses ``getText``; see ``fetch_text``
in ``base.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.services.knowledge.connectors.base import (
    Connector,
    LiteratureRecord,
    join_authors,
    snippet,
    xml_attr,
    xml_blocks,
    xml_text,
)

_BASE_URL = "https://export.arxiv.org/api/query"
_ABS_URL_RE = re.compile(r"^https?://arxiv\.org/abs/", re.IGNORECASE)


def _bare_id(id_url: str) -> str:
    return _ABS_URL_RE.sub("", (id_url or "").strip())


def _year_from_published(published: str | None) -> int | None:
    """arXiv's Atom ``<published>`` is an ISO-8601 timestamp (e.g.
    ``2026-05-25T00:00:00Z``); ``LiteratureRecord.year`` has no direct upstream
    field on this connector, so we derive it from the ISO year prefix."""
    if not published or len(published) < 4:
        return None
    try:
        return int(published[:4])
    except ValueError:
        return None


@dataclass(frozen=True)
class _Entry:
    entry_id: str
    title: str | None
    summary: str | None
    published: str | None
    authors: list[str]
    primary_category: str | None


def _parse_entries(xml: str) -> list[_Entry]:
    entries: list[_Entry] = []
    for block in xml_blocks(xml, "entry"):
        authors = [n for n in (xml_text(a, "name") for a in xml_blocks(block, "author")) if n]
        entries.append(
            _Entry(
                entry_id=xml_text(block, "id") or "",
                title=xml_text(block, "title"),
                summary=xml_text(block, "summary"),
                published=xml_text(block, "published"),
                authors=authors,
                primary_category=xml_attr(block, "arxiv:primary_category", "term"),
            )
        )
    return entries


def _to_record(entry: _Entry) -> LiteratureRecord:
    record_id = _bare_id(entry.entry_id)
    who = join_authors(entry.authors)
    meta = ". ".join(p for p in (who, entry.primary_category, (entry.published or "")[:10]) if p)
    return LiteratureRecord(
        source="arxiv",
        id=record_id or entry.entry_id,
        title=snippet(entry.title, 300) or record_id or "Untitled",
        authors=who,
        year=_year_from_published(entry.published),
        abstract_snippet=snippet(entry.summary) or (meta or None),
        url=entry.entry_id or (f"https://arxiv.org/abs/{record_id}" if record_id else None),
        venue=entry.primary_category,
    )


class ArxivConnector(Connector):
    """arXiv preprints — physics/math/CS/q-bio/... via the Atom query API."""

    source = "arxiv"

    def search(self, query: str, *, limit: int = 10) -> list[LiteratureRecord]:
        max_results = max(1, min(int(limit or 10), 50))
        xml = self._fetch_text(
            _BASE_URL,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
        )
        if not xml:
            return []
        try:
            return [_to_record(e) for e in _parse_entries(xml)]
        except Exception:  # noqa: BLE001 — malformed XML degrades to no results
            return []

    def fetch(self, record_id: str) -> LiteratureRecord | None:
        clean = _bare_id(record_id)
        xml = self._fetch_text(_BASE_URL, params={"id_list": clean, "max_results": 1})
        if not xml:
            return None
        try:
            entries = _parse_entries(xml)
        except Exception:  # noqa: BLE001
            return None
        return _to_record(entries[0]) if entries else None


__all__ = ["ArxivConnector"]
