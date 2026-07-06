"""Tests for backend.services.knowledge.connectors — the key-free literature
connectors (arXiv / OpenAlex / Semantic Scholar).

The suite is socket-hermetic (pytest-socket blocks every non-loopback host),
so every connector test injects a FAKE transport returning a canned in-memory
response instead of touching the network — the real seam
(``fetch_json_fn``/``fetch_text_fn``) that production code also uses.

Covers, per connector: (a) a canned response parses into well-formed
``LiteratureRecord``s with the right endpoint/params, (b) a blocked/failing
transport degrades to ``[]``/``None`` without raising. A final test proves
the REAL httpx-based default transport (no injection at all) also fails soft
under this hermetic environment — genuine, unmocked evidence of the
fail-soft contract.
"""

from __future__ import annotations

from backend.services.knowledge.connectors import (
    ArxivConnector,
    LiteratureRecord,
    OpenAlexConnector,
    SemanticScholarConnector,
    default_connectors,
    fetch_json,
    fetch_text,
)

# ---------------------------------------------------------------------------
# Canned fixtures — shaped exactly like the real APIs (endpoints/fields
# verified against the OpenScience TypeScript source before porting).
# ---------------------------------------------------------------------------

_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2605.15155v1</id>
    <title>Self-Distilled Agentic Reinforcement Learning</title>
    <summary>We introduce SDAR, a self-distilled agentic RL method that improves
    sample efficiency across ALFWorld, WebShop, and Search-QA.</summary>
    <published>2026-05-25T00:00:00Z</published>
    <updated>2026-05-25T00:00:00Z</updated>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <link title="pdf" href="http://arxiv.org/pdf/2605.15155v1" rel="related" type="application/pdf"/>
  </entry>
</feed>"""

_OPENALEX_WORK = {
    "id": "https://openalex.org/W4400000001",
    "doi": "https://doi.org/10.1000/xyz123",
    "display_name": "Self-Distilled Agentic Reinforcement Learning",
    "publication_year": 2026,
    "cited_by_count": 12,
    "abstract_inverted_index": {"We": [0], "introduce": [1], "SDAR": [2], "a": [3], "method": [4]},
    "authorships": [
        {"author": {"display_name": "Jane Doe"}},
        {"author": {"display_name": "John Smith"}},
    ],
    "primary_location": {
        "source": {"display_name": "arXiv"},
        "landing_page_url": "https://arxiv.org/abs/2605.15155",
    },
    "relevance_score": 0.98,
}
_OPENALEX_SEARCH_RESPONSE = {"results": [_OPENALEX_WORK], "meta": {"count": 1}}

_S2_PAPER = {
    "paperId": "abc123def456",
    "title": "Self-Distilled Agentic Reinforcement Learning",
    "abstract": "We introduce SDAR, a self-distilled agentic RL method.",
    "url": "https://www.semanticscholar.org/paper/abc123def456",
    "year": 2026,
    "venue": "arXiv preprint",
    "citationCount": 8,
    "authors": [{"name": "Jane Doe"}, {"name": "John Smith"}],
    "externalIds": {"ArXiv": "2605.15155"},
}
_S2_SEARCH_RESPONSE = {"total": 1, "data": [_S2_PAPER]}


class _RecordingFakeText:
    """Injectable ``fetch_text``-shaped fake: records calls, returns a canned body."""

    def __init__(self, body: str | None) -> None:
        self.body = body
        self.calls: list[dict] = []

    def __call__(self, url, *, params=None, headers=None, timeout_s=10.0):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.body


class _RecordingFakeJson:
    """Injectable ``fetch_json``-shaped fake: records calls, returns a canned dict."""

    def __init__(self, body: dict | None) -> None:
        self.body = body
        self.calls: list[dict] = []

    def __call__(self, url, *, params=None, headers=None, timeout_s=10.0):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.body


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def test_arxiv_search_parses_canned_atom_response():
    fake = _RecordingFakeText(_ARXIV_ATOM)
    connector = ArxivConnector(fetch_text_fn=fake)

    records = connector.search("SDAR agentic reinforcement learning", limit=5)

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, LiteratureRecord)
    assert record.source == "arxiv"
    assert record.id == "2605.15155v1"
    assert record.title == "Self-Distilled Agentic Reinforcement Learning"
    assert record.authors == "Jane Doe, John Smith"
    assert record.year == 2026
    assert "SDAR" in record.abstract_snippet
    assert record.url == "http://arxiv.org/abs/2605.15155v1"
    assert record.venue == "cs.LG"

    # Endpoint/query fidelity — the real export.arxiv.org Atom API shape.
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == "https://export.arxiv.org/api/query"
    assert call["params"]["search_query"] == "all:SDAR agentic reinforcement learning"
    assert call["params"]["max_results"] == 5
    assert call["params"]["sortBy"] == "relevance"


def test_arxiv_fetch_by_id_strips_abs_prefix():
    fake = _RecordingFakeText(_ARXIV_ATOM)
    connector = ArxivConnector(fetch_text_fn=fake)

    record = connector.fetch("https://arxiv.org/abs/2605.15155")

    assert record is not None
    assert record.id == "2605.15155v1"
    call = fake.calls[0]
    assert call["params"]["id_list"] == "2605.15155"
    assert call["params"]["max_results"] == 1


def test_arxiv_blocked_transport_degrades_to_empty():
    fake = _RecordingFakeText(None)  # simulates a timeout / blocked network
    connector = ArxivConnector(fetch_text_fn=fake)

    assert connector.search("anything", limit=5) == []
    assert connector.fetch("2605.15155") is None


def test_arxiv_malformed_xml_degrades_to_empty():
    fake = _RecordingFakeText("not xml at all { } <<<")
    connector = ArxivConnector(fetch_text_fn=fake)

    assert connector.search("anything", limit=5) == []


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def test_openalex_search_parses_canned_json_response():
    fake = _RecordingFakeJson(_OPENALEX_SEARCH_RESPONSE)
    connector = OpenAlexConnector(fetch_json_fn=fake)

    records = connector.search("SDAR reinforcement learning", limit=5)

    assert len(records) == 1
    record = records[0]
    assert record.source == "openalex"
    assert record.id == "W4400000001"
    assert record.title == "Self-Distilled Agentic Reinforcement Learning"
    assert record.authors == "Jane Doe, John Smith"
    assert record.year == 2026
    assert record.abstract_snippet == "We introduce SDAR a method"
    assert record.url == "https://openalex.org/W4400000001"
    assert record.venue == "arXiv"

    call = fake.calls[0]
    assert call["url"] == "https://api.openalex.org/works"
    assert call["params"]["search"] == "SDAR reinforcement learning"
    assert call["params"]["per-page"] == 5


def test_openalex_fetch_by_doi_uses_doi_path():
    fake = _RecordingFakeJson(_OPENALEX_WORK)
    connector = OpenAlexConnector(fetch_json_fn=fake)

    record = connector.fetch("10.1000/xyz123")

    assert record is not None
    assert record.id == "W4400000001"
    call = fake.calls[0]
    assert call["url"] == "https://api.openalex.org/works/doi:10.1000/xyz123"


def test_openalex_blocked_transport_degrades_to_empty():
    fake = _RecordingFakeJson(None)
    connector = OpenAlexConnector(fetch_json_fn=fake)

    assert connector.search("anything", limit=5) == []
    assert connector.fetch("W123") is None


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def test_semantic_scholar_search_parses_canned_json_response():
    fake = _RecordingFakeJson(_S2_SEARCH_RESPONSE)
    connector = SemanticScholarConnector(fetch_json_fn=fake)

    records = connector.search("SDAR reinforcement learning", limit=5)

    assert len(records) == 1
    record = records[0]
    assert record.source == "semantic_scholar"
    assert record.id == "abc123def456"
    assert record.title == "Self-Distilled Agentic Reinforcement Learning"
    assert record.authors == "Jane Doe, John Smith"
    assert record.year == 2026
    assert record.venue == "arXiv preprint"
    assert record.url == "https://www.semanticscholar.org/paper/abc123def456"

    call = fake.calls[0]
    assert call["url"] == "https://api.semanticscholar.org/graph/v1/paper/search"
    assert call["params"]["query"] == "SDAR reinforcement learning"
    assert call["headers"] is None  # key-free by default


def test_semantic_scholar_uses_optional_api_key_header(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-123")
    fake = _RecordingFakeJson(_S2_PAPER)
    connector = SemanticScholarConnector(fetch_json_fn=fake)

    record = connector.fetch("abc123def456")

    assert record is not None
    assert fake.calls[0]["headers"] == {"x-api-key": "test-key-123"}


def test_semantic_scholar_blocked_transport_degrades_to_empty():
    fake = _RecordingFakeJson(None)
    connector = SemanticScholarConnector(fetch_json_fn=fake)

    assert connector.search("anything", limit=5) == []
    assert connector.fetch("abc123") is None


# ---------------------------------------------------------------------------
# default_connectors() + the REAL default transport under the hermetic suite
# ---------------------------------------------------------------------------


def test_default_connectors_returns_all_three():
    connectors = default_connectors()
    sources = {c.source for c in connectors}
    assert sources == {"arxiv", "openalex", "semantic_scholar"}


def test_default_transport_fails_soft_when_network_blocked():
    """No injection at all — exercises the REAL httpx-based default transport.

    pytest-socket blocks every non-loopback connection in this suite, so this
    is genuine (unmocked) proof that a blocked network degrades to ``None``
    rather than raising, for both fetch_text and fetch_json.
    """
    assert fetch_text("https://example.com/nonexistent-path") is None
    assert fetch_json("https://example.com/nonexistent-path") is None
