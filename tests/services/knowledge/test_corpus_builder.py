"""Tests for backend.services.knowledge.corpus.builder — hermetic build flow.

Every network seam (S2 connector, search connector, PDF download, parser) is
injected as a fake, per the connectors' injectable-transport contract; the
suite is socket-hermetic. Flags come in via monkeypatch (conftest scrubs the
ambient OPENRESEARCH_* namespace).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from backend.services.knowledge.connectors.base import LiteratureRecord
from backend.services.knowledge.corpus.builder import (
    MAX_PAPERS,
    TargetPaper,
    build_corpus,
)
from backend.services.knowledge.corpus.store import CorpusStore, corpus_root

_TARGET = TargetPaper(
    project_id="prj_test",
    arxiv_id="2605.15155",
    title="Self-Distilled Agentic Reinforcement Learning",
    references=(
        {"arxiv_id": "2101.00001", "doi": None, "title": "Prior Work One", "raw_text": "[1] ..."},
        {"arxiv_id": None, "doi": "10.1000/xyz", "title": "Prior Work Two", "raw_text": "[2] ..."},
        {"arxiv_id": None, "doi": None, "title": "No id — dropped", "raw_text": "[3] ..."},
    ),
)


class _FakeS2:
    """Connector-shaped fake whose fetch() returns a record with neighbours."""

    def __init__(self, record: LiteratureRecord | None = None, raises: bool = False) -> None:
        self.record = record
        self.raises = raises
        self.fetch_calls: list[str] = []

    def fetch(self, record_id: str):
        self.fetch_calls.append(record_id)
        if self.raises:
            raise RuntimeError("s2 down")
        return self.record

    def search(self, query: str, *, limit: int = 10):  # pragma: no cover — unused
        return []


class _FakeSearch:
    def __init__(self, hits=()) -> None:
        self.hits = list(hits)
        self.search_calls: list[str] = []

    def search(self, query: str, *, limit: int = 10):
        self.search_calls.append(query)
        return self.hits[:limit]

    def fetch(self, record_id: str):  # pragma: no cover — unused
        return None


def _fake_s2_record() -> LiteratureRecord:
    return LiteratureRecord(
        source="semantic_scholar",
        id="s2target",
        title=_TARGET.title,
        references=(
            {"id": "s2ref1", "title": "Graph Reference", "arxiv_id": "2102.00002", "doi": None},
        ),
        citations=(
            {"id": "s2cit1", "title": "A Citing Paper", "arxiv_id": "2610.00003", "doi": None},
        ),
    )


def _fake_download(*, url: str, runs_root: Path, project_id: str, fetched_via: str, **_kw):
    dst = Path(runs_root) / project_id
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "raw_paper.pdf").write_bytes(b"%PDF-1.4 fake")
    return SimpleNamespace(sha256="fake")


class _FakeParser:
    def parse(self, *, project_id: str, paper_path: Path):
        return SimpleNamespace(full_text="parsed corpus paper text " * 10)


def test_flag_off_returns_none_and_touches_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_CORPUS", raising=False)
    report = build_corpus(runs_root=tmp_path, target=_TARGET, s2_connector=_FakeS2())
    assert report is None
    assert not corpus_root(tmp_path).exists()
    assert not (tmp_path / "prj_test").exists()


def test_flag_accepts_on_like_sibling_rlm_flags(tmp_path: Path, monkeypatch):
    """'on' is truthy for the whole literature flag family — a mixed vocabulary
    ('1' here, 'on' there) would half-enable the feature."""
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "on")
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_GROUNDING", raising=False)
    report = build_corpus(runs_root=tmp_path, target=_TARGET)
    assert report is not None
    assert report.papers_persisted == 2


def test_grounding_off_builds_local_only(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_GROUNDING", raising=False)
    s2 = _FakeS2(_fake_s2_record())
    search = _FakeSearch()

    report = build_corpus(
        runs_root=tmp_path, target=_TARGET, s2_connector=s2, search_connector=search
    )

    assert report is not None
    assert report.network_enabled is False
    assert s2.fetch_calls == []          # no network seam touched
    assert search.search_calls == []
    assert report.fetched_fulltext == 0
    # The two id-bearing parsed references still land (metadata-only).
    assert report.papers_persisted == 2
    spec = json.loads(Path(report.spec_path).read_text(encoding="utf-8"))
    ids = {p["id"] for p in spec["papers"]}
    assert ids == {"arxiv:2101.00001", "doi:10.1000/xyz"}


def test_full_build_ranks_relations_and_fetches(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    s2 = _FakeS2(_fake_s2_record())
    search_hit = LiteratureRecord(
        source="arxiv", id="2611.00004v1", title="A Search Hit About Agentic RL"
    )
    search = _FakeSearch([search_hit])

    report = build_corpus(
        runs_root=tmp_path,
        target=_TARGET,
        s2_connector=s2,
        search_connector=search,
        download_pdf_fn=_fake_download,
        parser=_FakeParser(),
    )

    assert report is not None
    assert s2.fetch_calls == ["ARXIV:2605.15155"]
    spec = json.loads(Path(report.spec_path).read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in spec["papers"]}
    # Adjacency ordering: direct refs above s2 refs above citers above search.
    assert by_id["arxiv:2101.00001"]["relation"] == "direct_ref"
    assert by_id["arxiv:2102.00002"]["relation"] == "s2_ref"
    assert by_id["arxiv:2610.00003"]["relation"] == "citer"
    assert by_id["arxiv:2611.00004"]["relation"] == "search"
    scores = [p["score"] for p in spec["papers"]]
    assert scores == sorted(scores, reverse=True)

    store = CorpusStore(corpus_root(tmp_path))
    store.initialize()
    # Edges: target cites its refs; the citer cites the target.
    conn = store.connection
    edges = {
        (r["src_id"], r["dst_id"])
        for r in conn.execute("SELECT src_id, dst_id FROM lit_relations").fetchall()
    }
    assert ("arxiv:2605.15155", "arxiv:2101.00001") in edges
    assert ("arxiv:2610.00003", "arxiv:2605.15155") in edges

    # Full text fetched for arXiv-hosted top candidates.
    assert report.fetched_fulltext >= 1
    row = store.get_paper("arxiv:2101.00001")
    assert row["fetched_level"] == "fulltext"
    parsed = store.paper_dir("arxiv:2101.00001") / "parsed_full_text.txt"
    assert parsed.exists()
    store.close()


def test_connector_failure_is_recorded_and_build_continues(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")

    report = build_corpus(
        runs_root=tmp_path,
        target=_TARGET,
        s2_connector=_FakeS2(raises=True),
        search_connector=_FakeSearch(),
        download_pdf_fn=_fake_download,
        parser=_FakeParser(),
    )

    assert report is not None
    assert any("s2 fetch failed" in e for e in report.errors)
    assert report.papers_persisted == 2  # parsed refs still landed


def test_dedupe_keeps_strongest_relation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    # S2 also returns the paper that is already a direct parsed reference.
    record = LiteratureRecord(
        source="semantic_scholar",
        id="s2target",
        title=_TARGET.title,
        references=(
            {"id": "dup", "title": "Prior Work One", "arxiv_id": "2101.00001", "doi": None},
        ),
    )
    report = build_corpus(
        runs_root=tmp_path,
        target=_TARGET,
        s2_connector=_FakeS2(record),
        search_connector=_FakeSearch(),
        download_pdf_fn=_fake_download,
        parser=_FakeParser(),
    )
    spec = json.loads(Path(report.spec_path).read_text(encoding="utf-8"))
    rows = [p for p in spec["papers"] if p["id"] == "arxiv:2101.00001"]
    assert len(rows) == 1
    assert rows[0]["relation"] == "direct_ref"  # the stronger relation wins


def test_fulltext_fetch_follows_final_score(tmp_path: Path, monkeypatch):
    """The FETCH_K budget is spent by final_score (adjacency + graph boost),
    not raw adjacency — a graph-boosted ref must win the single fetch slot."""
    from backend.services.knowledge.corpus import builder as builder_mod

    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_GROUNDING", "1")
    monkeypatch.setattr(builder_mod, "FETCH_K", 1)

    target = TargetPaper(
        project_id="prj_order",
        arxiv_id="2605.15155",
        title="Self-Distilled Agentic Reinforcement Learning",
        references=(
            {"arxiv_id": "2101.00001", "doi": None, "title": "Alpha", "raw_text": ""},
            {"arxiv_id": "2102.00002", "doi": None, "title": "Beta", "raw_text": ""},
        ),
    )
    # Pre-seed a co-citation edge: 2610.00003 cites BOTH the target (as the S2
    # citer below) and 2102.00002, so the latter earns a graph boost its
    # sibling direct_ref (identical adjacency score, smaller id) does not.
    store = CorpusStore(corpus_root(tmp_path))
    store.initialize()
    store.add_relation("arxiv:2610.00003", "arxiv:2102.00002")
    store.close()

    record = LiteratureRecord(
        source="semantic_scholar",
        id="s2target",
        title=target.title,
        citations=(
            {"id": "c1", "title": "A Citing Paper", "arxiv_id": "2610.00003", "doi": None},
        ),
    )
    report = build_corpus(
        runs_root=tmp_path,
        target=target,
        s2_connector=_FakeS2(record),
        search_connector=_FakeSearch(),
        download_pdf_fn=_fake_download,
        parser=_FakeParser(),
    )
    assert report is not None and report.fetched_fulltext == 1

    store = CorpusStore(corpus_root(tmp_path))
    store.initialize()
    # Adjacency-score-plus-id ordering alone would fetch 2101.00001.
    assert store.get_paper("arxiv:2102.00002")["fetched_level"] == "fulltext"
    assert store.get_paper("arxiv:2101.00001")["fetched_level"] == "metadata"
    store.close()


def test_cap_at_max_papers(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_LITERATURE_CORPUS", "1")
    monkeypatch.delenv("OPENRESEARCH_LITERATURE_GROUNDING", raising=False)
    many_refs = tuple(
        {"arxiv_id": f"21{i:02d}.{i:05d}", "doi": None, "title": f"Ref {i}", "raw_text": ""}
        for i in range(MAX_PAPERS + 20)
    )
    target = TargetPaper(project_id="prj_cap", arxiv_id="2605.15155", references=many_refs)
    report = build_corpus(runs_root=tmp_path, target=target)
    assert report.papers_ranked <= MAX_PAPERS
