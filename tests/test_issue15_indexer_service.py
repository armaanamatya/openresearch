"""Issue #15 — IndexerAppService integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId
from backend.messaging.event import _clear_registry_for_tests
from backend.services.context.indexer import (
    IndexerAppService,
    IndexerError,
    IndexState,
    SourcesProjection,
    StartIndexing,
)
from backend.services.ingestion.discovery import (
    ArtifactDiscoveryAppService,
    DiscoverArtifacts,
    RegexArtifactDiscoveryAdapter,
)
from backend.services.context.indexer.model import SourceKind
from backend.services.ingestion.intake import (
    FetchPaper,
    IntakeAppService,
    PdfPath,
    RegisterProject,
)
from backend.services.ingestion.intake.fetchers.pdf_path import PdfPathFetcher
from backend.services.ingestion.parser import (
    ParserAppService,
    StartParsing,
)
from backend.services.ingestion.parser.pymupdf_parser import PyMuPdfParser

fitz = pytest.importorskip("fitz")


def _re_register_all() -> None:
    from backend.messaging.event import register_event
    from backend.services.context.indexer.events import (
        ChunkCreated,
        IndexingCompleted,
        IndexingFailed,
        IndexingStarted,
        SourceRegistered,
    )
    from backend.services.ingestion.discovery.events import (
        ArtifactDiscovered,
        DiscoveryCompleted,
        DiscoveryFailed,
        DiscoveryStarted,
    )
    from backend.services.ingestion.intake.events import (
        PaperFetchFailed,
        PaperFetched,
        ProjectCreated,
    )
    from backend.services.ingestion.parser.events import (
        FigureExtracted,
        ParsingCompleted,
        ParsingFailed,
        ParsingStarted,
        ReferenceExtracted,
        SectionExtracted,
    )

    for cls in (
        ProjectCreated,
        PaperFetched,
        PaperFetchFailed,
        ParsingStarted,
        SectionExtracted,
        ReferenceExtracted,
        FigureExtracted,
        ParsingCompleted,
        ParsingFailed,
        IndexingStarted,
        SourceRegistered,
        ChunkCreated,
        IndexingCompleted,
        IndexingFailed,
        DiscoveryStarted,
        ArtifactDiscovered,
        DiscoveryCompleted,
        DiscoveryFailed,
    ):
        register_event(cls)


def _make_pdf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def store(tmp_path: Path):
    _clear_registry_for_tests()
    _re_register_all()
    s = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    yield s
    s.close()
    _clear_registry_for_tests()


@pytest.fixture
def parsed_project(store, tmp_path) -> str:
    """Set up a project that has reached PARSED state."""
    body = (
        "Abstract\nThe abstract.\n\n"
        "Introduction\nIntro body.\n\n"
        "Methods\nMethod body with detail.\n\n"
        "Results\nResults body.\n\n"
        "References\n\n[1] arXiv:1707.06347\n"
    )
    pdf = _make_pdf(tmp_path, body)
    runs = tmp_path / "runs"
    intake = IntakeAppService(
        store=store,
        fetchers={"pdf_path": PdfPathFetcher(runs_root=runs)},
    )
    pid = intake.register_project(RegisterProject(source=PdfPath(path=str(pdf))))
    intake.fetch_paper(FetchPaper(project_id=pid))
    parser = ParserAppService(store=store, parser=PyMuPdfParser(), runs_root=runs)
    parser.start_parsing(StartParsing(project_id=pid))
    return pid


@pytest.fixture
def indexer(store) -> IndexerAppService:
    return IndexerAppService(store=store)


# --- Happy path ------------------------------------------------------------


def test_start_indexing_writes_full_event_stream(store, indexer, parsed_project):
    success = indexer.start_indexing(StartIndexing(project_id=parsed_project))
    assert success is True

    index_id = AggregateId(f"{parsed_project}:index")
    types = [e.event_type for e in store.load(index_id)]
    assert types[0] == "indexing_started"
    assert types[-1] == "indexing_completed"
    assert "source_registered" in types
    assert "chunk_created" in types


def test_state_advances_to_indexed(indexer, parsed_project):
    indexer.start_indexing(StartIndexing(project_id=parsed_project))
    assert indexer.get_state(parsed_project) is IndexState.INDEXED


def test_idempotent_re_run(store, indexer, parsed_project):
    indexer.start_indexing(StartIndexing(project_id=parsed_project))
    indexer.start_indexing(StartIndexing(project_id=parsed_project))  # no-op
    index_id = AggregateId(f"{parsed_project}:index")
    types = [e.event_type for e in store.load(index_id)]
    assert types.count("indexing_started") == 1
    assert types.count("indexing_completed") == 1


# --- Determinism (Codex feedback baked in) ---------------------------------


def test_chunk_ids_stable_in_stored_events(store, indexer, parsed_project):
    indexer.start_indexing(StartIndexing(project_id=parsed_project))
    index_id = AggregateId(f"{parsed_project}:index")
    chunk_ids = [
        e.payload["chunk"]["id"]
        for e in store.load(index_id)
        if e.event_type == "chunk_created"
    ]
    # Stable order on re-load.
    chunk_ids_again = [
        e.payload["chunk"]["id"]
        for e in store.load(index_id)
        if e.event_type == "chunk_created"
    ]
    assert chunk_ids == chunk_ids_again
    assert len(set(chunk_ids)) == len(chunk_ids)  # all unique


# --- Projection ------------------------------------------------------------


def test_projection_is_rebuildable_from_event_log(store, indexer, parsed_project):
    indexer.start_indexing(StartIndexing(project_id=parsed_project))
    proj1 = SourcesProjection()
    indexer.project_into_projection(parsed_project, proj1)
    # Build a fresh projection separately.
    proj2 = SourcesProjection()
    indexer.project_into_projection(parsed_project, proj2)
    assert proj1.source_count == proj2.source_count
    assert proj1.chunk_count == proj2.chunk_count
    # Both projections expose the same SourceRefs.
    ids1 = sorted(s.id for s in proj1.list_sources())
    ids2 = sorted(s.id for s in proj2.list_sources())
    assert ids1 == ids2


def test_projection_separates_section_sources_from_reference_sources(
    indexer, parsed_project
):
    indexer.start_indexing(StartIndexing(project_id=parsed_project))
    proj = SourcesProjection()
    indexer.project_into_projection(parsed_project, proj)
    section_count = sum(
        1 for s in proj.list_sources() if s.kind is SourceKind.paper_section
    )
    reference_count = sum(
        1 for s in proj.list_sources() if s.kind is SourceKind.paper_reference
    )
    assert section_count >= 1
    assert reference_count >= 1


def test_chunks_for_source_returns_chunks_only_for_that_source(
    indexer, parsed_project
):
    indexer.start_indexing(StartIndexing(project_id=parsed_project))
    proj = SourcesProjection()
    indexer.project_into_projection(parsed_project, proj)
    section_sources = [
        s for s in proj.list_sources() if s.kind is SourceKind.paper_section
    ]
    src = section_sources[0]
    chunks = proj.chunks_for_source(src.id)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.source_id == src.id


def test_indexer_registers_discovered_external_artifacts(store, tmp_path):
    body = (
        "Abstract\n"
        "Code lives at https://github.com/openai/baselines and data at "
        "https://huggingface.co/datasets/ylecun/mnist.\n\n"
        "References\n\n[1] arXiv:1707.06347\n"
    )
    pdf = _make_pdf(tmp_path, body)
    runs = tmp_path / "runs"
    intake = IntakeAppService(
        store=store,
        fetchers={"pdf_path": PdfPathFetcher(runs_root=runs)},
    )
    pid = intake.register_project(RegisterProject(source=PdfPath(path=str(pdf))))
    intake.fetch_paper(FetchPaper(project_id=pid))
    ParserAppService(store=store, parser=PyMuPdfParser(), runs_root=runs).start_parsing(
        StartParsing(project_id=pid)
    )
    ArtifactDiscoveryAppService(
        store=store,
        adapters=[RegexArtifactDiscoveryAdapter()],
    ).discover(DiscoverArtifacts(project_id=pid))

    indexer = IndexerAppService(store=store)
    indexer.start_indexing(StartIndexing(project_id=pid))
    projection = SourcesProjection()
    indexer.project_into_projection(pid, projection)

    by_kind = {source.kind for source in projection.list_sources()}
    assert SourceKind.repository in by_kind
    assert SourceKind.dataset in by_kind
    external = [
        source
        for source in projection.list_sources()
        if source.kind in {SourceKind.repository, SourceKind.dataset}
    ]
    assert external
    assert all(projection.chunks_for_source(source.id) for source in external)


# --- Failure paths ---------------------------------------------------------


def test_indexing_unparsed_project_raises(store, indexer, tmp_path):
    """A project that hasn't reached PARSED can't be indexed."""
    body = "stuff"
    pdf = _make_pdf(tmp_path, body)
    runs = tmp_path / "runs"
    intake = IntakeAppService(
        store=store,
        fetchers={"pdf_path": PdfPathFetcher(runs_root=runs)},
    )
    pid = intake.register_project(RegisterProject(source=PdfPath(path=str(pdf))))
    # No fetch / parse.
    with pytest.raises(IndexerError, match="must be"):
        indexer.start_indexing(StartIndexing(project_id=pid))
