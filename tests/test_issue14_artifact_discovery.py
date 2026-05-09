"""Issue #14 — external artifact discovery adapters and event flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId
from backend.messaging.event import _clear_registry_for_tests
from backend.services.ingestion.discovery import (
    ArtifactDiscoveryAppService,
    DiscoverArtifacts,
    DiscoveredArtifactKind,
    DiscoveryState,
    RegexArtifactDiscoveryAdapter,
)
from backend.services.ingestion.intake import (
    FetchPaper,
    IntakeAppService,
    PdfPath,
    RegisterProject,
)
from backend.services.ingestion.intake.fetchers.pdf_path import PdfPathFetcher
from backend.services.ingestion.parser import ParserAppService, StartParsing
from backend.services.ingestion.parser.pymupdf_parser import PyMuPdfParser

fitz = pytest.importorskip("fitz")


def _re_register_all() -> None:
    from backend.messaging.event import register_event
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


def _parsed_project(store, tmp_path: Path) -> str:
    body = (
        "Abstract\n"
        "We evaluate code from https://github.com/openai/baselines and discuss\n"
        "known issue https://github.com/openai/baselines/issues/42.\n"
        "The dataset card is https://huggingface.co/datasets/ylecun/mnist.\n\n"
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
    return pid


def test_regex_adapter_discovers_repository_issue_and_dataset():
    text = (
        "Code: https://github.com/openai/baselines. "
        "Issue: https://github.com/openai/baselines/issues/42. "
        "Data: https://huggingface.co/datasets/ylecun/mnist."
    )
    artifacts = RegexArtifactDiscoveryAdapter().discover(project_id="prj_x", text=text)
    kinds = {artifact.kind for artifact in artifacts}
    locators = {artifact.locator for artifact in artifacts}

    assert DiscoveredArtifactKind.repository in kinds
    assert DiscoveredArtifactKind.issue in kinds
    assert DiscoveredArtifactKind.dataset in kinds
    assert "github:openai/baselines" in locators
    assert "github:openai/baselines#issue-42" in locators
    assert "huggingface:ylecun/mnist" in locators
    assert all(artifact.evidence_quote for artifact in artifacts)


def test_discovery_service_writes_rebuildable_event_stream(store, tmp_path):
    pid = _parsed_project(store, tmp_path)
    service = ArtifactDiscoveryAppService(
        store=store,
        adapters=[RegexArtifactDiscoveryAdapter()],
    )

    assert service.discover(DiscoverArtifacts(project_id=pid)) is True
    assert service.get_state(pid) is DiscoveryState.COMPLETED

    events = list(store.load(AggregateId(f"{pid}:discovery")))
    assert events[0].event_type == "discovery_started"
    assert events[-1].event_type == "discovery_completed"
    discovered = [event for event in events if event.event_type == "artifact_discovered"]
    assert len(discovered) == 3
    assert len(service.list_artifacts(pid)) == 3


def test_discovery_is_idempotent_after_completion(store, tmp_path):
    pid = _parsed_project(store, tmp_path)
    service = ArtifactDiscoveryAppService(
        store=store,
        adapters=[RegexArtifactDiscoveryAdapter()],
    )

    service.discover(DiscoverArtifacts(project_id=pid))
    service.discover(DiscoverArtifacts(project_id=pid))

    events = list(store.load(AggregateId(f"{pid}:discovery")))
    assert [event.event_type for event in events].count("discovery_started") == 1
    assert [event.event_type for event in events].count("discovery_completed") == 1
