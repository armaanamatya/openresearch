"""End-to-end smoke test for the vertical slice (Commit 5).

  PDF -> Project -> ParsedPaper -> SourceRefs/Chunks -> Workspace

Asserts:
  - The four services run in sequence to a successful WorkspaceReady.
  - The materialized workspace contains `claim_map` with non-empty
    evidence-grade citations resolving to real SourceIds.
  - Re-running on the same PDF produces the same project_id, the same
    SourceIds, and the same ChunkIds (idempotent re-ingest at the ID
    level — byte-identical replay is deferred per slice scope).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId
from backend.messaging.event import _clear_registry_for_tests
from backend.services.context.indexer import (
    IndexerAppService,
    SourcesProjection,
    StartIndexing,
)
from backend.services.context.workspace import (
    BuildWorkspace,
    WorkspaceAppService,
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
    from backend.services.context.indexer.events import (
        ChunkCreated,
        IndexingCompleted,
        IndexingFailed,
        IndexingStarted,
        SourceRegistered,
    )
    from backend.services.context.workspace.events import (
        CitationAttached,
        ToolInvoked,
        VariableEnriched,
        VariableLoaded,
        WorkspaceClosed,
        WorkspaceCreated,
        WorkspaceReady,
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
        WorkspaceCreated,
        VariableLoaded,
        VariableEnriched,
        CitationAttached,
        ToolInvoked,
        WorkspaceReady,
        WorkspaceClosed,
    ):
        register_event(cls)


def _make_pdf(tmp_path: Path, body: str, name: str = "ppo.pdf") -> Path:
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


PPO_BODY = (
    "Abstract\n"
    "We propose Proximal Policy Optimization, a family of policy\n"
    "gradient methods that alternate between sampling data and\n"
    "performing several epochs of optimization on the sampled data.\n\n"
    "Introduction\n"
    "Recent advances in deep reinforcement learning have demonstrated\n"
    "remarkable success in domains ranging from games to robotics.\n\n"
    "Methods\n"
    "We use a clipped surrogate objective to constrain policy updates.\n"
    "The clipping parameter epsilon controls how much the policy can\n"
    "change in each update.\n\n"
    "Experiments\n"
    "We evaluate PPO on continuous-control benchmarks including\n"
    "CartPole-v1 and MuJoCo locomotion tasks.\n\n"
    "Results\n"
    "PPO matches or exceeds the performance of TRPO with much simpler\n"
    "implementation and improved sample efficiency.\n\n"
    "References\n\n"
    "[1] Schulman et al. PPO arXiv:1707.06347\n\n"
    "[2] Schulman et al. TRPO arXiv:1502.05477\n"
)


def _services(tmp_path: Path):
    store = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    runs = tmp_path / "runs"
    intake = IntakeAppService(
        store=store,
        fetchers={"pdf_path": PdfPathFetcher(runs_root=runs)},
    )
    parser = ParserAppService(store=store, parser=PyMuPdfParser(), runs_root=runs)
    indexer = IndexerAppService(store=store)
    workspace = WorkspaceAppService(store=store, indexer=indexer)
    return store, intake, parser, indexer, workspace


def _run_pipeline(tmp_path: Path, pdf: Path) -> tuple[str, str]:
    store, intake, parser, indexer, workspace = _services(tmp_path)
    project_id = intake.register_project(
        RegisterProject(source=PdfPath(path=str(pdf)))
    )
    intake.fetch_paper(FetchPaper(project_id=project_id))
    parser.start_parsing(StartParsing(project_id=project_id))
    indexer.start_indexing(StartIndexing(project_id=project_id))
    workspace_id = workspace.build_workspace(BuildWorkspace(project_id=project_id))
    store.close()
    return project_id, workspace_id


# --- Setup -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_and_register():
    _clear_registry_for_tests()
    _re_register_all()
    yield
    _clear_registry_for_tests()


# --- Smoke -----------------------------------------------------------------


def test_e2e_pdf_to_ready_workspace(tmp_path: Path):
    pdf = _make_pdf(tmp_path, PPO_BODY)
    project_id, workspace_id = _run_pipeline(tmp_path, pdf)

    # Re-open the store to materialize the view.
    store = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    indexer = IndexerAppService(store=store)
    workspace = WorkspaceAppService(store=store, indexer=indexer)
    view = workspace.materialize_view(workspace_id)
    assert view.is_ready, "WorkspaceReady event must be present"
    claim_map = view.get("claim_map")
    assert claim_map is not None, "claim_map should be preloaded"

    # Citations are non-empty AND evidence-grade.
    assert len(claim_map.citations) >= 2
    proj = SourcesProjection()
    indexer.project_into_projection(project_id, proj)
    for cite in claim_map.citations:
        assert cite.source_id.startswith("src_")
        assert proj.get_source(cite.source_id) is not None, (
            "every citation must resolve to a real SourceRef"
        )
        assert cite.quote, "every citation must have a non-empty quote"
    store.close()


def test_e2e_idempotent_reingest_yields_identical_ids(tmp_path: Path):
    """Codex feedback resolution: re-ingesting the same PDF must
    produce the same project_id and the same set of SourceIds and
    ChunkIds. The slice does NOT promise byte-identical events
    (parse_duration_ms, occurred_at vary) — only ID stability."""
    pdf = _make_pdf(tmp_path, PPO_BODY)

    # Run 1.
    project_id_a, _ = _run_pipeline(tmp_path, pdf)

    # Capture run-1 IDs.
    store = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    proj_a = SourcesProjection()
    IndexerAppService(store=store).project_into_projection(project_id_a, proj_a)
    source_ids_a = sorted(s.id for s in proj_a.list_sources())
    # Pull chunk IDs from the index events directly.
    chunk_ids_a = sorted(
        e.payload["chunk"]["id"]
        for e in store.load(AggregateId(f"{project_id_a}:index"))
        if e.event_type == "chunk_created"
    )
    store.close()

    # Run 2 — fresh store directory.
    tmp_path2 = tmp_path / "second"
    tmp_path2.mkdir()
    project_id_b, _ = _run_pipeline(tmp_path2, pdf)

    # project_id is keyed on absolute path, so different cwd produces
    # different project_id even on the same PDF — confirm that's
    # consistent within the same tmp tree by re-running.
    # For ID equality, run on the SAME file in the SAME store from
    # scratch (drop the events.db and re-run inside tmp_path2).
    store2 = SqliteEventStore(f"sqlite:///{tmp_path2}/events.db")
    proj_b = SourcesProjection()
    IndexerAppService(store=store2).project_into_projection(project_id_b, proj_b)
    source_ids_b = sorted(s.id for s in proj_b.list_sources())
    chunk_ids_b = sorted(
        e.payload["chunk"]["id"]
        for e in store2.load(AggregateId(f"{project_id_b}:index"))
        if e.event_type == "chunk_created"
    )
    store2.close()

    # When the absolute PDF path is the same (it is — same `pdf`
    # variable), project_id and downstream IDs match.
    assert project_id_a == project_id_b
    assert source_ids_a == source_ids_b
    assert chunk_ids_a == chunk_ids_b


def test_e2e_cli_main_produces_summary(tmp_path: Path):
    """Smoke-test the CLI wrapper itself end-to-end."""
    from backend.cli import main

    pdf = _make_pdf(tmp_path, PPO_BODY)
    db_url = f"sqlite:///{tmp_path}/events.db"
    runs_root = str(tmp_path / "runs")

    code = main(
        [
            "--database-url", db_url,
            "--runs-root", runs_root,
            "ingest",
            str(pdf),
        ]
    )
    assert code == 0
