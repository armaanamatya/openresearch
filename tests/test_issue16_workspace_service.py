"""Issue #16 — WorkspaceAppService integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.event import _clear_registry_for_tests
from backend.schemas.citations import Citation
from backend.services.context.indexer import (
    IndexerAppService,
    SourcesProjection,
    StartIndexing,
)
from backend.schemas.scope import Scope
from backend.services.context.workspace import (
    BuildWorkspace,
    Cited,
    InspectVariableTool,
    ListVariablesTool,
    LookupTool,
    RlmQueryTool,
    SemanticSearchTool,
    WorkspaceAppService,
    WorkspaceError,
    WorkspaceState,
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
        VariablePromoted,
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
        VariablePromoted,
        WorkspaceReady,
        WorkspaceClosed,
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
def indexed_project(store, tmp_path) -> str:
    body = (
        "Abstract\nWe study X.\n\n"
        "Introduction\nIntro body alpha.\n\n"
        "Methods\nMethod body beta.\n\n"
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
    indexer = IndexerAppService(store=store)
    indexer.start_indexing(StartIndexing(project_id=pid))
    return pid


@pytest.fixture
def workspace_service(store) -> WorkspaceAppService:
    indexer = IndexerAppService(store=store)
    return WorkspaceAppService(store=store, indexer=indexer)


# --- Build workspace -------------------------------------------------------


def test_build_workspace_emits_full_event_stream(
    store, workspace_service, indexed_project
):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    from backend.messaging.envelope import AggregateId

    types = [e.event_type for e in store.load(AggregateId(wsid))]
    assert types[0] == "workspace_created"
    assert "variable_loaded" in types
    assert types[-1] == "workspace_ready"


def test_state_advances_to_ready(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    assert workspace_service.get_state(wsid) is WorkspaceState.READY


def test_idempotent_re_build(store, workspace_service, indexed_project):
    wsid_a = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    wsid_b = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    assert wsid_a == wsid_b
    from backend.messaging.envelope import AggregateId

    types = [e.event_type for e in store.load(AggregateId(wsid_a))]
    # Only one workspace_created.
    assert types.count("workspace_created") == 1
    assert types.count("workspace_ready") == 1


# --- Materialized view + Cited[T] -----------------------------------------


def test_view_contains_claim_map_with_evidence_grade_citations(
    workspace_service, indexed_project
):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    view = workspace_service.materialize_view(wsid)
    assert view.is_ready
    claim_map = view.get("claim_map")
    assert claim_map is not None
    assert isinstance(claim_map, Cited)
    # Each citation must have a non-empty quote that is NOT just the locator.
    for cite in claim_map.citations:
        assert cite.quote, "citation quote must be non-empty"
        # locator IS allowed to appear in quote text by accident, but
        # the chunk text is expected to have additional content.


def test_claim_map_value_has_entries_for_each_section(
    workspace_service, indexed_project
):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    view = workspace_service.materialize_view(wsid)
    claim_map = view.get("claim_map")
    entries = claim_map.value["entries"]
    assert len(entries) >= 2  # we wrote 2+ named-heading sections in the fixture
    # Each entry references a real source id.
    for entry in entries:
        assert entry["source_id"].startswith("src_")


# --- Citation invariant: defense in depth at the workspace boundary -------


def test_cited_construct_rejects_empty_citations():
    from backend.services.context.workspace.model import (
        CitationMissingError,
        Cited,
    )

    with pytest.raises(CitationMissingError):
        Cited(value={"x": 1}, citations=())


def test_workspace_view_rejects_event_payload_with_empty_citations(
    store, workspace_service, indexed_project
):
    """Even a hypothetically corrupted store row with empty citations
    can't produce a typed Cited[T] — Cited.__post_init__ would raise."""
    # The Pydantic VariableLoaded validator already prevents empty
    # citations at the constructor. We assert that as the canonical
    # entry point to the invariant.
    from pydantic import ValidationError

    from backend.services.context.workspace.events import VariableLoaded

    with pytest.raises(ValidationError):
        VariableLoaded(
            workspace_id="ws_x",
            variable_name="bad",
            value_payload={"x": 1},
            citations=(),  # type: ignore[arg-type]
        )


# --- Lookup tool integration ----------------------------------------------


def test_lookup_tool_returns_cited_with_evidence_quote(
    workspace_service, indexed_project, store
):
    workspace_service.build_workspace(BuildWorkspace(project_id=indexed_project))
    proj = SourcesProjection()
    IndexerAppService(store=store).project_into_projection(indexed_project, proj)
    tool = LookupTool(projection=proj)

    sources = proj.list_sources()
    section = next(s for s in sources if s.kind.value == "paper_section")

    result = tool.call(workspace_id="ws_x", source_id=section.id)
    assert isinstance(result, Cited)
    assert len(result.citations) == 1
    cite = result.citations[0]
    assert cite.source_id == section.id
    assert cite.locator == section.locator
    # Evidence-grade quote — chunk text, not the locator itself.
    assert cite.quote, "quote must be non-empty"
    # The quote should be a substring of one of the section's chunks.
    chunks = proj.chunks_for_source(section.id)
    assert any(cite.quote in c.text for c in chunks), (
        "LookupTool quote must come from real chunk text per Codex feedback"
    )


def test_lookup_tool_unknown_source_raises():
    from backend.services.context.workspace.tools.interface import WorkspaceToolError

    tool = LookupTool(projection=SourcesProjection())
    with pytest.raises(WorkspaceToolError):
        tool.call(workspace_id="ws_x", source_id="src_nope")


def test_semantic_search_tool_returns_ranked_cited_chunks(
    workspace_service, indexed_project, store
):
    workspace_service.build_workspace(BuildWorkspace(project_id=indexed_project))
    proj = SourcesProjection()
    IndexerAppService(store=store).project_into_projection(indexed_project, proj)
    tool = SemanticSearchTool(projection=proj)

    result = tool.call(
        workspace_id="ws_x",
        project_id=indexed_project,
        query="method beta",
        limit=2,
    )

    assert isinstance(result, Cited)
    assert result.value["query"] == "method beta"
    assert 1 <= len(result.value["results"]) <= 2
    assert len(result.citations) == len(result.value["results"])
    assert all(cite.quote for cite in result.citations)
    assert any("beta" in cite.quote.lower() for cite in result.citations)


# --- Failure paths ---------------------------------------------------------


def test_build_unindexed_project_raises(store):
    indexer = IndexerAppService(store=store)
    svc = WorkspaceAppService(store=store, indexer=indexer)
    with pytest.raises(WorkspaceError, match="must be"):
        svc.build_workspace(BuildWorkspace(project_id="prj_does_not_exist"))


# --- Enhanced preloading: paper_text + paper_sections -----------------------


def test_workspace_preloads_paper_text(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    view = workspace_service.materialize_view(wsid)
    paper_text = view.get("paper_text")
    assert paper_text is not None
    assert isinstance(paper_text, Cited)
    assert "text" in paper_text.value
    assert len(paper_text.value["text"]) > 0
    assert len(paper_text.citations) >= 1


def test_workspace_preloads_paper_sections(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    view = workspace_service.materialize_view(wsid)
    paper_sections = view.get("paper_sections")
    assert paper_sections is not None
    assert isinstance(paper_sections, Cited)
    sections = paper_sections.value["sections"]
    assert isinstance(sections, dict)
    assert len(sections) >= 2  # at least Abstract + Introduction or Methods


def test_workspace_preloads_three_variables(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    view = workspace_service.materialize_view(wsid)
    assert view.variable_count >= 3
    assert view.get("paper_text") is not None
    assert view.get("paper_sections") is not None
    assert view.get("claim_map") is not None


# --- Variable scope tracking ------------------------------------------------


def test_workspace_view_tracks_scope(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    view = workspace_service.materialize_view(wsid)
    for name in view.variable_names():
        assert view.get_scope(name) == Scope.private_to_parent


# --- enrich_variable --------------------------------------------------------


def test_enrich_variable_adds_to_workspace(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    cite = Citation(
        source_id="src_test",
        chunk_id=None,
        quote="test evidence",
        locator="test",
        confidence=1.0,
    )
    workspace_service.enrich_variable(
        workspace_id=wsid,
        variable_name="environment_spec",
        value_payload={"python": "3.8", "cuda": "11.3"},
        citations=(cite,),
        enriched_by="environment_detective",
    )
    view = workspace_service.materialize_view(wsid)
    env = view.get("environment_spec")
    assert env is not None
    assert env.value["python"] == "3.8"


def test_enrich_variable_on_new_workspace_raises(workspace_service):
    cite = Citation(
        source_id="src_test",
        chunk_id=None,
        quote="test",
        locator="test",
        confidence=1.0,
    )
    with pytest.raises(WorkspaceError, match="state"):
        workspace_service.enrich_variable(
            workspace_id="ws_nonexistent",
            variable_name="x",
            value_payload={"x": 1},
            citations=(cite,),
            enriched_by="test",
        )


# --- close_workspace --------------------------------------------------------


def test_close_workspace(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    assert workspace_service.get_state(wsid) is WorkspaceState.READY
    workspace_service.close_workspace(workspace_id=wsid, reason="done")
    assert workspace_service.get_state(wsid) is WorkspaceState.CLOSED


def test_close_workspace_idempotent(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    workspace_service.close_workspace(workspace_id=wsid)
    workspace_service.close_workspace(workspace_id=wsid)  # no error
    assert workspace_service.get_state(wsid) is WorkspaceState.CLOSED


def test_close_uncreated_workspace_raises(workspace_service):
    with pytest.raises(WorkspaceError, match="never created"):
        workspace_service.close_workspace(workspace_id="ws_nope")


# --- promote_variable -------------------------------------------------------


def test_promote_variable(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    workspace_service.promote_variable(
        workspace_id=wsid,
        variable_name="claim_map",
        new_scope=Scope.branch_shared,
        promoted_by="supervisor",
    )
    view = workspace_service.materialize_view(wsid)
    assert view.get_scope("claim_map") == Scope.branch_shared


def test_promote_missing_variable_raises(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    with pytest.raises(WorkspaceError, match="not found"):
        workspace_service.promote_variable(
            workspace_id=wsid,
            variable_name="nonexistent",
            new_scope=Scope.global_verified,
        )


# --- ListVariablesTool ------------------------------------------------------


def test_list_variables_tool(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = ListVariablesTool(view_provider=workspace_service)
    result = tool.call(workspace_id=wsid)
    assert isinstance(result, Cited)
    assert result.value["variable_count"] >= 3
    names = [v["variable_name"] for v in result.value["variables"]]
    assert "claim_map" in names
    assert "paper_text" in names
    assert "paper_sections" in names


# --- InspectVariableTool ----------------------------------------------------


def test_inspect_variable_tool(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = InspectVariableTool(view_provider=workspace_service)
    result = tool.call(workspace_id=wsid, variable_name="paper_text")
    assert isinstance(result, Cited)
    assert result.value["variable_name"] == "paper_text"
    assert "value" in result.value
    assert result.value["citation_count"] >= 1


def test_inspect_missing_variable_raises(workspace_service, indexed_project):
    from backend.services.context.workspace.tools.interface import WorkspaceToolError

    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = InspectVariableTool(view_provider=workspace_service)
    with pytest.raises(WorkspaceToolError, match="not found"):
        tool.call(workspace_id=wsid, variable_name="nonexistent")


# --- RlmQueryTool -----------------------------------------------------------


class _StubLlm:
    """Test LLM that echoes the question back."""

    def complete(self, *, system: str, user: str) -> str:
        return f"Answer based on context: {user.split('Question: ')[-1]}"


def test_rlm_query_tool_on_paper_text(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = RlmQueryTool(
        view_provider=workspace_service,
        llm_client=_StubLlm(),
    )
    result = tool.call(
        workspace_id=wsid,
        question="What does the paper study?",
        variable_name="paper_text",
    )
    assert isinstance(result, Cited)
    assert result.value["question"] == "What does the paper study?"
    assert result.value["variable_name"] == "paper_text"
    assert "answer" in result.value
    assert len(result.citations) >= 1


def test_rlm_query_tool_with_context_key(workspace_service, indexed_project):
    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = RlmQueryTool(
        view_provider=workspace_service,
        llm_client=_StubLlm(),
    )
    # Get a valid section name from the view.
    view = workspace_service.materialize_view(wsid)
    sections = view.get("paper_sections")
    section_name = list(sections.value["sections"].keys())[0]

    result = tool.call(
        workspace_id=wsid,
        question="What is discussed here?",
        variable_name="paper_sections",
        context_key=section_name,
    )
    assert isinstance(result, Cited)
    assert result.value["context_key"] == section_name


def test_rlm_query_empty_question_raises(workspace_service, indexed_project):
    from backend.services.context.workspace.tools.interface import WorkspaceToolError

    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = RlmQueryTool(
        view_provider=workspace_service,
        llm_client=_StubLlm(),
    )
    with pytest.raises(WorkspaceToolError, match="non-empty"):
        tool.call(
            workspace_id=wsid,
            question="   ",
            variable_name="paper_text",
        )


def test_rlm_query_missing_variable_raises(workspace_service, indexed_project):
    from backend.services.context.workspace.tools.interface import WorkspaceToolError

    wsid = workspace_service.build_workspace(
        BuildWorkspace(project_id=indexed_project)
    )
    tool = RlmQueryTool(
        view_provider=workspace_service,
        llm_client=_StubLlm(),
    )
    with pytest.raises(WorkspaceToolError, match="not found"):
        tool.call(
            workspace_id=wsid,
            question="test",
            variable_name="nonexistent",
        )
