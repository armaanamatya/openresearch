"""Layer 2 — Semantic Search with vector embeddings + BM25 fallback.

Tests both the ChromaEmbeddingStore (vector search) and the BM25 fallback
path in SemanticSearchTool. Verifies:
  - ChromaEmbeddingStore adds/queries chunks correctly
  - Vector search returns semantically similar results (not just keyword match)
  - BM25 fallback works when no embedding store is provided
  - SemanticSearchTool reports which backend was used
  - Citation invariant holds for both paths
  - Edge cases: empty queries, empty stores, limit bounds
  - Integration: full pipeline from indexer → embedding store → search tool
"""

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
from backend.services.context.indexer.model import Chunk, ChunkType
from backend.services.context.workspace import (
    BuildWorkspace,
    Cited,
    SemanticSearchTool,
    WorkspaceAppService,
)
from backend.services.context.workspace.tools.interface import WorkspaceToolError
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
chromadb = pytest.importorskip("chromadb")

from backend.services.context.semantic.store import (
    ChromaEmbeddingStore,
    SearchResult,
    try_create_chroma_store,
)


# --- Helpers ----------------------------------------------------------------


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


def _make_chunks() -> list[Chunk]:
    """Create test chunks with known semantic content."""
    return [
        Chunk(
            id="chk_001",
            source_id="src_abstract",
            project_id="prj_test",
            text="We propose a novel optimization method using Adam optimizer with learning rate warmup for training deep neural networks.",
            span=(0, 100),
            chunk_type=ChunkType.section,
        ),
        Chunk(
            id="chk_002",
            source_id="src_methods",
            project_id="prj_test",
            text="The training procedure uses stochastic gradient descent with momentum. Loss divergence was observed at higher learning rates.",
            span=(0, 100),
            chunk_type=ChunkType.section,
        ),
        Chunk(
            id="chk_003",
            source_id="src_results",
            project_id="prj_test",
            text="Our experiments on CartPole-v1 show that PPO achieves mean reward of 475 after 500k timesteps of training.",
            span=(0, 100),
            chunk_type=ChunkType.section,
        ),
        Chunk(
            id="chk_004",
            source_id="src_related",
            project_id="prj_test",
            text="Related work includes TRPO, A2C, and SAC algorithms for reinforcement learning in continuous action spaces.",
            span=(0, 100),
            chunk_type=ChunkType.section,
        ),
        Chunk(
            id="chk_005",
            source_id="src_env",
            project_id="prj_test",
            text="Environment setup requires Python 3.8, PyTorch 1.12 with CUDA 11.3, and gymnasium 0.26.",
            span=(0, 100),
            chunk_type=ChunkType.section,
        ),
    ]


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path):
    _clear_registry_for_tests()
    _re_register_all()
    s = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    yield s
    s.close()
    _clear_registry_for_tests()


@pytest.fixture
def chroma_store(request) -> ChromaEmbeddingStore:
    """In-memory Chroma store with a unique collection per test."""
    # Use the test node id to create a unique collection name.
    name = request.node.name.replace("[", "_").replace("]", "_")[:50]
    return ChromaEmbeddingStore(
        collection_name=f"test_{name}",
        persist_directory=None,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return _make_chunks()


@pytest.fixture
def projection_with_chunks(chunks: list[Chunk]) -> SourcesProjection:
    """A SourcesProjection with test chunks and matching sources."""
    from backend.services.context.indexer.model import SourceKind, SourceRef

    proj = SourcesProjection()
    sources = {
        "src_abstract": ("Abstract", SourceKind.paper_section),
        "src_methods": ("Methods", SourceKind.paper_section),
        "src_results": ("Results", SourceKind.paper_section),
        "src_related": ("Related Work", SourceKind.paper_section),
        "src_env": ("Environment", SourceKind.paper_section),
    }
    for sid, (locator, kind) in sources.items():
        proj.apply_source(SourceRef(
            id=sid,
            project_id="prj_test",
            kind=kind,
            locator=locator,
            upstream_id=sid,
        ))
    for chunk in chunks:
        proj.apply_chunk(chunk)
    return proj


@pytest.fixture
def indexed_project(store, tmp_path) -> str:
    body = (
        "Abstract\nWe study reinforcement learning.\n\n"
        "Introduction\nIntro body about policy optimization.\n\n"
        "Methods\nTraining uses gradient descent with momentum.\n\n"
        "Results\nPPO achieves high reward on CartPole.\n\n"
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


# =============================================================================
# ChromaEmbeddingStore unit tests
# =============================================================================


class TestChromaEmbeddingStore:
    def test_add_chunks(self, chroma_store, chunks):
        added = chroma_store.add_chunks(chunks)
        assert added == 5
        assert chroma_store.count == 5

    def test_add_empty_list(self, chroma_store):
        assert chroma_store.add_chunks([]) == 0
        assert chroma_store.count == 0

    def test_add_chunks_with_empty_text_skipped(self, chroma_store):
        empty_chunk = Chunk(
            id="chk_empty",
            source_id="src_x",
            project_id="prj_test",
            text="   ",
            span=(0, 0),
            chunk_type=ChunkType.section,
        )
        added = chroma_store.add_chunks([empty_chunk])
        assert added == 0

    def test_upsert_is_idempotent(self, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        chroma_store.add_chunks(chunks)  # re-add same chunks
        assert chroma_store.count == 5  # not 10

    def test_query_returns_results(self, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        results = chroma_store.query(text="reinforcement learning", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_scores_are_valid(self, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        results = chroma_store.query(text="optimizer", top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0
            assert r.chunk_id
            assert r.source_id

    def test_query_empty_store(self, chroma_store):
        results = chroma_store.query(text="anything", top_k=5)
        assert results == []

    def test_query_empty_text(self, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        results = chroma_store.query(text="", top_k=5)
        assert results == []

    def test_query_top_k_exceeds_count(self, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        results = chroma_store.query(text="training", top_k=100)
        assert len(results) == 5  # capped at total count

    def test_semantic_similarity_not_just_keywords(self, chroma_store, chunks):
        """Vector search should find semantically related chunks even
        without exact keyword matches."""
        chroma_store.add_chunks(chunks)
        # Query about "gradient explosion" — should find the chunk about
        # "loss divergence" (chk_002) since they're semantically related.
        results = chroma_store.query(text="gradient explosion instability", top_k=3)
        chunk_ids = [r.chunk_id for r in results]
        # The training/optimization chunks should rank higher than env setup.
        assert "chk_005" not in chunk_ids[:2], (
            "Environment setup chunk should not be the top result for gradient instability"
        )

    def test_metadata_preserved(self, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        results = chroma_store.query(text="reward", top_k=1)
        assert len(results) == 1
        assert results[0].source_id.startswith("src_")


class TestTryCreateChromaStore:
    def test_creates_successfully(self):
        store = try_create_chroma_store(collection_name="test_try")
        assert store is not None
        assert store.count == 0


# =============================================================================
# SemanticSearchTool with vector backend
# =============================================================================


class TestSemanticSearchToolVector:
    def test_reports_vector_backend(self, projection_with_chunks, chroma_store, chunks):
        chroma_store.add_chunks(chunks)
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=chroma_store,
        )
        assert tool.backend == "vector"

    def test_vector_search_returns_cited_results(
        self, projection_with_chunks, chroma_store, chunks
    ):
        chroma_store.add_chunks(chunks)
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=chroma_store,
        )
        result = tool.call(
            workspace_id="ws_x",
            query="reinforcement learning policy",
            limit=3,
        )
        assert isinstance(result, Cited)
        assert result.value["backend"] == "vector"
        assert len(result.value["results"]) > 0
        assert len(result.value["results"]) <= 3
        assert len(result.citations) == len(result.value["results"])

    def test_vector_search_citations_have_evidence(
        self, projection_with_chunks, chroma_store, chunks
    ):
        chroma_store.add_chunks(chunks)
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=chroma_store,
        )
        result = tool.call(
            workspace_id="ws_x",
            query="training procedure",
            limit=2,
        )
        for cite in result.citations:
            assert cite.quote, "citation quote must be non-empty"
            assert cite.source_id
            assert cite.locator

    def test_vector_search_result_has_backend_field(
        self, projection_with_chunks, chroma_store, chunks
    ):
        chroma_store.add_chunks(chunks)
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=chroma_store,
        )
        result = tool.call(workspace_id="ws_x", query="optimizer", limit=1)
        for entry in result.value["results"]:
            assert entry["backend"] == "vector"

    def test_vector_search_empty_query_raises(
        self, projection_with_chunks, chroma_store, chunks
    ):
        chroma_store.add_chunks(chunks)
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=chroma_store,
        )
        with pytest.raises(WorkspaceToolError, match="searchable text"):
            tool.call(workspace_id="ws_x", query="  ", limit=1)

    def test_vector_search_invalid_limit_raises(
        self, projection_with_chunks, chroma_store, chunks
    ):
        chroma_store.add_chunks(chunks)
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=chroma_store,
        )
        with pytest.raises(WorkspaceToolError, match="limit"):
            tool.call(workspace_id="ws_x", query="test", limit=0)


# =============================================================================
# SemanticSearchTool BM25 fallback
# =============================================================================


class TestSemanticSearchToolBM25Fallback:
    def test_reports_bm25_backend(self, projection_with_chunks):
        tool = SemanticSearchTool(projection=projection_with_chunks)
        assert tool.backend == "bm25"

    def test_bm25_returns_cited_results(self, projection_with_chunks):
        tool = SemanticSearchTool(projection=projection_with_chunks)
        result = tool.call(
            workspace_id="ws_x",
            query="training gradient",
            project_id="prj_test",
            limit=2,
        )
        assert isinstance(result, Cited)
        assert result.value["backend"] == "bm25"
        assert len(result.value["results"]) > 0

    def test_bm25_no_match_raises(self, projection_with_chunks):
        tool = SemanticSearchTool(projection=projection_with_chunks)
        with pytest.raises(WorkspaceToolError, match="no matching"):
            tool.call(
                workspace_id="ws_x",
                query="xyzzynonexistent",
                project_id="prj_test",
                limit=1,
            )

    def test_fallback_to_bm25_when_store_empty(self, projection_with_chunks):
        """If embedding store exists but is empty, fall back to BM25."""
        empty_store = ChromaEmbeddingStore(collection_name="empty_fallback_test")
        tool = SemanticSearchTool(
            projection=projection_with_chunks,
            embedding_store=empty_store,
        )
        result = tool.call(
            workspace_id="ws_x",
            query="training gradient",
            project_id="prj_test",
            limit=2,
        )
        assert result.value["backend"] == "bm25"


# =============================================================================
# Integration: full pipeline → embedding store → search
# =============================================================================


class TestSemanticSearchIntegration:
    def test_end_to_end_vector_search(self, store, indexed_project):
        """Full integration: index paper → embed chunks → vector search."""
        # Build the SourcesProjection from indexed events.
        indexer = IndexerAppService(store=store)
        proj = SourcesProjection()
        indexer.project_into_projection(indexed_project, proj)

        # Embed all indexed chunks into Chroma.
        chroma = ChromaEmbeddingStore(collection_name="integration_test")
        chunks = proj.list_chunks(indexed_project)
        assert len(chunks) > 0, "indexer should have produced chunks"
        chroma.add_chunks(chunks)
        assert chroma.count == len(chunks)

        # Search with the tool.
        tool = SemanticSearchTool(projection=proj, embedding_store=chroma)
        result = tool.call(
            workspace_id="ws_x",
            query="policy optimization reward",
            limit=3,
        )
        assert isinstance(result, Cited)
        assert result.value["backend"] == "vector"
        assert len(result.value["results"]) >= 1
        # All citations should have non-empty quotes.
        for cite in result.citations:
            assert cite.quote

    def test_bm25_still_works_in_integration(self, store, indexed_project):
        """BM25 path still works in the integrated pipeline."""
        indexer = IndexerAppService(store=store)
        proj = SourcesProjection()
        indexer.project_into_projection(indexed_project, proj)

        tool = SemanticSearchTool(projection=proj)
        result = tool.call(
            workspace_id="ws_x",
            query="reinforcement learning",
            project_id=indexed_project,
            limit=2,
        )
        assert isinstance(result, Cited)
        assert result.value["backend"] == "bm25"

    def test_existing_workspace_test_still_passes(
        self, store, indexed_project
    ):
        """The existing workspace build + semantic search test pattern
        from test_issue16 still works with the upgraded tool."""
        indexer = IndexerAppService(store=store)
        workspace_service = WorkspaceAppService(store=store, indexer=indexer)
        workspace_service.build_workspace(
            BuildWorkspace(project_id=indexed_project)
        )
        proj = SourcesProjection()
        indexer.project_into_projection(indexed_project, proj)

        # Without embedding store — BM25 path.
        tool = SemanticSearchTool(projection=proj)
        result = tool.call(
            workspace_id="ws_x",
            project_id=indexed_project,
            query="reinforcement",
            limit=2,
        )
        assert isinstance(result, Cited)
        assert all(cite.quote for cite in result.citations)
