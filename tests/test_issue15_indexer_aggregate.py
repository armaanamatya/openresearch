"""Issue #15 — IndexAggregate state machine."""

from __future__ import annotations

import pytest

from backend.services.context.indexer.aggregate import (
    IndexAggregate,
    IndexState,
    InvalidIndexTransition,
)
from backend.services.context.indexer.events import (
    ChunkCreated,
    IndexingCompleted,
    IndexingFailed,
    IndexingStarted,
    SourceRegistered,
)
from backend.services.context.indexer.model import (
    Chunk,
    ChunkType,
    SourceKind,
    SourceRef,
    chunk_id_for,
    source_id_for,
)


def _agg() -> IndexAggregate:
    return IndexAggregate.empty("prj_1")


def _src(upstream: str = "sec_a") -> SourceRef:
    sid = source_id_for(
        project_id="prj_1", kind=SourceKind.paper_section, upstream_id=upstream
    )
    return SourceRef(
        id=sid,
        project_id="prj_1",
        kind=SourceKind.paper_section,
        locator="Intro",
        upstream_id=upstream,
    )


def _chunk(src: SourceRef, text: str = "x", offset: int = 0) -> Chunk:
    cid = chunk_id_for(
        source_id=src.id,
        chunker_name="section",
        chunker_version="1",
        text=text,
        span=(offset, offset + len(text)),
        chunk_type=ChunkType.section,
    )
    return Chunk(
        id=cid,
        source_id=src.id,
        project_id="prj_1",
        text=text,
        span=(offset, offset + len(text)),
        chunk_type=ChunkType.section,
    )


def test_start_emits_indexing_started():
    agg = _agg()
    events = list(agg.handle_start("section", "1"))
    assert len(events) == 1
    assert isinstance(events[0], IndexingStarted)


def test_start_from_indexing_raises():
    agg = _agg()
    agg.apply(
        IndexingStarted(project_id="prj_1", chunker_name="section", chunker_version="1")
    )
    with pytest.raises(InvalidIndexTransition):
        agg.handle_start("section", "1")


def test_start_from_indexed_raises():
    agg = _agg()
    agg.apply(
        IndexingStarted(project_id="prj_1", chunker_name="section", chunker_version="1")
    )
    agg.apply(
        IndexingCompleted(
            project_id="prj_1",
            source_count=0,
            chunk_count=0,
            chunker_name="section",
            chunker_version="1",
        )
    )
    with pytest.raises(InvalidIndexTransition):
        agg.handle_start("section", "1")


def test_start_from_failed_allowed_for_retry():
    agg = _agg()
    agg.apply(
        IndexingStarted(project_id="prj_1", chunker_name="section", chunker_version="1")
    )
    agg.apply(
        IndexingFailed(
            project_id="prj_1",
            cause_kind="x",
            cause_message="y",
            retryable=True,
        )
    )
    events = list(agg.handle_start("section", "1"))
    assert isinstance(events[0], IndexingStarted)


def test_apply_source_registered_increments_source_count():
    agg = _agg()
    agg.apply(
        IndexingStarted(project_id="prj_1", chunker_name="section", chunker_version="1")
    )
    src = _src("sec_a")
    agg.apply(SourceRegistered(project_id="prj_1", source=src))
    assert agg.source_count == 1


def test_apply_chunk_created_increments_chunk_count():
    agg = _agg()
    agg.apply(
        IndexingStarted(project_id="prj_1", chunker_name="section", chunker_version="1")
    )
    src = _src("sec_a")
    agg.apply(ChunkCreated(project_id="prj_1", chunk=_chunk(src)))
    assert agg.chunk_count == 1


def test_full_lifecycle_via_apply_all():
    agg = _agg()
    src = _src("sec_a")
    events = [
        IndexingStarted(project_id="prj_1", chunker_name="section", chunker_version="1"),
        SourceRegistered(project_id="prj_1", source=src),
        ChunkCreated(project_id="prj_1", chunk=_chunk(src)),
        IndexingCompleted(
            project_id="prj_1",
            source_count=1,
            chunk_count=1,
            chunker_name="section",
            chunker_version="1",
        ),
    ]
    agg.apply_all(events)
    assert agg.state is IndexState.INDEXED
    assert agg.version == 4
    assert agg.source_count == 1
    assert agg.chunk_count == 1


# --- ID composition --------------------------------------------------------


def test_chunk_id_changes_when_chunker_version_changes():
    """Codex feedback: changing the chunker version must change the
    ChunkId so a new chunker run produces fresh chunks."""
    common = dict(
        source_id="src_x",
        text="hello",
        span=(0, 5),
        chunk_type=ChunkType.section,
    )
    a = chunk_id_for(chunker_name="section", chunker_version="1", **common)
    b = chunk_id_for(chunker_name="section", chunker_version="2", **common)
    assert a != b


def test_chunk_id_is_stable_for_same_inputs():
    common = dict(
        source_id="src_x",
        chunker_name="section",
        chunker_version="1",
        text="hello",
        span=(0, 5),
        chunk_type=ChunkType.section,
    )
    a = chunk_id_for(**common)
    b = chunk_id_for(**common)
    assert a == b


def test_source_id_is_stable_for_same_upstream_id():
    a = source_id_for(
        project_id="p", kind=SourceKind.paper_section, upstream_id="sec_x"
    )
    b = source_id_for(
        project_id="p", kind=SourceKind.paper_section, upstream_id="sec_x"
    )
    assert a == b


def test_source_id_differs_across_kind():
    a = source_id_for(
        project_id="p", kind=SourceKind.paper_section, upstream_id="x"
    )
    b = source_id_for(
        project_id="p", kind=SourceKind.paper_reference, upstream_id="x"
    )
    assert a != b
