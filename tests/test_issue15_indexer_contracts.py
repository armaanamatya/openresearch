"""Issue #15 — Context indexer with source mapping and chunk registration.

The indexer service itself is not implemented yet, but this file exercises
the foundation primitives that #15 depends on: Citation references to
source_id / chunk_id, multi-citation stacking on a single domain event,
and the typing surface that downstream chunk registration will rely on.

When the IndexerAppService and SourcesProjection land, this file expands
with their integration tests.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from backend.messaging.event import (
    DomainEvent,
    _clear_registry_for_tests,
    register_event,
)
from backend.schemas.citations import Citation, NonEmptyCitations


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def test_citation_targets_a_source_id():
    cite = Citation(
        source_id="src_paper_section_42",
        quote="PPO uses a clipped surrogate objective",
        locator="PPO §3.1",
    )
    assert cite.source_id == "src_paper_section_42"
    assert cite.chunk_id is None


def test_citation_can_target_chunk_within_source():
    cite = Citation(
        source_id="src_repo_readme",
        chunk_id="chk_l42_install",
        quote="pip install -r requirements.txt",
        locator="github.com/openai/baselines/README.md:42",
    )
    assert cite.chunk_id == "chk_l42_install"


def test_event_can_carry_multiple_citations_on_one_claim():
    """Event payload with NonEmptyCitations supports stacking evidence
    from paper + repo + issue thread on a single agent decision."""

    @register_event
    class SourceRegisteredLike(DomainEvent):
        event_type: ClassVar[str] = "source_registered_like"
        schema_version: ClassVar[int] = 1
        source_id: str
        citations: NonEmptyCitations

    ev = SourceRegisteredLike(
        source_id="src_test",
        citations=(
            Citation(source_id="src_paper", quote="q1", locator="paper §3"),
            Citation(source_id="src_repo", quote="q2", locator="readme:5"),
            Citation(source_id="src_issue", quote="q3", locator="gh#42"),
        ),
    )
    assert len(ev.citations) == 3
    sources = {c.source_id for c in ev.citations}
    assert sources == {"src_paper", "src_repo", "src_issue"}


def test_event_with_empty_citations_is_rejected_at_validation():
    """The citation invariant must hold for indexer-emitted events too —
    a SourceRegistered with no evidence is a bug, not a valid state."""

    @register_event
    class SourceRegisteredLike(DomainEvent):
        event_type: ClassVar[str] = "source_registered_like_2"
        schema_version: ClassVar[int] = 1
        source_id: str
        citations: NonEmptyCitations

    with pytest.raises(ValidationError):
        SourceRegisteredLike(source_id="src_x", citations=())  # type: ignore[arg-type]


def test_citations_with_different_confidence_are_distinct():
    a = Citation(source_id="s", quote="q", locator="l", confidence=0.9)
    b = Citation(source_id="s", quote="q", locator="l", confidence=0.7)
    assert a != b


def test_citation_low_confidence_still_valid_evidence():
    """Confidence < 1.0 is allowed — represents inferred / weak evidence
    (e.g., a heuristic match). NonEmptyCitations only requires presence,
    not high confidence."""
    c = Citation(source_id="s", quote="q", locator="l", confidence=0.4)
    assert c.confidence == 0.4
