"""Tests for backend.schemas.citations.

Verifies the citation invariant: NonEmptyCitations cannot be
constructed empty. This is the architectural keystone — without it,
the entire audit-trail thesis collapses.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from backend.schemas.citations import Citation, NonEmptyCitations


def test_citation_constructs_with_required_fields():
    c = Citation(source_id="src_1", quote="hello", locator="ppo §3")
    assert c.source_id == "src_1"
    assert c.quote == "hello"
    assert c.locator == "ppo §3"
    assert c.chunk_id is None
    assert c.confidence == 1.0


def test_citation_is_frozen():
    c = Citation(source_id="src_1", quote="hello", locator="x")
    with pytest.raises(ValidationError):
        c.source_id = "src_2"  # type: ignore[misc]


def test_citation_confidence_must_be_in_range():
    with pytest.raises(ValidationError):
        Citation(source_id="s", quote="q", locator="l", confidence=1.5)
    with pytest.raises(ValidationError):
        Citation(source_id="s", quote="q", locator="l", confidence=-0.1)


def test_non_empty_citations_rejects_empty_tuple():
    class Demo(BaseModel):
        cites: NonEmptyCitations

    with pytest.raises(ValidationError) as exc_info:
        Demo(cites=())  # type: ignore[arg-type]
    # Make sure the error mentions the constraint.
    assert "at least 1" in str(exc_info.value).lower() or "min_length" in str(exc_info.value)


def test_non_empty_citations_accepts_one_or_more():
    class Demo(BaseModel):
        cites: NonEmptyCitations

    one = Demo(cites=(Citation(source_id="s", quote="q", locator="l"),))
    assert len(one.cites) == 1

    three = Demo(
        cites=(
            Citation(source_id="a", quote="q1", locator="l1"),
            Citation(source_id="b", quote="q2", locator="l2"),
            Citation(source_id="c", quote="q3", locator="l3"),
        )
    )
    assert len(three.cites) == 3


def test_citation_model_construct_is_banned():
    """The third defense layer: Citation.model_construct raises
    InvariantBypassError. Even direct attempts to bypass Pydantic
    on a single Citation fail."""
    from backend.messaging.event import InvariantBypassError

    with pytest.raises(InvariantBypassError):
        Citation.model_construct(source_id="s", quote="q", locator="l")
