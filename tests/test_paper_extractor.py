"""Tests for the PaperExtractor hierarchy.

Uses the same synthetic-PDF pattern as test_issue13_parser_pymupdf.py.
Never makes real network/API calls — vision client is always mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.services.ingestion.parser.extractor import (
    NullExtractor,
    VisionAugmentingExtractor,
    extractor_from_settings,
)
from backend.services.ingestion.parser.interface import ParseResult
from backend.services.ingestion.parser.model import Figure, figure_id_for

fitz = pytest.importorskip("fitz")  # PyMuPDF — skip if not installed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(tmp_path: Path, body: str, name: str = "synthetic.pdf") -> Path:
    """Generate a one-page PDF with `body` as plain text."""
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _make_pdf_multipage(tmp_path: Path, pages: list[str], name: str = "multi.pdf") -> Path:
    """Generate a multi-page PDF; each string in `pages` goes on one page."""
    path = tmp_path / name
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_text(fitz.Point(50, 72), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _empty_result() -> ParseResult:
    return ParseResult(sections=(), references=(), figures=(), full_text="base text")


def _fake_client(available: bool, describe_return: str = "VISION_OUTPUT") -> MagicMock:
    client = MagicMock()
    client.is_available.return_value = available
    client.describe_page.return_value = describe_return
    return client


# ---------------------------------------------------------------------------
# NullExtractor
# ---------------------------------------------------------------------------


def test_null_extractor_returns_base_unchanged(tmp_path: Path):
    base = _empty_result()
    extractor = NullExtractor()
    result = extractor.extract(
        project_id="prj_test",
        paper_path=tmp_path / "ignored.pdf",
        base=base,
    )
    assert result is base


def test_null_extractor_metadata():
    e = NullExtractor()
    assert e.name == "null"
    assert e.version == "1"


# ---------------------------------------------------------------------------
# extractor_from_settings factory
# ---------------------------------------------------------------------------


def test_factory_returns_null_for_text_mode():
    settings = MagicMock()
    settings.paper_extraction_mode = "text"
    result = extractor_from_settings(settings)
    assert isinstance(result, NullExtractor)


def test_factory_returns_vision_augmenting_for_hybrid_mode():
    settings = MagicMock()
    settings.paper_extraction_mode = "hybrid"
    settings.paper_extraction_vision_model = "claude-sonnet-4-6"
    result = extractor_from_settings(settings)
    assert isinstance(result, VisionAugmentingExtractor)


# ---------------------------------------------------------------------------
# VisionAugmentingExtractor — client unavailable
# ---------------------------------------------------------------------------


def test_vision_extractor_unavailable_client_returns_base(tmp_path: Path):
    pdf = _make_pdf(tmp_path, "Some text here for the page body.")
    base = _empty_result()
    extractor = VisionAugmentingExtractor(_fake_client(available=False))
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=base)
    assert result is base


def test_vision_extractor_unavailable_client_does_not_raise(tmp_path: Path):
    pdf = _make_pdf(tmp_path, "text")
    extractor = VisionAugmentingExtractor(_fake_client(available=False))
    # must not raise under any circumstances
    extractor.extract(project_id="prj_x", paper_path=pdf, base=_empty_result())


# ---------------------------------------------------------------------------
# VisionAugmentingExtractor — text-rich page (above density threshold)
# ---------------------------------------------------------------------------


def test_vision_not_called_for_text_rich_page(tmp_path: Path):
    """A page with plenty of embedded text + no figures → zero vision calls."""
    # 200 characters of body text — well above TEXT_DENSITY_THRESHOLD=120.
    body = "This is a sufficiently long line of text. " * 5  # ~210 chars
    pdf = _make_pdf(tmp_path, body)
    client = _fake_client(available=True, describe_return="SHOULD_NOT_APPEAR")
    extractor = VisionAugmentingExtractor(client)
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=_empty_result())
    client.describe_page.assert_not_called()
    # full_text unchanged from base (no vision output appended)
    assert "SHOULD_NOT_APPEAR" not in result.full_text
    # result equals base (same content)
    assert result.full_text == "base text"


# ---------------------------------------------------------------------------
# VisionAugmentingExtractor — scanned (empty text) page
# ---------------------------------------------------------------------------


def test_vision_called_for_empty_text_page(tmp_path: Path):
    """A page with no embedded text is a vision candidate."""
    # Create a PDF with an empty page (no text inserted).
    pdf = _make_pdf_multipage(tmp_path, [""])  # one page, no text
    client = _fake_client(available=True, describe_return="VISION_TRANSCRIPT")
    extractor = VisionAugmentingExtractor(client)
    base = ParseResult(sections=(), references=(), figures=(), full_text="base text")
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=base)
    client.describe_page.assert_called_once()
    assert "VISION_TRANSCRIPT" in result.full_text


# ---------------------------------------------------------------------------
# VisionAugmentingExtractor — per-page describe_page raising
# ---------------------------------------------------------------------------


def test_vision_page_error_does_not_abort_extract(tmp_path: Path):
    """A failing describe_page call must not propagate — extract must not raise."""
    # Empty page triggers vision call.
    pdf = _make_pdf_multipage(tmp_path, ["", ""])  # two empty pages
    client = _fake_client(available=True)
    client.describe_page.side_effect = RuntimeError("simulated API failure")
    extractor = VisionAugmentingExtractor(client)
    # Must not raise.
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=_empty_result())
    # Returns base unchanged when no successful vision results.
    assert result is not None


def test_vision_outer_exception_returns_base(tmp_path: Path):
    """Even if something inside _extract raises, extract() returns base."""
    pdf = _make_pdf(tmp_path, "text")
    client = _fake_client(available=True)
    client.is_available.side_effect = RuntimeError("unexpected")
    extractor = VisionAugmentingExtractor(client)
    base = _empty_result()
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=base)
    assert result is base


# ---------------------------------------------------------------------------
# VisionAugmentingExtractor — figure page triggers vision call
# ---------------------------------------------------------------------------


def test_vision_called_for_figure_page(tmp_path: Path):
    """A page referenced by a Figure is a vision candidate even if text-rich."""
    # Two pages: page 0 is text-rich (won't trigger by density), page 0 has a figure.
    body = "This is a sufficiently long line of text. " * 5  # ~210 chars
    pdf = _make_pdf(tmp_path, body)

    fig = Figure(
        id=figure_id_for("prj_x", 0, "Figure 1"),
        project_id="prj_x",
        caption="Figure 1",
        page=0,
    )
    base = ParseResult(sections=(), references=(), figures=(fig,), full_text="base text")
    client = _fake_client(available=True, describe_return="FIG_VISION_OUTPUT")
    extractor = VisionAugmentingExtractor(client)
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=base)
    client.describe_page.assert_called_once()
    assert "FIG_VISION_OUTPUT" in result.full_text
    # The figure should be enriched with the description.
    assert result.figures[0].description == "FIG_VISION_OUTPUT"


# ---------------------------------------------------------------------------
# VisionAugmentingExtractor — figure description enrichment
# ---------------------------------------------------------------------------


def test_figure_description_enriched(tmp_path: Path):
    """Figures on vision-processed pages get their description field set."""
    pdf = _make_pdf_multipage(tmp_path, [""])  # empty page → vision candidate
    fig = Figure(
        id=figure_id_for("prj_x", 0, "Fig 1"),
        project_id="prj_x",
        caption="Fig 1",
        page=0,
        description=None,
    )
    base = ParseResult(sections=(), references=(), figures=(fig,), full_text="base")
    client = _fake_client(available=True, describe_return="RICH_DESCRIPTION")
    extractor = VisionAugmentingExtractor(client)
    result = extractor.extract(project_id="prj_x", paper_path=pdf, base=base)
    assert len(result.figures) == 1
    assert result.figures[0].description == "RICH_DESCRIPTION"
    # id must be stable (description does not affect figure_id_for).
    assert result.figures[0].id == fig.id
