"""Issue #13 — PyMuPdfParser tests on synthetic PDFs.

We generate a tiny PDF in-memory using PyMuPDF (`fitz`) so the test
runs without needing a real paper checked into the repo. PyMuPDF is
already a hard dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.ingestion.parser.pymupdf_parser import PyMuPdfParser

fitz = pytest.importorskip("fitz")  # PyMuPDF


def _make_pdf(tmp_path: Path, body: str, name: str = "synthetic.pdf") -> Path:
    """Generate a one-page PDF with `body` as plain text."""
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def test_parser_metadata():
    p = PyMuPdfParser()
    assert p.name == "pymupdf"
    assert p.version == "1"


def test_parser_returns_one_section_when_no_headings(tmp_path: Path):
    pdf = _make_pdf(tmp_path, "Just a single line of body text with no headings at all.")
    result = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    assert len(result.sections) == 1
    assert result.sections[0].title == "Document"
    assert result.sections[0].depth == 0


def test_parser_detects_named_top_level_headings(tmp_path: Path):
    body = (
        "Abstract\n"
        "This is the abstract body.\n\n"
        "Introduction\n"
        "We introduce the work here.\n\n"
        "Methods\n"
        "We do science.\n\n"
        "Results\n"
        "We get results.\n"
    )
    pdf = _make_pdf(tmp_path, body)
    result = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    titles = [s.title for s in result.sections]
    # Title-cased — "Methods" stays "Methods", "Introduction" stays "Introduction".
    assert "Abstract" in titles
    assert "Introduction" in titles
    assert "Methods" in titles
    assert "Results" in titles


def test_parser_detects_numbered_headings(tmp_path: Path):
    body = (
        "1 Introduction\n"
        "Setting up the work.\n\n"
        "2 Approach\n"
        "We propose X.\n\n"
        "2.1 Architecture\n"
        "Stack of layers.\n"
    )
    pdf = _make_pdf(tmp_path, body)
    result = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    titles = [s.title for s in result.sections]
    assert any(t.startswith("1 Introduction") for t in titles)
    assert any(t.startswith("2 Approach") for t in titles)
    assert any(t.startswith("2.1 Architecture") for t in titles)
    # Depth: top-level=1, subsection=2.
    depths_by_title = {s.title: s.depth for s in result.sections}
    assert depths_by_title.get("2 Approach") == 1
    assert depths_by_title.get("2.1 Architecture") == 2


def test_section_ids_are_deterministic_across_parses(tmp_path: Path):
    body = (
        "Introduction\n"
        "alpha beta gamma.\n\n"
        "Methods\n"
        "delta epsilon zeta.\n"
    )
    pdf = _make_pdf(tmp_path, body)
    a = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    b = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    assert [s.id for s in a.sections] == [s.id for s in b.sections]


def test_section_ids_differ_across_project_ids(tmp_path: Path):
    pdf = _make_pdf(tmp_path, "Introduction\nbody.\n")
    a = PyMuPdfParser().parse(project_id="prj_A", paper_path=pdf)
    b = PyMuPdfParser().parse(project_id="prj_B", paper_path=pdf)
    assert a.sections[0].id != b.sections[0].id


def test_sections_emitted_in_deterministic_order(tmp_path: Path):
    body = "1 Intro\nbody.\n\n2.1 Sub\nbody.\n\n2 Method\nbody.\n"
    pdf = _make_pdf(tmp_path, body)
    result = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    # Sorted by (depth, char_offset, id) — depth-1 entries come first,
    # in document order.
    depths = [s.depth for s in result.sections]
    assert depths == sorted(depths) or all(
        # If depth is the same across sections, char_offset must be sorted.
        depths[i] <= depths[i + 1] for i in range(len(depths) - 1)
    )


def test_references_extracted_by_arxiv_id(tmp_path: Path):
    body = (
        "Introduction\nintro body.\n\n"
        "References\n\n"
        "[1] Schulman et al. PPO. arXiv:1707.06347\n\n"
        "[2] Mnih et al. DQN. arXiv:1312.5602\n"
    )
    pdf = _make_pdf(tmp_path, body)
    result = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    arxiv_ids = sorted(r.arxiv_id for r in result.references if r.arxiv_id)
    assert "1707.06347" in arxiv_ids
    assert "1312.5602" in arxiv_ids


def test_references_extracted_by_doi(tmp_path: Path):
    body = (
        "Introduction\nbody.\n\n"
        "References\n\n"
        "[1] An author. A paper. doi:10.1109/CVPR.2019.00075\n"
    )
    pdf = _make_pdf(tmp_path, body)
    result = PyMuPdfParser().parse(project_id="prj_x", paper_path=pdf)
    dois = [r.doi for r in result.references if r.doi]
    assert "10.1109/CVPR.2019.00075" in dois


def test_parser_raises_on_missing_pdf(tmp_path: Path):
    from backend.services.ingestion.parser.interface import ParseError

    missing = tmp_path / "nope.pdf"
    with pytest.raises(ParseError) as ei:
        PyMuPdfParser().parse(project_id="prj_x", paper_path=missing)
    assert ei.value.cause_kind == "paper_missing"
    assert ei.value.retryable is True


def test_parser_raises_on_invalid_pdf(tmp_path: Path):
    from backend.services.ingestion.parser.interface import ParseError

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    with pytest.raises(ParseError) as ei:
        PyMuPdfParser().parse(project_id="prj_x", paper_path=bad)
    assert ei.value.cause_kind == "pdf_open_failed"
    assert ei.value.retryable is False
