"""Tests for HtmlPaperParser — arXiv LaTeXML HTML to clean ParseResult."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.ingestion.parser.html_parser import HtmlPaperParser
from backend.services.ingestion.parser.interface import ParseError

bs4 = pytest.importorskip("bs4")  # skip if beautifulsoup4 missing


_BODY_FILLER = " ".join(["word"] * 60)  # ~360 chars of prose per use

_SIMPLE_HTML = f"""\
<!DOCTYPE html>
<html>
<head><title>Test Paper</title></head>
<body>
  <script>alert('noise')</script>
  <nav>nav noise</nav>
  <section>
    <h2>Introduction</h2>
    <p>This is the introduction paragraph. It contains useful prose about the method and its applications in the real world. {_BODY_FILLER}</p>
    <p>A second paragraph with more words about the approach and results of the experiments. {_BODY_FILLER}</p>
  </section>
  <section>
    <h2>Methods</h2>
    <p>We describe the methodology here in detail with enough words to matter for understanding. {_BODY_FILLER}</p>
    <figure>
      <img src="fig1.png" alt="figure1" />
      <figcaption>Figure 1: The main architecture diagram for the system.</figcaption>
    </figure>
  </section>
  <section>
    <h2>Results</h2>
    <p>We achieve state of the art results on several benchmarks with our model and training procedure. {_BODY_FILLER}</p>
  </section>
</body>
</html>
"""

_JUNK_HTML = "<html><body><p>hi</p></body></html>"


def _write_html(tmp_path: Path, content: str, name: str = "raw_paper.html") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_html_parser_extracts_prose_and_figcaption(tmp_path: Path):
    """Paragraph text and figcaption text appear in full_text; script/nav do not."""
    html_path = _write_html(tmp_path, _SIMPLE_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)

    assert "introduction paragraph" in result.full_text.lower()
    assert "architecture diagram" in result.full_text.lower()
    # Script and nav noise must be absent
    assert "alert" not in result.full_text
    assert "nav noise" not in result.full_text


def test_html_parser_extracts_sections(tmp_path: Path):
    """Section titles and depths are parsed from <section><h2> structure."""
    html_path = _write_html(tmp_path, _SIMPLE_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)

    titles = [s.title for s in result.sections]
    assert "Introduction" in titles
    assert "Methods" in titles
    assert "Results" in titles


def test_html_parser_section_depth_from_heading_level(tmp_path: Path):
    """h2 → depth 2, h3 → depth 3."""
    html = """\
    <html><body>
    <section><h2>Top Level</h2><p>""" + "word " * 120 + """</p></section>
    <section><h3>Sub Level</h3><p>""" + "word " * 120 + """</p></section>
    </body></html>"""
    html_path = _write_html(tmp_path, html)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)

    depth_map = {s.title: s.depth for s in result.sections}
    assert depth_map.get("Top Level") == 2
    assert depth_map.get("Sub Level") == 3


def test_html_parser_raises_parse_error_on_empty_html(tmp_path: Path):
    """Empty / very short HTML raises ParseError with cause_kind html_parse_failed."""
    html_path = _write_html(tmp_path, _JUNK_HTML)
    with pytest.raises(ParseError) as exc_info:
        HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)
    assert exc_info.value.cause_kind == "html_parse_failed"


def test_html_parser_raises_parse_error_on_missing_file(tmp_path: Path):
    """Missing file raises ParseError, not FileNotFoundError."""
    missing = tmp_path / "does_not_exist.html"
    with pytest.raises(ParseError) as exc_info:
        HtmlPaperParser().parse(project_id="prj_test", paper_path=missing)
    assert exc_info.value.cause_kind == "html_parse_failed"


def test_html_parser_name_and_version():
    """Parser exposes expected name and version strings."""
    p = HtmlPaperParser()
    assert p.name == "arxiv-html"
    assert p.version == "1.0"


def test_html_parser_char_offsets_match_full_text(tmp_path: Path):
    """char_offset for each section is a valid offset into full_text."""
    html_path = _write_html(tmp_path, _SIMPLE_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)

    for section in result.sections:
        assert section.char_offset >= 0
        assert section.char_offset <= len(result.full_text)


# ---------------------------------------------------------------------------
# New slice-based invariant tests (review findings)
# ---------------------------------------------------------------------------

_NESTED_SECTION_HTML = """\
<!DOCTYPE html><html><body>
<section>
  <h2>Introduction</h2>
  <p>""" + "intro word " * 80 + """</p>
  <section>
    <h3>Subsection</h3>
    <p>""" + "sub word " * 80 + """</p>
  </section>
</section>
<section>
  <h2>Methods</h2>
  <p>""" + "method word " * 80 + """</p>
</section>
</body></html>"""

_ABSTRACT_FIRST_HTML = """\
<!DOCTYPE html><html><body>
<p>""" + "abstract word " * 60 + """</p>
<section>
  <h2>Introduction</h2>
  <p>""" + "intro word " * 80 + """</p>
</section>
<section>
  <h2>Methods</h2>
  <p>""" + "method word " * 80 + """</p>
</section>
</body></html>"""


def test_html_no_duplicate_text_in_nested_sections(tmp_path: Path):
    """Nested <section> elements do not cause subsection text to appear twice in full_text."""
    html_path = _write_html(tmp_path, _NESTED_SECTION_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)
    # "sub word" should appear exactly as many times as in the source.
    count = result.full_text.count("sub word")
    # Source has "sub word " * 80 = 80 occurrences.
    assert count == 80, f"Expected 80 occurrences of 'sub word', got {count}"


def test_html_abstract_text_included_in_full_text(tmp_path: Path):
    """Text before the first heading (abstract / front matter) appears in full_text."""
    html_path = _write_html(tmp_path, _ABSTRACT_FIRST_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)
    assert "abstract word" in result.full_text


def test_html_section_texts_tile_full_text(tmp_path: Path):
    """Concatenation of all section texts in char_offset order reconstructs full_text."""
    html_path = _write_html(tmp_path, _SIMPLE_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)
    # Sort sections by char_offset (document order, not depth-first).
    ordered = sorted(result.sections, key=lambda s: s.char_offset)
    reconstructed = "".join(s.text for s in ordered)
    assert reconstructed == result.full_text, (
        f"Tiling failed: reconstructed ({len(reconstructed)}) != full_text ({len(result.full_text)})"
    )


def test_html_section_char_offsets_are_correct(tmp_path: Path):
    """Each section's char_offset is the exact position of its text in full_text."""
    html_path = _write_html(tmp_path, _SIMPLE_HTML)
    result = HtmlPaperParser().parse(project_id="prj_test", paper_path=html_path)
    for section in result.sections:
        expected_slice = result.full_text[section.char_offset: section.char_offset + len(section.text)]
        assert expected_slice == section.text, (
            f"Section '{section.title}': full_text[{section.char_offset}:...] = {expected_slice!r}, "
            f"but section.text = {section.text!r}"
        )
