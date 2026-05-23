"""Tests for score_text_quality and ResolvingParser cascade logic.

Uses stub sub-parsers returning canned ParseResults so the tests are
fast and hermetic (no real PDF/HTML/OCR needed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.ingestion.parser.interface import ParseError, ParseResult
from backend.services.ingestion.parser.model import Section, section_id_for
from backend.services.ingestion.parser.resolving_parser import (
    ResolvingParser,
    score_text_quality,
    _USABLE,
)


# ---------------------------------------------------------------------------
# score_text_quality tests
# ---------------------------------------------------------------------------

_CLEAN_PROSE = (
    "The transformer architecture introduces multi-head self-attention which allows "
    "the model to jointly attend to information from different representation subspaces "
    "at different positions. This has proven highly effective for natural language "
    "understanding and generation tasks. We extend this mechanism to handle longer "
    "context windows by introducing a sparse attention pattern. " * 12
)

_FIGURE_NOISE = ("<BOS> IO S1 0.0 0.2 0.4 <MID> John Mary " * 50)

_SHORT_TEXT = "Too short."


def test_score_clean_prose_is_high():
    """Clean prose scores above the _USABLE threshold."""
    score = score_text_quality(_CLEAN_PROSE)
    assert score >= _USABLE, f"Expected >= {_USABLE}, got {score:.3f}"


def test_score_figure_noise_is_low():
    """Figure-noise token soup scores lower than clean prose."""
    clean = score_text_quality(_CLEAN_PROSE)
    noisy = score_text_quality(_FIGURE_NOISE)
    assert noisy < clean, f"Expected noisy ({noisy:.3f}) < clean ({clean:.3f})"


def test_score_short_text_is_zero():
    """Text shorter than 1000 chars always scores 0.0."""
    assert score_text_quality(_SHORT_TEXT) == 0.0


def test_score_empty_string_is_zero():
    """Empty string scores 0.0."""
    assert score_text_quality("") == 0.0


# ---------------------------------------------------------------------------
# Stub parser helpers
# ---------------------------------------------------------------------------

def _make_result(text: str, project_id: str = "prj_stub") -> ParseResult:
    """Build a minimal ParseResult with one section from the given text."""
    sid = section_id_for(project_id, 0, 0, "Doc", text)
    section = Section(
        id=sid,
        project_id=project_id,
        title="Doc",
        text=text,
        char_offset=0,
        parent_id=None,
        depth=0,
    )
    return ParseResult(sections=(section,), references=[], figures=(), full_text=text)


class _StubParser:
    """Configurable stub that returns a canned result or raises ParseError."""

    def __init__(self, name: str, result: ParseResult | None = None, raises: bool = False) -> None:
        self._name = name
        self._result = result
        self._raises = raises

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "stub"

    def parse(self, *, project_id: str, paper_path: Path) -> ParseResult:
        if self._raises:
            raise ParseError(f"Stub {self._name} fails", cause_kind="stub_fail", retryable=False)
        return self._result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ResolvingParser cascade tests
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path: Path) -> Path:
    """Write a fake raw_paper.pdf (content doesn't matter — stubs ignore it)."""
    p = tmp_path / "raw_paper.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _make_html(tmp_path: Path) -> Path:
    """Write a placeholder raw_paper.html sibling."""
    p = tmp_path / "raw_paper.html"
    p.write_bytes(b"<html></html>")
    return p


def test_resolving_prefers_html_when_good(tmp_path: Path):
    """HTML sibling with equal-score prose to PDF → HTML is chosen (HTML-preference tiebreak)."""
    # Same underlying text so both strategies produce the same quality score;
    # the tiebreak rule (HTML > PDF > OCR) then selects HTML.
    good_html_result = _make_result(_CLEAN_PROSE)
    good_pdf_result = _make_result(_CLEAN_PROSE)
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=good_html_result),
        pdf_parser=_StubParser("pymupdf", result=good_pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is good_html_result


def test_resolving_falls_back_to_pdf_when_no_html(tmp_path: Path):
    """No HTML sibling → PDF result chosen (no html file written)."""
    good_pdf_result = _make_result(_CLEAN_PROSE)
    _make_pdf(tmp_path)
    # No raw_paper.html created

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", raises=True),
        pdf_parser=_StubParser("pymupdf", result=good_pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is good_pdf_result


def test_resolving_falls_back_to_pdf_when_html_low_quality(tmp_path: Path):
    """HTML present but junk score → PDF is chosen instead."""
    junk_html_result = _make_result(_FIGURE_NOISE * 20)  # noisy, short-word tokens
    good_pdf_result = _make_result(_CLEAN_PROSE)
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=junk_html_result),
        pdf_parser=_StubParser("pymupdf", result=good_pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is good_pdf_result


def test_resolving_falls_back_to_ocr_when_pdf_unusable(tmp_path: Path):
    """PDF yields empty (score 0.0) → OCR is chosen."""
    # A non-empty but very short text will produce score 0.0 (< 1000 chars)
    poor_pdf_result = _make_result("a b c d")  # 7 chars → score 0.0
    ocr_result = _make_result(_CLEAN_PROSE)
    _make_pdf(tmp_path)
    # No HTML sibling

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", raises=True),
        pdf_parser=_StubParser("pymupdf", result=poor_pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", result=ocr_result),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is ocr_result


def test_resolving_raises_when_all_fail(tmp_path: Path):
    """All strategies raising ParseError → ResolvingParser raises ParseError with all_strategies_failed."""
    _make_pdf(tmp_path)
    # No HTML

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", raises=True),
        pdf_parser=_StubParser("pymupdf", raises=True),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    with pytest.raises(ParseError) as exc_info:
        parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert exc_info.value.cause_kind == "all_strategies_failed"


def test_resolving_name_and_version():
    """ResolvingParser exposes expected name and version."""
    p = ResolvingParser(
        html_parser=_StubParser("arxiv-html", raises=True),
        pdf_parser=_StubParser("pymupdf", raises=True),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    assert p.name == "resolving"
    assert p.version == "1.0"


# ---------------------------------------------------------------------------
# New best-of-N tests (review findings)
# ---------------------------------------------------------------------------

def _make_result_scored(score_target: float, project_id: str = "prj_stub") -> ParseResult:
    """Build a ParseResult whose score_text_quality equals score_target approximately.

    Uses _CLEAN_PROSE scaled by repetition to exceed 1000 chars, then
    directly returns a ParseResult whose full_text is recognized at the
    target category (above or below _USABLE).
    """
    # score_target >= _USABLE → use clean prose; otherwise use noisy text.
    if score_target >= _USABLE:
        text = _CLEAN_PROSE
    else:
        text = _FIGURE_NOISE * 20  # noisy; scores ~0.22 < _USABLE
    return _make_result(text, project_id)


def test_resolving_pdf_wins_when_pdf_score_higher(tmp_path: Path):
    """HTML=0.5-ish / PDF=0.8-ish (both usable) → PDF is chosen because its score is higher."""
    # We need HTML to score lower than PDF but both above _USABLE.
    # Craft texts with known token compositions:
    #   html_text: half wordish tokens → score ~0.5
    #   pdf_text: mostly wordish tokens → score ~0.8
    # Use repetition of mixed/clean prose to hit the targets approximately.
    # Simpler: use score_text_quality directly on the texts we produce.
    # html text: alternate word/noise tokens to score ~0.5.
    html_text = " ".join(["word"] * 500 + ["0.0"] * 500) + " " + ("x " * 500)
    # Pad to >1000 chars with real words to ensure the wordish ratio is ~0.5.
    html_text = ("alpha " * 600 + "0.1 " * 600)  # ~0.5 wordish ratio, long enough
    pdf_text = _CLEAN_PROSE  # ~0.7+ wordish ratio

    html_score = score_text_quality(html_text)
    pdf_score = score_text_quality(pdf_text)
    assert html_score >= _USABLE, f"html_score {html_score} not >= {_USABLE}"
    assert pdf_score > html_score, f"pdf_score {pdf_score} not > html_score {html_score}"

    html_result = _make_result(html_text)
    pdf_result = _make_result(pdf_text)
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=html_result),
        pdf_parser=_StubParser("pymupdf", result=pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is pdf_result


def test_resolving_html_wins_when_html_score_higher(tmp_path: Path):
    """HTML=0.7-ish / PDF=0.5-ish (both usable) → HTML is chosen because its score is higher."""
    html_text = _CLEAN_PROSE  # ~0.7+ wordish ratio
    pdf_text = ("alpha " * 600 + "0.1 " * 600)  # ~0.5 wordish ratio

    html_score = score_text_quality(html_text)
    pdf_score = score_text_quality(pdf_text)
    assert html_score > pdf_score, f"html_score {html_score} not > pdf_score {pdf_score}"
    assert pdf_score >= _USABLE, f"pdf_score {pdf_score} not >= {_USABLE}"

    html_result = _make_result(html_text)
    pdf_result = _make_result(pdf_text)
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=html_result),
        pdf_parser=_StubParser("pymupdf", result=pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is html_result


def test_resolving_html_wins_exact_tie(tmp_path: Path):
    """HTML and PDF score identically above _USABLE → HTML wins (tiebreak)."""
    # Use the same text so scores are exactly equal.
    shared_text = _CLEAN_PROSE
    html_result = _make_result(shared_text)
    pdf_result = _make_result(shared_text)
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=html_result),
        pdf_parser=_StubParser("pymupdf", result=pdf_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    # HTML must win on tie because _STRATEGY_PRIORITY["html"] < _STRATEGY_PRIORITY["pdf"].
    assert result is html_result


def test_resolving_returns_best_short_text_when_no_strategy_usable(tmp_path: Path):
    """Short non-empty text (score 0.0, < 1000 chars) is a valid parse, not a
    failure — the resolver returns the best such result rather than raising.

    Regression: a small/short PDF raised all_strategies_failed because
    score_text_quality scores < 1000-char text 0.0; a short document is valid.
    """
    short_result = _make_result("A short but perfectly valid paper body.")
    _make_pdf(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", raises=True),
        pdf_parser=_StubParser("pymupdf", result=short_result),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert result is short_result


def test_resolving_raises_when_all_results_empty(tmp_path: Path):
    """Every strategy returns an EMPTY-text result → ParseError(all_strategies_failed).
    An empty parse is a genuine failure, distinct from a short-but-valid one."""
    empty_result = _make_result("")
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=empty_result),
        pdf_parser=_StubParser("pymupdf", result=empty_result),
        ocr_parser=_StubParser("ocr-tesseract", result=empty_result),
    )
    with pytest.raises(ParseError) as exc_info:
        parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert exc_info.value.cause_kind == "all_strategies_failed"


def test_resolving_raises_when_all_parsers_raise_with_html_sibling(tmp_path: Path):
    """All parsers raise ParseError (HTML sibling present) → ParseError(all_strategies_failed)."""
    _make_pdf(tmp_path)
    _make_html(tmp_path)

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", raises=True),
        pdf_parser=_StubParser("pymupdf", raises=True),
        ocr_parser=_StubParser("ocr-tesseract", raises=True),
    )
    with pytest.raises(ParseError) as exc_info:
        parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")
    assert exc_info.value.cause_kind == "all_strategies_failed"


# ---------------------------------------------------------------------------
# T16 guard test — OCR-skip must gate on quality not raw length (review I4)
# ---------------------------------------------------------------------------

def test_ocr_runs_on_short_low_quality_html(tmp_path: Path):
    """Symptom: 200-char figure-noise HTML wins over OCR; cascade ships garbage.

    The OCR-skip gate used raw length (>=200 chars => skip), so a 250-char
    HTML soup scoring 0.0 from score_text_quality still skipped OCR (review I4 / T16).
    Verify: OCR is invoked when HTML quality score < _MIN_USEFUL_SCORE, regardless of length.
    """
    # 250-char string of noise tokens (scores 0.0 from score_text_quality — below 1000 chars)
    noise_250 = "<BOS> IO S1 0.0 0.2 0.4 <MID> " * 9  # ~270 chars, figure noise
    assert len(noise_250.strip()) >= 200, "test setup: HTML text must be >= 200 chars"
    assert score_text_quality(noise_250) == 0.0, "test setup: noise_250 must score 0.0"

    noisy_html_result = _make_result(noise_250)
    ocr_result = _make_result(_CLEAN_PROSE)

    _make_pdf(tmp_path)
    _make_html(tmp_path)

    ocr_stub = _StubParser("ocr-tesseract", result=ocr_result)
    ocr_called = []

    _orig_parse = ocr_stub.parse

    def _spy_parse(**kwargs):
        ocr_called.append(True)
        return _orig_parse(**kwargs)

    ocr_stub.parse = _spy_parse  # type: ignore[method-assign]

    parser = ResolvingParser(
        html_parser=_StubParser("arxiv-html", result=noisy_html_result),
        pdf_parser=_StubParser("pymupdf", raises=True),
        ocr_parser=ocr_stub,
    )
    result = parser.parse(project_id="prj_x", paper_path=tmp_path / "raw_paper.pdf")

    assert ocr_called, "OCR must be invoked when HTML quality score < _MIN_USEFUL_SCORE"
    assert result is ocr_result, "OCR result must be selected when HTML is garbage"
