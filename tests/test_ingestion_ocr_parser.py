"""Tests for OcrPaperParser — Tesseract OCR fallback for scanned PDFs."""

from __future__ import annotations

import io
from pathlib import Path

import pytest


def _tesseract_functional() -> bool:
    """Return True only if tesseract binary is present AND can actually OCR an image."""
    try:
        import io
        import pytesseract
        from PIL import Image

        pytesseract.get_tesseract_version()
        # Try a minimal OCR call to confirm language data is present.
        img = Image.new("RGB", (10, 10), color="white")
        pytesseract.image_to_string(img)
        return True
    except Exception:
        return False


pytesseract = pytest.importorskip("pytesseract")
fitz = pytest.importorskip("fitz")
PIL = pytest.importorskip("PIL")

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
from backend.services.ingestion.parser.interface import ParseError
from backend.services.ingestion.parser.ocr_parser import OcrPaperParser


def _make_scanned_pdf(tmp_path: Path, text: str) -> Path:
    """Create a single-page PDF containing a rasterized image of `text`.

    This simulates a scanned PDF: the text is rendered to a Pillow image,
    embedded as a JPEG in a new PDF via PyMuPDF. The PDF has no text layer.
    """
    # Render text to an image.
    img = Image.new("RGB", (800, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), text, fill="black")

    # Convert to PNG bytes.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Embed in a PDF page via PyMuPDF.
    doc = fitz.open()
    page = doc.new_page(width=800, height=200)
    page.insert_image(fitz.Rect(0, 0, 800, 200), stream=png_bytes)

    pdf_path = tmp_path / "scanned.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.mark.skipif(not _tesseract_functional(), reason="Tesseract not functional (binary absent or language data missing)")
def test_ocr_recovers_text_from_scanned_pdf(tmp_path: Path):
    """OCR parser extracts readable text from a rasterized single-page PDF."""
    text = "Hello World Test"
    pdf_path = _make_scanned_pdf(tmp_path, text)

    result = OcrPaperParser().parse(project_id="prj_ocr", paper_path=pdf_path)

    # OCR may not be letter-perfect; verify at least one expected word appears.
    assert any(word.lower() in result.full_text.lower() for word in ["hello", "world", "test"]), (
        f"Expected OCR output to contain a word from {text!r}; got: {result.full_text!r}"
    )
    assert len(result.sections) == 1
    assert result.sections[0].title == "Page 1"


def test_ocr_parser_raises_on_missing_file(tmp_path: Path):
    """OcrPaperParser raises ParseError with cause_kind ocr_failed for a missing PDF."""
    missing = tmp_path / "nope.pdf"
    with pytest.raises(ParseError) as exc_info:
        OcrPaperParser().parse(project_id="prj_ocr", paper_path=missing)
    assert exc_info.value.cause_kind == "ocr_failed"


def test_ocr_parser_name():
    """Parser name is 'ocr-tesseract'."""
    assert OcrPaperParser().name == "ocr-tesseract"


def test_rasterization_failure_surfaces_as_parse_error(tmp_path: Path, monkeypatch):
    """A RuntimeError from get_pixmap is surfaced as ParseError(cause_kind='ocr_failed'), not a raw exception."""
    import pytesseract as _pytesseract

    # Build a minimal 1-page PDF with a real text layer (fitz can open it).
    doc = fitz.open()
    doc.new_page(width=200, height=100)
    pdf_path = tmp_path / "minimal.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # Stub get_tesseract_version so it does not fail even without tesseract binary.
    monkeypatch.setattr(_pytesseract, "get_tesseract_version", lambda: "5.0.0")

    # Make get_pixmap raise a RuntimeError on any fitz Page instance.
    monkeypatch.setattr(fitz.Page, "get_pixmap", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rasterization boom")))

    with pytest.raises(ParseError) as exc_info:
        OcrPaperParser().parse(project_id="prj_raster_err", paper_path=pdf_path)

    assert exc_info.value.cause_kind == "ocr_failed"


def test_ocr_parser_version_is_cached(monkeypatch):
    """Symptom: tesseract version was fetched on every access to .version (review M6 / T29).

    Verify: accessing .version twice calls _get_tesseract_version at most once.
    """
    import backend.services.ingestion.parser.ocr_parser as ocr_mod

    call_count = []
    real_fn = ocr_mod._get_tesseract_version

    def counting_fn():
        call_count.append(1)
        return real_fn()

    monkeypatch.setattr(ocr_mod, "_get_tesseract_version", counting_fn)

    from backend.services.ingestion.parser.ocr_parser import OcrPaperParser
    parser = OcrPaperParser()
    _ = parser.version
    _ = parser.version
    assert len(call_count) <= 1, (
        f"_get_tesseract_version was called {len(call_count)} times on two .version accesses"
    )
