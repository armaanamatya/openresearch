"""OCR-based parser — fallback for scanned PDFs with no text layer.

Uses PyMuPDF to rasterize pages and pytesseract + Pillow for OCR.
One Section per page; no reference extraction.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from backend.services.ingestion.parser.interface import ParseError, ParseResult
from backend.services.ingestion.parser.model import Section, section_id_for

logger = logging.getLogger(__name__)

_PARSER_NAME = "ocr-tesseract"
_MIN_TEXT_CHARS = 200


def _get_tesseract_version() -> str:
    try:
        import pytesseract  # type: ignore[import-not-found]
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "unknown"


class OcrPaperParser:
    """Fallback parser for scanned PDFs.

    Rasterizes each page at 200 DPI via PyMuPDF, runs Tesseract OCR on
    each page image, and returns one Section per page.
    """

    @property
    def name(self) -> str:
        return _PARSER_NAME

    @property
    def version(self) -> str:
        return _get_tesseract_version()

    def parse(self, *, project_id: str, paper_path: Path) -> ParseResult:
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParseError(
                "pytesseract is not installed",
                cause_kind="ocr_failed",
                retryable=False,
            ) from exc

        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParseError(
                "PyMuPDF (fitz) is not installed",
                cause_kind="ocr_failed",
                retryable=False,
            ) from exc

        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParseError(
                "Pillow is not installed",
                cause_kind="ocr_failed",
                retryable=False,
            ) from exc

        if not paper_path.exists():
            raise ParseError(
                f"Paper path does not exist: {paper_path!s}",
                cause_kind="ocr_failed",
                retryable=False,
            )

        try:
            doc = fitz.open(str(paper_path))
        except Exception as exc:
            raise ParseError(
                f"Failed to open PDF for OCR: {exc}",
                cause_kind="ocr_failed",
                retryable=False,
            ) from exc

        try:
            # Check that pytesseract / the binary are available before processing pages.
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError as exc:
                raise ParseError(
                    "Tesseract binary not found",
                    cause_kind="ocr_failed",
                    retryable=False,
                ) from exc

            page_texts: list[str] = []
            for page in doc:
                try:
                    pixmap = page.get_pixmap(dpi=200)
                    png_bytes = pixmap.tobytes("png")
                    with Image.open(io.BytesIO(png_bytes)) as image:
                        text = pytesseract.image_to_string(image)
                    page_texts.append(text)
                except pytesseract.TesseractNotFoundError as exc:
                    raise ParseError(
                        "Tesseract binary not found during OCR",
                        cause_kind="ocr_failed",
                        retryable=False,
                    ) from exc
                except pytesseract.TesseractError as exc:
                    raise ParseError(
                        f"Tesseract OCR failed: {exc}",
                        cause_kind="ocr_failed",
                        retryable=False,
                    ) from exc
                except Exception as exc:
                    raise ParseError(
                        f"Page rasterization or image processing failed: {exc}",
                        cause_kind="ocr_failed",
                        retryable=False,
                    ) from exc
        finally:
            doc.close()

        full_text = "\n\n".join(page_texts)

        if len(full_text.strip()) < _MIN_TEXT_CHARS:
            raise ParseError(
                f"OCR produced only {len(full_text.strip())} chars (min {_MIN_TEXT_CHARS})",
                cause_kind="ocr_failed",
                retryable=False,
            )

        sections: list[Section] = []
        char_offset = 0
        for n, text in enumerate(page_texts, start=1):
            title = f"Page {n}"
            sid = section_id_for(project_id, 0, char_offset, title, text)
            sections.append(
                Section(
                    id=sid,
                    project_id=project_id,
                    title=title,
                    text=text,
                    char_offset=char_offset,
                    parent_id=None,
                    depth=0,
                )
            )
            char_offset += len(text) + 2  # +2 for "\n\n" join

        return ParseResult(
            sections=tuple(sections),
            references=[],
            figures=(),
            full_text=full_text,
        )


__all__ = ["OcrPaperParser"]
