"""PaperExtractor — optional augmentation pass that runs after the base Parser.

A PaperExtractor enriches a ParseResult with vision descriptions for
figures/tables/equations and OCR text for scanned (image-only) pages whose
embedded text layer is empty.

Contract: ``extract()`` MUST NEVER raise. A parse that succeeded must never
be turned into a failure by augmentation. On any error, ``base`` is returned
unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from backend.services.ingestion.parser.interface import ParseResult
from backend.services.ingestion.parser.model import Figure
from backend.services.ingestion.parser.vision import ClaudeVisionClient

logger = logging.getLogger(__name__)

# A page is a vision candidate if its embedded-text length is below this
# threshold — i.e. it is effectively scanned/image-only.
TEXT_DENSITY_THRESHOLD = 120

# Cap total vision API calls per document to bound cost and latency.
MAX_VISION_PAGES = 12


@runtime_checkable
class PaperExtractor(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...

    def extract(
        self,
        *,
        project_id: str,
        paper_path: Path,
        base: ParseResult,
    ) -> ParseResult:
        """Return an enriched ParseResult. MUST NOT raise under any circumstances."""


class NullExtractor:
    """No-op extractor. Preserves exact pre-feature behaviour."""

    @property
    def name(self) -> str:
        return "null"

    @property
    def version(self) -> str:
        return "1"

    def extract(
        self,
        *,
        project_id: str,
        paper_path: Path,
        base: ParseResult,
    ) -> ParseResult:
        return base


class VisionAugmentingExtractor:
    """Enriches ParseResult with per-page Claude vision descriptions.

    For each candidate page (scanned or referenced by a Figure), renders
    the page to PNG and calls the vision client. Merges results into a
    new ParseResult:
      - scanned-page transcriptions are appended to full_text
      - a "Visual Content" appendix section is appended listing per-page
        figure/table/equation descriptions
      - Figure objects are enriched with description text
    """

    @property
    def name(self) -> str:
        return "vision-augmenting"

    @property
    def version(self) -> str:
        return "1"

    def __init__(self, vision_client: ClaudeVisionClient) -> None:
        self._client = vision_client

    def extract(
        self,
        *,
        project_id: str,
        paper_path: Path,
        base: ParseResult,
    ) -> ParseResult:
        try:
            return self._extract(project_id=project_id, paper_path=paper_path, base=base)
        except Exception:
            logger.exception(
                "VisionAugmentingExtractor.extract failed for project %s; "
                "returning base ParseResult unchanged",
                project_id,
            )
            return base

    def _extract(
        self,
        *,
        project_id: str,
        paper_path: Path,
        base: ParseResult,
    ) -> ParseResult:
        if not self._client.is_available():
            logger.debug(
                "Vision client not available for project %s; skipping augmentation",
                project_id,
            )
            return base

        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("fitz (PyMuPDF) not importable; skipping vision augmentation")
            return base

        doc = fitz.open(str(paper_path))
        try:
            return self._process(doc, project_id=project_id, base=base)
        finally:
            doc.close()

    def _process(
        self,
        doc: object,
        *,
        project_id: str,
        base: ParseResult,
    ) -> ParseResult:

        # Pages referenced by existing figures (0-indexed).
        figure_pages: set[int] = {fig.page for fig in base.figures}

        # Build candidate list in page order, capped at MAX_VISION_PAGES.
        candidates: list[int] = []
        for page_obj in doc:  # type: ignore[union-attr]
            page_idx: int = page_obj.number  # 0-based
            text_len = len(page_obj.get_text("text").strip())
            is_scanned = text_len < TEXT_DENSITY_THRESHOLD
            if is_scanned or page_idx in figure_pages:
                candidates.append(page_idx)
            if len(candidates) >= MAX_VISION_PAGES:
                break

        if not candidates:
            return base

        # Collect per-page text hints from the base full_text is not
        # page-segmented, so we re-read directly from the doc for hints.
        page_texts: dict[int, str] = {}
        for page_obj in doc:  # type: ignore[union-attr]
            page_texts[page_obj.number] = page_obj.get_text("text").strip()

        vision_results: dict[int, str] = {}
        for page_idx in candidates:
            try:
                page_obj = doc[page_idx]  # type: ignore[index]
                png_bytes: bytes = page_obj.get_pixmap(dpi=150).tobytes("png")
                description = self._client.describe_page(
                    png_bytes=png_bytes,
                    page_number=page_idx + 1,  # 1-based for the prompt
                    text_hint=page_texts.get(page_idx, ""),
                )
                vision_results[page_idx] = description
            except Exception:
                logger.exception(
                    "Vision call failed for page %d of project %s; skipping page",
                    page_idx,
                    project_id,
                )

        if not vision_results:
            return base

        # Build enriched full_text.
        extra_text_parts: list[str] = []

        # (a) Scanned-page transcriptions appended to full_text.
        for page_idx in sorted(vision_results):
            if page_texts.get(page_idx, "") == "" or len(page_texts.get(page_idx, "")) < TEXT_DENSITY_THRESHOLD:
                extra_text_parts.append(
                    f"\n[Page {page_idx + 1} — vision transcription]\n"
                    + vision_results[page_idx]
                )

        # (b) Visual Content appendix.
        appendix_lines = ["\n\n[Visual Content]"]
        for page_idx in sorted(vision_results):
            appendix_lines.append(
                f"\nPage {page_idx + 1}:\n" + vision_results[page_idx]
            )
        extra_text_parts.append("\n".join(appendix_lines))

        new_full_text = base.full_text + "".join(extra_text_parts)

        # (c) Enrich figures with descriptions — match by page.
        # Build a mapping from page → description text for quick lookup.
        new_figures: list[Figure] = []
        for fig in base.figures:
            desc = vision_results.get(fig.page)
            if desc is not None:
                new_figures.append(fig.model_copy(update={"description": desc}))
            else:
                new_figures.append(fig)

        return ParseResult(
            sections=base.sections,
            references=base.references,
            figures=tuple(new_figures),
            full_text=new_full_text,
        )


def extractor_from_settings(settings: object) -> PaperExtractor:
    """Factory: returns the extractor appropriate for the current settings."""
    mode = getattr(settings, "paper_extraction_mode", "text")
    if mode == "hybrid":
        model = getattr(settings, "paper_extraction_vision_model", "claude-sonnet-4-6")
        return VisionAugmentingExtractor(ClaudeVisionClient(model=model))
    return NullExtractor()


__all__ = [
    "MAX_VISION_PAGES",
    "NullExtractor",
    "PaperExtractor",
    "TEXT_DENSITY_THRESHOLD",
    "VisionAugmentingExtractor",
    "extractor_from_settings",
]
