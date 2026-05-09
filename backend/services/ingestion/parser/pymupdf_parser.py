"""PyMuPDF-backed parser (#13).

Heuristic section detection:
  1. Numbered headings: ^N(.N)*  Title$ (e.g. "3.1 Method")
  2. Known top-level names: Abstract, Introduction, Methods, Results, …
  3. Fallback: a single section "Document" containing the full text
     (so the pipeline never blocks on parse heuristics).

Reference detection scans the References section for arXiv IDs and DOIs.

Runs in-process for the slice. Subprocess isolation (spec §8.10) is a
follow-up.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from backend.services.ingestion.parser.interface import (
    ParseError,
    ParseResult,
)
from backend.services.ingestion.parser.model import (
    Reference,
    Section,
    reference_id_for,
    section_id_for,
)

if TYPE_CHECKING:  # avoid hard import for environments without fitz
    pass


_PARSER_NAME = "pymupdf"
_PARSER_VERSION = "1"

# Numbered headings: 1, 1.2, 2.1.3 — followed by a Title (ASCII letters
# + digits + spaces + a few punctuation). Anchored to line start.
_HEADING_NUMBERED = re.compile(
    r"^(?P<num>\d+(?:\.\d+){0,3})\s+(?P<title>[A-Z][A-Za-z0-9 \-:&]{2,80})\s*$"
)

# Top-level conventional names; case-insensitive on the keyword but the
# line must be just the keyword (and maybe a trailing colon).
_HEADING_NAMED = re.compile(
    r"^(?P<title>(?:Abstract|Introduction|Background|Related\s+Work|"
    r"Method(?:s|ology)?|Approach|Experiments?|Results|Evaluation|"
    r"Discussion|Conclusions?|References|Bibliography|Appendix|"
    r"Acknowledgments?|Limitations))\s*:?\s*$",
    re.IGNORECASE,
)

# References section detection (case-insensitive, line-anchored).
_REF_HEADING = re.compile(r"^(References|Bibliography)\s*$", re.IGNORECASE)
_REF_LABEL_LINE = re.compile(
    r"^\s*(?:\[[^\]\n]{1,40}\]|\d{1,4}[.)])\s+"
)

# Patterns for individual references.
_ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\)]+)\b")


class PyMuPdfParser:
    """In-process parser using `fitz` (PyMuPDF)."""

    @property
    def name(self) -> str:
        return _PARSER_NAME

    @property
    def version(self) -> str:
        return _PARSER_VERSION

    def parse(self, *, project_id: str, paper_path: Path) -> ParseResult:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParseError(
                "PyMuPDF (`pymupdf` package) is not installed",
                cause_kind="pymupdf_missing",
                retryable=False,
            ) from exc

        if not paper_path.exists():
            raise ParseError(
                f"Paper path does not exist: {paper_path!s}",
                cause_kind="paper_missing",
                retryable=True,
            )

        try:
            doc = fitz.open(str(paper_path))
        except Exception as exc:  # PyMuPDF raises various exception types
            raise ParseError(
                f"Failed to open PDF: {exc}",
                cause_kind="pdf_open_failed",
                retryable=False,
            ) from exc

        full_text = self._extract_full_text(doc)
        sections = self._extract_sections(project_id, full_text)
        references = self._extract_references(project_id, full_text)

        # Sort sections deterministically: depth first (top-level before
        # subsections), then char_offset to preserve document order.
        # SectionChunker (commit 3) repeats this sort; doing it here makes
        # the parser's emit order itself deterministic, which is what the
        # event log captures.
        sorted_sections = sorted(sections, key=lambda s: (s.depth, s.char_offset, s.id))

        return ParseResult(
            sections=tuple(sorted_sections),
            references=tuple(references),
            figures=(),  # not extracted in this slice
            full_text=full_text,
        )

    # --- Internal -----------------------------------------------------------

    @staticmethod
    def _extract_full_text(doc: object) -> str:
        pages: list[str] = []
        for page in doc:  # type: ignore[union-attr]
            pages.append(page.get_text("text"))  # type: ignore[union-attr]
        return "\n".join(pages)

    @staticmethod
    def _extract_sections(project_id: str, full_text: str) -> list[Section]:
        lines = full_text.split("\n")
        # Pass 1: identify heading positions.
        heading_indices: list[tuple[int, str, int]] = []  # (line_idx, title, depth)
        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            m = _HEADING_NUMBERED.match(line)
            if m:
                num = m.group("num")
                depth = num.count(".") + 1  # "3" -> depth 1; "3.1" -> depth 2
                title = f"{num} {m.group('title').strip()}"
                heading_indices.append((i, title, depth))
                continue
            m = _HEADING_NAMED.match(line)
            if m:
                title = m.group("title").strip()
                # Normalize title casing for known names.
                title = title.title()
                heading_indices.append((i, title, 1))

        # Compute char_offset per line for offset attribution.
        char_offsets: list[int] = []
        running = 0
        for raw_line in lines:
            char_offsets.append(running)
            running += len(raw_line) + 1  # +1 for the '\n' join

        if not heading_indices:
            # Fallback: one big section. Better than blocking the pipeline.
            sid = section_id_for(project_id, 0, 0, "Document", full_text)
            return [
                Section(
                    id=sid,
                    project_id=project_id,
                    title="Document",
                    text=full_text,
                    char_offset=0,
                    parent_id=None,
                    depth=0,
                )
            ]

        # Pass 2: build sections from each heading to the next heading's start.
        sections: list[Section] = []
        for k, (line_idx, title, depth) in enumerate(heading_indices):
            start_line = line_idx + 1  # body starts after heading line
            end_line = (
                heading_indices[k + 1][0] if k + 1 < len(heading_indices) else len(lines)
            )
            body_lines = lines[start_line:end_line]
            body_text = "\n".join(body_lines).strip()
            char_offset = char_offsets[line_idx]
            sid = section_id_for(project_id, depth, char_offset, title, body_text)
            sections.append(
                Section(
                    id=sid,
                    project_id=project_id,
                    title=title,
                    text=body_text,
                    char_offset=char_offset,
                    parent_id=None,  # parent linkage deferred — flat sections fine for slice
                    depth=depth,
                )
            )
        return sections

    @staticmethod
    def _extract_references(project_id: str, full_text: str) -> list[Reference]:
        # Find the References / Bibliography heading and slice the rest.
        lines = full_text.split("\n")
        ref_start = None
        for i, raw in enumerate(lines):
            if _REF_HEADING.match(raw.strip()):
                ref_start = i + 1
                break
        if ref_start is None:
            return []

        ref_text = "\n".join(lines[ref_start:])
        blocks = PyMuPdfParser._split_reference_blocks(ref_text)

        refs: list[Reference] = []
        for block in blocks:
            arxiv_match = _ARXIV_RE.search(block)
            doi_match = _DOI_RE.search(block)
            if not arxiv_match and not doi_match:
                # Skip lines without a citation ID — too noisy otherwise.
                continue
            ref_id = reference_id_for(project_id, block)
            refs.append(
                Reference(
                    id=ref_id,
                    project_id=project_id,
                    raw_text=block,
                    arxiv_id=arxiv_match.group(1) if arxiv_match else None,
                    doi=doi_match.group(1) if doi_match else None,
                    title=None,
                )
            )
        return refs

    @staticmethod
    def _split_reference_blocks(ref_text: str) -> list[str]:
        """Split a references section into likely individual citations.

        PyMuPDF often collapses blank lines from generated or dense PDFs,
        so relying only on paragraph breaks can merge adjacent references.
        Start a new block when a line begins with a conventional reference
        label, while preserving wrapped continuation lines.
        """
        blocks: list[str] = []
        current: list[str] = []
        saw_label = False

        for raw_line in ref_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if current:
                    current.append("")
                continue

            starts_labeled_ref = _REF_LABEL_LINE.match(stripped) is not None
            if starts_labeled_ref:
                saw_label = True
                if current:
                    blocks.append("\n".join(current).strip())
                current = [stripped]
                continue

            current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        if saw_label:
            return [block for block in blocks if block]

        # Fallback for unlabeled bibliographies: paragraphs separated by
        # blank lines are the least surprising unit.
        return [b.strip() for b in re.split(r"\n\s*\n", ref_text) if b.strip()]


__all__ = ["PyMuPdfParser"]
