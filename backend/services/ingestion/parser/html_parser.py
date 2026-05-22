"""ArXiv HTML parser — extracts clean prose from LaTeXML-rendered HTML.

arXiv publishes a HTML version at https://arxiv.org/html/<id> for most
recent papers. Figures are rendered as images so they don't pollute the
text layer; the prose is clean and well-structured via <section> elements.

Section extraction is slice-based: the full document prose is built in a
single pass and sections are derived as non-overlapping slices of that text
anchored at heading boundaries, so concatenating all section texts
reconstructs full_text exactly with no duplication or loss.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.services.ingestion.parser.interface import ParseError, ParseResult
from backend.services.ingestion.parser.model import (
    Section,
    section_id_for,
)

logger = logging.getLogger(__name__)

_PARSER_NAME = "arxiv-html"
_PARSER_VERSION = "1.0"

# Elements whose entire subtree should be dropped (noise / non-prose).
_DROP_TAGS = {"script", "style", "nav", "header", "footer", "button", "svg", "figure", "img", "math"}

# Heading tag → depth mapping.
_HEADING_DEPTH: dict[str, int] = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}

_MIN_TEXT_CHARS = 500


class HtmlPaperParser:
    """Parser for arXiv LaTeXML HTML pages.

    Expects `paper_path` to point to an HTML file (e.g. raw_paper.html).
    Sections are derived as slices of the single-pass full_text build so
    section texts tile full_text exactly (no duplication, no loss).
    """

    @property
    def name(self) -> str:
        return _PARSER_NAME

    @property
    def version(self) -> str:
        return _PARSER_VERSION

    def parse(self, *, project_id: str, paper_path: Path) -> ParseResult:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParseError(
                "beautifulsoup4 is not installed",
                cause_kind="html_parse_failed",
                retryable=False,
            ) from exc

        if not paper_path.exists():
            raise ParseError(
                f"HTML path does not exist: {paper_path!s}",
                cause_kind="html_parse_failed",
                retryable=False,
            )

        try:
            html = paper_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ParseError(
                f"Could not read HTML file: {exc}",
                cause_kind="html_parse_failed",
                retryable=False,
            ) from exc

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            raise ParseError(
                f"BeautifulSoup failed to parse HTML: {exc}",
                cause_kind="html_parse_failed",
                retryable=False,
            ) from exc

        # Drop noise subtrees — but preserve figcaption text (useful prose).
        # For each <figure>, extract its <figcaption> text and inject a <p>
        # sibling after the <figure> so the caption survives the decompose.
        for fig in soup.find_all("figure"):
            cap = fig.find("figcaption")
            if cap:
                ct = cap.get_text(separator=" ", strip=True)
                if ct:
                    new_p = soup.new_tag("p")
                    new_p.string = ct
                    fig.insert_after(new_p)

        for tag_name in _DROP_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        full_text, sections = self._build_text_and_sections(project_id, soup)

        if len(full_text.strip()) < _MIN_TEXT_CHARS:
            raise ParseError(
                f"HTML yielded only {len(full_text.strip())} chars of text (min {_MIN_TEXT_CHARS})",
                cause_kind="html_parse_failed",
                retryable=False,
            )

        sorted_sections = sorted(sections, key=lambda s: (s.depth, s.char_offset, s.id))
        return ParseResult(
            sections=tuple(sorted_sections),
            references=(),
            figures=(),
            full_text=full_text,
        )

    # --- Internal -----------------------------------------------------------

    @staticmethod
    def _build_text_and_sections(project_id: str, soup: object) -> tuple[str, list[Section]]:
        """Single-pass slice-based section extraction.

        Walk the cleaned main content in document order. Body text tokens
        are accumulated into a flat list; when a heading (h1–h4) is
        encountered, record a boundary at the current accumulated character
        count. Heading text itself is NOT added to the body.

        After the walk, join tokens to get full_text. Sections are derived
        as exact slices of full_text between consecutive boundary offsets.

        Invariant: "".join(s.text for s in sections) == full_text when
        boundaries are chosen so slices partition the string without gaps.
        More precisely, the section texts tile full_text: each section's
        text is full_text[s.char_offset : next_offset] and the slices cover
        [0, len(full_text)] without overlap or gap.
        """
        from bs4 import NavigableString, Tag  # type: ignore[import-not-found]

        # Pick the main content container.
        root = (
            soup.find("article")  # type: ignore[union-attr]
            or soup.find("body")
            or soup
        )

        # tokens: list of (text_piece, is_boundary, boundary_title, boundary_depth)
        # We interleave boundary markers with text tokens.
        # Each boundary records the running character length so far (before
        # the space that would separate this token from the next).
        # Layout: full_text = " ".join(tokens); a boundary at position k means
        # the next section starts at len(" ".join(tokens[:k])) + (1 if k>0).
        #
        # Simpler: build text_parts (only text, not boundaries) and boundaries
        # as (text_parts_index_before_which_heading_appeared, title, depth).
        text_parts: list[str] = []
        # (index into text_parts at which the heading boundary appears, title, depth)
        boundaries: list[tuple[int, str, int]] = []

        def _walk(node: object) -> None:
            if isinstance(node, NavigableString):
                t = " ".join(str(node).split())  # normalize whitespace
                if t:
                    text_parts.append(t)
            elif isinstance(node, Tag):
                if node.name in _HEADING_DEPTH:
                    title = node.get_text(separator=" ", strip=True)
                    depth = _HEADING_DEPTH[node.name]
                    # Boundary: the next section starts at text_parts[len(text_parts)]
                    boundaries.append((len(text_parts), title, depth))
                else:
                    for child in node.children:
                        _walk(child)

        for child in root.children:  # type: ignore[union-attr]
            _walk(child)

        full_text = " ".join(text_parts)

        def _char_offset(parts_idx: int) -> int:
            """Character offset in full_text where text_parts[parts_idx] begins."""
            if parts_idx == 0:
                return 0
            if parts_idx >= len(text_parts):
                return len(full_text)
            # " ".join(text_parts[:parts_idx]) is the prefix; the next part
            # begins at len(prefix) + 1 (for the joining space).
            return len(" ".join(text_parts[:parts_idx])) + 1

        sections: list[Section] = []

        if not boundaries:
            # No headings — one monolithic section containing all prose.
            sid = section_id_for(project_id, 0, 0, "Document", full_text)
            return full_text, [
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

        # Text before the first heading → "Abstract" section at offset 0.
        first_offset = _char_offset(boundaries[0][0])
        abstract_text = full_text[:first_offset]
        if abstract_text:
            sid = section_id_for(project_id, 0, 0, "Abstract", abstract_text)
            sections.append(
                Section(
                    id=sid,
                    project_id=project_id,
                    title="Abstract",
                    text=abstract_text,
                    char_offset=0,
                    parent_id=None,
                    depth=0,
                )
            )

        # Sections from each heading boundary to the next.
        n = len(boundaries)
        for k, (parts_idx, title, depth) in enumerate(boundaries):
            this_offset = _char_offset(parts_idx)
            if k + 1 < n:
                next_offset = _char_offset(boundaries[k + 1][0])
            else:
                next_offset = len(full_text)
            section_text = full_text[this_offset:next_offset]

            sid = section_id_for(project_id, depth, this_offset, title, section_text)
            sections.append(
                Section(
                    id=sid,
                    project_id=project_id,
                    title=title,
                    text=section_text,
                    char_offset=this_offset,
                    parent_id=None,
                    depth=depth,
                )
            )

        return full_text, sections


__all__ = ["HtmlPaperParser"]
