"""ResolvingParser — best-of-N with HTML preference.

Always evaluates HTML (when the raw_paper.html sibling exists) and PDF.
OCR is attempted only when both HTML and PDF score below the _USABLE
threshold.

Selection logic:
  1. Among strategies that reached _USABLE, pick the highest score.
     On a tie, HTML beats PDF beats OCR (clean arXiv HTML reliably
     out-scores figure-noisy PDF, so HTML wins when equally good).
  2. If none reached _USABLE: pick the highest-scoring result whose score
     is > 0.0.
  3. If every strategy returned score 0.0 (empty / < 1000 chars) OR every
     strategy raised ParseError: raise ParseError(cause_kind="all_strategies_failed").

`score_text_quality` is a module-level helper used by both this resolver and
tests; it scores clean prose (wordish token ratio) and returns 0.0 for very
short text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.services.ingestion.parser.html_parser import HtmlPaperParser
from backend.services.ingestion.parser.interface import ParseError, ParseResult
from backend.services.ingestion.parser.ocr_parser import OcrPaperParser
from backend.services.ingestion.parser.pymupdf_parser import PyMuPdfParser

logger = logging.getLogger(__name__)

_PARSER_NAME = "resolving"
_PARSER_VERSION = "1.0"
_USABLE = 0.35

# A source carrying at least this many chars of real text makes slow OCR
# unnecessary — OCR is the scanned-PDF fallback, not a short-document fallback.
_MIN_USEFUL_CHARS = 200

# Compiled once; reused inside score_text_quality for efficiency.
_WORDISH_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")

# Tiebreak priority: lower index = higher preference.
_STRATEGY_PRIORITY: dict[str, int] = {"html": 0, "pdf": 1, "ocr": 2}


def score_text_quality(text: str) -> float:
    """Score text quality as the fraction of word-like tokens.

    Returns 0.0 for text shorter than 1 000 chars or with no tokens.
    Clean prose scores ~0.6–0.85; figure-noise-heavy text scores lower.
    This is a heuristic wordish-token ratio adequate for ranking clean HTML
    vs figure-noisy PDF — not a semantic quality measure.
    """
    tokens = text.split()
    if len(text.strip()) < 1000 or not tokens:
        return 0.0
    wordish = sum(1 for t in tokens if _WORDISH_RE.fullmatch(t))
    return wordish / len(tokens)


class ResolvingParser:
    """Composite Parser that picks the highest-quality source automatically.

    Evaluates HTML and PDF in parallel (both always attempted when HTML
    sibling exists). OCR is the last resort, only tried when both HTML and
    PDF score below _USABLE. The winning strategy is the one with the
    highest quality score among those that passed _USABLE; HTML beats PDF
    beats OCR on ties.

    Constructor accepts optional injected sub-parsers; default-constructs
    each one when not supplied (for production use). Inject stub parsers
    in tests for fast, hermetic coverage.
    """

    def __init__(
        self,
        *,
        html_parser: HtmlPaperParser | None = None,
        pdf_parser: PyMuPdfParser | None = None,
        ocr_parser: OcrPaperParser | None = None,
    ) -> None:
        self._html = html_parser if html_parser is not None else HtmlPaperParser()
        self._pdf = pdf_parser if pdf_parser is not None else PyMuPdfParser()
        self._ocr = ocr_parser if ocr_parser is not None else OcrPaperParser()

    @property
    def name(self) -> str:
        return _PARSER_NAME

    @property
    def version(self) -> str:
        return _PARSER_VERSION

    def parse(self, *, project_id: str, paper_path: Path) -> ParseResult:
        """Best-of-N parse: HTML + PDF always; OCR only when both fail _USABLE.

        Selection:
          - Among strategies scoring >= _USABLE, pick the highest; HTML beats
            PDF beats OCR on exact ties.
          - If none pass _USABLE, pick the highest score > 0.0.
          - If all scores are 0.0 or all strategies raised ParseError:
            raise ParseError(cause_kind="all_strategies_failed", retryable=False).
        """
        # Candidate list: (strategy_name, result_or_None, score, error_or_None)
        candidates: list[tuple[str, ParseResult | None, float, BaseException | None]] = []

        # 1. HTML sibling — always attempt if the file exists.
        html_path = paper_path.with_name("raw_paper.html")
        html_result: ParseResult | None = None
        html_score = 0.0
        html_error: BaseException | None = None
        if html_path.exists():
            try:
                html_result = self._html.parse(project_id=project_id, paper_path=html_path)
                html_score = score_text_quality(html_result.full_text)
            except ParseError as exc:
                html_error = exc
                logger.debug("HTML parser failed: %s", exc)
        else:
            html_error = ParseError("No HTML sibling found", cause_kind="html_parse_failed", retryable=False)
        candidates.append(("html", html_result, html_score, html_error))

        # 2. PDF (PyMuPDF) — always attempt.
        pdf_result: ParseResult | None = None
        pdf_score = 0.0
        pdf_error: BaseException | None = None
        try:
            pdf_result = self._pdf.parse(project_id=project_id, paper_path=paper_path)
            pdf_score = score_text_quality(pdf_result.full_text)
        except ParseError as exc:
            pdf_error = exc
            logger.debug("PDF parser failed: %s", exc)
        candidates.append(("pdf", pdf_result, pdf_score, pdf_error))

        # 3. OCR — the scanned-PDF fallback. Skip it whenever HTML or PDF
        # already gave a usable score OR a non-trivial amount of real text:
        # a short paper is not a scanned one, and OCR is slow.
        ocr_skipped = (
            html_score >= _USABLE
            or pdf_score >= _USABLE
            or (html_result is not None and len(html_result.full_text.strip()) >= _MIN_USEFUL_CHARS)
            or (pdf_result is not None and len(pdf_result.full_text.strip()) >= _MIN_USEFUL_CHARS)
        )
        ocr_result: ParseResult | None = None
        ocr_score = 0.0
        ocr_error: BaseException | None = None
        if not ocr_skipped:
            try:
                ocr_result = self._ocr.parse(project_id=project_id, paper_path=paper_path)
                ocr_score = score_text_quality(ocr_result.full_text)
            except ParseError as exc:
                ocr_error = exc
                logger.debug("OCR parser failed: %s", exc)
        else:
            ocr_error = ParseError("OCR skipped (HTML or PDF passed threshold)", cause_kind="ocr_skipped", retryable=False)
        candidates.append(("ocr", ocr_result, ocr_score, ocr_error))

        # 4. Choose best result.
        chosen_name, chosen_result = self._choose(candidates)
        self._log_choice(chosen_name, candidates, ocr_skipped=ocr_skipped)
        return chosen_result

    @staticmethod
    def _choose(
        candidates: list[tuple[str, ParseResult | None, float, BaseException | None]],
    ) -> tuple[str, ParseResult]:
        """Select the best result from candidates.

        Priority: highest score among >= _USABLE; on tie, HTML > PDF > OCR.
        Fallback: the best result that still carries non-empty text — a short
        document scores 0.0 (the < 1000-char rule) but is a valid parse.
        Failure: every result is empty / every strategy raised → ParseError.
        """
        # Usable candidates (score >= _USABLE and result is not None).
        usable = [
            (name, result, score)
            for name, result, score, _ in candidates
            if result is not None and score >= _USABLE
        ]
        if usable:
            # Sort: descending score; on tie, ascending priority (html=0 wins).
            best_name, best_result, _ = min(
                usable,
                key=lambda t: (-t[2], _STRATEGY_PRIORITY.get(t[0], 99)),
            )
            return best_name, best_result  # type: ignore[return-value]

        # No strategy reached _USABLE — fall back to the best result that still
        # carries real text. A short document scores 0.0 (score_text_quality's
        # < 1000-char rule) but is a valid parse, not a failure; only a
        # genuinely empty parse is a failure. Rank by score, then text length.
        non_empty = [
            (name, result, score)
            for name, result, score, _ in candidates
            if result is not None and result.full_text.strip()
        ]
        if non_empty:
            best_name, best_result, _ = min(
                non_empty,
                key=lambda t: (-t[2], -len(t[1].full_text), _STRATEGY_PRIORITY.get(t[0], 99)),
            )
            return best_name, best_result  # type: ignore[return-value]

        # Every strategy produced empty text or raised ParseError.
        raise ParseError(
            "All parsing strategies failed for this paper",
            cause_kind="all_strategies_failed",
            retryable=False,
        )

    @staticmethod
    def _log_choice(
        chosen: str | None,
        candidates: list[tuple[str, ParseResult | None, float, BaseException | None]],
        *,
        ocr_skipped: bool,
    ) -> None:
        """Emit an INFO line with the chosen strategy and all candidate scores."""
        score_parts: list[str] = []
        for name, result, score, _ in candidates:
            if ocr_skipped and name == "ocr":
                score_parts.append("ocr=skipped")
            elif result is not None:
                score_parts.append(f"{name}={score:.2f}")
            else:
                score_parts.append(f"{name}=err")
        scores_str = " ".join(score_parts)
        logger.info("resolving parser: chose %s (scores: %s)", chosen or "none", scores_str)


__all__ = ["ResolvingParser", "score_text_quality"]
