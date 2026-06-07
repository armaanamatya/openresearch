"""Tests for the paper-TITLE derivation in backend.cli.

Locks in the fix for the "paper_text" garbage title: the workspace claim map
hardcodes "paper_text" as its first-entry title, so demo_status.json used to
render `paperTitle: "paper_text"` for arXiv runs (notably SDAR 2605.15155).

These exercise the pure helpers extracted out of cmd_reproduce:
  - _derive_paper_title  — precedence: real first-line > good claim title >
                           readable <id> fallback; "paper_text" never wins.
  - _first_meaningful_line / _read_paper_text_first_line — first title-like
    line of the pre-extracted full paper text (OPENRESEARCH_PAPER_TEXT_PATH).
"""

from __future__ import annotations

from backend.cli import (
    _derive_paper_title,
    _first_meaningful_line,
    _is_noise_title,
    _read_paper_text_first_line,
)

SDAR_TITLE = "Self-Distilled Agentic Reinforcement Learning"


# ---------------------------------------------------------------------------
# _derive_paper_title — the three required scenarios
# ---------------------------------------------------------------------------

def test_paper_text_claim_plus_real_first_line_returns_real_line():
    """The SDAR live-run case: claim title is the "paper_text" placeholder but a
    real first line is available → return the real line, never "paper_text"."""
    out = _derive_paper_title(
        "arxiv",
        "2605.15155",
        claim_map_first_title="paper_text",
        paper_text_first_line=SDAR_TITLE,
    )
    assert out == SDAR_TITLE


def test_junk_claim_no_text_returns_arxiv_id():
    """Junk ("paper_text") claim title + no usable text → arXiv:<id>, the
    readable fallback (never "paper_text")."""
    out = _derive_paper_title(
        "arxiv",
        "2605.15155",
        claim_map_first_title="paper_text",
        paper_text_first_line="",
    )
    assert out == "arXiv:2605.15155"


def test_good_claim_title_is_kept():
    """A genuine (non-noise) claim-map title is preserved as before."""
    out = _derive_paper_title(
        "arxiv",
        "2605.15155",
        claim_map_first_title="Attention Is All You Need",
        paper_text_first_line="",
    )
    assert out == "Attention Is All You Need"


# ---------------------------------------------------------------------------
# _derive_paper_title — precedence + placeholder rejection
# ---------------------------------------------------------------------------

def test_real_first_line_beats_good_claim_title():
    """First-line real title outranks even a good claim title."""
    out = _derive_paper_title(
        "arxiv",
        "2605.15155",
        claim_map_first_title="Some Other Decent Title",
        paper_text_first_line=SDAR_TITLE,
    )
    assert out == SDAR_TITLE


def test_never_returns_paper_text_placeholder():
    """Belt-and-suspenders: even if both sources are placeholders, the result
    falls through to the readable id — never "paper_text"."""
    out = _derive_paper_title(
        "arxiv",
        "2605.15155",
        claim_map_first_title="paper_text",
        paper_text_first_line="paper_text",
    )
    assert out == "arXiv:2605.15155"
    assert out != "paper_text"


def test_pdf_fallback_normalizes_stem():
    out = _derive_paper_title(
        "pdf",
        "my_cool-paper",
        claim_map_first_title="paper_text",
        paper_text_first_line="",
    )
    assert out == "my cool paper"


def test_doi_fallback():
    out = _derive_paper_title(
        "doi",
        "10.1000/xyz",
        claim_map_first_title="",
        paper_text_first_line="",
    )
    assert out == "doi:10.1000/xyz"


def test_arxiv_first_line_placeholder_rejected_then_id_fallback():
    """A first line that is itself a placeholder is rejected too."""
    out = _derive_paper_title(
        "arxiv",
        "2605.15155",
        claim_map_first_title="",
        paper_text_first_line="Untitled Paper",
    )
    assert out == "arXiv:2605.15155"


# ---------------------------------------------------------------------------
# _is_noise_title — placeholder set
# ---------------------------------------------------------------------------

def test_is_noise_title_rejects_placeholders():
    for junk in (
        "paper_text", "PAPER_TEXT", " paper_text ", "paper text",
        "untitled", "Untitled Paper", "title", "abstract", "Introduction",
        "1 Introduction", "summary", "overview", "",
        "document", "Document",  # F-30: no-heading HTML-parser fallback section title
    ):
        assert _is_noise_title(junk) is True, junk


def test_is_noise_title_accepts_real_title():
    assert _is_noise_title(SDAR_TITLE) is False
    assert _is_noise_title("Attention Is All You Need") is False


# ---------------------------------------------------------------------------
# _first_meaningful_line — skips provenance/junk, picks the real title
# ---------------------------------------------------------------------------

def test_first_meaningful_line_skips_arxiv_header_and_blanks():
    text = (
        "\n"
        "arXiv:2605.15155v1 [cs.LG] 28 May 2026\n"
        "\n"
        f"{SDAR_TITLE}\n"
        "Anonymous Authors\n"
    )
    assert _first_meaningful_line(text) == SDAR_TITLE


def test_first_meaningful_line_skips_bare_arxiv_id_and_digit_lines():
    text = "2605.15155\n   \n12345\n----\n" + SDAR_TITLE + "\n"
    assert _first_meaningful_line(text) == SDAR_TITLE


def test_first_meaningful_line_empty_when_no_title_like_line():
    assert _first_meaningful_line("") == ""
    assert _first_meaningful_line("\n\n   \n") == ""
    assert _first_meaningful_line("123\n....\n") == ""


def test_first_meaningful_line_rejects_overlong_first_line():
    long_para = "x" * 500 + " words here that wrap on and on"
    assert _first_meaningful_line(long_para + "\n") == ""


# ---------------------------------------------------------------------------
# _read_paper_text_first_line — fail-soft file read
# ---------------------------------------------------------------------------

def test_read_paper_text_first_line_reads_real_file(tmp_path):
    p = tmp_path / "paper_text"
    p.write_text(
        "arXiv:2605.15155v1 [cs.LG]\n\n" + SDAR_TITLE + "\n\nAbstract\n",
        encoding="utf-8",
    )
    assert _read_paper_text_first_line(str(p)) == SDAR_TITLE


def test_read_paper_text_first_line_failsoft_on_missing(tmp_path):
    assert _read_paper_text_first_line(str(tmp_path / "nope.txt")) == ""
    assert _read_paper_text_first_line(None) == ""
    assert _read_paper_text_first_line("") == ""


def test_read_paper_text_first_line_failsoft_on_directory(tmp_path):
    # A directory path must not crash — is_file() is False.
    assert _read_paper_text_first_line(str(tmp_path)) == ""
