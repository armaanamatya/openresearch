"""Tests for the CLI paperTitle noise filter (F-30).

The corpus claim-map entry hard-codes title='paper_text' (_one_entry); the
no-heading HTML parser uses 'Document'. Neither is a real paper title, so the
demo_status paperTitle upgrade must reject both as noise — otherwise the lab UI
shows the literal placeholder 'paper_text' (the visible W-1 symptom).
"""
from __future__ import annotations

from backend.cli import _is_noise_title


def test_is_noise_title_rejects_paper_text_placeholder() -> None:
    # F-30: the literal corpus-entry placeholder must never become the title.
    assert _is_noise_title("paper_text") is True
    assert _is_noise_title("PAPER_TEXT") is True  # case-insensitive
    # The no-heading HTML fallback section title is also a placeholder.
    assert _is_noise_title("Document") is True


def test_is_noise_title_still_rejects_existing_noise() -> None:
    for noise in ("Abstract", "Introduction", "1 Introduction", "1. Introduction",
                  "Summary", "Overview", ""):
        assert _is_noise_title(noise) is True
    assert _is_noise_title(None) is True


def test_is_noise_title_keeps_a_real_title() -> None:
    assert _is_noise_title("Self-Distilled Agentic Reinforcement Learning") is False
    assert _is_noise_title("Attention Is All You Need") is False
