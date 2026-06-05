"""P1 / #7 Unit B — curated blocklist sources (paper_hints + --blacklist union).

The arXiv reproduce path loads neither a PaperBench bundle nor --blacklist, so the
arXiv-id-keyed paper_hints blocked_resources list is what guards the canonical SDAR
run. Regex auto-derivation was rejected (it would block framework deps like trl).
"""

from __future__ import annotations

from backend.agents.prompts.paper_hints import lookup_paper_hint
from backend.agents.runtime.base import RuntimeGuard
from backend.agents.schemas import PaperHint
from backend.cli import _resolve_blocked_terms

_SDAR_REPO = "https://github.com/BartekCupial/finetuning-RL-as-CL"


def test_paperhint_blocked_resources_defaults_empty():
    assert PaperHint().blocked_resources == []


def test_sdar_paper_hint_blocks_its_own_repo():
    hint = lookup_paper_hint("2605.15155")
    assert hint is not None
    assert _SDAR_REPO in hint.blocked_resources
    # Framework deps are deliberately NOT blocked (would break the reproduction).
    joined = " ".join(hint.blocked_resources).lower()
    assert "/trl" not in joined


def test_sdar_hint_version_suffix_still_blocks():
    """arXiv ids may arrive with a version suffix; lookup normalizes them."""
    hint = lookup_paper_hint("2605.15155v2")
    assert hint is not None and _SDAR_REPO in hint.blocked_resources


def test_resolve_blocked_terms_empty_when_nothing():
    assert _resolve_blocked_terms(None, None) == []


def test_resolve_blocked_terms_blacklist_arg_commalist():
    out = _resolve_blocked_terms("github.com/a/b, github.com/c/d", None)
    assert out == ["github.com/a/b", "github.com/c/d"]


def test_resolve_blocked_terms_paper_hint_only():
    out = _resolve_blocked_terms(None, lookup_paper_hint("2605.15155"))
    assert _SDAR_REPO in out


def test_resolve_blocked_terms_union_dedupes():
    """The same repo from both --blacklist and the hint appears exactly once."""
    out = _resolve_blocked_terms(_SDAR_REPO, lookup_paper_hint("2605.15155"))
    assert out.count(_SDAR_REPO) == 1


def test_sdar_blocklist_guards_repo_not_trl():
    """End-to-end shape (§4 acceptance): the curated SDAR blocklist, seeded into a
    RuntimeGuard, blocks the paper's repo but leaves huggingface/trl reachable."""
    guard = RuntimeGuard(blocked_terms=tuple(_resolve_blocked_terms(None, lookup_paper_hint("2605.15155"))))
    assert guard.find_blocked_term(f"git clone {_SDAR_REPO}.git") is not None
    assert guard.find_blocked_term("pip install git+https://github.com/huggingface/trl") is None
