"""P3 / §5c + superpowers B1 — the LLM grader prompt encodes the adversarial
stance: narration is an untrusted claim, score only from code/metrics, and any
positive score must cite concrete evidence. Guards against a silent revert.
"""

from __future__ import annotations

from backend.evals.paperbench.leaf_scorer import _SYSTEM_PROMPT


def test_grader_prompt_treats_narration_as_untrusted():
    p = _SYSTEM_PROMPT.lower()
    assert "unverified" in p or "optimistic" in p, "narration must be framed as an unverified claim"
    assert "narrative" in p


def test_grader_prompt_scores_from_measured_evidence():
    assert "metrics.json" in _SYSTEM_PROMPT
    assert "independently" in _SYSTEM_PROMPT.lower()


def test_grader_prompt_requires_citation_for_positive_scores():
    p = _SYSTEM_PROMPT.lower()
    assert "cite" in p, "any score >0 must cite concrete evidence (§5c)"
    # No concrete evidence ⇒ score 0.0 (the clamp instruction the grader self-applies).
    assert "0.0" in _SYSTEM_PROMPT


def test_grader_prompt_output_format_unchanged():
    """The leaf_id/score/justification contract must remain so _parse_batch_response
    keeps working (B1 is a stance change, not a format change)."""
    for field in ('"leaf_id"', '"score"', '"justification"'):
        assert field in _SYSTEM_PROMPT, field
