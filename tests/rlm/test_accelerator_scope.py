"""Guardrail for REPROLAB_ACCELERATOR_SCOPE — the grader/navigator split.

The accelerator must, by default, serve only context-navigation; the rubric grader
(ctx.llm_client) stays on the strong root model so a small accelerator never decides
the score. Only scope="all" offloads the grader too.
"""
from __future__ import annotations

from backend.agents.rlm.run import _accelerator_grader_offloaded


def test_default_keeps_grader_on_root_model():
    # Unset / None → "navigation" default → grader NOT offloaded.
    assert _accelerator_grader_offloaded(None) is False
    assert _accelerator_grader_offloaded("") is False
    assert _accelerator_grader_offloaded("navigation") is False


def test_all_offloads_grader():
    assert _accelerator_grader_offloaded("all") is True


def test_normalization_case_and_whitespace():
    assert _accelerator_grader_offloaded("ALL") is True
    assert _accelerator_grader_offloaded("  all  ") is True
    assert _accelerator_grader_offloaded("All") is True


def test_unknown_values_keep_grader_strong():
    # Anything that isn't "all" is treated as navigation-only (fail-safe for quality).
    for v in ("off", "nav", "navigation", "both", "garbage", "1"):
        assert _accelerator_grader_offloaded(v) is False
