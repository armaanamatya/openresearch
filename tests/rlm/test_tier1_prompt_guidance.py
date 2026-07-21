"""Tier-1 correctness prompt guidance (Fix A + Fix B).

Fix A: the rubric-gen prompt must force each reported result to be BOUND to its
exact architecture row, so a headline number belonging to a different model
(e.g. Shake-Shake's 2.56% vs WRN-28-10's 3.08% in the Cutout paper) is not
mis-targeted as the reproduction goal.

Fix B: the baseline-implementation prompt must bound the config set to the
paper's core claim(s), so the agent does not launch unrequested grid searches
that blow the per-cell time budget (which cost the Cutout run its verdict).
"""

from __future__ import annotations

from backend.agents.rlm.rubric_gen import _SYSTEM_PROMPT
from backend.agents.prompts.baseline_implementation import (
    BASELINE_IMPLEMENTATION_PROMPT,
)


def test_rubric_gen_binds_each_result_to_its_architecture():
    p = _SYSTEM_PROMPT.lower()
    assert "one row per architecture" in p
    assert "result match" in p
    # the metric must be tied to the exact model/config that produced it
    assert "exact configuration" in p or "exact model" in p


def test_baseline_impl_bounds_the_config_set():
    p = BASELINE_IMPLEMENTATION_PROMPT.lower()
    assert "experiment scope" in p
    assert "grid search" in p  # substring of "grid searches"
    assert "minimal set" in p
