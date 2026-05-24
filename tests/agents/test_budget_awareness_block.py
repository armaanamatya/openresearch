"""Pin the EXECUTION-BUDGET AWARENESS prompt block.

When the run-budget deadline is known (RunContext.remaining_s() set), the
implement_baseline agent's prompt must surface a budget-awareness block
telling it to scale train.py to fit. Without this the agent picks epoch
counts that overrun the sandbox kill, producing zero-metric timeouts.
"""

from __future__ import annotations

from backend.agents.baseline_implementation import (
    _budget_awareness_block,
    _compute_constraint_guidance,
)


def test_budget_block_empty_when_no_budget() -> None:
    assert _budget_awareness_block(None) == ""
    assert _budget_awareness_block(0) == ""
    assert _budget_awareness_block(-5.0) == ""


def test_budget_block_includes_rounded_seconds() -> None:
    out = _budget_awareness_block(893.7)
    assert "EXECUTION-BUDGET AWARENESS" in out
    # 893.7 rounds down to 890.
    assert "890" in out


def test_budget_block_threaded_into_compute_constraint_guidance() -> None:
    g = _compute_constraint_guidance(
        sandbox_mode="docker", gpu_mode=None, remaining_s=900.0
    )
    assert "EXECUTION-BUDGET AWARENESS" in g
    assert "900" in g
    # Other always-on blocks still present.
    assert "NO STUB" in g or "STUB" in g.upper()


def test_compute_constraint_guidance_skips_budget_when_not_provided() -> None:
    g = _compute_constraint_guidance(sandbox_mode="docker", gpu_mode=None)
    assert "EXECUTION-BUDGET AWARENESS" not in g
