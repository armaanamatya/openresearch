"""Tests for the undertrained-cell postflight guard.

Mirrors the style of test_all_models_failed_guard.py.

The guard reads ``OPENRESEARCH_MIN_TRAIN_STEPS`` (int; unset/<=0 → no-op).
It checks each per-model result leaf (ok-status only) for a recorded step count
below the floor and returns ``failure_class="undertrained"`` when found.

Shapes covered: flat (per_model[model] = leaf) and nested cells-route
(per_model[model] = {env: {baseline: leaf}}).

DEFAULT-OFF behind ``OPENRESEARCH_MIN_TRAIN_STEPS`` (unset or 0 → byte-for-byte today).
"""

from __future__ import annotations

from backend.agents.rlm.primitives import (
    _OK_STATUSES,
    _undertrained_cell_violation,
)
from backend.agents.baseline_implementation import _min_train_steps_block

_FLAG = "OPENRESEARCH_MIN_TRAIN_STEPS"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _flat_metrics(step: int | None = None, status: str = "ok") -> dict:
    """Flat per_model shape (per_model[model] = leaf)."""
    leaf: dict = {"status": status, "mean_reward": 0.05}
    if step is not None:
        leaf["step"] = step
    return {"per_model": {"qwen3_1_7b": leaf}}


def _nested_metrics(step: int | None = None, status: str = "ok") -> dict:
    """Nested cells-route shape (per_model[model][env][baseline] = leaf)."""
    leaf: dict = {"status": status, "mean_reward": 0.05}
    if step is not None:
        leaf["step"] = step
    return {
        "per_model": {
            "qwen3_1_7b": {
                "alfworld": {
                    "grpo": leaf,
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# OFF-state (min_steps=0) — must be byte-for-byte None
# ---------------------------------------------------------------------------

def test_off_flat_returns_none():
    """min_steps=0 → always None (flag unset / off-state)."""
    assert _undertrained_cell_violation(_flat_metrics(step=20), min_steps=0) is None


def test_off_nested_returns_none():
    assert _undertrained_cell_violation(_nested_metrics(step=20), min_steps=0) is None


def test_negative_min_steps_returns_none():
    """Negative min_steps treated as off."""
    assert _undertrained_cell_violation(_flat_metrics(step=1), min_steps=-5) is None


# ---------------------------------------------------------------------------
# ON-state — flat shape
# ---------------------------------------------------------------------------

def test_flat_undertrained_fires():
    """20-step ok cell with floor=120 → undertrained."""
    result = _undertrained_cell_violation(_flat_metrics(step=20), min_steps=120)
    assert result is not None
    cls, msg = result
    assert cls == "undertrained"
    assert "20" in msg
    assert "120" in msg
    assert "qwen3_1_7b" in msg


def test_flat_passing_cell_returns_none():
    """150-step ok cell with floor=120 → None (above floor)."""
    assert _undertrained_cell_violation(_flat_metrics(step=150), min_steps=120) is None


def test_flat_exactly_at_floor_returns_none():
    """Exactly floor=120 → passes (>= floor, not strictly above)."""
    assert _undertrained_cell_violation(_flat_metrics(step=120), min_steps=120) is None


def test_flat_no_step_field_returns_none():
    """No step field in leaf → cannot judge → None (conservative exemption)."""
    assert _undertrained_cell_violation(_flat_metrics(step=None), min_steps=120) is None


def test_flat_failed_cell_skipped():
    """A failed/errored leaf is skipped (different guard owns it)."""
    assert _undertrained_cell_violation(_flat_metrics(step=5, status="failed"), min_steps=120) is None


def test_flat_errored_cell_skipped():
    assert _undertrained_cell_violation(_flat_metrics(step=5, status="error"), min_steps=120) is None


def test_all_ok_statuses_accepted():
    """Every _OK_STATUSES value is accepted as a 'succeeded' leaf."""
    for s in _OK_STATUSES:
        leaf = {"status": s, "step": 5}
        m = {"per_model": {"qwen3": leaf}}
        result = _undertrained_cell_violation(m, min_steps=120)
        assert result is not None, f"Expected undertrained for status={s!r}"
        assert result[0] == "undertrained"


# ---------------------------------------------------------------------------
# ON-state — nested cells-route shape
# ---------------------------------------------------------------------------

def test_nested_undertrained_fires():
    """Nested cells-route 20-step cell → undertrained (the primary SDAR case)."""
    result = _undertrained_cell_violation(_nested_metrics(step=20), min_steps=120)
    assert result is not None
    cls, msg = result
    assert cls == "undertrained"
    assert "20" in msg
    assert "120" in msg


def test_nested_passing_cell_returns_none():
    assert _undertrained_cell_violation(_nested_metrics(step=150), min_steps=120) is None


def test_nested_no_step_field_returns_none():
    assert _undertrained_cell_violation(_nested_metrics(step=None), min_steps=120) is None


def test_nested_failed_cell_skipped():
    assert _undertrained_cell_violation(_nested_metrics(step=5, status="failed"), min_steps=120) is None


# ---------------------------------------------------------------------------
# Step-field aliases
# ---------------------------------------------------------------------------

def test_step_singular_accepted():
    """``step`` (the field the SDAR cells-route trainer emits) is recognised."""
    metrics = {"per_model": {"qwen3": {"status": "ok", "step": 19}}}
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"


def test_global_step_alias_accepted():
    """``global_step`` (already in _STEP_COUNT_KEYS) is accepted."""
    metrics = {"per_model": {"qwen3": {"status": "ok", "global_step": 10}}}
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"


def test_steps_plural_alias_accepted():
    """``steps`` (plural) field is accepted."""
    metrics = {"per_model": {"qwen3": {"status": "ok", "steps": 10}}}
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"


def test_total_steps_alias_accepted():
    metrics = {"per_model": {"qwen3": {"status": "ok", "total_steps": 10}}}
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"


def test_final_step_alias_accepted():
    metrics = {"per_model": {"qwen3": {"status": "ok", "final_step": 10}}}
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"


def test_train_steps_alias_accepted():
    """``train_steps`` (from _STEP_COUNT_KEYS) is accepted."""
    metrics = {"per_model": {"qwen3": {"status": "ok", "train_steps": 10}}}
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_per_model_returns_none():
    assert _undertrained_cell_violation({}, min_steps=120) is None


def test_empty_per_model_returns_none():
    assert _undertrained_cell_violation({"per_model": {}}, min_steps=120) is None


def test_per_model_not_a_dict_returns_none():
    assert _undertrained_cell_violation({"per_model": ["qwen3"]}, min_steps=120) is None


def test_one_cell_passing_one_failing():
    """First cell passes, second undertrains → undertrained fired on the short one."""
    metrics = {
        "per_model": {
            "qwen3_7b": {"status": "ok", "step": 150},   # passes
            "qwen3_1b": {"status": "ok", "step": 10},    # undertrains
        }
    }
    result = _undertrained_cell_violation(metrics, min_steps=120)
    assert result is not None
    assert result[0] == "undertrained"
    assert "qwen3_1b" in result[1]


# ---------------------------------------------------------------------------
# Guidance block tests
# ---------------------------------------------------------------------------

def test_guidance_block_empty_when_zero():
    assert _min_train_steps_block(0) == ""


def test_guidance_block_empty_when_negative():
    assert _min_train_steps_block(-1) == ""


def test_guidance_block_contains_step_count():
    block = _min_train_steps_block(120)
    assert "120" in block
    assert len(block) > 0


def test_guidance_block_mentions_undertrained():
    """Guidance must name the failure class so the executor knows why repair is triggered."""
    block = _min_train_steps_block(120)
    assert "undertrained" in block


def test_guidance_block_advises_not_to_reduce_steps():
    """The block must steer the executor away from hardcoding a shorter step count."""
    block = _min_train_steps_block(120)
    # Should mention the alternative (batch/seq/eval) not reducing steps
    assert "batch" in block.lower() or "scale" in block.lower()
