"""Pin the EXECUTION-BUDGET AWARENESS prompt block.

When the run-budget deadline is known (RunContext.remaining_s() set), the
implement_baseline agent's prompt must surface a budget-awareness block
telling it to scale train.py to fit. Without this the agent picks epoch
counts that overrun the sandbox kill, producing zero-metric timeouts.

The injection gate is OPENRESEARCH_BUDGET_AWARENESS_MODE (auto / always / never):
- auto  → inject on cost-bearing sandboxes (gcp / azure / aws) only
- always → inject regardless of sandbox
- never  → skip regardless
"""

from __future__ import annotations

import pytest

from backend.agents.baseline_implementation import (
    _budget_awareness_block,
    _compute_constraint_guidance,
)
import backend.config as _config


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    _config._settings_cache = None
    try:
        yield
    finally:
        _config._settings_cache = None


def test_budget_block_empty_when_no_budget() -> None:
    assert _budget_awareness_block(None) == ""
    assert _budget_awareness_block(0) == ""
    assert _budget_awareness_block(-5.0) == ""


def test_budget_block_includes_rounded_seconds() -> None:
    out = _budget_awareness_block(893.7)
    assert "EXECUTION-BUDGET AWARENESS" in out
    assert "890" in out


def test_auto_mode_skips_budget_on_local_docker(monkeypatch) -> None:
    """Default mode (auto) leaves local docker prompts free of budget framing."""
    monkeypatch.setenv("OPENRESEARCH_BUDGET_AWARENESS_MODE", "auto")
    g = _compute_constraint_guidance(sandbox_mode="docker", gpu_mode=None, remaining_s=900.0)
    assert "EXECUTION-BUDGET AWARENESS" not in g


def test_always_mode_injects_budget_even_on_local_docker(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_BUDGET_AWARENESS_MODE", "always")
    g = _compute_constraint_guidance(sandbox_mode="docker", gpu_mode=None, remaining_s=900.0)
    assert "EXECUTION-BUDGET AWARENESS" in g


def test_auto_mode_injects_budget_on_billed_local_vm(monkeypatch) -> None:
    """The supported paid-GCP path is `--sandbox local --billing-sandbox gcp`
    on a VM — the sandbox string alone looks free. The campaign driver marks
    billed GPU time by setting OPENRESEARCH_MAX_RUN_GPU_USD in the child env
    (campaign_policy), and auto mode must treat that as cost-bearing."""
    monkeypatch.setenv("OPENRESEARCH_BUDGET_AWARENESS_MODE", "auto")
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "5.0")
    g = _compute_constraint_guidance(sandbox_mode="local", gpu_mode=None, remaining_s=900.0)
    assert "EXECUTION-BUDGET AWARENESS" in g


def test_auto_mode_ignores_zero_or_junk_gpu_usd_marker(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_BUDGET_AWARENESS_MODE", "auto")
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "0")
    g = _compute_constraint_guidance(sandbox_mode="local", gpu_mode=None, remaining_s=900.0)
    assert "EXECUTION-BUDGET AWARENESS" not in g
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "not-a-number")
    g = _compute_constraint_guidance(sandbox_mode="local", gpu_mode=None, remaining_s=900.0)
    assert "EXECUTION-BUDGET AWARENESS" not in g


def test_never_mode_skips_budget_even_on_gcp(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_BUDGET_AWARENESS_MODE", "never")
    g = _compute_constraint_guidance(sandbox_mode="gcp", gpu_mode=None, remaining_s=900.0)
    assert "EXECUTION-BUDGET AWARENESS" not in g


def test_compute_constraint_guidance_skips_budget_when_remaining_s_unset(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_BUDGET_AWARENESS_MODE", "always")
    g = _compute_constraint_guidance(sandbox_mode="docker", gpu_mode=None)
    assert "EXECUTION-BUDGET AWARENESS" not in g
