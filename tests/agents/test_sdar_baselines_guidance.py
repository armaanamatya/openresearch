"""Tests for the SDAR baseline-coverage guidance block (BES Phase 1).

Spec: docs/superpowers/specs/2026-06-07-bes-integration/phase-1-coverage-completion.md

Covers:
  - Opt-in OFF parity: with REPROLAB_SDAR_BASELINES unset, _compute_constraint_guidance
    output is byte-identical to the output with REPROLAB_SDAR_BASELINES=0 (block NOT
    injected) — absolute default-OFF behaviour preservation.
  - Opt-in ON injection: with REPROLAB_SDAR_BASELINES=1 the _SDAR_BASELINES_BLOCK is
    present and names the three missing baselines (opsd / skill_sd / rlsd), the
    populated skill_context, the curves.json artifact (gate_mean/gap/opsd_loss/reward),
    and the ZJU-REAL/SDAR provenance link.
  - Sequencing-trap guard: the injected block does NOT add ALFWorld / WebShop env
    activation guidance (Search-QA baselines + provenance/curves ONLY).

The env var is read inside _compute_constraint_guidance on every call, so no module
reload is needed between toggles (mirror of the RL-scaffold test pattern).
"""

from __future__ import annotations

import pytest

from backend.agents.baseline_implementation import (
    _SDAR_BASELINES_BLOCK,
    _compute_constraint_guidance,
)

_FLAG = "REPROLAB_SDAR_BASELINES"


# ---------------------------------------------------------------------------
# Opt-in OFF parity
# ---------------------------------------------------------------------------
def test_opt_in_off_guidance_unchanged(monkeypatch):
    """With REPROLAB_SDAR_BASELINES unset, guidance must be byte-identical to =0."""
    monkeypatch.delenv(_FLAG, raising=False)
    baseline_off = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    monkeypatch.setenv(_FLAG, "0")
    off_explicit = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert baseline_off == off_explicit, (
        "Guidance with REPROLAB_SDAR_BASELINES unset vs =0 must be identical"
    )
    # The block must be absent from both.
    assert "SDAR BASELINE COVERAGE" not in baseline_off
    assert "SDAR BASELINE COVERAGE" not in off_explicit
    assert _SDAR_BASELINES_BLOCK not in baseline_off


@pytest.mark.parametrize("falsey", ["", "0", "false", "no", "off"])
def test_falsey_values_do_not_inject(monkeypatch, falsey):
    """Only 1/true/yes inject; any other value leaves guidance block-free."""
    monkeypatch.delenv(_FLAG, raising=False)
    base = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    monkeypatch.setenv(_FLAG, falsey)
    got = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert got == base, f"REPROLAB_SDAR_BASELINES={falsey!r} must not inject the block"
    assert "SDAR BASELINE COVERAGE" not in got


# ---------------------------------------------------------------------------
# Opt-in ON injection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("truthy", ["1", "true", "yes"])
def test_opt_in_on_injects_block(monkeypatch, truthy):
    """With REPROLAB_SDAR_BASELINES truthy, the block + key strings must appear."""
    monkeypatch.setenv(_FLAG, truthy)
    guidance = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert _SDAR_BASELINES_BLOCK in guidance, "Block must be injected verbatim when opt-in"
    assert "SDAR BASELINE COVERAGE" in guidance

    # The three missing baselines, as the exact cell strings.
    assert "baseline='opsd'" in guidance
    assert "baseline='skill_sd'" in guidance
    assert "baseline='rlsd'" in guidance
    # Standalone OPSD recipe: OPSD loss only, no GRPO RL term.
    assert "grpo_weight=0.0" in guidance
    # Skill-SD: populated skill_context prompt slot.
    assert "skill_context" in guidance
    # cells.json carries the baseline; aggregate nests per_model[model][env][baseline].
    assert "cells.json" in guidance
    assert "per_model[model][env][baseline]" in guidance


def test_curves_json_guidance_present(monkeypatch):
    """Component C: per-step curves.json with the four required series."""
    monkeypatch.setenv(_FLAG, "1")
    guidance = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert "curves.json" in guidance
    for series in ("gate_mean", "gap", "opsd_loss", "reward"):
        assert series in guidance, f"curves.json series {series!r} must be named"


def test_provenance_link_present(monkeypatch):
    """Component C: explicit ZJU-REAL/SDAR reference-repo provenance link."""
    monkeypatch.setenv(_FLAG, "1")
    guidance = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert "ZJU-REAL/SDAR" in guidance
    assert "github.com/ZJU-REAL/SDAR" in guidance


def test_off_vs_on_differ(monkeypatch):
    """OFF and ON guidance must differ, and ON must be the longer (block added)."""
    monkeypatch.delenv(_FLAG, raising=False)
    off = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    monkeypatch.setenv(_FLAG, "1")
    on = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert off != on, "Guidance must differ between opt-in OFF and ON"
    assert len(on) > len(off), "ON guidance must be longer (block injected)"
    # The ON delta is EXACTLY the block: removing the block from ON yields OFF.
    # (The block is injected mid-prompt — later blocks follow it — so this is a
    # remove-and-compare, not a suffix check.)
    assert _SDAR_BASELINES_BLOCK in on
    assert on.replace(_SDAR_BASELINES_BLOCK, "", 1) == off
    assert len(on) - len(off) == len(_SDAR_BASELINES_BLOCK)


# ---------------------------------------------------------------------------
# Sequencing-trap guard — Search-QA only, NO ALFWorld/WebShop env activation
# ---------------------------------------------------------------------------
def test_block_does_not_activate_alfworld_or_webshop(monkeypatch):
    """CRITICAL: the block must not add ALFWorld/WebShop env-activation guidance.

    Activating an env that cannot learn converts excluded leaves into counted
    zeros. Phase 1 is Search-QA baselines + provenance/curves ONLY. The block's
    only mentions of ALFWorld/WebShop must be the explicit 'do NOT add' guards.
    """
    block_lower = _SDAR_BASELINES_BLOCK.lower()
    # Search-QA must be the named target env.
    assert "search-qa" in block_lower or "searchqa" in block_lower

    # Any ALFWorld/WebShop mention must be a negative instruction (a 'do NOT'
    # guard), never an activation directive.
    for env_name in ("alfworld", "webshop"):
        for line in _SDAR_BASELINES_BLOCK.splitlines():
            if env_name in line.lower():
                assert "do not" in line.lower() or "not add" in line.lower(), (
                    f"Line mentioning {env_name!r} must be a negative guard, got: {line!r}"
                )
