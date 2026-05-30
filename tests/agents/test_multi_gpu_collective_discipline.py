"""Fix B (2026-05-30): multi-GPU guidance must forbid gating a model COLLECTIVE
behind is_main_process.

A rank-0-only zero-shot generate() before accelerator.wait_for_everyone()
deadlocked all 4 ranks of prj_09047604e591d969 → 600 s NCCL timeout → SIGABRT
(exit -6). The old guidance literally said "guard one-time prep with
is_main_process then wait_for_everyone" with no scoping, which the agent
over-applied to a GPU collective. These tests pin the corrected guidance.
"""
from __future__ import annotations

from backend.agents.baseline_implementation import _RUNTIME_DETECTION_BLOCK as B


def test_collective_discipline_directive_present():
    assert "COLLECTIVE DISCIPLINE" in B
    assert "NEVER put a model call inside" in B
    assert "is_main_process" in B  # still referenced (WRONG/RIGHT contrast + I/O gating)


def test_collective_discipline_has_wrong_and_right_example():
    assert "WRONG:" in B and "RIGHT:" in B
    assert "DEADLOCK" in B
    assert "gather" in B  # the fix: gather across ranks, not a rank-0-only forward


def test_setup_once_is_scoped_to_pure_io_only():
    # The dangerous standalone "guard prep with is_main_process + wait_for_everyone"
    # is now explicitly scoped to PURE I/O (downloads), with a cross-reference that
    # a model call must NEVER be gated this way.
    assert "PURE-I/O" in B
    assert "downloads touch no" in B
    assert "NEVER gate a model call this way" in B


def test_generate_must_be_lockstep():
    assert "ALL ranks must call it together" in B
