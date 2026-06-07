"""Tests for the cell-contract guidance (2026-05-31 OOM/GPU remediation, comp 3).

`_compute_constraint_guidance` must, on the harness-owned cell path (local/docker
backend with >=1 GPU), inject the GPU-budget brief + single-cell trainer contract
and SUPPRESS the torchrun/DDP multi-GPU guidance (which would re-create the
cuda:0 matrix-stacking that OOM'd the 2026-05-31 run). The memory-discipline block
is always-on regardless of backend.
"""

from __future__ import annotations

from backend.agents.baseline_implementation import _compute_constraint_guidance as guidance


def _cell(num_gpus=8, vram=23.68, backend="local"):
    return {"backend_kind": backend, "num_gpus": num_gpus, "per_gpu_vram_gb": vram}


def test_cell_path_injects_contract_budget_and_suppresses_torchrun():
    g = guidance("local", "max", gpu_cell_budget=_cell(), gpu_parallelism="multi", gpu_visible_count=8)
    assert "SINGLE-CELL TRAINER CONTRACT" in g
    assert "cells.json" in g and "train_cell.py" in g
    assert "per-cell budget is ONE GPU = 24 GB" in g
    assert "harness-owned cell matrix (one GPU per cell)" in g
    # The torchrun multi-GPU instruction must NOT appear — it fights one-cell-per-GPU.
    assert "torchrun --standalone" not in g
    assert "OPENRESEARCH_CELL_PARAMS" in g and "OPENRESEARCH_CELL_BATCH_SCALE" in g


def test_memory_discipline_is_always_on_even_without_gpu():
    cpu = guidance("local", "off", gpu_cell_budget=None, gpu_parallelism="auto")
    assert "GPU MEMORY DISCIPLINE" in cpu
    assert "log_softmax" in cpu  # the forbidden fp32 full-vocab pattern is named
    # No cell contract / budget brief off the cell path.
    assert "SINGLE-CELL TRAINER CONTRACT" not in cpu
    assert "per-cell budget is ONE GPU" not in cpu


def test_runpod_backend_is_not_the_cell_path():
    # Cloud uses the legacy torchrun/escalation path; the cell runner build target
    # is local/docker only. backend_kind=runpod must NOT trigger the cell contract.
    g = guidance("runpod", "max", gpu_cell_budget=_cell(backend="runpod"), gpu_parallelism="multi", gpu_visible_count=8)
    assert "SINGLE-CELL TRAINER CONTRACT" not in g


def test_zero_gpus_is_not_the_cell_path():
    g = guidance("local", "max", gpu_cell_budget=_cell(num_gpus=0), gpu_parallelism="auto")
    assert "SINGLE-CELL TRAINER CONTRACT" not in g


def test_unknown_vram_uses_conservative_budget_brief():
    g = guidance("local", "max", gpu_cell_budget=_cell(vram=0.0), gpu_parallelism="auto", gpu_visible_count=4)
    assert "SINGLE-CELL TRAINER CONTRACT" in g  # still the cell path (num_gpus>=1)
    assert "per-card VRAM unknown" in g          # conservative variant, no bogus "0 GB"
    assert "ONE GPU = 0 GB" not in g


def test_no_budget_kwarg_is_byte_identical_to_legacy_plus_memory_block():
    # Backward-compat: omitting gpu_cell_budget must not inject any cell-path text.
    g = guidance("docker", "auto", gpu_parallelism="auto", gpu_visible_count=2)
    assert "SINGLE-CELL TRAINER CONTRACT" not in g
    assert "GPU MEMORY DISCIPLINE" in g  # the one always-on addition
