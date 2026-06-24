"""Heterogeneous per-cell GPU counts — run_matrix and capacity_gate tests.

Feature 6c: each cell may declare ``"gpus": K`` to claim K physical GPUs for
model-parallel sharding.  When no cell uses ``"gpus"`` the schedule is
byte-equivalent to the prior uniform behaviour.

Safety/no-overlap invariant: no two concurrently-running cells may share a GPU.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import backend.agents.rlm.gpu_cell_runner as gcr
from backend.agents.rlm.cell_matrix import capacity_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub(records: list, sleep_s: float = 0.05):
    """Return a _run_cell_subprocess stub that records timing and GPU ids."""
    lock = threading.Lock()

    def stub(*, cell, cell_script, gpu_id, output_dir, batch_scale,
             grad_checkpoint, timeout_s, log_path):
        t_start = time.monotonic()
        time.sleep(sleep_s)
        t_end = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps({"status": "ok", "cell_id": cell.get("id")}), encoding="utf-8"
        )
        with lock:
            records.append((cell.get("id"), frozenset(gpu_id.split(",")), t_start, t_end))
        return 0, ""

    return stub


def _intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    """True iff [a_start, a_end] and [b_start, b_end] have a non-trivial overlap."""
    return a_start < b_end and b_start < a_end


# ---------------------------------------------------------------------------
# Safety / no-overlap test (the money invariant)
# ---------------------------------------------------------------------------

def test_no_gpu_overlap_heterogeneous(tmp_path):
    """No two concurrently-running cells may share a GPU.

    Mix of single-GPU cells and two 2-GPU cells on an 8-GPU pool.
    """
    records: list = []
    stub = _make_stub(records, sleep_s=0.06)

    cells = [{"id": f"c{i}"} for i in range(6)]
    cells.append({"id": "big", "gpus": 2})
    cells.append({"id": "big2", "gpus": 2})

    with patch.object(gcr, "_run_cell_subprocess", stub), \
         patch.object(gcr, "_load_metrics", lambda d: {"ok": 1}):
        res = gcr.run_matrix(
            cells, "x.py", output_root=str(tmp_path),
            gpus=[str(i) for i in range(8)],
        )

    # Every cell has a result
    assert set(res) == {c["id"] for c in cells}, "missing cells in results"
    assert all(r["status"] == "ok" for r in res.values()), (
        f"non-ok cells: {[(cid, r['status']) for cid, r in res.items() if r['status'] != 'ok']}"
    )

    # The 2-GPU cells each got exactly 2 distinct GPU ids
    big_ids = next(gpu_ids for cell_id, gpu_ids, _, _ in records if cell_id == "big")
    big2_ids = next(gpu_ids for cell_id, gpu_ids, _, _ in records if cell_id == "big2")
    assert len(big_ids) == 2, f"'big' cell should have 2 GPUs, got {big_ids}"
    assert len(big2_ids) == 2, f"'big2' cell should have 2 GPUs, got {big2_ids}"

    # No two overlapping cells share a GPU id
    for i, (cid_a, gpus_a, t0_a, t1_a) in enumerate(records):
        for cid_b, gpus_b, t0_b, t1_b in records[i + 1:]:
            if _intervals_overlap(t0_a, t1_a, t0_b, t1_b):
                shared = gpus_a & gpus_b
                assert not shared, (
                    f"cells '{cid_a}' and '{cid_b}' ran concurrently "
                    f"and shared GPU(s): {shared}"
                )


# ---------------------------------------------------------------------------
# Uniform byte-equivalence test
# ---------------------------------------------------------------------------

def test_uniform_gpus_per_cell_no_sharing(tmp_path):
    """When no cell declares 'gpus', single-GPU assignment, no overlap."""
    records: list = []
    stub = _make_stub(records, sleep_s=0.05)

    cells = [{"id": f"c{i}"} for i in range(8)]

    with patch.object(gcr, "_run_cell_subprocess", stub), \
         patch.object(gcr, "_load_metrics", lambda d: {"ok": 1}):
        res = gcr.run_matrix(
            cells, "x.py", output_root=str(tmp_path),
            gpus=["0", "1", "2", "3"],
            gpus_per_cell=1,
        )

    assert set(res) == {c["id"] for c in cells}
    assert all(r["status"] == "ok" for r in res.values())

    # Each cell ran on exactly ONE gpu id
    for cell_id, gpu_ids, _, _ in records:
        assert len(gpu_ids) == 1, f"cell {cell_id!r} expected 1 GPU, got {gpu_ids}"

    # No GPU is held by two temporally-overlapping cells — the real no-sharing
    # invariant. (A global "<= 4 concurrent" count measured from wall-clock
    # timestamps is timing-racy under the thread pool and intermittently observes
    # 5-6 near scheduling boundaries; the per-GPU disjointness below is what the
    # 4-GPU / gpus_per_cell=1 placement actually guarantees, and is deterministic.)
    for i, (cell_a, gpus_a, t0_a, t1_a) in enumerate(records):
        for j, (cell_b, gpus_b, t0_b, t1_b) in enumerate(records):
            if i == j:
                continue
            if _intervals_overlap(t0_a, t1_a, t0_b, t1_b):
                shared = set(gpus_a) & set(gpus_b)
                assert not shared, (
                    f"cells {cell_a!r} and {cell_b!r} overlapped in time while "
                    f"sharing GPU(s) {shared}"
                )


# ---------------------------------------------------------------------------
# capacity_gate — per-cell GPU budget tests
# ---------------------------------------------------------------------------

def _cell(model_key, env, baseline, *, est_vram_gb=None, gpus=None, seed=42):
    c = {
        "id": f"{model_key}__{baseline}__{env}__s{seed}",
        "model_key": model_key, "baseline": baseline, "env": env, "seed": seed,
    }
    if est_vram_gb is not None:
        c["est_vram_gb"] = est_vram_gb
    if gpus is not None:
        c["gpus"] = gpus
    return c


class TestCapacityGateHeterogeneous:
    def test_2gpu_cell_fits_that_wouldnt_on_1gpu(self):
        """est=60 GB, gpus=2, per_gpu=40 GB → budget=80 GB, 60×1.25=75 ≤ 80 → KEPT."""
        cells = [_cell("big", "env", "base", est_vram_gb=60.0, gpus=2)]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=40.0)
        assert len(kept) == 1
        assert skipped == []
        assert gaps == []

    def test_2gpu_cell_kept_single_gpu_cell_dropped(self):
        """Same model: gpus=2 → 80 GB budget (fits); gpus=1 → 40 GB (dropped)."""
        # gpus=1 version: 60×1.25=75 > 40 → dropped
        cells_1gpu = [_cell("big", "env", "base", est_vram_gb=60.0, gpus=1)]
        kept, gaps, skipped = capacity_gate(cells_1gpu, per_gpu_vram_gb=40.0)
        assert kept == []
        assert skipped == ["big"]

        # gpus=2 version: 60×1.25=75 ≤ 80 → kept
        cells_2gpu = [_cell("big", "env", "base", est_vram_gb=60.0, gpus=2)]
        kept2, gaps2, skipped2 = capacity_gate(cells_2gpu, per_gpu_vram_gb=40.0)
        assert len(kept2) == 1
        assert skipped2 == []

    def test_no_gpus_field_uses_default_gpus_per_cell(self):
        """Cell without 'gpus' field uses default_gpus_per_cell."""
        # default_gpus_per_cell=2 means budget = 40×2 = 80 GB → 60×1.25=75 ≤ 80 → kept
        cells = [_cell("big", "env", "base", est_vram_gb=60.0)]  # no gpus field
        kept, gaps, skipped = capacity_gate(
            cells, per_gpu_vram_gb=40.0, default_gpus_per_cell=2)
        assert len(kept) == 1
        assert skipped == []

    def test_no_gpus_field_default_1_byte_identical(self):
        """No 'gpus' field + default_gpus_per_cell=1 replicates the prior single-GPU drop."""
        # est=60, per_gpu=40, 60×1.25=75 > 40 → dropped (same as before)
        cells = [_cell("big", "env", "base", est_vram_gb=60.0)]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=40.0, default_gpus_per_cell=1)
        assert kept == []
        assert skipped == ["big"]

    def test_existing_single_gpu_behaviour_unchanged(self):
        """Prior tests: 24 GB, 7B (28 GB) dropped, 1.7B (14 GB) kept."""
        cells = [
            _cell("qwen3_1_7b", "search_qa", "sdar", est_vram_gb=14.0),
            _cell("qwen2_5_7b", "search_qa", "sdar", est_vram_gb=28.0),
        ]
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=23.68)
        assert {c["model_key"] for c in kept} == {"qwen3_1_7b"}
        assert skipped == ["qwen2_5_7b"]
        assert len(gaps) == 1
        assert "per-GPU budget" in gaps[0]["reason"]

    def test_non_numeric_gpus_falls_back_to_default(self):
        """Non-numeric 'gpus' field is treated as default_gpus_per_cell."""
        cells = [_cell("big", "env", "base", est_vram_gb=60.0)]
        cells[0]["gpus"] = "two"  # non-numeric
        # Falls back to default_gpus_per_cell=1 → 60×1.25=75 > 40 → dropped
        kept, gaps, skipped = capacity_gate(cells, per_gpu_vram_gb=40.0, default_gpus_per_cell=1)
        assert kept == []
        assert skipped == ["big"]
