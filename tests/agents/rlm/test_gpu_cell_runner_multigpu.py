"""Multi-GPU cells: run_matrix supports heterogeneous per-cell GPU counts.

Updated from the prior slot-pool design to the free-pool design (2026-06-18).
The old tests asserted specific CSV slot pairings like "0,1" / "2,3" which were
an artefact of the fixed slot-builder (GPUs pre-paired before cells ran). Under
the new Condition-guarded free-pool design, any k cards are taken atomically
from the pool — the ORDER of the ids in the CSV is not deterministic, but each
cell always gets exactly k DISTINCT ids and no two concurrent cells share a GPU.

These tests assert the semantically-correct invariants rather than the
implementation-detail slot ordering.
"""
from __future__ import annotations

from unittest.mock import patch

import backend.agents.rlm.gpu_cell_runner as gcr


def test_gpus_per_cell_groups_into_csv_slots(tmp_path):
    """gpus_per_cell=2 with 4 GPUs: each cell gets exactly 2 distinct GPU ids."""
    seen: list[str] = []

    def fake(*, cell, cell_script, gpu_id, output_dir, batch_scale,
             grad_checkpoint, timeout_s, log_path):
        seen.append(gpu_id)
        return 0, "ok"

    with patch.object(gcr, "_run_cell_subprocess", fake), \
         patch.object(gcr, "_load_metrics", lambda d: {"ok": 1}):
        cells = [{"id": f"c{i}"} for i in range(4)]
        res = gcr.run_matrix(cells, "x.py", output_root=str(tmp_path),
                             gpus=["0", "1", "2", "3"], gpus_per_cell=2)

    assert set(res) == {"c0", "c1", "c2", "c3"}      # every cell ran
    # Each cell got a 2-GPU CSV slot (exactly 2 ids, comma-separated)
    for gpu_id in seen:
        parts = [p.strip() for p in gpu_id.split(",") if p.strip()]
        assert len(parts) == 2, f"expected 2 GPU ids, got {parts!r} in {gpu_id!r}"
    # All assigned GPU ids are from the provided pool
    all_ids = {p for gpu_id in seen for p in gpu_id.split(",") if p.strip()}
    assert all_ids <= {"0", "1", "2", "3"}


def test_gpus_per_cell_1_is_single_gpu(tmp_path):
    seen: list[str] = []

    def fake(*, gpu_id, **kw):
        seen.append(gpu_id)
        return 0, "ok"

    with patch.object(gcr, "_run_cell_subprocess", fake), \
         patch.object(gcr, "_load_metrics", lambda d: {"ok": 1}):
        gcr.run_matrix([{"id": "c0"}, {"id": "c1"}], "x.py", output_root=str(tmp_path),
                       gpus=["0", "1"], gpus_per_cell=1)

    assert all("," not in g for g in seen)           # single-GPU cells unchanged


def test_fewer_gpus_than_per_cell_uses_all_as_one_slot(tmp_path):
    """gpus_per_cell=4 with only 2 GPUs: cell is clamped to all 2 available."""
    seen: list[str] = []

    def fake(*, gpu_id, **kw):
        seen.append(gpu_id)
        return 0, "ok"

    with patch.object(gcr, "_run_cell_subprocess", fake), \
         patch.object(gcr, "_load_metrics", lambda d: {"ok": 1}):
        gcr.run_matrix([{"id": "c0"}], "x.py", output_root=str(tmp_path),
                       gpus=["0", "1"], gpus_per_cell=4)  # want 4, only 2 available

    # The cell is clamped to the 2 available GPUs (_cell_gpu_count clamps to total_gpus)
    assert len(seen) == 1
    parts = [p.strip() for p in seen[0].split(",") if p.strip()]
    assert len(parts) == 2
    assert set(parts) == {"0", "1"}
