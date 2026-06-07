"""Multi-GPU cells: run_matrix groups GPUs into slots of `gpus_per_cell` so a cell
can device_map-shard a large model across several cards (2026-06-02)."""
from __future__ import annotations

from unittest.mock import patch

import backend.agents.rlm.gpu_cell_runner as gcr


def test_gpus_per_cell_groups_into_csv_slots(tmp_path):
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
    assert all("," in g for g in seen)               # each got a multi-GPU CSV slot
    assert set(seen) <= {"0,1", "2,3"}               # only the 2-GPU slots assigned


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
    seen: list[str] = []

    def fake(*, gpu_id, **kw):
        seen.append(gpu_id)
        return 0, "ok"

    with patch.object(gcr, "_run_cell_subprocess", fake), \
         patch.object(gcr, "_load_metrics", lambda d: {"ok": 1}):
        gcr.run_matrix([{"id": "c0"}], "x.py", output_root=str(tmp_path),
                       gpus=["0", "1"], gpus_per_cell=4)  # want 4, only 2 available

    assert seen == ["0,1"]                            # one slot of all available GPUs
