"""Tests for gpu_cell_runner — harness-owned multi-GPU cell scheduler.

All tests mock subprocess so no real GPUs or torch are required.  The suite
verifies:

  * GPU pool schedules N cells across N gpus, each cell getting exactly one id.
  * Concurrency never exceeds len(gpus).
  * OOM-retry path: a cell that OOMs once then succeeds → status ok, retries=1,
    OPENRESEARCH_CELL_BATCH_SCALE carried to retry.
  * Retries exhausted → oom_failed without raising.
  * discover_visible_gpus parses UUID and integer-index forms.
  * Results aggregation is complete (every input cell present in output dict).
"""
from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import patch

import pytest

import backend.agents.rlm.gpu_cell_runner as gcr
from backend.agents.rlm.gpu_cell_runner import (
    _is_oom,
    _load_metrics,
    discover_visible_gpus,
    run_matrix,
)


# ---------------------------------------------------------------------------
# discover_visible_gpus
# ---------------------------------------------------------------------------

class TestDiscoverVisibleGpus:
    def test_integer_indices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        assert discover_visible_gpus() == ["0", "1", "2", "3"]

    def test_uuid_form(self, monkeypatch):
        uuid = "GPU-abc12345-1234-5678-9abc-def012345678"
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
        assert discover_visible_gpus() == [uuid]

    def test_multiple_uuids(self, monkeypatch):
        u1 = "GPU-aaaa-bbbb"
        u2 = "GPU-cccc-dddd"
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{u1},{u2}")
        assert discover_visible_gpus() == [u1, u2]

    def test_unset_falls_back_to_nvidia_smi(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch("subprocess.check_output", return_value="0\n1\n2\n"):
            result = discover_visible_gpus()
        assert result == ["0", "1", "2"]

    def test_nodevfiles_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NODEVFILES")
        with patch("subprocess.check_output", return_value="0\n"):
            result = discover_visible_gpus()
        assert result == ["0"]

    def test_minus_one_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        with patch("subprocess.check_output", return_value="0\n"):
            result = discover_visible_gpus()
        assert result == ["0"]

    def test_nvidia_smi_failure_falls_back_to_zero(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = discover_visible_gpus()
        assert result == ["0"]

    def test_empty_string_falls_back(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        with patch("subprocess.check_output", return_value="0\n1\n"):
            result = discover_visible_gpus()
        assert result == ["0", "1"]


# ---------------------------------------------------------------------------
# _is_oom
# ---------------------------------------------------------------------------

class TestIsOom:
    @pytest.mark.parametrize("msg", [
        "CUDA out of memory. Tried to allocate 20 GiB",
        "CUDA error: out of memory",
        "OutOfMemoryError",
        "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB",
        "torch.cuda.OutOfMemoryError: out of memory",
    ])
    def test_positive_signatures(self, msg):
        assert _is_oom(msg) is True

    @pytest.mark.parametrize("msg", [
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied",
        "ValueError: bad input",
        "",
        "CUDA device available",
    ])
    def test_negative(self, msg):
        assert _is_oom(msg) is False


# ---------------------------------------------------------------------------
# _load_metrics
# ---------------------------------------------------------------------------

class TestLoadMetrics:
    def test_loads_valid_json(self, tmp_path):
        (tmp_path / "metrics.json").write_text('{"acc": 0.9}', encoding="utf-8")
        assert _load_metrics(tmp_path) == {"acc": 0.9}

    def test_missing_file_returns_none(self, tmp_path):
        assert _load_metrics(tmp_path) is None

    def test_corrupt_json_returns_none(self, tmp_path):
        (tmp_path / "metrics.json").write_text("{bad json", encoding="utf-8")
        assert _load_metrics(tmp_path) is None


# ---------------------------------------------------------------------------
# Helpers for mocking _run_cell_subprocess
# ---------------------------------------------------------------------------

def _make_ok_subprocess(metrics: dict | None = None):
    """Return a fake _run_cell_subprocess that writes metrics.json and returns 0."""
    def _fake(*, cell, cell_script, gpu_id, output_dir, batch_scale, grad_checkpoint,
               timeout_s, log_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        if metrics is not None:
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
        return 0, "Training complete."
    return _fake


def _make_oom_then_ok_subprocess(fail_count: int = 1, metrics: dict | None = None):
    """Return a fake that OOMs ``fail_count`` times then succeeds."""
    call_counts: dict[str, int] = {}

    def _fake(*, cell, cell_script, gpu_id, output_dir, batch_scale, grad_checkpoint,
               timeout_s, log_path):
        cid = cell.get("id", "?")
        n = call_counts.get(cid, 0)
        call_counts[cid] = n + 1
        output_dir.mkdir(parents=True, exist_ok=True)
        if n < fail_count:
            return 1, "CUDA out of memory. Tried to allocate 2 GiB"
        # Success
        if metrics is not None:
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
        return 0, "Training complete."

    return _fake, call_counts


def _make_always_oom_subprocess():
    """Return a fake that always OOMs."""
    def _fake(*, cell, cell_script, gpu_id, output_dir, batch_scale, grad_checkpoint,
               timeout_s, log_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return 1, "CUDA out of memory. Tried to allocate 10 GiB"
    return _fake


# ---------------------------------------------------------------------------
# run_matrix — basic scheduling
# ---------------------------------------------------------------------------

class TestRunMatrixScheduling:
    def test_empty_cells_returns_empty(self, tmp_path):
        result = run_matrix([], "train_cell.py", output_root=tmp_path)
        assert result == {}

    def test_single_cell_ok(self, tmp_path):
        cells = [{"id": "c0", "model": "qwen3-1.7b", "baseline": "grpo", "env": "search_qa"}]
        with patch.object(gcr, "_run_cell_subprocess", _make_ok_subprocess({"acc": 0.5})):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0"])
        assert results["c0"]["status"] == "ok"
        assert results["c0"]["metrics"] == {"acc": 0.5}
        assert results["c0"]["retries"] == 0

    def test_all_cells_present_in_result(self, tmp_path):
        cells = [{"id": f"cell_{i}"} for i in range(5)]
        with patch.object(gcr, "_run_cell_subprocess", _make_ok_subprocess()):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"])
        assert set(results.keys()) == {f"cell_{i}" for i in range(5)}

    def test_each_cell_gets_exactly_one_gpu(self, tmp_path):
        """Each subprocess call must receive a single gpu_id from the pool."""
        cells = [{"id": f"c{i}"} for i in range(4)]
        assigned_gpus: list[str] = []
        lock = threading.Lock()

        def _record(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                    grad_checkpoint, timeout_s, log_path):
            with lock:
                assigned_gpus.append(gpu_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _record):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1", "2", "3"])

        # Every assigned gpu_id must be a single value from the pool.
        pool = {"0", "1", "2", "3"}
        for gid in assigned_gpus:
            assert gid in pool, f"Unexpected gpu_id {gid!r} outside pool"
        assert len(assigned_gpus) == 4

    def test_concurrency_never_exceeds_gpu_count(self, tmp_path):
        """The number of simultaneously running subprocess calls never exceeds len(gpus)."""
        cells = [{"id": f"c{i}"} for i in range(6)]
        gpus = ["0", "1", "2"]
        concurrent_peak = [0]
        active_count = [0]
        lock = threading.Lock()
        barrier = threading.Barrier(3, timeout=10)

        def _concurrent(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                        grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            with lock:
                active_count[0] += 1
                if active_count[0] > concurrent_peak[0]:
                    concurrent_peak[0] = active_count[0]
            # Synchronise the first 3 workers so they all run at once.
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
            with lock:
                active_count[0] -= 1
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _concurrent):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=gpus)

        assert concurrent_peak[0] <= len(gpus), (
            f"Concurrency peak {concurrent_peak[0]} exceeded gpu count {len(gpus)}"
        )

    def test_gpu_id_in_result(self, tmp_path):
        cells = [{"id": "cx"}]
        captured: dict = {}

        def _capture(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                     grad_checkpoint, timeout_s, log_path):
            captured["gpu"] = gpu_id
            output_dir.mkdir(parents=True, exist_ok=True)
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _capture):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["2"])

        assert results["cx"]["gpu"] == "2"
        assert captured["gpu"] == "2"


# ---------------------------------------------------------------------------
# run_matrix — OOM retry path
# ---------------------------------------------------------------------------

class TestRunMatrixOomRetry:
    def test_oom_once_then_ok_gives_status_ok(self, tmp_path):
        cells = [{"id": "r0"}]
        fake, call_counts = _make_oom_then_ok_subprocess(fail_count=1, metrics={"f1": 0.7})
        with patch.object(gcr, "_run_cell_subprocess", fake):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0"], max_oom_retries=2
            )
        assert results["r0"]["status"] == "ok"
        assert results["r0"]["retries"] == 1
        assert results["r0"]["metrics"] == {"f1": 0.7}
        assert call_counts["r0"] == 2  # first attempt + one retry

    def test_oom_retry_passes_batch_scale(self, tmp_path):
        """On retry 1 the subprocess must receive batch_scale=0.5."""
        cells = [{"id": "bs0"}]
        batch_scales_seen: list[Any] = []
        lock = threading.Lock()

        def _fake(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                  grad_checkpoint, timeout_s, log_path):
            with lock:
                batch_scales_seen.append(batch_scale)
            output_dir.mkdir(parents=True, exist_ok=True)
            if len(batch_scales_seen) == 1:
                return 1, "CUDA out of memory."
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _fake):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0"], max_oom_retries=2
            )

        assert results["bs0"]["status"] == "ok"
        assert batch_scales_seen[0] is None    # first attempt: no batch_scale
        assert batch_scales_seen[1] == 0.5     # first retry: 0.5

    def test_oom_retry_passes_grad_checkpoint(self, tmp_path):
        """On retry the subprocess must receive grad_checkpoint=True."""
        cells = [{"id": "gc0"}]
        grad_checkpoints_seen: list[Any] = []
        lock = threading.Lock()

        def _fake(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                  grad_checkpoint, timeout_s, log_path):
            with lock:
                grad_checkpoints_seen.append(grad_checkpoint)
            output_dir.mkdir(parents=True, exist_ok=True)
            if len(grad_checkpoints_seen) == 1:
                return 1, "CUDA out of memory."
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _fake):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0"], max_oom_retries=2)

        assert grad_checkpoints_seen[0] is False
        assert grad_checkpoints_seen[1] is True

    def test_retries_exhausted_gives_oom_failed(self, tmp_path):
        cells = [{"id": "fail0"}]
        with patch.object(gcr, "_run_cell_subprocess", _make_always_oom_subprocess()):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0"], max_oom_retries=2
            )
        r = results["fail0"]
        assert r["status"] == "oom_failed"
        assert r["retries"] == 2  # exhausted all retries
        assert r["error"] is not None

    def test_retries_exhausted_does_not_raise(self, tmp_path):
        """Matrix must complete normally even when all retries are exhausted."""
        cells = [{"id": f"fail{i}"} for i in range(3)]
        with patch.object(gcr, "_run_cell_subprocess", _make_always_oom_subprocess()):
            # Must not raise.
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1"], max_oom_retries=1
            )
        assert len(results) == 3
        for r in results.values():
            assert r["status"] == "oom_failed"

    def test_non_oom_failure_gives_error_status(self, tmp_path):
        cells = [{"id": "e0"}]

        def _fail(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                  grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            return 1, "SyntaxError: invalid syntax"

        with patch.object(gcr, "_run_cell_subprocess", _fail):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0"])

        assert results["e0"]["status"] == "error"
        assert results["e0"]["retries"] == 0  # non-OOM: no retry attempted

    def test_mixed_cells_ok_and_failed(self, tmp_path):
        """Healthy cells must still be ok even when others fail."""
        ok_cell = {"id": "ok0"}
        bad_cell = {"id": "bad0"}
        cells = [ok_cell, bad_cell]

        def _mixed(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                   grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            if cell["id"] == "bad0":
                return 1, "CUDA out of memory."
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _mixed):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0"], max_oom_retries=0
            )

        assert results["ok0"]["status"] == "ok"
        assert results["bad0"]["status"] == "oom_failed"


# ---------------------------------------------------------------------------
# run_matrix — results aggregation completeness
# ---------------------------------------------------------------------------

class TestRunMatrixResultsCompleteness:
    def test_result_count_matches_cell_count(self, tmp_path):
        n = 8
        cells = [{"id": f"cell_{i}"} for i in range(n)]
        with patch.object(gcr, "_run_cell_subprocess", _make_ok_subprocess()):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path, gpus=["0", "1", "2", "3"]
            )
        assert len(results) == n

    def test_result_dict_has_required_keys(self, tmp_path):
        cells = [{"id": "chk0"}]
        with patch.object(gcr, "_run_cell_subprocess", _make_ok_subprocess({"x": 1})):
            results = run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0"])
        r = results["chk0"]
        assert {"status", "metrics", "gpu", "retries", "error"} <= set(r.keys())

    def test_max_parallel_respected(self, tmp_path):
        """max_parallel=1 means cells run serially (peak concurrent=1)."""
        cells = [{"id": f"s{i}"} for i in range(4)]
        peak = [0]
        active = [0]
        lock = threading.Lock()

        def _serial(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                    grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            with lock:
                active[0] += 1
                if active[0] > peak[0]:
                    peak[0] = active[0]
            # Brief pause to allow racing workers to show up.
            import time; time.sleep(0.01)
            with lock:
                active[0] -= 1
            return 0, ""

        with patch.object(gcr, "_run_cell_subprocess", _serial):
            results = run_matrix(
                cells, "train_cell.py", output_root=tmp_path,
                gpus=["0", "1", "2", "3"], max_parallel=1,
            )

        assert peak[0] == 1
        assert len(results) == 4

    def test_log_file_created_per_cell(self, tmp_path):
        cells = [{"id": "log0"}, {"id": "log1"}]

        def _log_writer(*, cell, cell_script, gpu_id, output_dir, batch_scale,
                        grad_checkpoint, timeout_s, log_path):
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text("some output\n", encoding="utf-8")
            return 0, "some output\n"

        with patch.object(gcr, "_run_cell_subprocess", _log_writer):
            run_matrix(cells, "train_cell.py", output_root=tmp_path, gpus=["0"])

        assert (tmp_path / "log0.log").exists()
        assert (tmp_path / "log1.log").exists()
