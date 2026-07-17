"""Tests for backend.agents.rlm.cpu_class — pure CPU-vs-GPU cell classification.

No network, no K8s, no GPU — pure-stdlib logic module.
"""
from __future__ import annotations

from backend.agents.rlm import cpu_class


# ---------------------------------------------------------------------------
# requires_gpu — hard signals override a soft accelerator="cpu" declaration
# ---------------------------------------------------------------------------

class TestRequiresGpuHardOverride:
    def test_est_vram_gb_overrides_cpu_declaration(self):
        cell = {"accelerator": "cpu", "est_vram_gb": 24.0}
        assert cpu_class.requires_gpu(cell) is True

    def test_verl_framework_overrides_cpu_declaration(self):
        cell = {"accelerator": "cpu", "framework": "verl"}
        assert cpu_class.requires_gpu(cell) is True

    def test_verl_image_key_overrides_cpu_declaration(self):
        cell = {"accelerator": "cpu", "image_key": "VERL"}
        assert cpu_class.requires_gpu(cell) is True

    def test_distributed_flag_overrides_cpu_declaration(self):
        cell = {"accelerator": "cpu", "distributed": True}
        assert cpu_class.requires_gpu(cell) is True

    def test_nproc_per_node_overrides_cpu_declaration(self):
        cell = {"accelerator": "cpu", "nproc_per_node": 4}
        assert cpu_class.requires_gpu(cell) is True


class TestRequiresGpuSoftDeclaration:
    def test_accelerator_cpu_with_no_hard_signal_is_cpu(self):
        cell = {"accelerator": "cpu"}
        assert cpu_class.requires_gpu(cell) is False

    def test_accelerator_gpu_is_gpu(self):
        cell = {"accelerator": "gpu"}
        assert cpu_class.requires_gpu(cell) is True

    def test_unknown_accelerator_is_conservative_gpu(self):
        cell = {}
        assert cpu_class.requires_gpu(cell) is True

    def test_zero_est_vram_gb_is_not_a_hard_signal(self):
        cell = {"accelerator": "cpu", "est_vram_gb": 0}
        assert cpu_class.requires_gpu(cell) is False

    def test_unrelated_framework_is_not_a_hard_signal(self):
        cell = {"accelerator": "cpu", "framework": "pytorch"}
        assert cpu_class.requires_gpu(cell) is False


# ---------------------------------------------------------------------------
# run_is_cpu_class
# ---------------------------------------------------------------------------

class TestRunIsCpuClass:
    def test_all_cpu_cells_true(self):
        cells = [{"accelerator": "cpu"}, {"accelerator": "cpu"}]
        assert cpu_class.run_is_cpu_class(cells) is True

    def test_mixed_cells_false(self):
        cells = [{"accelerator": "cpu"}, {"accelerator": "gpu"}]
        assert cpu_class.run_is_cpu_class(cells) is False

    def test_empty_cells_false(self):
        assert cpu_class.run_is_cpu_class([]) is False


# ---------------------------------------------------------------------------
# all_cells_infra_failed
# ---------------------------------------------------------------------------

class TestAllCellsInfraFailed:
    def test_all_error_true(self):
        results = {
            "c0": {"status": "error", "error": "boom"},
            "c1": {"status": "error", "error": "boom2"},
        }
        assert cpu_class.all_cells_infra_failed(results) is True

    def test_one_ok_false(self):
        results = {
            "c0": {"status": "error", "error": "boom"},
            "c1": {"status": "ok", "error": None},
        }
        assert cpu_class.all_cells_infra_failed(results) is False

    def test_empty_false(self):
        assert cpu_class.all_cells_infra_failed({}) is False
