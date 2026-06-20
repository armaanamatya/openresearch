"""Tests for backend/agents/rlm/metric_reality_smoke.py (P2 — §4.2).

Covers the pure evaluate_smoke_trace contract + the three review fixes:
  #1 launched-but-no-trace is JUDGED (not fail-open); only won't-spawn fails open.
  #2 a constant PRIMARY loss masked by a varying kl must FAIL.
  #3 an all-zero reward with a varying loss must PASS (2-step cold start).
"""

from __future__ import annotations

import backend.agents.rlm.metric_reality_smoke as mrs


# ---------------------------------------------------------------------------
# evaluate_smoke_trace (pure)
# ---------------------------------------------------------------------------

class TestEvaluateSmokeTrace:
    def test_one_record_with_real_loss_passes(self):
        # Relaxed to >=1 record: one step with loss>0 passes (slow-rollout RL).
        v = mrs.evaluate_smoke_trace([{"loss": 0.5}], None)
        assert v["ok"] is True

    def test_zero_records_fails(self):
        v = mrs.evaluate_smoke_trace([], None)
        assert v["ok"] is False and v["failure_class"] == "smoke_metrics_unreal"

    def test_one_record_zero_loss_fails(self):
        # A single step with loss==0.0 still catches the v6 disconnected-loss failure.
        v = mrs.evaluate_smoke_trace([{"loss": 0.0}], None)
        assert v["ok"] is False

    def test_single_dict_with_real_loss_passes(self):
        # A single-dict trace normalizes to one record; loss>0 passes.
        v = mrs.evaluate_smoke_trace({"loss": 0.5}, None)
        assert v["ok"] is True

    def test_non_list_non_dict_fails(self):
        v = mrs.evaluate_smoke_trace(42, None)
        assert v["ok"] is False

    def test_varying_primary_loss_passes(self):
        v = mrs.evaluate_smoke_trace([{"loss": 0.5}, {"loss": 0.4}], None)
        assert v["ok"] is True

    def test_constant_primary_loss_fails(self):
        v = mrs.evaluate_smoke_trace([{"loss": 0.5}, {"loss": 0.5}], None)
        assert v["ok"] is False

    def test_all_zero_primary_loss_fails(self):
        v = mrs.evaluate_smoke_trace([{"loss": 0.0}, {"loss": 0.0}], None)
        assert v["ok"] is False

    def test_fix2_constant_loss_masked_by_varying_kl_fails(self):
        # The PRIMARY loss is constant; only kl varies → must FAIL (no masking).
        v = mrs.evaluate_smoke_trace(
            [{"loss": 0.5, "kl": 0.1}, {"loss": 0.5, "kl": 0.2}], None
        )
        assert v["ok"] is False, "constant primary loss masked by varying kl must fail"

    def test_fix3_all_zero_reward_with_varying_loss_passes(self):
        # Sparse-reward cold start: reward 0.0,0.0 is legitimate at 2 steps.
        v = mrs.evaluate_smoke_trace(
            [{"loss": 0.5, "mean_reward": 0.0}, {"loss": 0.4, "mean_reward": 0.0}], None
        )
        assert v["ok"] is True, "all-zero reward must not fail the smoke (cold start)"

    def test_vram_below_floor_fails(self):
        v = mrs.evaluate_smoke_trace([{"loss": 0.5}, {"loss": 0.4}], peak_vram_gb=0.1)
        assert v["ok"] is False

    def test_vram_above_floor_passes(self):
        v = mrs.evaluate_smoke_trace([{"loss": 0.5}, {"loss": 0.4}], peak_vram_gb=8.0)
        assert v["ok"] is True

    def test_bad_grad_norm_fails(self):
        v = mrs.evaluate_smoke_trace(
            [{"loss": 0.5, "grad_norm": 1.0}, {"loss": 0.4, "grad_norm": 0.0}], None
        )
        assert v["ok"] is False

    def test_fallback_pooled_when_no_primary_key(self):
        # No primary loss key; only "kl" present and varying → fallback lenient passes.
        v = mrs.evaluate_smoke_trace([{"kl": 0.1}, {"kl": 0.2}], None)
        assert v["ok"] is True

    def test_fallback_pooled_constant_fails(self):
        v = mrs.evaluate_smoke_trace([{"kl": 0.1}, {"kl": 0.1}], None)
        assert v["ok"] is False


# ---------------------------------------------------------------------------
# run_metric_reality_smoke (I/O, mocked)
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    (tmp_path / "train_cell.py").write_text("print('x')", encoding="utf-8")
    cells = [{"id": "c0", "model_key": "qwen3_1.7b", "env": "alfworld"}]
    return cells


class TestRunSmoke:
    def test_flag_off_is_ok(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_METRIC_REALITY_SMOKE", raising=False)
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=None)
        assert v["ok"] is True

    def test_no_gpus_fail_open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: [])
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=None)
        assert v["ok"] is True

    def test_natural_exit_no_trace_is_judged(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # launched=True, NATURAL exit (timed_out=False), no trace → JUDGED (codex Area-4).
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: (None, 8.0, True, False))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is False and v["failure_class"] == "smoke_metrics_unreal"

    def test_timeout_no_trace_is_inconclusive_fail_open(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # TIMED OUT before any record (slow-rollout env) → inconclusive, fail-open.
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: (None, None, True, True))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is True

    def test_timeout_with_bad_partial_is_judged(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # Timed out but produced a partial trace whose loss is 0.0 → still judged.
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: ([{"loss": 0.0}], 8.0, True, True))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is False

    def test_fix1_all_wont_spawn_fail_open(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # launched=False everywhere → fail-open.
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: (None, None, False, False))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is True

    def test_good_trace_passes(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        monkeypatch.setattr(
            mrs, "_run_one_smoke_cell",
            lambda *a, **k: ([{"loss": 0.5}, {"loss": 0.4}], 8.0, True, False),
        )
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is True

    def test_bad_trace_judged(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        monkeypatch.setattr(
            mrs, "_run_one_smoke_cell",
            lambda *a, **k: ([{"loss": 0.0}, {"loss": 0.0}], 8.0, True, False),
        )
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is False
