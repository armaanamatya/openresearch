"""Tests for backend/agents/rlm/metric_reality_smoke.py (P2 — §4.2).

Covers the pure evaluate_smoke_trace contract + the three review fixes:
  #1 launched-but-no-trace is JUDGED (not fail-open); only won't-spawn fails open.
  #2 a constant PRIMARY loss masked by a varying kl must FAIL.
  #3 an all-zero reward with a varying loss must PASS (2-step cold start).
"""

from __future__ import annotations

import math
import textwrap

import backend.agents.rlm.metric_reality_smoke as mrs
from backend.agents.rlm.metric_reality_smoke import _run_one_smoke_cell


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
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: (None, 8.0, True, False, 0, ""))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is False and v["failure_class"] == "smoke_metrics_unreal"

    def test_timeout_no_trace_is_inconclusive_fail_open(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # TIMED OUT before any record (slow-rollout env) → inconclusive, fail-open.
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: (None, None, True, True, None, ""))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is True

    def test_timeout_with_bad_partial_is_judged(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # Timed out but produced a partial trace whose loss is 0.0 → still judged.
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: ([{"loss": 0.0}], 8.0, True, True, None, ""))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is False

    def test_fix1_all_wont_spawn_fail_open(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        # launched=False everywhere → fail-open.
        monkeypatch.setattr(mrs, "_run_one_smoke_cell", lambda *a, **k: (None, None, False, False, None, ""))
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is True

    def test_good_trace_passes(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        monkeypatch.setattr(
            mrs, "_run_one_smoke_cell",
            lambda *a, **k: ([{"loss": 0.5}, {"loss": 0.4}], 8.0, True, False, 0, ""),
        )
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is True

    def test_bad_trace_judged(self, tmp_path, monkeypatch):
        cells = _setup(tmp_path)
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        monkeypatch.setattr(
            mrs, "_run_one_smoke_cell",
            lambda *a, **k: ([{"loss": 0.0}, {"loss": 0.0}], 8.0, True, False, 0, ""),
        )
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=cells)
        assert v["ok"] is False


# ---------------------------------------------------------------------------
# _run_one_smoke_cell (real subprocess — no GPU needed; stub ignores GPU)
# ---------------------------------------------------------------------------

class TestRunOneSmokeCell:
    def test_crash_surfaces_traceback_and_rc(self, tmp_path):
        """A cell that crashes must return exit-code + traceback tail in the 6-tuple."""
        stub = tmp_path / "cell.py"
        stub.write_text(
            textwrap.dedent("""\
                import sys
                sys.stderr.write("BOOM-CRASH-MARKER\\n")
                sys.exit(3)
            """),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        cell = {"id": "c0"}
        trace, peak_vram, launched, timed_out, returncode, output_tail = _run_one_smoke_cell(
            cell, stub, "0", out_dir, timeout_s=30
        )
        assert launched is True
        assert timed_out is False
        assert returncode == 3
        assert "BOOM-CRASH-MARKER" in output_tail

    def test_clean_exit_no_trace(self, tmp_path):
        """A cell that exits 0 without writing any trace file."""
        stub = tmp_path / "cell.py"
        stub.write_text(
            textwrap.dedent("""\
                import sys
                sys.exit(0)
            """),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        cell = {"id": "c0"}
        trace, peak_vram, launched, timed_out, returncode, output_tail = _run_one_smoke_cell(
            cell, stub, "0", out_dir, timeout_s=30
        )
        assert launched is True
        assert timed_out is False
        assert returncode == 0
        assert trace is None


# ---------------------------------------------------------------------------
# FIX 3 — crash detail propagates end-to-end through run_metric_reality_smoke.
# ---------------------------------------------------------------------------

class TestCrashDetailPropagation:
    def test_crash_marker_and_exit_code_reach_verdict_detail(self, tmp_path, monkeypatch):
        """A crashing cell's traceback + exit code must reach the SmokeVerdict detail
        (proves the diagnosis reaches repair_context — the point of Fix A)."""
        (tmp_path / "train_cell.py").write_text(
            textwrap.dedent("""\
                import sys
                sys.stderr.write("BOOM-PROPAGATE-7\\n")
                raise SystemExit(3)
            """),
            encoding="utf-8",
        )
        (tmp_path / "cells.json").write_text(
            '{"cells":[{"id":"c0"}]}', encoding="utf-8"
        )
        monkeypatch.setenv("OPENRESEARCH_METRIC_REALITY_SMOKE", "1")
        monkeypatch.setattr(mrs, "_get_available_gpu_ids", lambda: ["0"])
        v = mrs.run_metric_reality_smoke(ctx=object(), code_dir=tmp_path, cells=None)
        assert v["ok"] is False
        assert v["failure_class"] == "smoke_metrics_unreal"
        assert "BOOM-PROPAGATE-7" in v["detail"]
        assert "exit 3" in v["detail"]


# ---------------------------------------------------------------------------
# FIX 1 — token-aware loss (pure evaluate_smoke_trace).
# ---------------------------------------------------------------------------

class TestTokenAwareLoss:
    def test_train_loss_recognized_and_varying_passes(self):
        v = mrs.evaluate_smoke_trace(
            [{"train_loss": 0.5}, {"train_loss": 0.4}], peak_vram_gb=8.0
        )
        assert v["ok"] is True

    def test_train_loss_all_zero_fails(self):
        # Teeth preserved: an all-zero recognized loss is still rejected.
        v = mrs.evaluate_smoke_trace([{"train_loss": 0.0}, {"train_loss": 0.0}], None)
        assert v["ok"] is False

    def test_val_loss_no_grad_fails_with_training_signal_hint(self):
        # val_loss is NOT a training loss; with no grad evidence the no-signal
        # rejection fires.
        v = mrs.evaluate_smoke_trace([{"val_loss": 0.5}, {"val_loss": 0.4}], None)
        assert v["ok"] is False
        detail = v["detail"].lower()
        assert "training-loss key" in detail or "backprop" in detail


# ---------------------------------------------------------------------------
# FIX 2 — grad evidence sufficiency (pure evaluate_smoke_trace).
# ---------------------------------------------------------------------------

class TestGradEvidenceSufficiency:
    def test_no_loss_positive_grad_passes(self):
        v = mrs.evaluate_smoke_trace(
            [{"reward": 0.0, "grad_norm": 0.7}, {"reward": 0.0, "grad_norm": 0.6}], None
        )
        assert v["ok"] is True

    def test_no_loss_zero_grad_fails(self):
        # grad present but 0 → Requirement 4 rejects <=0 grad.
        v = mrs.evaluate_smoke_trace([{"reward": 0.0, "grad_norm": 0.0}], None)
        assert v["ok"] is False

    def test_no_loss_no_grad_fails_with_actionable_hint(self):
        v = mrs.evaluate_smoke_trace([{"reward": 0.5}, {"reward": 0.6}], None)
        assert v["ok"] is False
        assert "log a per-step loss (or grad_norm)" in v["detail"]


# ---------------------------------------------------------------------------
# RL-aware smoke — GRPO/PPO objectives legitimately log ~0 loss/grad on a tiny
# degenerate slice (group-relative advantage collapses to 0). The supervised
# loss>0/varies + grad>0 teeth must NOT fire for RL; only non-finite is fatal.
# The VRAM floor + finite checks stay; supervised teeth are a REGRESSION guard.
# ---------------------------------------------------------------------------

class TestRLAwareSmoke:
    def test_rl_detector(self):
        assert mrs._is_rl_objective([{"l_grpo": 0.0}]) is True
        assert mrs._is_rl_objective([{"grpo_loss": 0.1}]) is True
        # reward alone is NOT an RL/PG loss marker.
        assert mrs._is_rl_objective([{"reward": 0.0, "loss": 0.1}]) is False
        assert mrs._is_rl_objective([{"train_loss": 0.1}]) is False
        # a ppo token anywhere in a key counts.
        assert mrs._is_rl_objective([{"ppo_clip": 0.1}]) is True

    def test_rl_zero_loss_passes(self):
        # Degenerate RL slice: l_grpo/loss/grad_norm all 0.0 is LEGITIMATE.
        v = mrs.evaluate_smoke_trace(
            [
                {"l_grpo": 0.0, "loss": 0.0, "grad_norm": 0.0},
                {"l_grpo": 0.0, "loss": 0.0, "grad_norm": 0.0},
            ],
            peak_vram_gb=8.0,
        )
        assert v["ok"] is True

    def test_rl_constant_loss_passes(self):
        v = mrs.evaluate_smoke_trace(
            [
                {"l_grpo": 0.5, "grad_norm": 0.0},
                {"l_grpo": 0.5, "grad_norm": 0.0},
            ],
            peak_vram_gb=8.0,
        )
        assert v["ok"] is True

    def test_rl_nonfinite_loss_fails(self):
        # Non-finite loss is fatal even for RL (diverged / not real).
        v = mrs.evaluate_smoke_trace(
            [{"l_grpo": math.nan, "loss": math.nan}], peak_vram_gb=8.0
        )
        assert v["ok"] is False

    def test_rl_low_vram_fails(self):
        # VRAM floor teeth preserved for RL.
        v = mrs.evaluate_smoke_trace([{"l_grpo": 0.0, "loss": 0.0}], peak_vram_gb=0.1)
        assert v["ok"] is False

    def test_supervised_zero_loss_still_fails(self):
        # REGRESSION guard: supervised all-zero loss still rejected.
        v = mrs.evaluate_smoke_trace(
            [{"train_loss": 0.0}, {"train_loss": 0.0}], peak_vram_gb=8.0
        )
        assert v["ok"] is False

    def test_supervised_constant_loss_still_fails(self):
        v = mrs.evaluate_smoke_trace(
            [{"train_loss": 0.5}, {"train_loss": 0.5}], peak_vram_gb=8.0
        )
        assert v["ok"] is False
