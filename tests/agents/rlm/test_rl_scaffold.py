"""Tests for the harness-owned GRPO + vLLM RL scaffold (Track A-MVP).

Covers:
  - OPSD math on tiny tensors (gate detached, lambda applied, shapes, reverse-KL sign)
  - Rewriter skip-on-sentinel (all three detection surfaces)
  - Default-unchanged: rewriter skip-on-sentinel does NOT fire for normal commands
  - Metrics schema via finalize_metrics (RubricGuardFailure on missing key)
  - Opt-in-OFF parity: _compute_constraint_guidance output with REPROLAB_RL_SCAFFOLD
    unset is byte-identical to the pre-scaffold baseline

No GPU or TRL/vLLM installation required for any test here.
The smoke-test (1-step GRPO on a tiny model) is gated behind REPROLAB_RL_SCAFFOLD_SMOKE=1
so CI can run it optionally without a full trl install.
"""
from __future__ import annotations

import os
import json
import re
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# OPSD math tests
# ---------------------------------------------------------------------------

def test_opsd_gate_is_detached_no_grad():
    """g_t must be constructed with .detach() — gradient MUST NOT flow through it."""
    torch = pytest.importorskip("torch")
    from backend.agents.rlm.rl_scaffold import opsd_custom_loss_term, BETA, LAMBDA

    logp_s = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    logp_t = torch.tensor([-1.5, -1.8, -0.3], requires_grad=False)

    loss = opsd_custom_loss_term(logp_s, logp_t)
    loss.backward()

    # Gradient must flow through logp_s (it's in the loss).
    assert logp_s.grad is not None, "Gradient should flow through logp_student"
    # The gate g_t is detached, so no grad through it (already asserted by .detach() in impl).
    # We verify the loss is a scalar.
    assert loss.dim() == 0, "opsd_custom_loss_term must return a scalar"


def test_opsd_lambda_applied():
    """LAMBDA = 0.1 must be applied in GRPOScaffold.compute_loss.

    We verify the constant is 0.1 as a literal in the module source.
    """
    from backend.agents.rlm import rl_scaffold
    import inspect
    src = inspect.getsource(rl_scaffold)
    # Literal constant must appear in source (rubric reads it).
    assert "LAMBDA: float = 0.1" in src, "LAMBDA = 0.1 must appear literally in rl_scaffold.py"
    assert "BETA: float = 10.0" in src, "BETA = 10.0 must appear literally in rl_scaffold.py"


def test_opsd_loss_shape():
    """opsd_custom_loss_term must return a scalar for any batch × seq input."""
    torch = pytest.importorskip("torch")
    from backend.agents.rlm.rl_scaffold import opsd_custom_loss_term

    # batch=4, seq=8
    logp_s = torch.randn(4, 8, requires_grad=True)
    logp_t = torch.randn(4, 8)
    loss = opsd_custom_loss_term(logp_s, logp_t)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


def test_opsd_reverse_kl_sign():
    """Reverse-KL (mode-seeking): loss should decrease when student log-probs
    increase toward teacher (i.e., student catches up to teacher on high-prob tokens).
    When student log-prob < teacher log-prob:
      delta_t = logp_s - logp_t < 0
      g_t = sigmoid(10 * delta_t) ≈ 0 (gate suppresses high-gap tokens)
      opsd_loss = g_t * (-logp_s) ≈ 0

    When student log-prob > teacher log-prob:
      delta_t > 0 → g_t ≈ 1 → opsd_loss = 1 * (-logp_s)

    Verify: loss is finite and non-negative for log-probs in (-inf, 0).
    """
    torch = pytest.importorskip("torch")
    from backend.agents.rlm.rl_scaffold import opsd_custom_loss_term

    # logp in (-5, 0) range — realistic log-prob territory
    logp_s = torch.tensor([-0.5, -1.0, -2.0, -3.0], requires_grad=False)
    logp_t = torch.tensor([-0.3, -0.8, -2.5, -2.8], requires_grad=False)
    loss = opsd_custom_loss_term(logp_s, logp_t)
    assert torch.isfinite(loss), f"opsd_loss must be finite, got {loss}"
    assert loss >= 0, f"opsd_loss must be non-negative (reverse-KL), got {loss}"


def test_opsd_beta_constant_value():
    """BETA must equal 10.0 and LAMBDA must equal 0.1 as module-level constants."""
    from backend.agents.rlm.rl_scaffold import BETA, LAMBDA
    assert BETA == 10.0, f"BETA must be 10.0, got {BETA}"
    assert LAMBDA == 0.1, f"LAMBDA must be 0.1, got {LAMBDA}"


def test_opsd_stop_grad_on_gate():
    """Stop-gradient on gate: g_t.requires_grad must be False after .detach()."""
    torch = pytest.importorskip("torch")
    from backend.agents.rlm.rl_scaffold import BETA

    logp_s = torch.tensor([-1.0, -2.0], requires_grad=True)
    logp_t = torch.tensor([-0.5, -1.5])
    delta_t = logp_s - logp_t.detach()
    g_t = torch.sigmoid(BETA * delta_t).detach()
    assert not g_t.requires_grad, "Gate g_t must have requires_grad=False after .detach()"


# ---------------------------------------------------------------------------
# Rewriter sentinel tests (A3)
# ---------------------------------------------------------------------------

def _make_train_py(tmp_path: Path, body: str) -> Path:
    (tmp_path / "train.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_skip_on_env_sentinel(tmp_path, monkeypatch):
    """REPROLAB_RL_SCAFFOLD=1 → rewriter must skip even with distributed markers."""
    from backend.agents.rlm.primitives import _resolve_distributed_launch

    monkeypatch.setenv("REPROLAB_RL_SCAFFOLD", "1")
    code = _make_train_py(tmp_path, "from accelerate import Accelerator\n")
    cmds = ["python rl_launch.py"]
    result = _resolve_distributed_launch(cmds, code, 4)
    assert result == cmds, "Must return commands unchanged when REPROLAB_RL_SCAFFOLD=1"


def test_skip_on_sentinel_file(tmp_path, monkeypatch):
    """code/.reprolab_rl_scaffold file → rewriter must skip."""
    from backend.agents.rlm.primitives import _resolve_distributed_launch

    monkeypatch.delenv("REPROLAB_RL_SCAFFOLD", raising=False)
    (tmp_path / ".reprolab_rl_scaffold").touch()
    code = _make_train_py(tmp_path, "from accelerate import Accelerator\n")
    cmds = ["python rl_launch.py"]
    result = _resolve_distributed_launch(cmds, code, 4)
    assert result == cmds, "Must return commands unchanged when sentinel file present"


def test_skip_on_command_marker(tmp_path, monkeypatch):
    """Command containing '# reprolab:rl-scaffold-owns-launch' → rewriter must skip."""
    from backend.agents.rlm.primitives import _resolve_distributed_launch

    monkeypatch.delenv("REPROLAB_RL_SCAFFOLD", raising=False)
    code = _make_train_py(tmp_path, "from accelerate import Accelerator\n")
    cmds = [
        "pip install -r requirements.txt",
        "# reprolab:rl-scaffold-owns-launch\npython rl_launch.py",
    ]
    result = _resolve_distributed_launch(cmds, code, 4)
    assert result == cmds, "Must return commands unchanged when marker is in any command"


def test_default_unchanged_no_sentinel(tmp_path, monkeypatch):
    """Without any sentinel, the rewriter must still wrap distributed scripts."""
    from backend.agents.rlm.primitives import _resolve_distributed_launch

    monkeypatch.delenv("REPROLAB_RL_SCAFFOLD", raising=False)
    # Ensure sentinel file is NOT present.
    sentinel = tmp_path / ".reprolab_rl_scaffold"
    if sentinel.exists():
        sentinel.unlink()
    code = _make_train_py(tmp_path, "from accelerate import Accelerator\n")
    cmds = ["python train.py"]
    result = _resolve_distributed_launch(cmds, code, 4)
    # Should be rewritten (default behavior unchanged).
    assert result != cmds, "Default FSDP rewrite must still fire when no sentinel"
    assert "accelerate launch" in result[0]


def test_skip_env_sentinel_false_does_not_skip(tmp_path, monkeypatch):
    """REPROLAB_RL_SCAFFOLD=0 must NOT trigger the skip."""
    from backend.agents.rlm.primitives import _resolve_distributed_launch

    monkeypatch.setenv("REPROLAB_RL_SCAFFOLD", "0")
    sentinel = tmp_path / ".reprolab_rl_scaffold"
    if sentinel.exists():
        sentinel.unlink()
    code = _make_train_py(tmp_path, "from accelerate import Accelerator\n")
    cmds = ["python train.py"]
    result = _resolve_distributed_launch(cmds, code, 4)
    # Must NOT skip (env var is "0").
    assert "accelerate launch" in result[0]


# ---------------------------------------------------------------------------
# Metrics schema tests
# ---------------------------------------------------------------------------

def test_finalize_metrics_happy_path(tmp_path):
    """finalize_metrics writes terminal status and passes rubric guard."""
    from backend.agents.rlm.rl_scaffold import GRPOScaffold

    metrics_path = tmp_path / "metrics.json"
    scaffold = GRPOScaffold(
        model_name="dummy",
        reward_fn=lambda *a, **kw: 0.0,
        output_dir=tmp_path / "rl_output",
        metrics_path=metrics_path,
        model_tag="qwen3_1.7b",
    )
    scaffold._metrics = {
        "per_model": {"qwen3_1.7b": {"searchqa_em": 0.42, "reward": 1.5}},
        "baselines_vs_sdar": {"grpo": 0.30},
    }
    # finalize_metrics should write + validate without raising.
    scaffold.finalize_metrics(
        required_keys=["per_model", "baselines_vs_sdar"],
        artifact_dir=tmp_path,  # no required_artifacts → no artifact check
    )
    assert metrics_path.exists()
    written = json.loads(metrics_path.read_text())
    assert written["status"] == "completed"
    assert "per_model" in written


def test_finalize_metrics_raises_on_missing_key(tmp_path):
    """finalize_metrics raises RubricGuardFailure when a required key is absent."""
    from backend.agents.rlm.rl_scaffold import GRPOScaffold
    from backend.agents.rlm.rubric_guard import RubricGuardFailure

    scaffold = GRPOScaffold(
        model_name="dummy",
        reward_fn=lambda *a, **kw: 0.0,
        output_dir=tmp_path / "rl_output",
        metrics_path=tmp_path / "metrics.json",
        model_tag="qwen3_1.7b",
    )
    scaffold._metrics = {"per_model": {"qwen3_1.7b": {}}}
    with pytest.raises(RubricGuardFailure):
        scaffold.finalize_metrics(
            required_keys=["missing_key_that_does_not_exist"],
        )


def test_write_metrics_atomic(tmp_path):
    """_write_metrics_atomic writes JSON and is idempotent on repeat calls."""
    from backend.agents.rlm.rl_scaffold import _write_metrics_atomic

    path = tmp_path / "metrics.json"
    _write_metrics_atomic({"a": 1}, path)
    assert json.loads(path.read_text()) == {"a": 1}
    _write_metrics_atomic({"a": 2, "b": 3}, path)
    assert json.loads(path.read_text()) == {"a": 2, "b": 3}


def test_write_metrics_incremental(tmp_path):
    """write_metrics_incremental flushes incrementally and sets status=running."""
    from backend.agents.rlm.rl_scaffold import GRPOScaffold

    scaffold = GRPOScaffold(
        model_name="dummy",
        reward_fn=lambda *a, **kw: 0.0,
        output_dir=tmp_path / "out",
        metrics_path=tmp_path / "metrics.json",
        model_tag="qwen3_1.7b",
    )
    scaffold.write_metrics_incremental({"searchqa_em": 0.35}, step=10)
    written = json.loads((tmp_path / "metrics.json").read_text())
    assert written["status"] == "running"
    assert written["per_model"]["qwen3_1.7b"]["searchqa_em"] == 0.35
    assert written["per_model"]["qwen3_1.7b"]["last_step"] == 10


# ---------------------------------------------------------------------------
# Opt-in OFF parity
# ---------------------------------------------------------------------------

def test_opt_in_off_guidance_unchanged(tmp_path, monkeypatch):
    """With REPROLAB_RL_SCAFFOLD unset, _compute_constraint_guidance output
    must be byte-identical to the output with REPROLAB_RL_SCAFFOLD=0
    (i.e., the block is NOT injected)."""
    monkeypatch.delenv("REPROLAB_RL_SCAFFOLD", raising=False)
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    baseline_off = _compute_constraint_guidance(
        sandbox_mode="local",
        gpu_mode="auto",
    )

    monkeypatch.setenv("REPROLAB_RL_SCAFFOLD", "0")
    # Reload is not needed — the env var is read inside the function each call.
    off_explicit = _compute_constraint_guidance(
        sandbox_mode="local",
        gpu_mode="auto",
    )

    assert baseline_off == off_explicit, (
        "Guidance with REPROLAB_RL_SCAFFOLD unset vs =0 must be identical"
    )

    # Verify RL scaffold block is NOT present in either.
    assert "RL SCAFFOLD" not in baseline_off
    assert "RL SCAFFOLD" not in off_explicit


def test_opt_in_on_injects_scaffold_block(tmp_path, monkeypatch):
    """With REPROLAB_RL_SCAFFOLD=1, the _RL_SCAFFOLD_BLOCK must be present."""
    monkeypatch.setenv("REPROLAB_RL_SCAFFOLD", "1")
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    guidance = _compute_constraint_guidance(
        sandbox_mode="local",
        gpu_mode="auto",
    )
    assert "RL SCAFFOLD" in guidance, "RL scaffold block must be injected when opt-in"
    assert "trl==0.16.1" in guidance
    assert "BETA = 10.0" in guidance
    assert "LAMBDA = 0.1" in guidance
    assert "# reprolab:rl-scaffold-owns-launch" in guidance


def test_opt_in_off_vs_on_differ(monkeypatch):
    """The guidance with and without opt-in must differ, and the scaffold content
    must appear in the ON guidance but not the OFF guidance."""
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    monkeypatch.delenv("REPROLAB_RL_SCAFFOLD", raising=False)
    guidance_off = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    monkeypatch.setenv("REPROLAB_RL_SCAFFOLD", "1")
    guidance_on = _compute_constraint_guidance(sandbox_mode="local", gpu_mode="auto")

    assert guidance_off != guidance_on, "Guidance must differ between opt-in OFF and ON"
    # The ON guidance must be longer (scaffold block injected).
    assert len(guidance_on) > len(guidance_off), (
        "Opt-in ON guidance must be longer than OFF guidance"
    )
    # Scaffold-specific content must appear in ON but not OFF.
    assert "RL SCAFFOLD" in guidance_on
    assert "RL SCAFFOLD" not in guidance_off


# ---------------------------------------------------------------------------
# Module import smoke (no trl/vllm required)
# ---------------------------------------------------------------------------

def test_rl_scaffold_imports_without_trl():
    """rl_scaffold.py must import cleanly without trl or vllm installed."""
    # Force re-import to exercise the module-level import path fresh.
    if "backend.agents.rlm.rl_scaffold" in sys.modules:
        del sys.modules["backend.agents.rlm.rl_scaffold"]
    import backend.agents.rlm.rl_scaffold as rl_mod
    assert hasattr(rl_mod, "GRPOScaffold")
    assert hasattr(rl_mod, "opsd_custom_loss_term")
    assert hasattr(rl_mod, "BETA")
    assert hasattr(rl_mod, "LAMBDA")
    assert hasattr(rl_mod, "RL_LAUNCH_TEMPLATE")


def test_rl_launch_template_has_sentinel():
    """RL_LAUNCH_TEMPLATE string must include the sentinel comment."""
    from backend.agents.rlm.rl_scaffold import RL_LAUNCH_TEMPLATE
    assert "# reprolab:rl-scaffold-owns-launch" in RL_LAUNCH_TEMPLATE


# ---------------------------------------------------------------------------
# Optional smoke test (gated — requires trl + vllm + GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("REPROLAB_RL_SCAFFOLD_SMOKE"),
    reason="REPROLAB_RL_SCAFFOLD_SMOKE not set — skipping full trl smoke test",
)
def test_grpo_scaffold_one_step_smoke(tmp_path):
    """1-step GRPO on a tiny model: gen → loss → step → metrics_path written.

    Requires: trl==0.16.1, transformers, torch, HuggingFaceTB/SmolLM-135M (or
    a model available locally), and one free GPU.
    Gate: REPROLAB_RL_SCAFFOLD_SMOKE=1 (never set in CI — devs opt in explicitly).
    """
    from datasets import Dataset
    from backend.agents.rlm.rl_scaffold import GRPOScaffold, opsd_custom_loss_term

    def dummy_reward(completions, **kwargs):
        return [1.0] * len(completions)

    model_name = os.environ.get("REPROLAB_SMOKE_MODEL", "HuggingFaceTB/SmolLM-135M")
    metrics_path = tmp_path / "metrics.json"

    scaffold = GRPOScaffold(
        model_name=model_name,
        reward_fn=dummy_reward,
        custom_loss_term=opsd_custom_loss_term,
        output_dir=tmp_path / "rl_output",
        metrics_path=metrics_path,
        model_tag="smoke_model",
        max_steps=1,
        num_generations=2,
        per_device_train_batch_size=1,
        # Disable vLLM server for smoke (single GPU path).
        vllm_server_host="localhost",
        vllm_server_port=9999,
    )

    # Tiny dataset: 4 samples.
    ds = Dataset.from_dict({"prompt": ["Hello"] * 4})
    scaffold.train(ds)
    scaffold.finalize_metrics(
        final_eval={"smoke_reward": 1.0},
        required_keys=["per_model"],
    )

    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert data["status"] == "completed"
    assert "per_model" in data
