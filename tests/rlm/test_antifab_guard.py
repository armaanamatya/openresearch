"""Tests for the anti-fabrication guard.

Part 1: AST detector (_check_no_fabrication wired through scan_code_dir).
Part 2: VRAM evidence guard (fabrication_suspected degradation in run_matrix).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(code_dir: Path, name: str, body: str) -> Path:
    p = code_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _hard_fab(violations) -> list:
    """Return only hard violations whose detail mentions fabrication-related keywords."""
    return [
        v for v in violations
        if v.severity == "hard"
        and any(
            kw in (v.detail or "").lower()
            for kw in ("stub", "fabricat", "hardcoded", "random", "literal", "log-prob")
        )
    ]


# ---------------------------------------------------------------------------
# Part 1 — AST detector
# ---------------------------------------------------------------------------


class TestAstStubModel:
    """(a) Stub model alongside real-model-load signal."""

    def test_linear_stub_with_from_pretrained_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2-1.5B"
# use a stub that satisfies NO-STUB by declaring the exact intended model
model = nn.Linear(1, 1).to("cuda")

output = model(torch.randn(4, 1))
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 fabrication violation, got: {violations}"
        assert any("stub" in v.detail.lower() or "linear" in v.detail.lower() for v in fab)

    def test_identity_stub_with_auto_model_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train.py", """\
from transformers import AutoModel
import torch.nn as nn

model = nn.Identity()
# real load would be: model = AutoModel.from_pretrained("bert-base")
x = model(inputs)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 fabrication violation, got: {violations}"

    def test_no_flag_without_real_model_signal(self, tmp_path: Path, monkeypatch) -> None:
        """A plain Linear layer in a file with no from_pretrained/AutoModel — not suspicious."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "model.py", """\
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) == 0, f"Expected 0 fab violations for clean code, got: {violations}"


class TestAstRandomLogprobs:
    """(b) Random log-probs assigned to logp/teacher/student variable."""

    def test_randn_to_student_logp_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
import torch

student_logp = torch.randn(len(sequence_ids))
teacher_logp = student_logp + torch.randn_like(student_logp) * 0.01
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 fabrication violation, got: {violations}"
        assert any("logp" in v.detail.lower() or "student" in v.detail.lower() for v in fab)

    def test_torch_rand_to_teacher_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train.py", """\
import torch

teacher_log_prob = torch.rand(batch_size, seq_len)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 fabrication violation for teacher_log_prob, got: {violations}"

    def test_np_random_normal_to_logprob_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train.py", """\
import numpy as np

log_prob = np.random.normal(0, 1, size=(32,))
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 violation for np.random.normal→log_prob, got: {violations}"

    def test_randn_to_unrelated_variable_not_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """torch.randn assigned to a non-logprob variable is NOT suspicious."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train.py", """\
import torch

noise = torch.randn(32, 768)
x = noise + embeddings
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) == 0, f"Expected 0 violations for randn→noise, got: {violations}"


class TestAstHardcodedMetric:
    """(c) Hardcoded metric assigned from a literal or literal-conditional."""

    def test_literal_conditional_metric_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
model_key = cell_params["model_key"]
env = cell_params["env"]

# Table-1 numbers hardcoded
final_metric = 0.844 if "3b" in model_key else 0.539

metrics = {"metric": final_metric}
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 violation for hardcoded metric, got: {violations}"

    def test_plain_literal_metric_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
success_rate = 0.716
metrics = {"success_rate": success_rate}
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 violation for hardcoded success_rate, got: {violations}"

    def test_dict_subscript_metric_literal_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
results = {}
results["metric"] = 0.844
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected >=1 violation for results['metric']=literal, got: {violations}"

    def test_metric_computed_from_model_not_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """A metric computed from actual model outputs is NOT flagged."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
from transformers import AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id)

total_correct = 0
total = 0
for batch in dataloader:
    outputs = model(**batch)
    preds = outputs.logits.argmax(-1)
    total_correct += (preds == batch["labels"]).sum().item()
    total += len(batch["labels"])

accuracy = total_correct / total
metrics = {"accuracy": accuracy}
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) == 0, f"Expected 0 fab violations for clean trainer, got: {fab}"

    def test_dict_literal_hardcoded_metric_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """GAP-2: a metric literal INSIDE a dict literal — the grok-4.3 evasion that
        the per-target ``metric = 0.844`` / ``results["metric"] = 0.844`` shapes miss."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
metrics = {"success_rate": 0.844, "accuracy": 0.448}
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected dict-literal hardcoded metric flagged, got: {violations}"

    def test_return_dict_literal_hardcoded_metric_is_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """GAP-2: the same fabrication via ``return {...}`` (no assignment target)."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
def run():
    return {"accuracy": 0.844}
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) >= 1, f"Expected returned dict-literal hardcoded metric flagged, got: {violations}"

    def test_dict_literal_metric_init_not_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """A zero/round init inside a dict literal is NOT a fabricated result."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
metrics = {"success_rate": 0.0, "accuracy": 1.0}
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) == 0, f"Init dict literal should not be flagged, got: {fab}"


class TestVramEvidenceHelpers:
    """GAP-1: the shared VRAM-evidence decision + the GPU-training-claim predicate,
    used by BOTH the cells route and the monolithic run_experiment path."""

    def test_verdict_low_vram_with_claim_is_fabrication(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        from backend.agents.rlm.gpu_cell_runner import vram_evidence_verdict
        assert vram_evidence_verdict(0.01, claims_gpu_training=True) is True

    def test_verdict_real_gpu_run_not_fabrication(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        from backend.agents.rlm.gpu_cell_runner import vram_evidence_verdict
        assert vram_evidence_verdict(8.0, claims_gpu_training=True) is False

    def test_verdict_unmeasured_vram_never_flags(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        from backend.agents.rlm.gpu_cell_runner import vram_evidence_verdict
        assert vram_evidence_verdict(None, claims_gpu_training=True) is False

    def test_verdict_no_gpu_claim_never_flags(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        from backend.agents.rlm.gpu_cell_runner import vram_evidence_verdict
        assert vram_evidence_verdict(0.01, claims_gpu_training=False) is False

    def test_verdict_guard_disabled_never_flags(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "0")
        from backend.agents.rlm.gpu_cell_runner import vram_evidence_verdict
        assert vram_evidence_verdict(0.01, claims_gpu_training=True) is False

    def test_claim_device_cuda(self) -> None:
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training
        assert metrics_claim_gpu_training({"success_rate": 0.844, "device": "cuda"}) is True

    def test_claim_model_id_under_model_key(self) -> None:
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training
        assert metrics_claim_gpu_training({"model": "Qwen/Qwen2.5-3B-Instruct"}) is True

    def test_claim_nested_device(self) -> None:
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training
        assert metrics_claim_gpu_training({"per_model": {"qwen3": {"device": "GPU"}}}) is True

    def test_cpu_metrics_not_claimed(self) -> None:
        """A CPU-only run reporting accuracy/loss with device=cpu is NOT a GPU claim."""
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training
        assert metrics_claim_gpu_training({"accuracy": 0.91, "loss": 0.3, "device": "cpu"}) is False

    def test_path_date_fraction_not_misread_as_model(self) -> None:
        """A '/'-bearing value NOT under a model key (path/date/fraction) must not be
        read as a model claim — the false-positive guard for CPU-only papers."""
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training
        assert metrics_claim_gpu_training(
            {"output_dir": "outputs/run_123", "timestamp": "2026/06/18", "ratio": "3/4"}
        ) is False

    def test_non_dict_metrics_fail_soft(self) -> None:
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training
        assert metrics_claim_gpu_training(None) is False
        assert metrics_claim_gpu_training("just a string") is False


class TestAstGuardDisabled:
    """Respects OPENRESEARCH_ANTIFAB_GUARD=0."""

    def test_flag_disabled_skips_fabrication_check(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "0")
        _write(tmp_path, "train_cell.py", """\
import torch.nn as nn
from transformers import AutoModelForCausalLM
model_id = "Qwen/Qwen2-1.5B"
model = nn.Linear(1, 1)
final_metric = 0.844 if True else 0.539
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) == 0, f"ANTIFAB_GUARD=0 should suppress all fab violations, got: {fab}"


class TestAstCleanRealTrainer:
    """A clean real-trainer (from_pretrained + computed metric) must NOT be flagged."""

    def test_real_trainer_not_flagged(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

model_id = os.environ.get("OPENRESEARCH_CELL_PARAMS", "{}")
params = json.loads(model_id)
model_name = params.get("model_key", "Qwen/Qwen2-1.5B")

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()
total_loss = 0.0
for step, batch in enumerate(train_loader):
    input_ids = batch["input_ids"].to("cuda")
    labels = batch["labels"].to("cuda")
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += loss.item()

avg_loss = total_loss / max(1, step + 1)
reward_mean = evaluate(model, eval_loader)

output_dir = os.environ.get("OPENRESEARCH_CELL_OUTPUT_DIR", ".")
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump({"metric": reward_mean, "loss": avg_loss}, f)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        violations = scan_code_dir(tmp_path)
        fab = _hard_fab(violations)
        assert len(fab) == 0, (
            f"A clean real trainer should not be flagged by the anti-fab guard. Got: {fab}"
        )


class TestAstNoFalsePositives:
    """Conservative-fix regressions: legitimate small layers + metric inits must NOT flag."""

    def test_value_head_not_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """A GRPO value head ``nn.Linear(hidden, 1)`` alongside a real model load is legitimate."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
import torch.nn as nn
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
hidden_size = model.config.hidden_size
value_head = nn.Linear(hidden_size, 1)
proj = nn.Linear(4096, 256)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        fab = _hard_fab(scan_code_dir(tmp_path))
        assert len(fab) == 0, f"A value head / projection must NOT be flagged. Got: {fab}"

    def test_metric_init_literal_not_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """A plain ``success_rate = 0.0`` initialization is not a hardcoded-metric fabrication."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train.py", """\
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
success_rate = 0.0
final_metric = 0.0
for step in range(150):
    success_rate = train_step(model)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        fab = _hard_fab(scan_code_dir(tmp_path))
        assert len(fab) == 0, f"A plain metric init must NOT be flagged. Got: {fab}"

    def test_value_head_does_not_mask_real_stub(self, tmp_path: Path, monkeypatch) -> None:
        """A genuine ``model = nn.Linear(1, 1)`` stub IS still flagged even with a legit value head present."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        _write(tmp_path, "train_cell.py", """\
import torch.nn as nn
from transformers import AutoModelForCausalLM  # declared but not used

model = nn.Linear(1, 1).to("cuda")
value_head = nn.Linear(2048, 1)
""")
        from backend.agents.rlm.preflight_ast import scan_code_dir
        fab = _hard_fab(scan_code_dir(tmp_path))
        assert len(fab) >= 1, f"The real stub model must still be flagged. Got: {fab}"


# ---------------------------------------------------------------------------
# Part 2 — VRAM evidence guard
# ---------------------------------------------------------------------------


def _make_cell(cell_id: str = "cell_0") -> dict[str, Any]:
    return {"id": cell_id, "model_key": "qwen-1.5b", "env": "alfworld"}


def _make_cell_script(tmp_path: Path) -> Path:
    """Write a minimal cell script that exits 0 immediately."""
    p = tmp_path / "train_cell.py"
    p.write_text(
        "import json, os, sys\n"
        "od = os.environ.get('OPENRESEARCH_CELL_OUTPUT_DIR', '.')\n"
        "os.makedirs(od, exist_ok=True)\n"
        "with open(os.path.join(od, 'metrics.json'), 'w') as f:\n"
        "    json.dump({'metric': 0.844}, f)\n",
        encoding="utf-8",
    )
    return p


def _make_vram_mocks(baseline_mib: float | None, peak_mib: float | None):
    """Return context managers that mock VRAM sampling.

    ``_sample_vram_mib`` returns the baseline on the first call (before launch)
    and ``peak_mib`` on subsequent calls.  ``_poll_peak_vram_daemon`` appends
    ``peak_mib`` to the readings list once (simulating one poll iteration).
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx(gcr):
        _sample_calls = [0]

        def _fake_sample(gpu_id):
            _sample_calls[0] += 1
            return baseline_mib

        def _fake_poll(gpu_id, interval_s, stop_flag, readings):
            if peak_mib is not None:
                readings.append(peak_mib)
            # Don't loop — just record one reading and return.

        with (
            patch.object(gcr, "_sample_vram_mib", side_effect=_fake_sample),
            patch.object(gcr, "_poll_peak_vram_daemon", side_effect=_fake_poll),
        ):
            yield

    return _ctx


class TestVramGuard:
    """Part-2: VRAM fabrication degradation in run_matrix.

    We mock both ``_run_cell_subprocess`` (to control returncode/output) and the
    VRAM sampling helpers (``_sample_vram_mib`` for baseline, ``_poll_peak_vram_daemon``
    for the during-run poll).  This avoids any real nvidia-smi dependency while
    exercising the net-delta logic (peak - baseline).
    """

    def test_low_vram_success_is_degraded(self, tmp_path: Path, monkeypatch) -> None:
        """A cell with returncode=0 but net VRAM < threshold → fabrication_suspected + status=failed.

        Scenario: baseline=500 MiB, 2 polls → max=501 MiB → net delta=1 MiB = 0.001 GiB < 1.5 GiB.
        We need >=2 readings to trigger the guard (short runs with <=1 reading → fail-soft/None).
        """
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_MIN_VRAM_GB", "1.5")

        from backend.agents.rlm import gpu_cell_runner

        cell = _make_cell()
        script = _make_cell_script(tmp_path)

        # baseline=500 MiB (background GPU), 2 during-run polls → 501, 500.5 MiB (stub adds ~1 MiB)
        # net delta = (501 - 500) / 1024 ≈ 0.001 GiB < threshold 1.5 GiB → fabrication_suspected
        with (
            patch.object(gpu_cell_runner, "_run_cell_subprocess", return_value=(0, "training complete\n")),
            patch.object(gpu_cell_runner, "_sample_vram_mib", return_value=500.0),
            patch.object(gpu_cell_runner, "_poll_peak_vram_daemon",
                         side_effect=lambda gpu_id, interval_s, stop_flag, readings: readings.extend([501.0, 500.5])),
        ):
            results = gpu_cell_runner.run_matrix(
                [cell],
                str(script),
                output_root=str(tmp_path / "out"),
                gpus=["0"],
            )

        r = results.get("cell_0", {})
        assert r.get("fabrication_suspected") is True, (
            f"Expected fabrication_suspected=True (net=0.001 GiB < threshold), got: {r}"
        )
        assert r.get("status") == "failed", (
            f"Expected status='failed' for fabricated cell, got: {r.get('status')!r}"
        )
        # peak_vram_gb = net delta = (501 - 500) / 1024 ≈ 0.001 GiB
        assert r.get("peak_vram_gb") is not None
        assert r["peak_vram_gb"] < 0.01

    def test_normal_vram_success_is_not_degraded(self, tmp_path: Path, monkeypatch) -> None:
        """A cell with returncode=0 and large net VRAM → NOT flagged.

        Scenario: baseline=500 MiB, 2 during-run polls → max=20980 MiB → net delta=20480 MiB = 20.0 GiB.
        """
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_MIN_VRAM_GB", "1.5")

        from backend.agents.rlm import gpu_cell_runner

        cell = _make_cell()
        script = _make_cell_script(tmp_path)

        with (
            patch.object(gpu_cell_runner, "_run_cell_subprocess", return_value=(0, "epoch 1 done\n")),
            patch.object(gpu_cell_runner, "_sample_vram_mib", return_value=500.0),
            patch.object(gpu_cell_runner, "_poll_peak_vram_daemon",
                         side_effect=lambda gpu_id, interval_s, stop_flag, readings: readings.extend([20980.0, 20900.0])),
        ):
            results = gpu_cell_runner.run_matrix(
                [cell],
                str(script),
                output_root=str(tmp_path / "out"),
                gpus=["0"],
            )

        r = results.get("cell_0", {})
        assert r.get("fabrication_suspected") is False, (
            f"Expected fabrication_suspected=False for high net-VRAM cell, got: {r}"
        )
        assert r.get("status") == "ok", (
            f"Expected status='ok' for high net-VRAM cell, got: {r.get('status')!r}"
        )
        assert r.get("peak_vram_gb") == pytest.approx(20.0, abs=0.5)

    def test_none_vram_never_flagged(self, tmp_path: Path, monkeypatch) -> None:
        """When nvidia-smi is absent (no readings), peak_vram_gb=None → NOT flagged."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_MIN_VRAM_GB", "1.5")

        from backend.agents.rlm import gpu_cell_runner

        cell = _make_cell()
        script = _make_cell_script(tmp_path)

        # nvidia-smi absent: baseline=None, poll appends nothing.
        with (
            patch.object(gpu_cell_runner, "_run_cell_subprocess", return_value=(0, "done\n")),
            patch.object(gpu_cell_runner, "_sample_vram_mib", return_value=None),
            patch.object(gpu_cell_runner, "_poll_peak_vram_daemon",
                         side_effect=lambda gpu_id, interval_s, stop_flag, readings: None),
        ):
            results = gpu_cell_runner.run_matrix(
                [cell],
                str(script),
                output_root=str(tmp_path / "out"),
                gpus=["0"],
            )

        r = results.get("cell_0", {})
        assert r.get("fabrication_suspected") is False, (
            f"Expected fabrication_suspected=False when VRAM unavailable, got: {r}"
        )
        assert r.get("status") == "ok", (
            f"Expected status='ok' when VRAM measurement unavailable, got: {r.get('status')!r}"
        )
        assert r.get("peak_vram_gb") is None

    def test_guard_disabled_low_vram_not_degraded(self, tmp_path: Path, monkeypatch) -> None:
        """With OPENRESEARCH_ANTIFAB_GUARD=0, low VRAM cells are NOT degraded."""
        monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "0")

        from backend.agents.rlm import gpu_cell_runner

        cell = _make_cell()
        script = _make_cell_script(tmp_path)

        # Even with near-zero net VRAM, guard=0 means no check.
        with (
            patch.object(gpu_cell_runner, "_run_cell_subprocess", return_value=(0, "done\n")),
            # Don't mock samplers — guard is disabled, they should never be called.
        ):
            results = gpu_cell_runner.run_matrix(
                [cell],
                str(script),
                output_root=str(tmp_path / "out"),
                gpus=["0"],
            )

        r = results.get("cell_0", {})
        assert r.get("status") == "ok", (
            f"Expected status='ok' when ANTIFAB_GUARD=0, got: {r.get('status')!r}"
        )
        assert not r.get("fabrication_suspected"), (
            f"Expected fabrication_suspected=False/absent when guard disabled, got: {r}"
        )
