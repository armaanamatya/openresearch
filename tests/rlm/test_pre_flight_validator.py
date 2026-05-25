"""Tests for backend.agents.rlm.pre_flight_validator.

Pinned guarantees:

  * No paper_targets → empty list (fail-soft default).
  * Variant present (any case / underscore form) → no violation.
  * Variant missing AND not omitted → hard violation.
  * Variant honestly omitted via metrics["omitted"]["<id>"] → no violation.
  * Surrogate class name (TinyMLP, MockModel, …) → hard violation per match.
  * Real architecture name (Qwen3Model, ResNet50, …) → no violation.
  * Dataset subset detected (range(0, 4000) when full is 60000) → hard.
  * Required metric key absent → soft violation (NOT hard).
  * Required artifact name absent → soft violation.
  * train.py SyntaxError → one hard violation with line number.
  * Completes in <500 ms on a typical paper directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from backend.agents.rlm.pre_flight_validator import (
    PreFlightViolation,
    validate_code_pre_flight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _hard(vs: list[PreFlightViolation]) -> list[PreFlightViolation]:
    return [v for v in vs if v.severity == "hard"]


def _soft(vs: list[PreFlightViolation]) -> list[PreFlightViolation]:
    return [v for v in vs if v.severity == "soft"]


# ---------------------------------------------------------------------------
# Fail-soft defaults
# ---------------------------------------------------------------------------


def test_no_paper_targets_returns_empty(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "print('hi')\n")
    assert validate_code_pre_flight(tmp_path, None) == []


def test_empty_paper_targets_returns_empty(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "print('hi')\n")
    assert validate_code_pre_flight(tmp_path, {}) == []


def test_no_code_files_returns_empty(tmp_path: Path) -> None:
    # No train.py and no exp_*.py — fall-through is empty.
    targets = {"variants_required": ["baseline"]}
    assert validate_code_pre_flight(tmp_path, targets) == []


def test_missing_code_dir_returns_empty(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist"
    targets = {"variants_required": ["baseline"]}
    assert validate_code_pre_flight(nonexistent, targets) == []


# ---------------------------------------------------------------------------
# Variants check
# ---------------------------------------------------------------------------


def test_variant_present_in_code_no_violation(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "MODEL = 'qwen3_1_7b'\n")
    out = validate_code_pre_flight(
        tmp_path, {"variants_required": ["qwen3_1_7b"]},
    )
    assert _hard(out) == []


def test_variant_present_with_hyphen_form_no_violation(tmp_path: Path) -> None:
    # YAML says qwen3_1_7b; code says "Qwen3-1.7B" (HF repo id form).
    _write(tmp_path / "train.py", 'MODEL_ID = "Qwen/Qwen3-1.7B-Instruct"\n')
    out = validate_code_pre_flight(
        tmp_path, {"variants_required": ["qwen3_1_7b"]},
    )
    assert _hard(out) == []


def test_variant_missing_and_not_omitted_hard_violation(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "MODEL = 'qwen3_1_7b'\n")
    out = validate_code_pre_flight(
        tmp_path, {"variants_required": ["qwen3_1_7b", "qwen2_5_7b"]},
    )
    hard = _hard(out)
    assert len(hard) == 1
    v = hard[0]
    assert v.area == "Experiment execution and reproducibility"
    assert "qwen2_5_7b" in v.detail
    assert "missing" in v.detail.lower()


def test_variant_honestly_omitted_no_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        'MODEL = "qwen3_1_7b"\n'
        'metrics = {"omitted": {"qwen2_5_7b": "tight compute budget"}}\n',
    )
    out = validate_code_pre_flight(
        tmp_path, {"variants_required": ["qwen3_1_7b", "qwen2_5_7b"]},
    )
    assert _hard(out) == []


def test_variant_check_skips_non_string_entries(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "MODEL = 'baseline'\n")
    out = validate_code_pre_flight(
        tmp_path,
        {"variants_required": ["baseline", None, 42, ""]},  # type: ignore[list-item]
    )
    assert _hard(out) == []


# ---------------------------------------------------------------------------
# Dataset-size check
# ---------------------------------------------------------------------------


def test_dataset_range_subset_hard_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "for i in range(0, 4000):\n    pass\n",
    )
    out = validate_code_pre_flight(
        tmp_path,
        {"variants_required": [], "train_size_full": 60000},
    )
    hard = _hard(out)
    assert len(hard) == 1
    v = hard[0]
    assert v.area == "Data fidelity and preparation"
    assert "4000" in v.detail
    assert "60000" in v.detail


def test_dataset_slice_subset_hard_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "trainset = list(range(60000))\n"
        "subset = trainset[:4000]\n",
    )
    out = validate_code_pre_flight(
        tmp_path, {"train_size_full": 60000},
    )
    hard = _hard(out)
    assert len(hard) == 1
    assert hard[0].area == "Data fidelity and preparation"


def test_dataset_kwarg_subset_hard_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "from torchvision.datasets import MNIST\n"
        "ds = MNIST(root='.', train=True, num_samples=2000)\n",
    )
    out = validate_code_pre_flight(
        tmp_path, {"train_size_full": 60000},
    )
    hard = _hard(out)
    assert len(hard) == 1
    assert "2000" in hard[0].detail


def test_dataset_at_full_size_no_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "for i in range(0, 60000):\n    pass\n",
    )
    out = validate_code_pre_flight(
        tmp_path, {"train_size_full": 60000},
    )
    # 60000 is not < floor (90% of 60000 = 54000), so no violation.
    assert _hard(out) == []


def test_dataset_no_full_size_no_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "for i in range(0, 100):\n    pass\n",
    )
    # No train_size_full → no contract → no dataset violation.
    out = validate_code_pre_flight(
        tmp_path, {"variants_required": []},
    )
    assert _hard(out) == []


def test_dataset_small_numbers_ignored(tmp_path: Path) -> None:
    # Batch size, layer dim, etc. — small literals must not trigger.
    _write(
        tmp_path / "train.py",
        "batch_size = 32\n"
        "for epoch in range(0, 10):\n    pass\n"
        "for i in range(0, 60000):\n    pass\n",
    )
    out = validate_code_pre_flight(
        tmp_path, {"train_size_full": 60000},
    )
    assert _hard(out) == []


# ---------------------------------------------------------------------------
# Surrogate detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_name",
    [
        "TinyMLP",
        "MockModel",
        "DummyNet",
        "SmokeTestModel",
        "FakeTransformer",
        "StubNet",
        "ToyResNet",
        "SurrogateModel",
    ],
)
def test_surrogate_class_name_detected(tmp_path: Path, class_name: str) -> None:
    _write(
        tmp_path / "train.py",
        f"import torch.nn as nn\n\nclass {class_name}(nn.Module):\n    pass\n",
    )
    out = validate_code_pre_flight(tmp_path, {"variants_required": []})
    hard = _hard(out)
    assert len(hard) == 1
    v = hard[0]
    assert v.area == "Method fidelity to the paper"
    assert class_name in v.detail


def test_real_architecture_name_no_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "import torch.nn as nn\n\n"
        "class Qwen3Model(nn.Module):\n    pass\n"
        "class ResNet50(nn.Module):\n    pass\n",
    )
    out = validate_code_pre_flight(tmp_path, {"variants_required": []})
    assert _hard(out) == []


def test_multiple_surrogates_produce_multiple_violations(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        "class TinyMLP: pass\n"
        "class MockOptimizer: pass\n",
    )
    out = validate_code_pre_flight(tmp_path, {"variants_required": []})
    hard = _hard(out)
    assert len(hard) == 2


def test_surrogate_detection_walks_exp_files(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "from exp_a import run\nrun()\n")
    _write(tmp_path / "exp_a.py", "class TinyResNet: pass\n")
    out = validate_code_pre_flight(tmp_path, {"variants_required": []})
    hard = _hard(out)
    assert len(hard) == 1
    assert "TinyResNet" in hard[0].detail


# ---------------------------------------------------------------------------
# Required metric keys (soft)
# ---------------------------------------------------------------------------


def test_required_key_present_no_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        'metrics = {}\n'
        'metrics["mnist_baseline_final_acc"] = 0.985\n',
    )
    out = validate_code_pre_flight(
        tmp_path,
        {"required_metrics_keys": ["mnist_baseline_final_acc"]},
    )
    assert _soft(out) == []
    assert _hard(out) == []


def test_required_key_missing_soft_violation(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "print('no metrics here')\n")
    out = validate_code_pre_flight(
        tmp_path,
        {"required_metrics_keys": ["mnist_baseline_final_acc"]},
    )
    soft = _soft(out)
    assert len(soft) == 1
    v = soft[0]
    assert v.area == "Evaluation protocol and metric correctness"
    assert "mnist_baseline_final_acc" in v.detail
    # Soft violations must NOT escalate to hard.
    assert _hard(out) == []


# ---------------------------------------------------------------------------
# Required artifact names (soft)
# ---------------------------------------------------------------------------


def test_required_artifact_present_no_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "train.py",
        'open("training_curves.json", "w").write("{}")\n',
    )
    out = validate_code_pre_flight(
        tmp_path,
        {"required_artifacts": ["training_curves.json"]},
    )
    assert _soft(out) == []


def test_required_artifact_glob_present(tmp_path: Path) -> None:
    # The glob "fig_*.png" should be satisfied by "fig_" appearing anywhere.
    _write(
        tmp_path / "train.py",
        'plt.savefig("fig_mnist_dropout.png")\n',
    )
    out = validate_code_pre_flight(
        tmp_path,
        {"required_artifacts": ["fig_*.png"]},
    )
    assert _soft(out) == []


def test_required_artifact_missing_soft_violation(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "print('nothing')\n")
    out = validate_code_pre_flight(
        tmp_path,
        {"required_artifacts": ["README.md"]},
    )
    soft = _soft(out)
    assert len(soft) == 1
    assert soft[0].area == "Artifact completeness and provenance"
    assert "README.md" in soft[0].detail


# ---------------------------------------------------------------------------
# SyntaxError fail-soft
# ---------------------------------------------------------------------------


def test_syntax_error_produces_one_hard_violation(tmp_path: Path) -> None:
    _write(tmp_path / "train.py", "def broken(:\n    pass\n")
    out = validate_code_pre_flight(
        tmp_path, {"variants_required": ["baseline"]},
    )
    hard = _hard(out)
    # At least one hard violation describing the syntax error must appear.
    syn_violations = [v for v in hard if "SyntaxError" in v.detail]
    assert len(syn_violations) == 1
    assert "train.py" in syn_violations[0].detail


# ---------------------------------------------------------------------------
# Performance budget
# ---------------------------------------------------------------------------


def test_validator_completes_under_500ms_on_realistic_input(tmp_path: Path) -> None:
    # Synthesise a realistic train.py + a few exp_*.py files. 5 KB each is
    # representative of an agent's output; total ~30 KB across files.
    realistic_body = (
        "import torch\nimport torch.nn as nn\n\n"
        + "class RealModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.l = nn.Linear(784, 10)\n    def forward(self, x):\n        return self.l(x)\n\n"
        + "metrics = {}\nmetrics['acc'] = 0.99\n"
        + "# padding\n" * 200
    )
    _write(tmp_path / "train.py", realistic_body)
    for i in range(4):
        _write(tmp_path / f"exp_{i}.py", realistic_body)

    targets = {
        "variants_required": ["baseline", "dropout", "bn"],
        "train_size_full": 60000,
        "required_metrics_keys": ["mnist_baseline_final_acc", "cifar10_baseline_final_acc"],
        "required_artifacts": ["README.md", "training_curves.json", "fig_*.png"],
    }
    t0 = time.perf_counter()
    validate_code_pre_flight(tmp_path, targets)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"pre-flight took {elapsed:.3f}s (budget 0.5 s)"


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tensor device-mismatch — pins the Dropout 2026-05-24 Exp 1+3 crash class
# ---------------------------------------------------------------------------


_DROPOUT_BAD = """\
import torch

class AdamOptimizer:
    def __init__(self, params):
        self.params = list(params)
        self.m = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad.data
            self.m[i] = 0.9 * self.m[i] + 0.1 * g  # crashes if mixed device


def main():
    model = MyModel()
    opt = AdamOptimizer(model.parameters())  # built BEFORE .to() → bug
    run_epochs(model, loader, opt, torch.device("cuda"), 10)


def run_epochs(model, loader, optimizer, device, epochs):
    model.to(device)  # ← THE BUG: optimizer was built before this
    for _ in range(epochs):
        for xb, yb in loader:
            xb.to(device)  # this is OK (batch tensor) — should NOT fire
            optimizer.step()
"""

_FIXED_PATTERN = """\
import torch


def main():
    model = MyModel()
    device = torch.device("cuda")
    model.to(device)
    optimizer = AdamOptimizer(model.parameters())
    train(model, optimizer, device)


def train(model, optimizer, device):
    # No .to() here — model was moved BEFORE optimizer construction.
    for _ in range(10):
        optimizer.step()
"""


def test_tensor_device_mismatch_catches_dropout_bug(tmp_path: Path) -> None:
    """The exact 2026-05-24 Dropout Exp 1+3 pattern must be hard-blocked."""
    _write(tmp_path / "train.py", _DROPOUT_BAD)
    out = validate_code_pre_flight(tmp_path, {})  # NO paper_targets — invariant check
    hard = _hard(out)
    assert len(hard) >= 1
    bug = next((v for v in hard if "to(...)" in v.detail and "run_epochs" in v.detail), None)
    assert bug is not None, f"didn't catch the run_epochs(...) bug, got: {[v.detail[:60] for v in hard]}"
    assert "cuda:0 and cpu" in bug.detail
    assert "BEFORE" in bug.hint  # standard PyTorch idiom hint


def test_tensor_device_mismatch_no_false_positive_on_correct_idiom(tmp_path: Path) -> None:
    """The standard PyTorch idiom (model.to() at outer scope, optimizer
    constructed after) must NOT trigger the violation."""
    _write(tmp_path / "train.py", _FIXED_PATTERN)
    out = validate_code_pre_flight(tmp_path, {})
    tensor_v = [v for v in out if "cuda:0 and cpu" in v.detail]
    assert tensor_v == []


def test_tensor_device_mismatch_no_violation_when_no_optimizer_param(tmp_path: Path) -> None:
    """A function with .to(device) but no `optimizer` parameter is innocent."""
    body = """\
def setup(model, device):
    model.to(device)
    return model
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    tensor_v = [v for v in out if "cuda:0 and cpu" in v.detail]
    assert tensor_v == []


def test_tensor_device_mismatch_detects_opt_arg_alias(tmp_path: Path) -> None:
    """The shorthand `opt` is also a smoking-gun parameter name."""
    body = """\
def setup():
    model.parameters()  # marks model as the model
def train_loop(model, loader, opt, device):
    model.to(device)
    opt.step()
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    hard = _hard(out)
    assert any("train_loop" in v.detail for v in hard)


def test_tensor_device_mismatch_handles_string_cuda_arg(tmp_path: Path) -> None:
    """`.to(\"cuda\")` (string literal) is also caught."""
    body = """\
def setup():
    _ = model.parameters()
def go(model, optimizer):
    model.to("cuda")
    optimizer.step()
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    assert any("to(...)" in v.detail and "go" in v.detail for v in _hard(out))


def test_tensor_device_mismatch_handles_torch_device_call(tmp_path: Path) -> None:
    """`.to(torch.device(\"cuda\"))` is caught via the call-attr-call form."""
    body = """\
import torch
def setup():
    _ = model.parameters()
def go(model, optimizer):
    model.to(torch.device("cuda"))
    optimizer.step()
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    assert any("to(...)" in v.detail and "go" in v.detail for v in _hard(out))


def test_tensor_device_mismatch_catches_dot_cuda_form(tmp_path: Path) -> None:
    """`model.cuda()` is the same bug as `model.to('cuda')` — must be caught."""
    body = """\
def main():
    model = MyModel()
    opt = AdamOptimizer(model.parameters())  # built BEFORE .cuda()
    go(model, opt)

def go(model, optimizer):
    model.cuda()  # ← same bug, different syntax
    optimizer.step()
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    hard = _hard(out)
    assert any("model.cuda(...)" in v.detail for v in hard), \
        f"didn't catch .cuda() form: {[v.detail[:80] for v in hard]}"


def test_tensor_device_mismatch_skips_batch_tensors(tmp_path: Path) -> None:
    """xb.to(device) and yb.to(device) inside a train loop are FINE —
    they're batch tensors, not the model. Must not flag them as bugs."""
    body = """\
def main():
    model = MyModel().to(device)
    opt = AdamOptimizer(model.parameters())

def run_epochs(model, loader, optimizer, device):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.step()
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    tensor_v = [v for v in _hard(out) if "cuda:0 and cpu" in v.detail]
    assert tensor_v == [], f"false positive on batch tensor: {[v.detail[:80] for v in tensor_v]}"


def test_tensor_device_mismatch_fail_soft_on_syntax_error(tmp_path: Path) -> None:
    """A file with a SyntaxError emits the syntax violation, but the
    tensor-device check shouldn't crash the whole pre-flight."""
    _write(tmp_path / "train.py", "def f(:\n")  # broken
    out = validate_code_pre_flight(tmp_path, {})
    # No tensor-device violation (can't parse), and pre-flight still returned
    # something (the syntax violation, or empty in fail-soft case).
    tensor_v = [v for v in out if "cuda:0 and cpu" in v.detail]
    assert tensor_v == []  # check didn't crash, just had nothing to walk


def test_tensor_device_mismatch_works_without_paper_targets(tmp_path: Path) -> None:
    """Invariant check fires even when paper_targets is None."""
    _write(tmp_path / "train.py", _DROPOUT_BAD)
    out = validate_code_pre_flight(tmp_path, None)
    assert any("cuda:0 and cpu" in v.detail for v in _hard(out))


# ---------------------------------------------------------------------------
# Learning-rate sanity — pins the Dropout 2026-05-25 NaN-training regression
# ---------------------------------------------------------------------------


def test_lr_kwarg_above_one_is_hard_blocked(tmp_path: Path) -> None:
    """Adam(lr=10) — the exact 2026-05-25 Dropout pattern — must be blocked."""
    body = """\
import torch
opt = torch.optim.Adam(model.parameters(), lr=10.0)
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    hard = _hard(out)
    assert any("lr=10.0" in v.detail for v in hard), \
        f"didn't catch lr=10.0: {[v.detail[:80] for v in hard]}"


def test_lr_assignment_above_one_is_hard_blocked(tmp_path: Path) -> None:
    body = """\
learning_rate = 5.0
lr = 10
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    hard = _hard(out)
    # Both must be flagged.
    assert any("learning_rate=5.0" in v.detail for v in hard)
    assert any("lr=10" in v.detail for v in hard)


def test_lr_dict_literal_above_one_is_hard_blocked(tmp_path: Path) -> None:
    """The Dropout pattern — lr inside a config dict in a list of configs."""
    body = """\
MLP_OPT_CONFIGS_STOCH = [
    {"lr": 10.0, "momentum": 0.9},
    {"lr": 0.01, "momentum": 0.5},
]
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    hard = _hard(out)
    assert any("lr=10.0" in v.detail for v in hard)
    # The sane lr=0.01 must NOT be flagged.
    assert not any("lr=0.01" in v.detail for v in hard)


def test_lr_within_sane_range_not_flagged(tmp_path: Path) -> None:
    body = """\
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    lr_v = [v for v in _hard(out) if "lr=" in v.detail]
    assert lr_v == []


def test_lr_one_is_borderline_accepted(tmp_path: Path) -> None:
    """lr=1.0 is the upper bound — accepted (some scheduled-LR codes hit it)."""
    body = "opt = SGD(model.parameters(), lr=1.0)\n"
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    assert not any("lr=1.0" in v.detail for v in _hard(out))


def test_lr_too_small_is_blocked(tmp_path: Path) -> None:
    """lr=1e-9 effectively disables learning — block."""
    body = "opt = Adam(model.parameters(), lr=1e-9)\n"
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    assert any("lr=" in v.detail for v in _hard(out))


def test_lr_aliases_caught(tmp_path: Path) -> None:
    """The check covers common LR aliases (alpha, base_lr, max_lr, etc.)."""
    body = """\
alpha = 2.5
base_lr = 5.0
max_lr = 3.0
init_lr = 8.0
initial_lr = 6.0
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    hard = _hard(out)
    aliases = ("alpha", "base_lr", "max_lr", "init_lr", "initial_lr")
    for a in aliases:
        assert any(f"{a}=" in v.detail for v in hard), f"missed alias: {a}"


def test_lr_negative_literal_caught(tmp_path: Path) -> None:
    """``-1e-3`` via UnaryOp(USub, Constant) — must be parsed correctly."""
    body = "opt = Adam(model.parameters(), lr=-1e-3)\n"
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    # -1e-3 is technically below the lower bound (negative); block it.
    assert any("lr=-0.001" in v.detail for v in _hard(out))


def test_lr_non_literal_value_skipped(tmp_path: Path) -> None:
    """When lr is a Name/Expr (not a literal), we can't statically tell — skip."""
    body = """\
lr = some_function()
opt = Adam(model.parameters(), lr=cfg.learning_rate)
"""
    _write(tmp_path / "train.py", body)
    out = validate_code_pre_flight(tmp_path, {})
    assert not any("lr=" in v.detail for v in _hard(out))


def test_violation_to_dict_carries_all_fields() -> None:
    v = PreFlightViolation(
        severity="hard",
        area="Method fidelity to the paper",
        detail="surrogate detected",
        hint="use the real model",
    )
    assert v.to_dict() == {
        "severity": "hard",
        "area": "Method fidelity to the paper",
        "detail": "surrogate detected",
        "hint": "use the real model",
    }
