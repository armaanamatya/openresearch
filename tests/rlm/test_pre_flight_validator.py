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
