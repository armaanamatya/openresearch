"""Tests for the torch-redundancy guardrail.

Symptom: every recent v10-class RunPod run failed with
``ModuleNotFoundError: matplotlib`` because pip aborted mid-stream while
re-downloading the 755 MB torch wheel from PyPI.  Root cause: requirements.txt
listed torch even though the ``runpod/pytorch`` base image already has it.

Two guarantees pinned here:

1. ``synthesize_requirements_txt`` strips torch/torchvision/torchaudio when
   given a ``runpod/pytorch*`` base image, but keeps them on any other base
   (e.g. ``python:3.11-slim``).

2. ``validate_code_pre_flight`` raises a hard pre-flight violation when an
   existing ``requirements.txt`` re-installs torch on a runpod/pytorch base —
   covers the case where the agent writes its own requirements.txt and
   bypasses the auto-derive.
"""

from __future__ import annotations

from pathlib import Path

from backend.agents.rlm import requirements_derive as rd
from backend.agents.rlm.pre_flight_validator import validate_code_pre_flight


# ---------------------------------------------------------------------------
# Auto-derive: strip torch on runpod/pytorch
# ---------------------------------------------------------------------------


def test_synthesize_strips_torch_on_runpod_pytorch_base() -> None:
    df = """\
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
RUN pip install matplotlib==3.8.0 numpy==1.26.4 tqdm==4.66.0
"""
    out = rd.synthesize_requirements_txt(
        df, base_image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    )
    # Check actual PACKAGE LINES (not comments) for torch — comments may legitimately
    # mention "torch/torchvision/torchaudio stripped".
    pkg_lines = [l for l in out.splitlines() if l and not l.startswith("#")]
    pkg_blob = "\n".join(pkg_lines).lower()
    assert "torch==" not in pkg_blob
    assert "torchvision" not in pkg_blob
    assert "torchaudio" not in pkg_blob
    assert "matplotlib==3.8.0" in pkg_blob
    assert "numpy==1.26.4" in pkg_blob
    assert "tqdm==4.66.0" in pkg_blob


def test_synthesize_keeps_torch_on_python_slim_base() -> None:
    df = """\
RUN pip install --no-cache-dir torch==2.2.0
RUN pip install matplotlib==3.8.0
"""
    out = rd.synthesize_requirements_txt(df, base_image="python:3.11-slim")
    assert "torch==2.2.0" in out
    assert "matplotlib==3.8.0" in out


def test_synthesize_keeps_torch_when_no_base_image() -> None:
    df = "RUN pip install torch==2.2.0 matplotlib==3.8.0\n"
    out = rd.synthesize_requirements_txt(df, base_image=None)
    assert "torch==2.2.0" in out


def test_synthesize_strips_unpinned_torch() -> None:
    df = "RUN pip install torch torchvision matplotlib\n"
    out = rd.synthesize_requirements_txt(df, base_image="runpod/pytorch:foo")
    pkg_lines = [l for l in out.splitlines() if l and not l.startswith("#")]
    pkg_blob = "\n".join(pkg_lines).lower()
    assert "matplotlib" in pkg_blob
    # Strict: package lines must not have torch or torchvision as their own entry
    assert "torch\n" not in pkg_blob + "\n"
    assert "torchvision" not in pkg_blob


def test_ensure_passes_base_image_through(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (tmp_path / "Dockerfile").write_text(
        "RUN pip install torch==2.2.0 matplotlib==3.8.0 numpy==1.26.4\n"
    )
    out = rd.ensure_requirements_txt(
        code_dir,
        base_image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    )
    assert out is not None
    text = out.read_text()
    assert "torch==2.2.0" not in text
    assert "matplotlib==3.8.0" in text
    assert "numpy==1.26.4" in text


# ---------------------------------------------------------------------------
# Pre-flight guardrail: hard violation when agent writes torch in requirements
# ---------------------------------------------------------------------------


def test_pre_flight_blocks_torch_in_requirements_on_runpod_pytorch(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text(
        "torch==2.2.0\nmatplotlib==3.8.0\n"
    )
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None,
        base_image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    )
    torch_v = [v for v in violations if "torch" in v.detail.lower()]
    assert len(torch_v) == 1
    assert torch_v[0].severity == "hard"
    assert "pre-installed" in torch_v[0].detail.lower()


def test_pre_flight_blocks_all_three_torch_packages(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text(
        "torch==2.2.0\ntorchvision==0.17.0\ntorchaudio==2.2.0\nmatplotlib==3.8.0\n"
    )
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None,
        base_image="runpod/pytorch:devel",
    )
    torch_v = [v for v in violations if "Experiment execution" in v.area]
    assert len(torch_v) == 3


def test_pre_flight_no_violation_on_python_slim(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text("torch==2.2.0\nmatplotlib==3.8.0\n")
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None, base_image="python:3.11-slim",
    )
    torch_v = [v for v in violations if "torch" in v.detail.lower()]
    assert torch_v == []


def test_pre_flight_skips_torch_check_when_no_base_image(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text("torch==2.2.0\n")
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None, base_image=None,
    )
    torch_v = [v for v in violations if "torch" in v.detail.lower()]
    assert torch_v == []


def test_pre_flight_skips_torch_check_when_no_requirements(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None,
        base_image="runpod/pytorch:devel",
    )
    torch_v = [v for v in violations if "torch" in v.detail.lower()]
    assert torch_v == []


def test_pre_flight_handles_unversioned_torch(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text("torch\nmatplotlib\n")
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None,
        base_image="runpod/pytorch:devel",
    )
    torch_v = [v for v in violations if "torch" in v.detail.lower()]
    assert len(torch_v) == 1


def test_pre_flight_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text(
        "# torch should be in here\n\n# but not actually\nmatplotlib\n"
    )
    violations = validate_code_pre_flight(
        code_dir, paper_targets=None,
        base_image="runpod/pytorch:devel",
    )
    torch_v = [v for v in violations if "torch" in v.detail.lower()]
    assert torch_v == []
