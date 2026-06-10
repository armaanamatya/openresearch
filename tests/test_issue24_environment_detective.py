"""Tests for Issue #24: Environment Detective Agent.

Validates:
- Offline mode generates valid Dockerfile for PPO
- Framework inference works correctly
- Assumptions are generated for all inferred values
- Output files are written correctly
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.environment_detective import (
    run_offline,
    _generate_dockerfile,
    _infer_framework,
    _infer_python_version,
    _dataset_packages,
    _generate_assumptions,
)
from backend.agents.schemas import (
    Assumption,
    DatasetRequirement,
    EnvironmentSpec,
    MetricSpec,
    PaperClaimMap,
    RiskLevel,
    TrainingRecipe,
)


def _ppo_claim_map() -> PaperClaimMap:
    """Fixture: PPO paper claim map."""
    return PaperClaimMap(
        core_contribution="Proximal Policy Optimization for reinforcement learning",
        claims=[{"method": "PPO", "dataset": "CartPole-v1", "metric": "reward", "expected_result": ">=475"}],
        datasets=[
            DatasetRequirement(name="CartPole-v1", source="Gymnasium", download_method="bundled"),
        ],
        metrics=[MetricSpec(name="mean_reward", definition="Mean over 100 episodes", target_value="475")],
        model_architecture="2-layer MLP with 64 hidden units",
        training_recipe=TrainingRecipe(optimizer="Adam", learning_rate="3e-4"),
        evaluation_protocol="100 episodes, report mean reward",
        hardware_clues=[],
    )


class TestRunOffline:
    def test_produces_environment_spec(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert isinstance(result, EnvironmentSpec)

    def test_generates_pytorch_dockerfile(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert "FROM python:3.11-slim" in result.dockerfile
        assert "torch==" in result.dockerfile
        assert "pip install" in result.dockerfile

    def test_includes_gymnasium_for_cartpole(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert "gymnasium" in result.pip_packages

    def test_uses_cpu_torch_index(self, tmp_path: Path):
        """PPO on CartPole doesn't need GPU."""
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert "download.pytorch.org/whl/cpu" in result.dockerfile

    def test_framework_is_pytorch(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert result.framework == "pytorch"
        assert result.framework_version == "2.2.0"

    def test_python_version(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert result.python_version == "3.11"

    def test_generates_assumptions(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert len(result.assumptions) >= 2
        ids = [a.assumption_id for a in result.assumptions]
        assert all(id.startswith("ENV") for id in ids)

    def test_assumption_for_framework_version(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        fw_assumptions = [a for a in result.assumptions if "pytorch" in a.detail.lower()]
        assert len(fw_assumptions) >= 1
        assert fw_assumptions[0].chosen_value == "2.2.0"

    def test_writes_dockerfile_to_disk(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        dockerfile = tmp_path / "prj_ppo" / "Dockerfile"
        assert dockerfile.exists()
        content = dockerfile.read_text()
        assert "FROM python:" in content

    def test_writes_spec_json_to_disk(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        spec_path = tmp_path / "prj_ppo" / "environment_spec.json"
        assert spec_path.exists()
        data = json.loads(spec_path.read_text())
        reconstructed = EnvironmentSpec(**data)
        assert reconstructed.framework == "pytorch"

    def test_includes_matplotlib_for_plots(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _ppo_claim_map())
        assert "matplotlib" in result.pip_packages


class TestFrameworkInference:
    def test_pytorch_default(self):
        claim_map = PaperClaimMap(core_contribution="PPO RL method")
        fw, ver = _infer_framework(claim_map)
        assert fw == "pytorch"

    def test_tensorflow_detected(self):
        claim_map = PaperClaimMap(core_contribution="Using TensorFlow for training")
        fw, ver = _infer_framework(claim_map)
        assert fw == "tensorflow"

    def test_jax_detected(self):
        claim_map = PaperClaimMap(core_contribution="JAX-based implementation with Flax")
        fw, ver = _infer_framework(claim_map)
        assert fw == "jax"


class TestDockerfileGeneration:
    def test_basic_dockerfile(self):
        packages = {"torch": "2.2.0", "numpy": "1.26.4"}
        df = _generate_dockerfile("3.11", packages)
        assert "FROM python:3.11-slim" in df
        assert "torch==2.2.0" in df
        assert "numpy==1.26.4" in df
        assert "WORKDIR /workspace" in df

    def test_torch_uses_cpu_index(self):
        packages = {"torch": "2.2.0"}
        df = _generate_dockerfile("3.11", packages)
        assert "download.pytorch.org/whl/cpu" in df

    def test_torch_uses_cpu_index_for_gpu_mode_off(self):
        packages = {"torch": "2.2.0"}
        df = _generate_dockerfile("3.11", packages, gpu_mode="off")
        assert "download.pytorch.org/whl/cpu" in df

    def test_torch_uses_cpu_index_for_gpu_mode_auto(self):
        # --gpu-mode auto does NOT trigger LocalDocker GPU passthrough, so the
        # container won't see CUDA either way. Keep the smaller CPU wheel.
        packages = {"torch": "2.2.0"}
        df = _generate_dockerfile("3.11", packages, gpu_mode="auto")
        assert "download.pytorch.org/whl/cpu" in df

    def test_torch_uses_cuda_wheel_for_gpu_mode_prefer(self, monkeypatch):
        # prefer/max only select the CUDA wheel when the HOST has an NVIDIA GPU;
        # effective_gpu_mode downgrades to the CPU wheel on a GPU-less host
        # (CI runners and macOS dev boxes). Pin host-GPU present so this test
        # deterministically exercises the passthrough path it's meant to cover.
        monkeypatch.setattr(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            lambda: True,
        )
        packages = {"torch": "2.2.0"}
        df = _generate_dockerfile("3.11", packages, gpu_mode="prefer")
        # No --index-url override → default PyPI ships CUDA wheel.
        assert "download.pytorch.org/whl/cpu" not in df
        assert "torch==2.2.0" in df

    def test_torch_uses_cuda_wheel_for_gpu_mode_max(self, monkeypatch):
        # See sibling test above: pin host-GPU present (GPU-less hosts downgrade).
        monkeypatch.setattr(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            lambda: True,
        )
        packages = {"torch": "2.2.0"}
        df = _generate_dockerfile("3.11", packages, gpu_mode="max")
        assert "download.pytorch.org/whl/cpu" not in df
        assert "torch==2.2.0" in df

    def test_runpod_sandbox_uses_pytorch_base_image(self):
        packages = {"torch": "2.1.0", "transformers": "4.44.0", "numpy": "1.26.4"}
        df = _generate_dockerfile("3.10", packages, sandbox_mode="runpod")
        assert "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04" in df
        assert "FROM runpod/pytorch" in df

    def test_runpod_sandbox_skips_torch_install(self):
        packages = {"torch": "2.1.0", "transformers": "4.44.0"}
        df = _generate_dockerfile("3.10", packages, sandbox_mode="runpod")
        # torch is pre-installed in the base image; must not reinstall
        lines = [l.strip() for l in df.splitlines() if "pip install" in l and "torch==" in l]
        assert lines == [], f"torch should not be pip-installed on runpod base; found: {lines}"

    def test_runpod_sandbox_installs_other_packages(self):
        packages = {"torch": "2.1.0", "transformers": "4.44.0", "numpy": "1.26.4"}
        df = _generate_dockerfile("3.10", packages, sandbox_mode="runpod")
        assert "transformers==4.44.0" in df
        assert "numpy==1.26.4" in df

    def test_non_runpod_sandbox_uses_slim_base(self):
        packages = {"torch": "2.1.0"}
        for mode in (None, "local", "docker", "auto"):
            df = _generate_dockerfile("3.10", packages, sandbox_mode=mode)
            assert "FROM python:3.10-slim" in df, f"expected slim base for sandbox_mode={mode!r}"

    def test_non_torch_packages_separate(self):
        packages = {"torch": "2.2.0", "gymnasium": "0.29.1"}
        df = _generate_dockerfile("3.11", packages)
        # torch should be in its own RUN with CPU index
        # gymnasium should be in a separate RUN
        lines = df.split("\n")
        torch_line = [l for l in lines if "torch==" in l][0]
        assert "--index-url" in torch_line


class TestDatasetPackages:
    def test_cartpole_needs_gymnasium(self):
        claim_map = PaperClaimMap(
            core_contribution="test",
            datasets=[DatasetRequirement(name="CartPole-v1")],
        )
        pkgs = _dataset_packages(claim_map)
        assert "gymnasium" in pkgs

    def test_cifar_needs_torchvision(self):
        claim_map = PaperClaimMap(
            core_contribution="test",
            datasets=[DatasetRequirement(name="CIFAR-10")],
        )
        pkgs = _dataset_packages(claim_map)
        assert "torchvision" in pkgs
