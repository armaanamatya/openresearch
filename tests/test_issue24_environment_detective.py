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


from backend.agents.environment_detective import (
    run_offline,
    _generate_dockerfile,
    _infer_framework,
    _dataset_packages,
    _normalize_dockerfile_from,
    _RUNPOD_PYTORCH_BASE,
)
from backend.agents.schemas import (
    DatasetRequirement,
    EnvironmentSpec,
    MetricSpec,
    PaperClaimMap,
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


# ---------------------------------------------------------------------------
# _normalize_dockerfile_from — wiring tests (pure unit, no docker/network)
# ---------------------------------------------------------------------------


class TestNormalizeDockerfileFrom:
    """Tests for the _normalize_dockerfile_from wiring function.

    Covers:
    - Known-good FROM → returned unchanged
    - Hallucinated / unknown FROM → normalised to configured fallback
    - Missing / malformed FROM (ok=False) → fallback FROM prepended
    - Devel hint → Dockerfile left unchanged (advisory only)
    - Exception in validate_from_base → original returned unchanged (fail-soft)
    """

    def test_known_good_from_python_slim_unchanged(self):
        df = "FROM python:3.11-slim\nRUN pip install numpy\n"
        result = _normalize_dockerfile_from(df)
        assert result == df

    def test_known_good_from_runpod_unchanged(self):
        df = f"FROM {_RUNPOD_PYTORCH_BASE}\nRUN echo hi\n"
        result = _normalize_dockerfile_from(df)
        assert result == df

    def test_known_good_from_nvidia_cuda_unchanged(self):
        df = "FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04\nRUN pip install torch\n"
        result = _normalize_dockerfile_from(df)
        assert result == df

    def test_hallucinated_from_normalised_to_fallback(self):
        df = "FROM hallucinated/image:v99\nRUN pip install torch\n"
        result = _normalize_dockerfile_from(df)
        # The FROM line must now reference the fallback, not the garbage image.
        first_from = next(
            ln for ln in result.splitlines() if ln.strip().upper().startswith("FROM ")
        )
        assert "hallucinated/image:v99" not in first_from
        assert _RUNPOD_PYTORCH_BASE in first_from

    def test_hallucinated_from_rest_of_dockerfile_preserved(self):
        df = "FROM totally-made-up:latest\nRUN pip install torch\nWORKDIR /code\n"
        result = _normalize_dockerfile_from(df)
        assert "pip install torch" in result
        assert "WORKDIR /code" in result

    def test_hallucinated_from_as_alias_preserved(self):
        # FROM <bad_image> AS base  →  FROM <fallback> AS base
        df = "FROM totally-made-up:latest AS base\nRUN pip install numpy\n"
        result = _normalize_dockerfile_from(df)
        first_from = next(
            ln for ln in result.splitlines() if ln.strip().upper().startswith("FROM ")
        )
        assert "AS base" in first_from
        assert _RUNPOD_PYTORCH_BASE in first_from

    def test_missing_from_fallback_prepended(self):
        df = "RUN pip install numpy\nWORKDIR /code\n"
        result = _normalize_dockerfile_from(df)
        assert result.splitlines()[0].strip().startswith("FROM ")
        assert _RUNPOD_PYTORCH_BASE in result.splitlines()[0]

    def test_malformed_from_no_image_fallback_used(self):
        df = "FROM\nRUN pip install numpy\n"
        result = _normalize_dockerfile_from(df)
        # Should have a valid FROM line now.
        assert any(
            ln.strip().upper().startswith("FROM ") and len(ln.strip().split()) >= 2
            for ln in result.splitlines()
        )

    def test_devel_hint_does_not_modify_dockerfile(self):
        # flash-attn on a -runtime- base triggers devel_hint; we log only, no swap.
        df = (
            "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04\n"
            "RUN pip install flash-attn\n"
        )
        result = _normalize_dockerfile_from(df)
        # The -runtime- base must still be there (we did not swap to -devel-).
        assert "-runtime-" in result
        assert result == df

    def test_exception_in_validator_returns_original(self, monkeypatch):
        """validate_from_base raising must not propagate — fail-soft."""
        import backend.agents.environment_detective as ed

        def _raise(dockerfile, **kwargs):
            raise RuntimeError("unexpected error in validator")

        monkeypatch.setattr(ed, "validate_from_base", _raise)
        df = "FROM python:3.11-slim\nRUN pip install numpy\n"
        result = _normalize_dockerfile_from(df)
        assert result == df

    def test_run_offline_known_good_from_preserved(self, tmp_path):
        """run_offline produces a known-good FROM; normalization must leave it alone."""
        from backend.agents.schemas import PaperClaimMap

        claim = PaperClaimMap(core_contribution="test")
        spec = run_offline("prj_norm", tmp_path, claim)
        # The returned dockerfile must contain a valid FROM line.
        first_from = next(
            ln for ln in spec.dockerfile.splitlines()
            if ln.strip().upper().startswith("FROM ")
        )
        assert first_from.strip().startswith("FROM ")
        # Must not be the fallback-replaced form on this path (it was already correct).
        assert "python:" in first_from or "runpod/pytorch" in first_from
