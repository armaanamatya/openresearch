"""Environment Detective Agent — infers runtime environment and generates Dockerfiles.

Provides:
  - ``run_offline()`` — deterministic Dockerfile generation from claim map (no LLM, used by RLM path)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from backend.agents.schemas import (
    Assumption,
    EnvironmentSpec,
    PaperClaimMap,
    RiskLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known environment templates
# ---------------------------------------------------------------------------

_FRAMEWORK_COMPATIBILITY = {
    "pytorch": {
        "2.2.0": {"python": "3.11", "cuda": "12.1"},
        "2.1.0": {"python": "3.11", "cuda": "11.8"},
        "2.0.0": {"python": "3.10", "cuda": "11.7"},
        "1.13.0": {"python": "3.10", "cuda": "11.6"},
    },
    "tensorflow": {
        "2.15.0": {"python": "3.11", "cuda": "12.2"},
        "2.14.0": {"python": "3.11", "cuda": "11.8"},
    },
}

_DATASET_PACKAGES = {
    "CartPole-v1": ["gymnasium>=0.29.1"],
    "CIFAR-10": ["torchvision>=0.17.0"],
    "MNIST": ["torchvision>=0.17.0"],
    "MuJoCo": ["gymnasium[mujoco]>=0.29.1"],
    "Atari": ["gymnasium[atari]>=0.29.1", "ale-py"],
}


def run_offline(
    project_id: str,
    runs_root: Path,
    paper_claim_map: PaperClaimMap,
    artifact_index: dict[str, Any] | None = None,
    *,
    gpu_mode: str | None = None,
    sandbox_mode: str | None = None,
) -> EnvironmentSpec:
    """Deterministic environment inference without LLM.

    Uses the claim map's hardware clues, datasets, and training recipe
    to generate a Dockerfile.

    ``gpu_mode`` (``"off"``, ``"auto"``, ``"prefer"``, ``"max"``, or ``None``)
    drives the torch wheel selection: ``prefer``/``max`` picks the default
    CUDA-capable PyPI wheel so a GPU-bearing sandbox can actually use the
    card.  Other values keep the CPU-only wheel (smaller image, faster build).

    ``sandbox_mode`` (``"runpod"``, ``"docker"``, ``"local"``, or ``None``)
    affects the Dockerfile base image. When ``"runpod"``, the generated
    Dockerfile uses ``runpod/pytorch`` as its base (torch pre-installed) and
    skips reinstalling torch — avoids the 1800s build timeout caused by
    downloading the ~2.5 GB CUDA wheel inside the Docker build step.
    """
    # Determine framework from training recipe
    framework, framework_version = _infer_framework(paper_claim_map)
    python_version = _infer_python_version(framework, framework_version)

    # Collect pip packages
    pip_packages: dict[str, str] = {}
    pip_packages.update(_framework_packages(framework, framework_version))
    pip_packages.update(_dataset_packages(paper_claim_map))
    pip_packages.update(_utility_packages())

    # Generate assumptions for each inferred value
    assumptions = _generate_assumptions(
        paper_claim_map, framework, framework_version, python_version,
    )

    # Generate Dockerfile — wheel selection follows gpu_mode; base image
    # follows sandbox_mode (runpod uses pre-built pytorch image, others use slim).
    dockerfile = _generate_dockerfile(
        python_version, pip_packages, gpu_mode=gpu_mode, sandbox_mode=sandbox_mode,
    )

    spec = EnvironmentSpec(
        dockerfile=dockerfile,
        python_version=python_version,
        framework=framework,
        framework_version=framework_version,
        pip_packages=pip_packages,
        assumptions=assumptions,
        compatibility_notes=f"Generated for {framework}=={framework_version} on CPU.",
    )

    # Write to disk
    out_dir = Path(runs_root) / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (out_dir / "environment_spec.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Environment spec written to %s", out_dir)

    return spec


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_framework(claim_map: PaperClaimMap) -> tuple[str, str]:
    """Infer ML framework and version from claim map."""
    all_text = (
        claim_map.core_contribution
        + " "
        + claim_map.model_architecture
        + " "
        + claim_map.training_recipe.optimizer
    ).lower()

    # Check recipe's other_hparams for framework clues
    recipe_text = str(claim_map.training_recipe.other_hparams).lower()

    if "tensorflow" in all_text or "tf." in all_text or "keras" in all_text:
        return "tensorflow", "2.15.0"
    if "jax" in all_text or "flax" in all_text:
        return "jax", "0.4.25"

    # Default to PyTorch (most common in RL/ML papers)
    return "pytorch", "2.2.0"


def _infer_python_version(framework: str, framework_version: str) -> str:
    """Infer Python version from framework compatibility."""
    compat = _FRAMEWORK_COMPATIBILITY.get(framework, {})
    version_info = compat.get(framework_version, {})
    return version_info.get("python", "3.11")


def _framework_packages(framework: str, version: str) -> dict[str, str]:
    """Get framework pip packages."""
    if framework == "pytorch":
        return {
            "torch": version,
            "numpy": "1.26.4",
        }
    if framework == "tensorflow":
        return {
            "tensorflow": version,
            "numpy": "1.26.4",
        }
    if framework == "jax":
        return {
            "jax": version,
            "jaxlib": version,
            "numpy": "1.26.4",
        }
    return {"numpy": "1.26.4"}


def _dataset_packages(claim_map: PaperClaimMap) -> dict[str, str]:
    """Get packages needed for datasets."""
    packages: dict[str, str] = {}
    for dataset in claim_map.datasets:
        deps = _DATASET_PACKAGES.get(dataset.name, [])
        for dep in deps:
            # Parse "package>=version" format
            match = re.match(r"([a-zA-Z0-9_-]+)(?:\[.*?\])?(?:>=|==)(.+)", dep)
            if match:
                pkg_name = dep.split(">=")[0].split("==")[0]
                pkg_version = match.group(2)
                packages[pkg_name] = pkg_version
            else:
                packages[dep] = ""
    return packages


def _utility_packages() -> dict[str, str]:
    """Common utility packages."""
    return {
        "matplotlib": "3.8.0",
        "tqdm": "4.66.0",
    }


def _generate_assumptions(
    claim_map: PaperClaimMap,
    framework: str,
    framework_version: str,
    python_version: str,
) -> list[Assumption]:
    """Generate assumptions for all inferred environment details."""
    assumptions: list[Assumption] = []
    idx = 1

    # Framework version assumption
    hardware_clues = claim_map.hardware_clues
    has_framework_evidence = any(
        framework in clue.lower() for clue in hardware_clues
    )
    assumptions.append(Assumption(
        assumption_id=f"ENV{idx:03d}",
        detail=f"{framework} version not explicitly stated in paper",
        chosen_value=framework_version,
        evidence=hardware_clues if has_framework_evidence else ["Inferred from paper date and compatibility"],
        risk=RiskLevel.low if has_framework_evidence else RiskLevel.medium,
    ))
    idx += 1

    # Python version assumption
    assumptions.append(Assumption(
        assumption_id=f"ENV{idx:03d}",
        detail="Python version not specified",
        chosen_value=python_version,
        evidence=[f"Compatible with {framework}=={framework_version}"],
        risk=RiskLevel.low,
    ))
    idx += 1

    # CPU vs GPU assumption
    gpu_mentioned = any("gpu" in c.lower() or "cuda" in c.lower() for c in hardware_clues)
    assumptions.append(Assumption(
        assumption_id=f"ENV{idx:03d}",
        detail="GPU requirement",
        chosen_value="GPU" if gpu_mentioned else "CPU only (no GPU required)",
        evidence=hardware_clues if gpu_mentioned else ["No GPU mentioned in paper"],
        risk=RiskLevel.low,
    ))

    return assumptions


_RUNPOD_PYTORCH_BASE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04"


def _generate_dockerfile(
    python_version: str,
    pip_packages: dict[str, str],
    *,
    gpu_mode: str | None = None,
    sandbox_mode: str | None = None,
) -> str:
    """Generate a working Dockerfile.

    When ``sandbox_mode == "runpod"``, uses the pre-built RunPod PyTorch base
    image which has torch 2.1 pre-installed — skipping the ~2.5 GB CUDA wheel
    download that caused the 1800s build timeout (BUG-LR-009).

    For all other sandbox modes, wheel selection for torch/torchvision/torchaudio
    is delegated to select_torch_index_url, which honours both gpu_mode and the
    host's actual NVIDIA capability.
    """
    use_runpod_base = (
        isinstance(sandbox_mode, str) and sandbox_mode.lower() == "runpod"
    )

    if use_runpod_base:
        # RunPod base has torch 2.1 + CUDA 11.8 pre-installed; don't reinstall.
        other_packages = []
        for pkg, version in sorted(pip_packages.items()):
            if pkg not in ("torch", "torchvision", "torchaudio"):
                other_packages.append(f"{pkg}=={version}" if version else pkg)

        lines = [
            f"FROM {_RUNPOD_PYTORCH_BASE}",
            "",
            "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
        ]
        if other_packages:
            lines.append(f"RUN pip install --no-cache-dir {' '.join(other_packages)}")
        lines.extend([
            "WORKDIR /code",
            "ENV PYTHONUNBUFFERED=1",
            'CMD ["python", "train.py"]',
        ])
        return "\n".join(lines) + "\n"

    from backend.services.runtime.gpu_resolution import select_torch_index_url

    lines = [
        f"FROM python:{python_version}-slim",
        "",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        "    git \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
    ]

    torch_packages = []
    other_packages = []
    for pkg, version in sorted(pip_packages.items()):
        if pkg in ("torch", "torchvision", "torchaudio"):
            torch_packages.append(f"{pkg}=={version}" if version else pkg)
        else:
            other_packages.append(f"{pkg}=={version}" if version else pkg)

    if torch_packages:
        index_url = select_torch_index_url(gpu_mode)
        if index_url is None:
            lines.append(
                f"RUN pip install --no-cache-dir {' '.join(torch_packages)}"
            )
        else:
            lines.append(
                f"RUN pip install --no-cache-dir {' '.join(torch_packages)} "
                f"--index-url {index_url}"
            )

    if other_packages:
        lines.append(f"RUN pip install --no-cache-dir {' '.join(other_packages)}")

    lines.extend([
        "",
        "WORKDIR /workspace",
        "COPY . /workspace/",
        "",
        'CMD ["python", "train.py"]',
    ])

    return "\n".join(lines) + "\n"


