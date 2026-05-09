"""Environment Detective Agent — infers runtime environment and generates Dockerfiles.

Provides:
  - ``run_offline()`` — deterministic Dockerfile generation from claim map (no LLM)
  - ``run_with_sdk()`` — full LLM-powered environment inference
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from backend.agents.runtime.base import AgentRuntime, ProviderName
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
) -> EnvironmentSpec:
    """Deterministic environment inference without LLM.

    Uses the claim map's hardware clues, datasets, and training recipe
    to generate a Dockerfile.
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

    # Generate Dockerfile
    dockerfile = _generate_dockerfile(python_version, pip_packages)

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
    (out_dir / "Dockerfile").write_text(dockerfile)
    (out_dir / "environment_spec.json").write_text(spec.model_dump_json(indent=2))
    logger.info("Environment spec written to %s", out_dir)

    return spec


async def run_with_sdk(
    project_id: str,
    runs_root: Path,
    paper_claim_map: PaperClaimMap,
    artifact_index: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
) -> EnvironmentSpec:
    """Full LLM-powered environment detection via the configured agent runtime."""
    from backend.agents.runtime.invoke import collect_agent_text

    project_dir = Path(runs_root) / project_id
    context = {
        "paper_claim_map": paper_claim_map.model_dump(),
        "artifact_index": artifact_index or {},
    }

    prompt = (
        f"Build the Docker environment for project {project_id}.\n"
        f"Context:\n```json\n{json.dumps(context, indent=2)}\n```\n"
        f"Write Dockerfile and environment_spec.json to {project_dir}/"
    )

    full_text = await collect_agent_text(
        "environment-detective",
        prompt,
        project_dir=project_dir,
        model=model,
        provider=provider,
        runtime=runtime,
        max_turns=20,
    )

    # Try to read the written spec
    spec_path = project_dir / "environment_spec.json"
    if spec_path.exists():
        data = json.loads(spec_path.read_text())
        return EnvironmentSpec(**data)

    # Parse from output
    data = _extract_json(full_text)
    spec = EnvironmentSpec(**data)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Dockerfile").write_text(spec.dockerfile)
    spec_path.write_text(spec.model_dump_json(indent=2))
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


def _generate_dockerfile(python_version: str, pip_packages: dict[str, str]) -> str:
    """Generate a working Dockerfile."""
    lines = [
        f"FROM python:{python_version}-slim",
        "",
        "# System packages",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        "    git \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
        "# Python packages",
    ]

    # Group torch install separately for CPU index
    torch_packages = []
    other_packages = []
    for pkg, version in sorted(pip_packages.items()):
        if pkg in ("torch", "torchvision", "torchaudio"):
            torch_packages.append(f"{pkg}=={version}" if version else pkg)
        else:
            other_packages.append(f"{pkg}=={version}" if version else pkg)

    if torch_packages:
        lines.append(
            f"RUN pip install --no-cache-dir {' '.join(torch_packages)} "
            f"--index-url https://download.pytorch.org/whl/cpu"
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


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from text."""
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace_start : i + 1])
    raise ValueError(f"No JSON found: {text[:200]}")
