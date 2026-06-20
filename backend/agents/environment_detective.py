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

    # Validate the FROM base and normalize hallucinated/unknown bases before use.
    # Fail-soft: any exception leaves the Dockerfile unchanged.
    dockerfile = _normalize_dockerfile_from(dockerfile)

    spec = EnvironmentSpec(
        dockerfile=dockerfile,
        python_version=python_version,
        framework=framework,
        framework_version=framework_version,
        pip_packages=pip_packages,
        assumptions=assumptions,
        # BUG-NEW-047 (ported 2026-06-10 from the archived gepa branch): the old
        # "on CPU." phrasing made the root conclude compute_scope=CPU-only.
        compatibility_notes=(
            f"Generated for {framework}=={framework_version} on local CPU dev machine. "
            "NOTE: this spec describes the LOCAL environment used to derive package requirements — "
            "the GPU execution environment (RunPod/local CUDA) provides the GPUs. Do NOT use this "
            "note to conclude that experiments run CPU-only; set compute_scope from the GPU plan."
        ),
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

# ---------------------------------------------------------------------------
# FROM-base static validator
# ---------------------------------------------------------------------------

# Known-good base image name prefixes (family prefixes, not full tags).
# A FROM line whose image starts with one of these is accepted as-is.
_KNOWN_GOOD_BASE_FAMILIES: tuple[str, ...] = (
    "runpod/pytorch",
    "pytorch/pytorch",
    "nvidia/cuda",
    "nvcr.io/nvidia",
    "python:",
    "python ",   # covers "python AS ..." without a tag
    "ubuntu:",
    "ubuntu ",
    "debian:",
    "debian ",
    "tensorflow/tensorflow",
    "rocm/pytorch",
    "continuumio/",
    "mambaorg/",
    "quay.io/",
    "gcr.io/",
    "us-central1-docker.pkg.dev/",
    "europe-west4-docker.pkg.dev/",
    "scratch",
    "busybox",
    "alpine",
)

# Requirements packages that indicate CUDA-compilation headers are needed.
# When any of these appear in a requirements list the devel-vs-runtime hint
# prefers a -devel- base over a -runtime- base.
_CUDA_COMPILE_PKGS: frozenset[str] = frozenset({
    "bitsandbytes",
    "flash-attn",
    "flash_attn",
    "deepspeed",
    "apex",
    "xformers",
    "triton",
    "pynvml",
    "nvcc",
})


def _extract_from_image(dockerfile: str) -> str | None:
    """Return the base image token from the first non-comment/non-ARG FROM line.

    Returns ``None`` when no FROM line is found (empty or comments-only file).
    """
    for raw in dockerfile.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        upper = stripped.upper()
        if upper.startswith("ARG"):
            continue
        if upper.startswith("FROM "):
            parts = stripped.split()
            if len(parts) >= 2:
                return parts[1]  # the image token (may include tag)
            return None  # malformed FROM with no image
        break  # first non-blank/non-comment/non-ARG line that is not FROM
    return None


def _is_known_good_base(image: str) -> bool:
    """Return True when *image* matches a known-good base image family.

    Comparison is case-insensitive; variable references (``$VAR`` / ``${VAR}``)
    are treated as trusted (the user knows what they set).
    """
    if not image:
        return False
    if image.startswith("$"):
        return True  # ARG-interpolated image — trust the caller
    lower = image.lower()
    return any(lower.startswith(fam.lower()) for fam in _KNOWN_GOOD_BASE_FAMILIES)


def _requirements_need_devel(requirements_text: str) -> bool:
    """Return True when *requirements_text* contains packages that need CUDA headers.

    This is intentionally conservative — only returns True when a
    known CUDA-compile package name is found as a whole word.  Package
    names with version specifiers (``flash-attn>=2.0``) are matched too.
    """
    lower = requirements_text.lower()
    return any(
        re.search(r"\b" + re.escape(pkg) + r"\b", lower)
        for pkg in _CUDA_COMPILE_PKGS
    )


def _suggest_devel_base(base_image: str) -> str:
    """Swap *-runtime-* for *-devel-* in a RunPod/NVIDIA base tag.

    Only touches images that already carry ``-runtime-``; all other
    images are returned unchanged.  Never raises.
    """
    if "-runtime-" in base_image:
        return base_image.replace("-runtime-", "-devel-", 1)
    return base_image


class FromValidationResult:
    """Result of :func:`validate_from_base`."""

    __slots__ = ("ok", "image", "warning", "suggested_image", "devel_hint")

    def __init__(
        self,
        *,
        ok: bool,
        image: str | None,
        warning: str | None = None,
        suggested_image: str | None = None,
        devel_hint: bool = False,
    ) -> None:
        self.ok = ok
        self.image = image
        self.warning = warning
        self.suggested_image = suggested_image
        self.devel_hint = devel_hint

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FromValidationResult(ok={self.ok!r}, image={self.image!r}, "
            f"warning={self.warning!r}, suggested_image={self.suggested_image!r}, "
            f"devel_hint={self.devel_hint!r})"
        )


def validate_from_base(
    dockerfile: str,
    *,
    requirements_text: str = "",
    fallback_base: str | None = None,
) -> FromValidationResult:
    """Statically validate the FROM base image in *dockerfile*.

    Runs at detect/generate time — before any ``docker build`` — to catch
    hallucinated or malformed FROM lines cheaply.

    Rules (all fail-soft; a non-None ``suggested_image`` is advisory only):

    (a) **Malformed / empty FROM** — no FROM line found, or the FROM has no
        image token → ``ok=False``, ``warning`` set.

    (b) **Known-good base families** — ``runpod/pytorch``, ``pytorch/pytorch``,
        ``nvidia/cuda``, ``python:``, ``ubuntu:``, ``tensorflow/tensorflow``,
        official cloud registries, etc. → accepted unchanged (``ok=True``).

    (c) **Unknown / hallucinated base** — image does NOT match any known family →
        ``ok=True`` but ``warning`` set + ``suggested_image = fallback_base``
        (caller decides whether to swap).  Conservative: no legitimate custom
        base is blocked.

    (d) **Devel-vs-runtime hint** — when *requirements_text* contains
        CUDA-compilation packages (bitsandbytes / flash-attn / deepspeed /
        apex / xformers / triton) AND the chosen base contains ``-runtime-``,
        ``devel_hint=True`` is set and ``suggested_image`` is updated to the
        ``-devel-`` variant (only when the base itself is otherwise accepted).

    Parameters
    ----------
    dockerfile:
        Full Dockerfile text (may be agent-generated).
    requirements_text:
        Optional pip requirements.txt / requirements list text; used for the
        devel-vs-runtime hint.
    fallback_base:
        Default base image to suggest when the inferred FROM is unknown.
        Defaults to ``_RUNPOD_PYTORCH_BASE``.
    """
    if fallback_base is None:
        fallback_base = _RUNPOD_PYTORCH_BASE

    image = _extract_from_image(dockerfile)

    # (a) malformed / empty
    if image is None:
        return FromValidationResult(
            ok=False,
            image=None,
            warning=(
                "Dockerfile has no FROM line (or FROM is missing the image token). "
                f"Consider using: {fallback_base}"
            ),
            suggested_image=fallback_base,
        )

    # (b) known-good base — accepted as-is (may still trigger devel hint)
    if _is_known_good_base(image):
        result = FromValidationResult(ok=True, image=image)
        # (d) devel-vs-runtime hint
        if requirements_text and _requirements_need_devel(requirements_text):
            devel = _suggest_devel_base(image)
            if devel != image:
                result.devel_hint = True
                result.suggested_image = devel
                result.warning = (
                    f"Requirements contain CUDA-compilation packages "
                    f"(bitsandbytes/flash-attn/deepspeed/apex/xformers/triton) but "
                    f"the base image uses -runtime- CUDA headers. "
                    f"Consider switching to: {devel}"
                )
                logger.warning(
                    "validate_from_base: devel hint for %r → %r", image, devel
                )
        return result

    # (c) unknown / hallucinated base
    logger.warning(
        "validate_from_base: unrecognised FROM base %r; "
        "suggesting fallback %r (caller decides whether to swap)",
        image,
        fallback_base,
    )
    return FromValidationResult(
        ok=True,  # fail-soft — never block a legitimate custom base
        image=image,
        warning=(
            f"FROM base {image!r} is not a recognised base image family. "
            f"If this is a custom registry image, ignore this warning. "
            f"Otherwise consider: {fallback_base}"
        ),
        suggested_image=fallback_base,
    )


def _normalize_dockerfile_from(dockerfile: str) -> str:
    """Apply validate_from_base and normalize the FROM line if needed.

    Rules:
    - Malformed / missing FROM (ok=False): replace with a minimal fallback
      Dockerfile whose first line is ``FROM <fallback_base>``, preserving the
      remainder of the original.
    - Unknown / hallucinated base (ok=True, suggested_image set, not devel hint
      only): swap the FROM line's image token to the fallback; leave everything
      else untouched.
    - Known-good base: return unchanged.
    - Devel hint: log only; no structural change (advisory).
    - Any exception: return the original unchanged (fail-soft).
    """
    try:
        result = validate_from_base(dockerfile)
        if not result.ok:
            # Malformed or missing FROM — replace the entire FROM line with fallback.
            fallback = result.suggested_image or _RUNPOD_PYTORCH_BASE
            logger.warning(
                "run_offline: Dockerfile has no valid FROM line; "
                "replacing with fallback base %r",
                fallback,
            )
            # Prepend the fallback FROM, dropping any existing broken FROM line.
            lines = dockerfile.splitlines(keepends=True)
            cleaned = [
                ln for ln in lines
                if not ln.strip().upper().startswith("FROM")
            ]
            return f"FROM {fallback}\n" + "".join(cleaned)

        if result.suggested_image and not result.devel_hint:
            # Unknown / hallucinated base — swap the image token only.
            logger.warning(
                "run_offline: FROM base %r not recognised; "
                "normalising to configured fallback %r",
                result.image,
                result.suggested_image,
            )
            # Replace the first FROM line's image token in-place.
            new_lines = []
            replaced = False
            for line in dockerfile.splitlines(keepends=True):
                stripped = line.strip()
                if not replaced and stripped.upper().startswith("FROM "):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        # Preserve the rest of the FROM (e.g. AS alias).
                        suffix = " ".join(parts[2:])
                        new_from = f"FROM {result.suggested_image}"
                        if suffix:
                            new_from += f" {suffix}"
                        # Preserve original line ending.
                        ending = "\n" if line.endswith("\n") else ""
                        new_lines.append(new_from + ending)
                        replaced = True
                        continue
                new_lines.append(line)
            return "".join(new_lines)

        if result.devel_hint and result.warning:
            # Advisory only — log but do not modify the Dockerfile.
            logger.warning("run_offline: devel hint: %s", result.warning)

    except Exception:  # noqa: BLE001
        logger.warning(
            "run_offline: validate_from_base raised unexpectedly; "
            "leaving Dockerfile unchanged",
            exc_info=True,
        )

    return dockerfile


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


