"""GPU-mode resolution — single source of truth.

Previously the rule "is this gpu_mode going to actually pass a GPU through?"
was duplicated as ``gpu_mode in {"prefer", "max"}`` across three layers:

  * LocalDockerBackend.create_sandbox (device_requests gate)
  * environment_detective._generate_dockerfile (torch wheel index gate)
  * baseline_implementation prompt policy overlays

A new mode (``always`` for a future bare-metal sandbox, etc.) had to be added
in every call site. Worse, none of those layers checked whether the *host*
actually has an NVIDIA GPU — a ``--gpu-mode prefer`` run on a Mac-on-Mac dev
loop would install CUDA torch into a container that has no GPU device.

This module centralises the policy and adds host capability sniffing so the
effective mode degrades gracefully when the host can't honour the request.

Public API:

* :func:`is_gpu_passthrough_mode` — pure string-to-bool predicate.
* :func:`host_supports_nvidia_gpu` — cached ``nvidia-smi -L`` probe.
* :func:`effective_gpu_mode` — requested mode downgraded to fit the host.
* :func:`select_torch_index_url` — pip ``--index-url`` for ``torch``-family
  wheels, or ``None`` to mean "use the default PyPI index".

Determinism: every public function is pure given a fixed environment. The
``nvidia-smi`` probe is cached process-wide via ``lru_cache`` so repeated
calls during a single run pay the subprocess cost exactly once.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode predicates
# ---------------------------------------------------------------------------

# gpu_mode values that REQUEST a docker --gpus all passthrough. Add new modes
# here (e.g. future bare-metal sandbox) and every layer picks them up.
_GPU_PASSTHROUGH_MODES: Final[frozenset[str]] = frozenset({"prefer", "max"})

# pip --index-url for the CPU-only torch wheel.
CPU_TORCH_INDEX_URL: Final[str] = "https://download.pytorch.org/whl/cpu"


def is_gpu_passthrough_mode(mode: str | None) -> bool:
    """True iff this gpu_mode requests an actual GPU device on the container.

    Case-insensitive; ``None`` and unknown values return False (safe default).
    """
    return (mode or "").lower() in _GPU_PASSTHROUGH_MODES


# ---------------------------------------------------------------------------
# Host capability probe
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _probe_nvidia_smi() -> bool:
    """Run ``nvidia-smi -L`` and cache the result for the process lifetime.

    Returns True when at least one NVIDIA GPU is enumerated. Any error from
    the subprocess (missing binary, timeout, non-zero exit, empty output) is
    treated as "no GPU" — a strictly safe-by-default contract.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("gpu_resolution: nvidia-smi probe failed (%s); host has no GPU", exc)
        return False
    if result.returncode != 0:
        logger.debug("gpu_resolution: nvidia-smi exit=%s; host has no GPU", result.returncode)
        return False
    stdout = (result.stdout or "").strip()
    # nvidia-smi -L output shape: "GPU 0: NVIDIA GeForce RTX 2060 (UUID: ...)"
    has_gpu = bool(stdout) and stdout.startswith("GPU ")
    if not has_gpu:
        logger.debug("gpu_resolution: nvidia-smi returned empty / non-GPU output: %r", stdout[:80])
    return has_gpu


def host_supports_nvidia_gpu() -> bool:
    """True when the host has at least one NVIDIA GPU visible to nvidia-smi.

    Result is cached process-wide. Tests can call
    ``gpu_resolution._probe_nvidia_smi.cache_clear()`` to reset the cache
    between fixtures.
    """
    return _probe_nvidia_smi()


# ---------------------------------------------------------------------------
# Effective resolution
# ---------------------------------------------------------------------------


def effective_gpu_mode(requested: str | None) -> str:
    """Return the gpu_mode that will actually be honoured on this host.

    ``prefer`` / ``max`` are downgraded to ``auto`` when the host has no
    NVIDIA GPU — callers should NOT install CUDA torch into a container that
    will never see a CUDA device. Other modes pass through unchanged.

    ``None`` is normalised to ``"auto"``.
    """
    canonical = (requested or "auto").lower()
    if canonical in _GPU_PASSTHROUGH_MODES and not host_supports_nvidia_gpu():
        logger.info(
            "gpu_resolution: gpu_mode=%r requested but host has no NVIDIA GPU "
            "— downgrading to 'auto' (no CUDA passthrough, CPU torch wheel)",
            requested,
        )
        return "auto"
    return canonical


def select_torch_index_url(gpu_mode: str | None) -> str | None:
    """Return the pip ``--index-url`` for ``torch`` (and family).

    * ``None`` — use the default PyPI index, which ships a CUDA-capable wheel
      on linux/x86_64. This is the right answer when the container will
      actually receive a CUDA device.
    * :data:`CPU_TORCH_INDEX_URL` — the explicit CPU-only wheel index. Smaller
      image, faster build, no CUDA runtime baked in.

    The choice rides on :func:`effective_gpu_mode` so a ``--gpu-mode prefer``
    request on a GPU-less host still gets the CPU wheel.
    """
    if is_gpu_passthrough_mode(effective_gpu_mode(gpu_mode)):
        return None  # default PyPI ships CUDA wheel
    return CPU_TORCH_INDEX_URL


__all__ = [
    "CPU_TORCH_INDEX_URL",
    "effective_gpu_mode",
    "host_supports_nvidia_gpu",
    "is_gpu_passthrough_mode",
    "select_torch_index_url",
]
