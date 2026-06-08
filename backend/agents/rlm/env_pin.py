"""D6a — hermetic core-package pinning + conflict neutralization (backend-agnostic).

The 2026-06-07 Adam run lost its first experiment to ``torch_redundancy``: the agent's
generated ``requirements.txt`` re-pinned ``torch`` to a version that fought the
driver-compatible build the harness installs. The existing protection only fired on
the runpod pytorch base (``requirements_derive._strip_preinstalled`` and
``pre_flight_validator._check_requirements_torch_redundancy`` both gate on
``runpod/pytorch``); local + docker were unprotected.

This module is the single source of truth for which packages the HARNESS owns — it
installs a known-good build of these BEFORE the agent's deps, so the agent must not
re-pin them to a conflicting version. Pure stdlib so it can be unit-tested and reused
from any backend path. Wiring is intentionally separate (callers in the requirements
path) so this can be reviewed/tested in isolation.

Fail-soft by construction: only a well-formed denylisted requirement line is dropped;
anything we cannot confidently parse is kept verbatim, so a parser miss can never
break a working install. The per-paper escape hatch (``allow_override``) honors an
exotic torch/CUDA need.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Core packages the harness installs a known-good build of BEFORE the agent's deps.
# Re-pinning any of these to a conflicting version is the torch_redundancy /
# version_conflict / cuda_driver_mismatch failure class. Names are PEP 503 canonical.
CORE_DENYLIST: frozenset[str] = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "numpy",
    }
)


@dataclass(frozen=True)
class PinSet:
    """A proven-compatible core pin set for one base tag."""

    torch: str  # "" means the base image already carries torch (runpod) — never install a pin
    torchvision: str
    torchaudio: str | None
    numpy: str  # deliberately loose (e.g. "numpy<2") — only an upper bound for ABI safety


# base_tag -> proven pins. Refresh quarterly like gpu_catalog.py.
#   "cu121"  — the local 8×A5000 host's driver-compatible build (matches the torch
#              index pin in primitives.py's local requirements bootstrap, and the
#              torch 2.5.1 / torchvision 0.20.1 / numpy 1.26.4 set validated 2026-06-07).
#   "runpod" — runpod/pytorch images ship their own torch; we only STRIP, never pin.
COMPAT_MATRIX: dict[str, PinSet] = {
    "cu121": PinSet(
        torch="torch==2.5.1",
        torchvision="torchvision==0.20.1",
        torchaudio="torchaudio==2.5.1",
        numpy="numpy<2",
    ),
    "runpod": PinSet(torch="", torchvision="", torchaudio=None, numpy="numpy<2"),
}


def _canon(name: str) -> str:
    """PEP 503 canonical name: lower-case, runs of -/_/. collapsed to a single -."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _req_name(line: str) -> str | None:
    """Canonical package name from a requirements line, or None for a blank / comment /
    option line (``-r foo``, ``--index-url …``, ``-e .``)."""
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-"):
        return None
    m = re.match(r"^([A-Za-z0-9_.\-]+)", s)
    return _canon(m.group(1)) if m else None


def base_tag_for(sandbox_mode: str | None, base_image: str | None) -> str:
    """Map a sandbox/base-image to a COMPAT_MATRIX tag. Unknown → "" (fail-soft: the
    caller falls back to today's behavior — the agent's own torch)."""
    img = (base_image or "").lower()
    if "runpod/pytorch" in img:
        return "runpod"
    if (sandbox_mode or "").lower() in ("local", "docker"):
        return "cu121"
    return ""


def harden_requirements(
    lines: list[str], *, base_tag: str, allow_override: bool = False
) -> tuple[list[str], list[str]]:
    """Neutralize core-package lines the harness owns. Returns ``(kept, dropped)``.

    ``allow_override`` (a paper opting into an exotic torch/CUDA build) → no-op.
    An unknown ``base_tag`` (no PinSet) → no-op, since the harness installs no pin and
    so does not own the package on that backend.
    """
    if allow_override or base_tag not in COMPAT_MATRIX:
        return list(lines), []
    denyl = {_canon(n) for n in CORE_DENYLIST}
    kept: list[str] = []
    dropped: list[str] = []
    for ln in lines:
        name = _req_name(ln)
        if name is not None and name in denyl:
            dropped.append(ln.strip())
        else:
            kept.append(ln)
    return kept, dropped


def pin_install_specs(base_tag: str) -> list[str]:
    """The exact core pins to install BEFORE the agent's deps for ``base_tag``.

    Empty for runpod (image already carries torch) and for an unknown tag (fail-soft).
    """
    ps = COMPAT_MATRIX.get(base_tag)
    if ps is None or not ps.torch:
        return []
    specs = [ps.torch, ps.torchvision]
    if ps.torchaudio:
        specs.append(ps.torchaudio)
    return specs


__all__ = [
    "CORE_DENYLIST",
    "COMPAT_MATRIX",
    "PinSet",
    "base_tag_for",
    "harden_requirements",
    "pin_install_specs",
]
