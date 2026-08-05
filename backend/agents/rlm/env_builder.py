"""Deterministic env-image assembly for build-on-miss (design
spec docs/history/specs/2026-07-08-root-driven-adaptive-environment-build-design.md
§4). Pure — no cloud/subprocess/network; the Cloud Build submit lives in
env_builder's sibling cloud_build.py, wired by build_environment (later increment).
"""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    framework: str  # "verl" | "unknown" | ...
    base_image: str  # the validated floor image to build FROM
    build_layers: tuple[str, ...]  # extra pip deps to BAKE (compiled/heavy or accumulated-missing)
    assertions: tuple[str, ...]  # build-time `python3 -c "<stmt>"` import gates
    reason: str = ""


# Representative import gates per known framework — enough that a broken
# interpreter/ABI (the rc1/flash-attn class of failure) fails the CPU Cloud
# Build at $0 instead of on a GPU node. Keep in sync with the validated
# docker/gke-cell-verl recipe (memory reference_gke_verl_cell_image).
_FRAMEWORK_ASSERTIONS: dict[str, tuple[str, ...]] = {
    "verl": (
        "import torch",
        "import vllm",
        "import tensordict",
        "import flash_attn",
        "from math_verify import parse",
    ),
}

# A few pip-package -> import-name fixups so an assertion for a baked dep is
# importable (extend as needed; default is the package name verbatim).
_IMPORT_NAME_OVERRIDES: dict[str, str] = {
    "flash-attn": "flash_attn",
    "math-verify": "math_verify",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "pillow": "PIL",
}


def _import_name(pip_pkg: str) -> str:
    """Best-effort import name for a pip requirement (strips version/extras)."""
    # strip version specifiers and extras: "flash-attn==2.7.4" -> "flash-attn",
    # "uvicorn[standard]" -> "uvicorn"
    base = pip_pkg.strip()
    for sep in ("==", ">=", "<=", "~=", ">", "<", "[", ";", " "):
        if sep in base:
            base = base.split(sep, 1)[0].strip()
    return _IMPORT_NAME_OVERRIDES.get(base.lower(), base)


def resolve_env_spec(
    *, framework: str, base_image: str, extra_deps: tuple[str, ...] = ()
) -> EnvSpec:
    """Build an EnvSpec: FROM base_image, bake `extra_deps`, assert the framework
    base stack + each baked dep imports. Deterministic — deps are deduped+sorted so
    the same request always yields the same content_hash (cache stability)."""
    layers = tuple(sorted({d.strip() for d in extra_deps if d and d.strip()}))
    assertions = tuple(_FRAMEWORK_ASSERTIONS.get(framework, ()))
    assertions += tuple(f"import {_import_name(d)}" for d in layers)
    reason = (
        f"framework={framework} base={base_image} "
        f"+{len(layers)} baked dep(s)" if layers else f"framework={framework} base={base_image} (no extra deps)"
    )
    return EnvSpec(framework=framework, base_image=base_image, build_layers=layers, assertions=assertions, reason=reason)


def _reject_control_chars(spec: EnvSpec) -> None:
    """Fail fast if any token carries a newline/carriage-return/NUL. ``shlex.quote``
    guards SHELL tokenization but NOT Dockerfile INSTRUCTION boundaries — a newline
    embedded in ``base_image`` / a dep / an assertion would split ``FROM``/``RUN``
    into extra attacker-controlled instructions. These tokens come from config +
    framework detection (not end-user text), so any control char is malformed
    input, never legitimate — reject it rather than render an injected Dockerfile."""
    bad = ("\n", "\r", "\0")
    for label, value in (
        ("base_image", spec.base_image),
        *(("build_layer", d) for d in spec.build_layers),
        *(("assertion", a) for a in spec.assertions),
    ):
        if any(ch in value for ch in bad):
            raise ValueError(
                f"env_builder: {label} contains a control character (newline/NUL); "
                f"refusing to render a Dockerfile from {value!r}"
            )


def assemble_dockerfile(spec: EnvSpec) -> str:
    """Render the Dockerfile: FROM base, one pip layer for the baked deps, then a
    build-time import-assertion RUN per assertion (each fails the build cheaply on
    CPU if the stack is broken). No CMD/ENTRYPOINT — the cell entrypoint is owned
    by the base image (gke-cell-verl). Raises ``ValueError`` if any token carries a
    control char that could inject an extra Dockerfile instruction."""
    _reject_control_chars(spec)
    lines = [f"FROM {spec.base_image}"]
    if spec.build_layers:
        pkgs = " ".join(shlex.quote(p) for p in spec.build_layers)
        lines.append(f"RUN pip install --no-cache-dir {pkgs}")
    for stmt in spec.assertions:
        # single-line `python3 -c` (Cloud Build's non-BuildKit builder chokes on heredocs)
        lines.append(f"RUN python3 -c {shlex.quote(stmt)}")
    return "\n".join(lines) + "\n"


def content_hash(spec: EnvSpec) -> str:
    """Stable 12-char hex over (base_image, build_layers, assertions) — the cache
    key. Two EnvSpecs that assemble to the same image share a tag (no rebuild)."""
    h = hashlib.sha256()
    h.update(spec.base_image.encode())
    h.update(b"\x00")
    h.update("|".join(spec.build_layers).encode())
    h.update(b"\x00")
    h.update("|".join(spec.assertions).encode())
    return h.hexdigest()[:12]


def built_image_ref(registry: str, framework: str, digest: str) -> str:
    """Full Artifact Registry ref for a built env image. `registry` is e.g.
    'us-central1-docker.pkg.dev/proj/reprolab'. Tag is content-addressed."""
    reg = registry.rstrip("/")
    fw = framework or "env"
    return f"{reg}/env-{fw}:{digest}"
