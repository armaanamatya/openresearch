"""Auto-derive a ``requirements.txt`` from a Dockerfile.

The local-docker sandbox path BUILDS an image from the agent's Dockerfile,
which installs every pip dep listed in ``RUN pip install ...`` lines.
The RunPod sandbox path is structurally different: it boots a pre-built
RunPod PyTorch image and only installs what's in ``requirements.txt`` via
the backend bootstrap.  If the agent forgets to write a ``requirements.txt``
(every implement_baseline iteration is a fresh sub-agent invocation —
nothing forces the discipline) the RunPod run silently misses every dep
that wasn't in the base image, e.g. ``matplotlib``.

This module reads the Dockerfile, parses ``RUN pip install ...`` lines,
and synthesises a ``requirements.txt`` containing the union of every
package spec found.  Backend hook: ``_execute_in_sandbox`` calls this
before bootstrap when sandbox is runpod AND ``requirements.txt`` is
missing.  Idempotent — repeated calls overwrite with the same content.

Design contract:

  * Pure function of the Dockerfile string — no I/O side-effects beyond
    the optional write_path argument.
  * Robust to multi-line RUN blocks via ``\\`` continuations and ``&&``
    chaining (the deterministic env_detective output uses both shapes).
  * Filters out non-package shell args (``--no-cache-dir``,
    ``--index-url <url>``, etc.) so the synthesized requirements.txt is
    clean.
  * Stable order — packages are deduped + sorted so the output is
    deterministic and content-hash-friendly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# Flags / options that may appear inline in ``pip install`` invocations.
# Either standalone (``--no-cache-dir``) or with a value (``--index-url URL``).
_PIP_FLAGS_NO_VALUE: frozenset[str] = frozenset({
    "--no-cache-dir",
    "--quiet",
    "-q",
    "--user",
    "--upgrade",
    "-U",
    "--pre",
    "--force-reinstall",
})
_PIP_FLAGS_WITH_VALUE: frozenset[str] = frozenset({
    "--index-url",
    "-i",
    "--extra-index-url",
    "--find-links",
    "-f",
    "--target",
    "-t",
    "--constraint",
    "-c",
    "--requirement",
    "-r",
    "--proxy",
    "--cert",
})

# Match a complete `RUN ... pip install ...` block, supporting ``\`` line
# continuations and ``&& other-stuff`` chains.  The capture group is the
# ``pip install`` argument tail; trailing ``&& other-stuff`` is dropped via
# a second split below.
_PIP_INSTALL_BLOCK = re.compile(
    r"RUN\s+(?:[^\n]*?\b)?pip(?:3)?\s+install\s+(.*?)(?=\n[A-Z][A-Z]+\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _unwrap_continuations(block: str) -> str:
    """Collapse ``\\\n`` line continuations into single spaces."""
    return re.sub(r"\\\s*\n\s*", " ", block)


def _drop_chained_commands(block: str) -> str:
    """Trim everything after the first ``&&`` (the next shell command)."""
    return block.split("&&", 1)[0]


def _tokenize(block: str) -> list[str]:
    """Whitespace-split, filter empties."""
    return [t for t in block.replace("\n", " ").split() if t.strip()]


def _is_package_spec(token: str) -> bool:
    """Heuristic: package specs are non-flag, non-URL tokens.

    Includes versioned specs (``torch==2.2.0``), unversioned (``numpy``),
    PEP 508 extras (``alfworld[full]``), and VCS pins (``git+https://...``)
    are intentionally excluded — they belong in a Dockerfile note, not
    requirements.txt.
    """
    if not token:
        return False
    if token.startswith("-"):
        return False
    if token.startswith(("http://", "https://", "git+", "file:", ".")):
        return False
    return True


def parse_pip_packages_from_dockerfile(dockerfile_text: str) -> list[str]:
    """Return a deduplicated, sorted list of pip package specs from a Dockerfile.

    Each spec preserves its version pin when present.  Index-URL flags and
    other pip options are stripped — the consumer ``requirements.txt`` is a
    plain dependency list, not a shell invocation.
    """
    packages: set[str] = set()
    for match in _PIP_INSTALL_BLOCK.finditer(dockerfile_text):
        block = match.group(1)
        block = _unwrap_continuations(block)
        block = _drop_chained_commands(block)
        tokens = _tokenize(block)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in _PIP_FLAGS_WITH_VALUE:
                i += 2  # skip the value too
                continue
            if tok in _PIP_FLAGS_NO_VALUE:
                i += 1
                continue
            if tok.startswith("-"):
                # Unknown flag form ``--foo=bar`` or ``--foo`` — drop it
                # alongside any next arg if it doesn't carry ``=``.
                if "=" in tok:
                    i += 1
                else:
                    i += 1  # conservative — only skip the flag itself
                continue
            if _is_package_spec(tok):
                packages.add(tok)
            i += 1
    return sorted(packages)


def synthesize_requirements_txt(dockerfile_text: str) -> str:
    """Return the full text of a synthesized requirements.txt."""
    packages = parse_pip_packages_from_dockerfile(dockerfile_text)
    if not packages:
        return ""
    header = (
        "# Auto-derived from Dockerfile by backend.agents.rlm.requirements_derive\n"
        "# Source of truth for pip installs is the Dockerfile; this file is\n"
        "# regenerated on every runpod bootstrap when missing.\n"
    )
    return header + "\n".join(packages) + "\n"


def ensure_requirements_txt(
    code_dir: Path,
    dockerfile_path: Path | None = None,
) -> Path | None:
    """If ``requirements.txt`` is missing under ``code_dir``, synthesize one
    from the project's Dockerfile.

    Returns the path to the synthesized requirements.txt (whether newly
    written or already present), or ``None`` when no Dockerfile is available
    OR no pip packages were parsed.  Fail-soft on every error.

    The Dockerfile path defaults to ``code_dir.parent / "Dockerfile"``,
    matching the layout used by ``environment_detective.run_offline`` and
    by ``run_experiment``'s rebuild-from-Dockerfile logic.
    """
    req_path = code_dir / "requirements.txt"
    if req_path.exists():
        return req_path

    if dockerfile_path is None:
        dockerfile_path = code_dir.parent / "Dockerfile"
    if not dockerfile_path.exists():
        return None

    try:
        text = dockerfile_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("requirements_derive: cannot read Dockerfile: %s", exc)
        return None

    content = synthesize_requirements_txt(text)
    if not content:
        return None

    try:
        req_path.write_text(content, encoding="utf-8")
        logger.info(
            "requirements_derive: wrote %d-byte requirements.txt synthesized from %s",
            len(content),
            dockerfile_path,
        )
        return req_path
    except OSError as exc:
        logger.warning("requirements_derive: cannot write requirements.txt: %s", exc)
        return None


__all__ = [
    "ensure_requirements_txt",
    "parse_pip_packages_from_dockerfile",
    "synthesize_requirements_txt",
]
