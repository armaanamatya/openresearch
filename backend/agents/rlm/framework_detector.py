"""Deterministic training-framework fingerprinting for execute-mode reproduction.

Pure, no I/O beyond reading files under ``code_path`` — no LLM, no network, no
GPU. This is the first stage of the deterministic execute-mode pipeline (see
``docs/superpowers/specs/2026-07-07-deterministic-any-paper-execute-mode-design.md``
§3/§5): before any launch/reward extraction is attempted, decide WHICH framework
adapter (if any) applies to a cloned repo.

Increment 1 only detects ``verl`` (the UCPO-validated adapter,
``scripts/ucpo_execute_cell/train_cell.py``). Adding a framework later
(``hf_trainer``, ``accelerate``, ...) is one additional branch below — the
function's shape (fingerprint signals -> confidence) does not change.
"""

from __future__ import annotations

from pathlib import Path

# Hydra config keys that only appear in a verl (or verl-shaped) training launch.
_VERL_SIGNATURE_TOKENS: tuple[str, ...] = ("actor_rollout_ref", "algorithm.adv_estimator")

# Directories that are never useful to scan and can be large (vendored deps,
# VCS metadata, caches) — skipping them keeps detection fast without changing
# the result (none of these ever hold an author-written launch script).
_SKIP_DIR_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv"}

_SCAN_GLOB_PATTERNS: tuple[str, ...] = ("**/*.sh", "**/*.py")


def _is_skipped(code_dir: Path, path: Path) -> bool:
    parts = path.relative_to(code_dir).parts[:-1]
    return any(part in _SKIP_DIR_NAMES for part in parts)


def _iter_candidate_files(code_dir: Path):
    seen: set[Path] = set()
    for pattern in _SCAN_GLOB_PATTERNS:
        for path in code_dir.glob(pattern):
            if path in seen or not path.is_file() or _is_skipped(code_dir, path):
                continue
            seen.add(path)
            yield path


def _find_bundled_verl_dir(code_dir: Path) -> Path | None:
    """A top-level ``verl/`` directory that is itself a pip project."""
    candidate = code_dir / "verl"
    if candidate.is_dir() and (
        (candidate / "setup.py").is_file() or (candidate / "pyproject.toml").is_file()
    ):
        return candidate
    return None


def _find_signature_files(code_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in _iter_candidate_files(code_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(token in text for token in _VERL_SIGNATURE_TOKENS):
            hits.append(str(path.relative_to(code_dir)))
    return sorted(hits)


def detect_framework(code_path: str | Path) -> tuple[str, float, dict]:
    """Detect the training framework a cloned repo uses, deterministically.

    Returns ``(framework, confidence, evidence)``:
      - ``("verl", 0.9, evidence)`` — a bundled ``verl/`` pip project AND a verl
        hydra-signature token somewhere under ``code_path``.
      - ``("verl", 0.7, evidence)`` — a verl hydra-signature token but no
        bundled ``verl/`` directory (e.g. the repo expects ``pip install verl``
        from PyPI/git rather than vendoring it).
      - ``("unknown", 0.0, evidence)`` — neither signal fired (Increment 1 has
        no other adapters yet).

    ``evidence`` always carries ``bundled_verl_dir`` (bool) and
    ``signature_files`` (list of paths, relative to ``code_path``, that
    referenced a verl signature token) so a caller can audit *why*.
    """
    code_dir = Path(code_path)
    evidence: dict = {"bundled_verl_dir": False, "signature_files": []}
    if not code_dir.is_dir():
        return "unknown", 0.0, evidence

    verl_dir = _find_bundled_verl_dir(code_dir)
    evidence["bundled_verl_dir"] = verl_dir is not None

    signature_files = _find_signature_files(code_dir)
    evidence["signature_files"] = signature_files

    if signature_files:
        if verl_dir is not None:
            return "verl", 0.9, evidence
        return "verl", 0.7, evidence

    return "unknown", 0.0, evidence
