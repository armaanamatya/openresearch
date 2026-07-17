"""Repo-first STRUCTURE-ONLY grounding of an already-cloned author repo.

Flag-gated on the EXISTING ``OPENRESEARCH_USE_AUTHOR_REPO`` master gate
(``backend.agents.rlm.feature_flags.use_author_repo``) -- this module does not
introduce a new flag. It also does NOT resolve or clone anything: that is
``backend.services.ingestion.repo.{resolver,provisioner}``'s job, driven from
``run.py::_resolve_and_clone_repo`` which persists ``rlm_state/repo_spec.json``
and clones the repo onto disk. This module reads that already-cloned repo dir
and summarizes it into a small, deterministic grounding context -- no network
calls, no git, no LLM calls.

STRUCTURE/VALUES ONLY: the output names which files exist and copies bounded
literal values out of them (declared entry-point filenames, declared
dependency names, a short README excerpt). It NEVER asserts, computes, or
implies anything about whether a reproduction attempt passed -- callers must
not treat any key here as a verdict/score signal.

Wiring into ``run.py`` is a later phase; this module is standalone today.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from backend.agents.rlm.feature_flags import use_author_repo

# --- deterministic detection globs -----------------------------------------
_ENTRY_POINT_GLOBS = ("train*.py", "main*.py", "run*.py", "setup.py", "pyproject.toml")
_REQUIREMENTS_GLOBS = ("requirements*.txt", "environment*.yml", "environment*.yaml", "pyproject.toml")
_CONFIG_GLOBS = ("*.yaml", "*.yml", "*.json", "*.toml")
_EXCLUDED_DIR_PARTS = frozenset({".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"})
_TOP_TWO_LEVELS_MAX_DEPTH = 1  # depth 0 = top-level file, depth 1 = one dir down
_README_HEAD_CHARS = 500

PROVENANCE = "author_repo"

_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+")


def _matches_any(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _is_excluded(rel: Path) -> bool:
    return any(part in _EXCLUDED_DIR_PARTS for part in rel.parts[:-1])


def _iter_repo_files(repo_dir: Path):
    """Yield ``(rel, abs_path)`` for real files under ``repo_dir``, sorted, symlink-safe.

    Mirrors ``repo.manifest.build_manifest``'s traversal guards (skip symlinks,
    skip anything that resolves outside ``repo_dir``, skip vendored/VCS dirs).
    Fail-soft: an unreadable/missing ``repo_dir`` yields nothing.
    """
    try:
        if not repo_dir.is_dir():
            return
        repo_root = repo_dir.resolve()
    except OSError:
        return
    try:
        candidates = sorted(repo_dir.rglob("*"))
    except OSError:
        return
    for abs_path in candidates:
        if abs_path.is_symlink():
            continue
        try:
            if not abs_path.is_file():
                continue
        except OSError:
            continue
        try:
            resolved = abs_path.resolve()
            if resolved != repo_root and repo_root not in resolved.parents:
                continue
        except OSError:
            continue
        rel = abs_path.relative_to(repo_dir)
        if _is_excluded(rel):
            continue
        yield rel, abs_path


def _bare_pkg_name(spec: str) -> str:
    """Strip a requirement spec down to its bare package name (best-effort)."""
    spec = spec.strip()
    if not spec or spec.startswith("#"):
        return ""
    spec = spec.split(";", 1)[0].strip()  # drop environment markers
    m = _PKG_NAME_RE.match(spec)
    return m.group(0) if m else ""


def _parse_requirements_txt(text: str) -> set[str]:
    names: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = _bare_pkg_name(line)
        if name:
            names.add(name)
    return names


def _parse_environment_yml(text: str) -> set[str]:
    """Best-effort conda ``environment.yml`` dependency extraction (fail-soft, no hard yaml dep)."""
    names: set[str] = set()
    try:
        import yaml  # noqa: PLC0415 -- lazy import, mirrors paper_invariants.py's fail-soft pattern
        data = yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001 -- a missing/broken yaml parser must never break grounding
        return names
    if not isinstance(data, dict):
        return names
    deps = data.get("dependencies")
    if not isinstance(deps, list):
        return names
    for item in deps:
        if isinstance(item, str):
            bare = item.split("=")[0].strip()
            if bare:
                names.add(bare)
        elif isinstance(item, dict):
            pip_deps = item.get("pip")
            if isinstance(pip_deps, list):
                for pip_item in pip_deps:
                    if isinstance(pip_item, str):
                        name = _bare_pkg_name(pip_item)
                        if name:
                            names.add(name)
    return names


def _parse_pyproject_deps(path: Path) -> set[str]:
    """Best-effort PEP 621 / Poetry dependency extraction from ``pyproject.toml``."""
    names: set[str] = set()
    try:
        import tomllib  # noqa: PLC0415 -- stdlib (py311+), lazy to keep import cost local
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- malformed toml must never break grounding
        return names
    project = data.get("project") if isinstance(data, dict) else None
    if isinstance(project, dict):
        for dep in project.get("dependencies") or []:
            if isinstance(dep, str):
                name = _bare_pkg_name(dep)
                if name:
                    names.add(name)
    tool = data.get("tool") if isinstance(data, dict) else None
    poetry = tool.get("poetry") if isinstance(tool, dict) else None
    if isinstance(poetry, dict):
        for dep_name in poetry.get("dependencies") or {}:
            if isinstance(dep_name, str) and dep_name.lower() != "python":
                names.add(dep_name)
    return names


def _extract_packages(name: str, abs_path: Path) -> set[str]:
    lname = name.lower()
    try:
        if lname == "pyproject.toml":
            return _parse_pyproject_deps(abs_path)
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    if lname.startswith("requirements") and lname.endswith(".txt"):
        return _parse_requirements_txt(text)
    if lname.startswith("environment") and (lname.endswith(".yml") or lname.endswith(".yaml")):
        return _parse_environment_yml(text)
    return set()


def _readme_head(repo_dir: Path, max_chars: int = _README_HEAD_CHARS) -> str:
    try:
        top_level = list(repo_dir.iterdir()) if repo_dir.is_dir() else []
    except OSError:
        return ""
    candidates = sorted(
        (p for p in top_level if p.is_file() and p.name.lower().startswith("readme")),
        key=lambda p: p.name,
    )
    if not candidates:
        return ""
    try:
        text = candidates[0].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:max_chars].strip()


def ground_from_repo(repo_dir: Path, *, max_files: int = 40) -> dict[str, Any]:
    """Summarize the cloned repo at ``repo_dir`` into a bounded, deterministic structure.

    Returns ``{entry_points, requirements, key_configs, readme_summary,
    module_tree, provenance}``. Every list is sorted and capped at
    ``max_files`` entries; ``provenance`` is always the literal
    ``"author_repo"``. STRUCTURE/VALUES ONLY -- never asserts a reproduction
    pass. Fail-soft: a missing/unreadable ``repo_dir`` degrades to an
    all-empty structure rather than raising.
    """
    repo_dir = Path(repo_dir)
    entry_points: set[str] = set()
    requirement_files: set[str] = set()
    packages: set[str] = set()
    key_configs: set[str] = set()
    module_tree: list[str] = []

    for rel, abs_path in _iter_repo_files(repo_dir):
        rel_posix = rel.as_posix()
        name = abs_path.name
        depth = len(rel.parts) - 1  # 0 = top-level file

        module_tree.append(rel_posix)

        if _matches_any(name, _ENTRY_POINT_GLOBS):
            entry_points.add(rel_posix)

        if _matches_any(name, _REQUIREMENTS_GLOBS):
            requirement_files.add(rel_posix)
            packages.update(_extract_packages(name, abs_path))

        if depth <= _TOP_TWO_LEVELS_MAX_DEPTH and _matches_any(name, _CONFIG_GLOBS):
            key_configs.add(rel_posix)

    return {
        "entry_points": sorted(entry_points)[:max_files],
        "requirements": {
            "files": sorted(requirement_files)[:max_files],
            "packages": sorted(packages)[:max_files],
        },
        "key_configs": sorted(key_configs)[:max_files],
        "readme_summary": _readme_head(repo_dir),
        "module_tree": sorted(module_tree)[:max_files],
        "provenance": PROVENANCE,
    }


def repo_first_enabled() -> bool:
    """``True`` iff the EXISTING ``OPENRESEARCH_USE_AUTHOR_REPO`` master gate is on."""
    return use_author_repo()


def _resolve_repo_dir(project_dir: Path) -> Path | None:
    """Probe for an already-cloned repo dir: ``rlm_state/repo_spec.json``'s path, else ``code/``.

    Never resolves/clones -- read-only probing of what ``_resolve_and_clone_repo``
    (or, in execute/adapt mode, the implementer) already put on disk. Returns
    ``None`` when neither candidate is a real directory.
    """
    repo_spec_path = project_dir / "rlm_state" / "repo_spec.json"
    if repo_spec_path.is_file():
        try:
            spec = json.loads(repo_spec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            spec = None
        if isinstance(spec, dict):
            spec_path = spec.get("path")
            if spec_path:
                candidate = Path(spec_path)
                if candidate.is_dir():
                    return candidate

    code_dir = project_dir / "code"
    if code_dir.is_dir():
        return code_dir

    return None


def write_grounding(project_dir: Path) -> Path | None:
    """Ground the project's already-cloned author repo and persist the snapshot.

    Off-flag (``OPENRESEARCH_USE_AUTHOR_REPO`` unset/falsy): returns ``None``
    and writes nothing -- byte-identical to today. On-flag: probes for a
    cloned repo dir (``rlm_state/repo_spec.json``'s ``path``, else ``code/``);
    if neither exists, returns ``None`` and writes nothing. Otherwise writes
    ``rlm_state/repo_grounding.json`` (atomic write) and returns its path.
    Fail-soft: never raises into the caller.
    """
    if not repo_first_enabled():
        return None
    try:
        project_dir = Path(project_dir)
        repo_dir = _resolve_repo_dir(project_dir)
        if repo_dir is None:
            return None

        grounding = ground_from_repo(repo_dir)

        rlm_state = project_dir / "rlm_state"
        rlm_state.mkdir(parents=True, exist_ok=True)
        out_path = rlm_state / "repo_grounding.json"
        fd, tmp_name = tempfile.mkstemp(dir=str(rlm_state), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(grounding, indent=2, sort_keys=True))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, out_path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
        return out_path
    except Exception:  # noqa: BLE001 -- grounding is best-effort, never breaks the caller
        return None
