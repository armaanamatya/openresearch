"""Constant-size repository manifest.

The MANIFEST — not the raw tree — is what enters the root's context (RLM
Algorithm 1 invariant). A hard byte ceiling on ``as_context()`` mirrors the
``context_map.MAX_BYTES`` cap. The root navigates deeper via ``inspect_repository``.
Pure stdlib; fail-soft.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILES = 200
MAX_DEPTH = 4
MAX_CONTEXT_BYTES = 8192
KEY_FILE_EXCERPT_CHARS = 800

# Glob patterns for the files whose names + short excerpts go in the manifest.
_KEY_FILE_GLOBS = (
    "README*", "requirements*.txt", "setup.py", "pyproject.toml",
    "environment*.yml", "environment*.yaml", "train.py", "main.py", "run.py",
)
_EXCLUDED_DIR_PARTS = frozenset({".git", "__pycache__", ".venv", "node_modules"})


@dataclass
class RepoManifest:
    path: str
    commit_sha: str | None
    file_tree: list[str] = field(default_factory=list)
    key_files: dict[str, str] = field(default_factory=dict)
    size_mb: float = 0.0
    lfs_skipped: bool = True

    def as_context(self) -> dict:
        """Return a JSON-serializable dict bounded by ``MAX_CONTEXT_BYTES``.

        Provenance (commit_sha/path/size_mb/lfs_skipped) is preserved first; the
        file_tree and key_files are trimmed until the serialized payload fits.
        """
        import json

        base = {
            "path": self.path,
            "commit_sha": self.commit_sha,
            "size_mb": self.size_mb,
            "lfs_skipped": self.lfs_skipped,
        }
        tree = list(self.file_tree)
        keys = dict(self.key_files)
        while True:
            payload = {**base, "file_tree": tree, "key_files": keys}
            if len(json.dumps(payload).encode("utf-8")) <= MAX_CONTEXT_BYTES:
                return payload
            # Trim the largest contributor first: drop a key-file excerpt, else a tree entry.
            if keys:
                largest = max(keys, key=lambda k: len(keys[k]))
                if len(keys[largest]) > 80:
                    keys[largest] = keys[largest][:80]
                    continue
                keys.pop(largest)
                continue
            if len(tree) > 1:
                tree = tree[: len(tree) // 2]
                continue
            # Floor: provenance only.
            return base


def _is_excluded(rel: Path) -> bool:
    return any(part in _EXCLUDED_DIR_PARTS for part in rel.parts[:-1])


def _is_key_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in _KEY_FILE_GLOBS)


def build_manifest(
    repo_dir: Path,
    *,
    commit_sha: str | None = None,
    size_mb: float = 0.0,
    lfs_skipped: bool = True,
) -> RepoManifest:
    """Walk ``repo_dir`` into a capped manifest (≤MAX_FILES, depth <MAX_DEPTH)."""
    repo_dir = Path(repo_dir)
    file_tree: list[str] = []
    key_files: dict[str, str] = {}
    try:
        repo_root = repo_dir.resolve()
        for abs_path in sorted(repo_dir.rglob("*")):
            if abs_path.is_symlink():
                continue
            if not abs_path.is_file():
                continue
            try:
                if abs_path.resolve() != repo_root and repo_root not in abs_path.resolve().parents:
                    continue
            except OSError:
                continue
            rel = abs_path.relative_to(repo_dir)
            if _is_excluded(rel):
                continue
            depth = len(rel.parts) - 1  # 0 for a top-level file
            if depth >= MAX_DEPTH:
                continue
            rel_posix = rel.as_posix()
            if len(file_tree) < MAX_FILES:
                file_tree.append(rel_posix)
            if _is_key_file(abs_path.name) and abs_path.name not in key_files:
                try:
                    key_files[abs_path.name] = abs_path.read_text(
                        encoding="utf-8", errors="replace"
                    )[:KEY_FILE_EXCERPT_CHARS]
                except OSError:
                    pass
    except OSError:
        pass
    return RepoManifest(
        path=str(repo_dir),
        commit_sha=commit_sha,
        file_tree=file_tree,
        key_files=key_files,
        size_mb=size_mb,
        lfs_skipped=lfs_skipped,
    )
