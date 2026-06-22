"""Shallow, fail-soft git clone of the resolved repository.

IO; sandbox-agnostic — always runs on the orchestrator host (which has git + TLS
certs). Any failure (bad spec, network/egress blocked, private/auth, 404,
oversize, timeout) returns ``None`` so the run proceeds from-scratch. A blocked
clone NEVER aborts a run.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from backend.config import Settings
from backend.services.ingestion.repo.manifest import RepoManifest, build_manifest
from backend.services.ingestion.repo.resolver import RepoSpec

logger = logging.getLogger(__name__)


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


class RepoProvisioner:
    """Clone ``spec.url`` into ``dest``; return a manifest or ``None`` (fail-soft)."""

    @staticmethod
    def clone(spec: RepoSpec, dest: Path) -> RepoManifest | None:
        if not spec or not spec.url:
            return None
        dest = Path(dest)
        # Instantiate fresh so env-var monkeypatching in tests is picked up.
        settings = Settings()
        timeout_s = int(getattr(settings, "repo_clone_timeout_s", 300))
        max_mb = int(getattr(settings, "repo_clone_max_mb", 2048))
        lfs = bool(getattr(settings, "repo_clone_lfs", False))

        # Clean any stale destination so the clone is deterministic.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")  # never block on credentials
        if not lfs:
            env["GIT_LFS_SKIP_SMUDGE"] = "1"

        cmd = [
            "git", "clone", "--depth", "1", "--no-tags",
            "--config", "credential.helper=",  # disable interactive auth prompts
            spec.url, str(dest),
        ]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("repo clone failed (%s): %s", type(exc).__name__, spec.url)
            shutil.rmtree(dest, ignore_errors=True)
            return None
        if proc.returncode != 0:
            logger.warning("repo clone non-zero (%s): %s", proc.returncode, (proc.stderr or "")[:300])
            shutil.rmtree(dest, ignore_errors=True)
            return None

        size_mb = _dir_size_mb(dest)
        if max_mb > 0 and size_mb > max_mb:  # 0 disables the cap (codebase convention)
            logger.warning("repo clone oversize %.1f MB > %d MB: %s", size_mb, max_mb, spec.url)
            shutil.rmtree(dest, ignore_errors=True)
            return None

        commit_sha: str | None = None
        try:
            rev = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=dest, env=env,
                capture_output=True, text=True, timeout=30,
            )
            if rev.returncode == 0:
                commit_sha = rev.stdout.strip() or None
        except (subprocess.TimeoutExpired, OSError):
            commit_sha = None

        return build_manifest(dest, commit_sha=commit_sha, size_mb=size_mb, lfs_skipped=not lfs)
