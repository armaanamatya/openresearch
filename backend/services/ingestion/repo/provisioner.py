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
            if p.is_symlink():
                # Never follow a symlink into out-of-copy data (a repo may symlink
                # a multi-GB dataset from a cache dir; following it wrongly balloons
                # the measured size and trips the oversize cap). copytree preserves
                # the link (symlinks=True), so it costs ~0 bytes in the copy.
                continue
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
        try:
            # Instantiate fresh so env-var monkeypatching in tests is picked up.
            settings = Settings()
            timeout_s = int(getattr(settings, "repo_clone_timeout_s", 300))
            max_mb = int(getattr(settings, "repo_clone_max_mb", 2048))
            lfs = bool(getattr(settings, "repo_clone_lfs", False))
            local_path = str(getattr(settings, "repo_local_path", "") or "").strip()
            pin_commit = str(getattr(settings, "repo_commit", "") or "").strip()

            if local_path:
                local_dir = Path(local_path)
                if local_dir.is_dir():
                    return RepoProvisioner._reuse_local(
                        local_dir, dest, pin_commit=pin_commit, size_mb_cap=max_mb,
                        lfs=lfs,
                    )
                logger.warning(
                    "repo_local_path set but not a directory (%s); falling back to clone: %s",
                    local_path, spec.url,
                )

            # Clean any stale destination so the clone is deterministic.
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.parent.mkdir(parents=True, exist_ok=True)

            env = dict(os.environ)
            env["GIT_TERMINAL_PROMPT"] = "0"   # force: never block on credential prompts
            env["GIT_ASKPASS"] = "true"        # any askpass call returns empty -> fail fast
            env["GIT_SSH_COMMAND"] = "ssh -oBatchMode=yes -oStrictHostKeyChecking=accept-new"
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
        except Exception:  # noqa: BLE001 — any failure returns None (documented contract)
            logger.warning("repo clone unexpected failure (non-fatal): %s", spec.url, exc_info=True)
            shutil.rmtree(dest, ignore_errors=True)
            return None

    @staticmethod
    def _reuse_local(
        local_dir: Path, dest: Path, *, pin_commit: str, size_mb_cap: int, lfs: bool,
    ) -> RepoManifest | None:
        """Seed ``dest`` from a pre-staged local clone instead of a network clone.

        Copies ``local_dir`` (INCLUDING ``.git`` so a commit pin can be checked
        out) into ``dest``. Best-effort checkout of ``pin_commit`` — a failure
        just warns (the staged repo is already presumed to be at the pin).
        Fail-soft: any error returns ``None`` so the caller falls back to scratch.
        """
        try:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # symlinks=True PRESERVES symlinks instead of following them — a staged
            # repo may symlink a multi-GB dataset from the cache dir (e.g. SDAR's
            # webshop/data/items_shuffle.json -> /mnt/.../webshop/data, 5.2 GB);
            # following it balloons the copy past the oversize cap and forces a
            # from-scratch fallback. The absolute links still resolve on the host;
            # the execute-mode code/ seed excludes symlinks anyway.
            shutil.copytree(local_dir, dest, dirs_exist_ok=True, symlinks=True)

            env = dict(os.environ)
            if pin_commit:
                try:
                    checkout = subprocess.run(
                        ["git", "-C", str(dest), "checkout", pin_commit],
                        env=env, capture_output=True, text=True, timeout=60,
                    )
                    if checkout.returncode != 0:
                        logger.warning(
                            "repo_commit checkout failed (%s); using staged HEAD: %s",
                            pin_commit, (checkout.stderr or "")[:300],
                        )
                except (subprocess.TimeoutExpired, OSError) as exc:
                    logger.warning(
                        "repo_commit checkout raised (%s); using staged HEAD: %s",
                        type(exc).__name__, pin_commit,
                    )

            commit_sha: str | None = None
            try:
                rev = subprocess.run(
                    ["git", "-C", str(dest), "rev-parse", "HEAD"],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                if rev.returncode == 0:
                    commit_sha = rev.stdout.strip() or None
            except (subprocess.TimeoutExpired, OSError):
                commit_sha = None

            size_mb = _dir_size_mb(dest)
            if size_mb_cap > 0 and size_mb > size_mb_cap:  # 0 disables the cap
                logger.warning(
                    "local repo reuse oversize %.1f MB > %d MB: %s", size_mb, size_mb_cap, local_dir,
                )
                shutil.rmtree(dest, ignore_errors=True)
                return None

            return build_manifest(dest, commit_sha=commit_sha, size_mb=size_mb, lfs_skipped=not lfs)
        except Exception:  # noqa: BLE001 — any failure returns None (documented contract)
            logger.warning("local repo reuse unexpected failure (non-fatal): %s", local_dir, exc_info=True)
            shutil.rmtree(dest, ignore_errors=True)
            return None
