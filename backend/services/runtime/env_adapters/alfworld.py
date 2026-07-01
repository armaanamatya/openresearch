"""AlfworldAdapter — ALFWorld game-data provisioning (Phase 1a).

Part of the provisioning-seam refactor (see
``docs/superpowers/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``):
lifts ``EnvCacheManager.ensure_alfworld`` and its helper functions
(``_alfworld_has_games`` / ``_resolve_console_script`` /
``_default_alfworld_downloader``) out of ``env_cache.py`` verbatim, behind the
:class:`~backend.services.runtime.env_adapters.base.EnvironmentAdapter`
contract. ``env_cache.py`` itself is untouched by this unit; a later task
rewrites it as a facade delegating to this adapter.

The on-disk state key stays ``"alfworld"`` (== :attr:`AlfworldAdapter.key`) so
a warm SDAR cache disk is byte-compatible across the refactor.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.base import (
    EnvironmentAdapter,
    EnvSetupResult,
    ProvisionCtx,
    SmokeResult,
    _fail,
)

logger = logging.getLogger(__name__)

__all__ = ["AlfworldAdapter"]

_ALIASES = {"alfworld", "alf world", "alf-world"}


def _alfworld_has_games(data_dir: "Path | str", *, max_scan: int = 200_000) -> bool:
    """Return True iff at least one ``traj_data.json`` game file exists under ``data_dir``.

    Mirrors ``alfworld_env.ALFWorldEnv._has_any_games`` but lives here (stdlib-only,
    no alfworld import) so ``provision`` can verify the download without pulling
    in the alfworld package at module scope.  Bounded walk so a pathological tree never
    hangs.  Fail-soft: any OS error → False (treats as 'no games').
    """
    scanned = 0
    try:
        for _dp, _dns, filenames in os.walk(str(data_dir)):
            if "traj_data.json" in filenames:
                return True
            scanned += len(filenames) + 1
            if scanned >= max_scan:
                break
    except OSError:
        return False
    return False


def _resolve_console_script(name: str) -> str | None:
    """Resolve a venv console script (e.g. ``alfworld-download``) to an abs path.

    Console scripts install next to the interpreter (``<venv>/bin/<name>``) but
    that dir is not necessarily on a child process's PATH, so resolve by abs path
    first and fall back to a PATH lookup. Returns ``None`` if not found.
    """
    import shutil

    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


def _default_alfworld_downloader(cache_dir: Path) -> None:
    """Run ``alfworld-download`` into ``cache_dir`` (real path; injected in tests).

    ``ALFWORLD_DATA`` controls the download target. The console script is resolved
    by abs path (it may not be on the child's PATH); a missing script raises, which
    ``provision`` converts into a verified Exclusion.
    """
    import subprocess  # local import: only the real path needs it

    exe = _resolve_console_script("alfworld-download")
    if not exe:
        raise FileNotFoundError(
            "alfworld-download console script not found next to the interpreter "
            f"({Path(sys.executable).parent}) or on PATH"
        )
    env = {**os.environ, "ALFWORLD_DATA": str(cache_dir)}
    subprocess.run([exe], check=True, env=env, timeout=3600)


class AlfworldAdapter(EnvironmentAdapter):
    """Idempotent, fcntl-locked ALFWorld game-data provisioning.

    BES Phase 4A (A1) note — two distinct "once" caches, do not conflate: this
    adapter is the host-shared **game-data** cache (the multi-GB
    ``alfworld-download``), idempotent across runs/cells. The per-cell **env
    object** reuse (build ``AlfredTWEnv`` once and ``reset()`` it in place
    across episodes) lives in ``alfworld_env.ALFWorldEnv`` behind
    ``OPENRESEARCH_ALFWORLD_ENV_REUSE`` — a rollout-loop concern, not a data-cache
    one. The data cache that A1's env reuse builds on top of is exactly this
    adapter's output.
    """

    key = "alfworld"

    def __init__(
        self,
        cache: AssetCache,
        *,
        downloader: Callable[[Path], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = cache
        self._downloader = downloader or _default_alfworld_downloader
        self._clock = clock

    def applies(self, env_name: str) -> bool:
        return (env_name or "").strip().lower() in _ALIASES

    def provision(self, ctx: ProvisionCtx) -> EnvSetupResult:
        """Download ALFWorld once into the shared cache; reuse on later calls."""
        display_name = ctx.display_name or "ALFWorld"
        data_dir = self._cache.cache_dir / "alfworld"
        try:
            with self._cache.locked_state() as state:
                rec = state.get(self.key) or {}
                if rec.get("ready") and Path(rec.get("data_path", "")).exists():
                    # Game-file presence check: a stale/partial download could have
                    # written the directory with no traj_data.json games, in which
                    # case cells would produce info["unavailable"]=True and 0.0
                    # rewards that are counted as real results (the silent
                    # zero-score failure). Re-verify game files; if missing, fall
                    # through to re-download.
                    if _alfworld_has_games(Path(rec["data_path"])):
                        return EnvSetupResult(env=display_name, ok=True,
                                              data_path=rec["data_path"], detail="cache hit")
                    logger.warning(
                        "env_adapters.alfworld: cache hit but no games found under %r; "
                        "re-downloading to recover", rec["data_path"]
                    )
                    # Clear stale state so a re-download and fresh state record follow.
                    state.pop(self.key, None)
                data_dir.mkdir(parents=True, exist_ok=True)
                self._downloader(data_dir)  # injected; real path runs alfworld-download
                # Verify game files are actually present after the download completes.
                # A successful alfworld-download that yields no games (network blip,
                # partial fetch) must be treated as a failure — otherwise cells
                # inherit ALFWORLD_DATA pointing at an empty dir and produce 0.0
                # rewards that are counted in the score instead of being excluded
                # as env_setup_failed.
                if not _alfworld_has_games(data_dir):
                    reason = (
                        f"alfworld-download ran without error but no games (traj_data.json) "
                        f"found under {data_dir!r}; download may be incomplete. "
                        "Re-run with a clean cache (delete the alfworld/ subdir)."
                    )
                    logger.warning("env_adapters.alfworld: %s", reason)
                    return _fail(display_name, reason, evidence=str(data_dir))
                state[self.key] = {"ready": True, "data_path": str(data_dir),
                                   "downloaded_at": self._clock()}
                return EnvSetupResult(env=display_name, ok=True,
                                      data_path=str(data_dir), detail="downloaded")
        except Exception as exc:  # noqa: BLE001 — fail-soft into a verified Exclusion
            logger.warning("env_adapters.alfworld: setup failed: %s", exc)
            return _fail(display_name, f"alfworld-download failed: {type(exc).__name__}: {exc}",
                        evidence=str(exc)[:200])

    def smoke(self, ctx: ProvisionCtx) -> SmokeResult:
        """Cheap post-provision liveness: at least one game file on disk.

        Prefers the cached state's ``data_path`` (the authoritative record
        written by :meth:`provision`); falls back to ``ctx.display_name`` when
        no cached record exists (e.g. a caller probing a known path directly).
        """
        with self._cache.locked_state() as state:
            rec = state.get(self.key) or {}
            data_path = rec.get("data_path") or None
        if not data_path:
            data_path = ctx.display_name or None
        if not data_path:
            return SmokeResult(ok=False, detail="no ALFWorld data path known")
        ok = _alfworld_has_games(data_path)
        return SmokeResult(ok=ok, detail="games present" if ok else "no games found")
