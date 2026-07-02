"""AssetCache — generalized crash-safe, fcntl-locked keyed state store.

Phase 1a of the provisioning-seam refactor (see
``docs/superpowers/plans/2026-07-01-phase-1a-1b-provisioning-seam-and-gates.md``):
this module lifts the crash-safe state-file substrate out of
``backend/services/runtime/env_cache.py``'s ``EnvCacheManager`` so it can back
ANY :class:`~backend.services.runtime.env_adapters.base.EnvironmentAdapter`,
not just the three SDAR-specific environments ``env_cache.py`` originally
hard-coded.

``default_cache_dir``, ``_pid_alive``, and the locked-state I/O
(``locked_state``/``_read_state``/``_write_state``) are moved **verbatim**
from ``env_cache.py`` — same semantics, same on-disk filenames
(``env_cache_state.json`` / ``.env_cache.lock``) — so a warm SDAR cache disk
on an existing host stays byte-compatible across this refactor. ``env_cache.py``
itself is untouched by this unit; it re-exports these symbols in a later task.

Stdlib-only: ``contextlib``, ``fcntl``, ``json``, ``os``, ``tempfile``, ``pathlib``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

__all__ = ["AssetCache", "default_cache_dir", "_pid_alive"]


def default_cache_dir() -> Path:
    """Resolve the shared env-cache dir from ``OPENRESEARCH_ENV_CACHE_DIR`` or default."""
    override = os.environ.get("OPENRESEARCH_ENV_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    runs_root = os.environ.get("OPENRESEARCH_RUNS_ROOT", "").strip() or "runs"
    return (Path(runs_root) / ".cache" / "envs").resolve()


def _pid_alive(pid: int) -> bool:
    """True iff ``pid`` is a live process (signal 0 probe). Mirrors the GPU allocator."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


class AssetCache:
    """Idempotent, fcntl-locked, keyed JSON state store shared by every adapter.

    Generalizes ``EnvCacheManager``'s private ``_locked_state``/``_read_state``/
    ``_write_state`` trio into a standalone, adapter-agnostic primitive: any
    :class:`~backend.services.runtime.env_adapters.base.EnvironmentAdapter`
    (ALFWorld / WebShop / Search-QA today; any future paper-specific asset
    tomorrow) opens the SAME on-disk state dict under an exclusive lock,
    mutates its own key, and has it persisted atomically on exit.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.cache_dir / "env_cache_state.json"
        self._lock_path = self.cache_dir / ".env_cache.lock"

    # --- locked state I/O (mirrors local_gpu_allocator's fcntl discipline) ----

    @contextlib.contextmanager
    def locked_state(self) -> Iterator[dict[str, Any]]:
        """Yield the mutable state dict under an exclusive lock; persist on exit."""
        with open(self._lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state()
                yield state
                self._write_state(state)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _read_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, prefix=".env_cache_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, self._state_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
