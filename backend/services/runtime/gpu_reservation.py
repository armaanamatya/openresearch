"""Best-effort GPU reservation for a shared multi-GPU host.

Parks a tiny CUDA holder (``gpu_holder``) on currently-FREE GPUs so other users'
"find a free card" tooling skips them, and records the holds in a file-locked
registry. Reservations are:

* **safe for other users** — only ever placed on cards that are free *at
  reservation time* (``free_devices`` excludes any card with a foreign process or
  heavyweight memory use); we never evict ``zby22``/``bgu9`` et al.;
* **TTL-bounded** — each hold auto-releases after ``ttl_seconds`` (and the
  registry reaps expired/dead holds), so a forgotten reservation can't strand
  cards forever — good citizenship even while "disregarding etiquette";
* **transparent to our own runs** — the holders' PIDs are exported via
  :meth:`holder_pids`, which the :class:`LocalGpuAllocator` consumes as
  ``own_pids`` so OUR reproductions can still lease and use a reserved card;
* **crash-safe** — the registry is atomic-write + ``fcntl``-locked, and dead
  holders (PID gone) are reaped lazily, mirroring ``local_gpu_allocator``.

Etiquette note: holding cards you aren't actively using is antisocial on a shared
box. This is opt-in and TTL-bounded by design.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

from backend.services.runtime.local_gpu_allocator import (
    GpuDevice,
    discover_gpus,
    free_devices,
)

try:
    import fcntl as _fcntl

    _FCNTL_AVAILABLE = True
except ImportError:  # pragma: no cover - non-POSIX
    _fcntl = None  # type: ignore[assignment]
    _FCNTL_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["GpuReservation", "GpuReservationManager"]

_DEFAULT_HOLD_MIB = 256
_DEFAULT_TTL_SECONDS = 12 * 3600  # 12h — bounded so a forgotten hold self-releases
_HOLD_CONFIRM_SECONDS = 4.0  # wait this long for a holder to prove it grabbed the card


@dataclass(frozen=True)
class GpuReservation:
    """One reserved GPU held by a ``gpu_holder`` subprocess."""

    uuid: str
    index: int
    pid: int
    hold_mib: int
    reserved_at: str  # ISO-8601 UTC
    expires_at: str  # ISO-8601 UTC, or "" for no expiry


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True


class GpuReservationManager:
    """File-backed, crash-safe manager for GPU reservations on the local host.

    Args:
        registry_path: JSON registry file (a sibling ``.lock`` is used for the
            exclusive ``fcntl`` lock). Parent dir is created on first write.
        repo_root: working directory for the spawned holder (so
            ``python -m backend.services.runtime.gpu_holder`` resolves). Defaults
            to two levels up from this file's package root.
        python_executable: interpreter for the holder (defaults to the current).
    """

    def __init__(
        self,
        registry_path: Path,
        *,
        repo_root: Path | None = None,
        python_executable: str | None = None,
        lease_state_path: Path | None = None,
    ) -> None:
        self._registry_path = Path(registry_path)
        self._lock_path = self._registry_path.with_suffix(self._registry_path.suffix + ".lock")
        self._repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self._python = python_executable or sys.executable
        # The allocator's lease file (sibling by default): a card another run has
        # leased may be physically idle right now, so we must not reserve over it.
        self._lease_state_path = (
            Path(lease_state_path) if lease_state_path
            else self._registry_path.parent / ".gpu_leases.json"
        )

    # ------------------------------------------------------------------ locking
    @contextmanager
    def _locked(self) -> Generator[None, None, None]:
        if not _FCNTL_AVAILABLE:
            yield
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self._lock_path, "a")
        try:
            _fcntl.flock(fd.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(fd.fileno(), _fcntl.LOCK_UN)
        finally:
            fd.close()

    def _load(self) -> dict[str, dict]:
        if not self._registry_path.exists():
            return {}
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("gpu_reservation: unreadable registry (%s); resetting", exc)
            return {}

    def _save(self, state: dict[str, dict]) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(self._registry_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self._registry_path)

    def _leased_uuids(self) -> frozenset[str]:
        """UUIDs currently leased by the allocator (so we don't reserve over a run)."""
        try:
            data = json.loads(self._lease_state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return frozenset()
        out: set[str] = set()
        if isinstance(data, dict):
            for entry in data.values():
                if isinstance(entry, dict):
                    out.update(entry.get("gpu_uuids", []))
        return frozenset(out)

    # ------------------------------------------------------------------ reaping
    @staticmethod
    def _expired(entry: dict, now: datetime) -> bool:
        exp = entry.get("expires_at") or ""
        if not exp:
            return False
        try:
            return now >= datetime.fromisoformat(exp)
        except ValueError:
            return False

    def _reap_inplace(self, state: dict[str, dict], now: datetime) -> list[str]:
        """Drop dead (PID gone) and expired (TTL) holds; SIGTERM expired holders."""
        dropped: list[str] = []
        for uuid, entry in list(state.items()):
            pid = int(entry.get("pid", -1))
            if not _is_pid_alive(pid):
                dropped.append(uuid)
                del state[uuid]
                continue
            if self._expired(entry, now):
                _terminate(pid)
                dropped.append(uuid)
                del state[uuid]
                logger.info("gpu_reservation: reaped expired hold on %s (pid %d)", uuid, pid)
        return dropped

    # ------------------------------------------------------------------- public
    def list_reservations(self) -> list[GpuReservation]:
        """Live reservations (reaps dead/expired first)."""
        with self._locked():
            state = self._load()
            if self._reap_inplace(state, datetime.now(timezone.utc)):
                self._save(state)
            return [_to_res(u, e) for u, e in state.items()]

    def holder_pids(self) -> frozenset[int]:
        """PIDs of live holders — feed to ``LocalGpuAllocator(...).acquire(own_pids=...)``."""
        return frozenset(r.pid for r in self.list_reservations())

    def reserved_uuids(self) -> frozenset[str]:
        return frozenset(r.uuid for r in self.list_reservations())

    def reserve(
        self,
        *,
        count: int | None = None,
        uuids: "list[str] | None" = None,
        all_free: bool = False,
        hold_mib: int = _DEFAULT_HOLD_MIB,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> list[GpuReservation]:
        """Reserve free GPUs. Exactly one selector: ``count``, ``uuids``, or ``all_free``.

        Only cards that are free *right now* (excluding other users and already-
        reserved cards) are targeted. Returns the reservations actually created.
        """
        if sum(x is not None and x is not False for x in (count, uuids, all_free)) != 1:
            raise ValueError("reserve() takes exactly one of count=, uuids=, all_free=True")

        with self._locked():
            state = self._load()
            now = datetime.now(timezone.utc)
            self._reap_inplace(state, now)

            already = set(state.keys())
            own = frozenset(int(e["pid"]) for e in state.values())
            leased = self._leased_uuids()
            devices = discover_gpus()
            free = [
                d for d in free_devices(devices, own_pids=own)
                if d.uuid not in already and d.uuid not in leased
            ]
            by_uuid = {d.uuid: d for d in devices}

            targets = self._select_targets(free, by_uuid, count, uuids, all_free, already)

            created: list[GpuReservation] = []
            for dev in targets:
                res = self._spawn_holder(dev, hold_mib, ttl_seconds, now)
                if res is not None:
                    state[res.uuid] = _to_dict(res)
                    created.append(res)
            if created:
                self._save(state)
            return created

    def _select_targets(
        self,
        free: list[GpuDevice],
        by_uuid: dict[str, GpuDevice],
        count: int | None,
        uuids: "list[str] | None",
        all_free: bool,
        already: set[str],
    ) -> list[GpuDevice]:
        if uuids is not None:
            out: list[GpuDevice] = []
            free_uuids = {d.uuid for d in free}
            for u in uuids:
                if u in already:
                    logger.info("gpu_reservation: %s already reserved — skipping", u)
                    continue
                if u not in free_uuids:
                    logger.warning("gpu_reservation: %s is not free right now — skipping", u)
                    continue
                out.append(by_uuid[u])
            return out
        if all_free:
            return free
        return free[: max(0, int(count or 0))]

    def _spawn_holder(
        self, dev: GpuDevice, hold_mib: int, ttl_seconds: int, now: datetime
    ) -> GpuReservation | None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev.uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        log_dir = self._registry_path.parent / ".cache"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gpu_holder_{dev.index}.log"
        cmd = [
            self._python, "-m", "backend.services.runtime.gpu_holder",
            "--uuid", dev.uuid, "--mib", str(hold_mib), "--ttl-seconds", str(ttl_seconds),
        ]
        try:
            logf = open(log_path, "ab")
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(self._repo_root),
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,  # outlive this process / the CLI
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("gpu_reservation: failed to spawn holder for %s: %s", dev.uuid, exc)
            return None

        # Confirm the holder actually grabbed the card (it exits 2 on failure).
        deadline = time.time() + _HOLD_CONFIRM_SECONDS
        while time.time() < deadline:
            rc = proc.poll()
            if rc is not None:
                logger.error("gpu_reservation: holder for GPU %d exited rc=%s (see %s)",
                             dev.index, rc, log_path)
                return None
            time.sleep(0.25)

        expires = "" if ttl_seconds <= 0 else (now + timedelta(seconds=ttl_seconds)).isoformat()
        logger.info("gpu_reservation: reserved GPU %d (%s) pid=%d ttl=%ss",
                    dev.index, dev.uuid[:20], proc.pid, ttl_seconds or "none")
        return GpuReservation(
            uuid=dev.uuid, index=dev.index, pid=proc.pid, hold_mib=hold_mib,
            reserved_at=now.isoformat(), expires_at=expires,
        )

    def release(self, *, uuids: "list[str] | None" = None, all: bool = False) -> list[str]:
        """Release reservations (SIGTERM the holders). Returns released UUIDs."""
        if (uuids is None) == (not all):
            raise ValueError("release() takes exactly one of uuids= or all=True")
        with self._locked():
            state = self._load()
            targets = list(state.keys()) if all else [u for u in (uuids or []) if u in state]
            for u in targets:
                _terminate(int(state[u].get("pid", -1)))
                del state[u]
            self._save(state)
            if targets:
                logger.info("gpu_reservation: released %d hold(s): %s", len(targets), targets)
            return targets


def _terminate(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:  # pragma: no cover
        logger.warning("gpu_reservation: not permitted to SIGTERM pid %d", pid)


def _to_dict(r: GpuReservation) -> dict:
    return {
        "index": r.index, "pid": r.pid, "hold_mib": r.hold_mib,
        "reserved_at": r.reserved_at, "expires_at": r.expires_at,
    }


def _to_res(uuid: str, e: dict) -> GpuReservation:
    return GpuReservation(
        uuid=uuid, index=int(e.get("index", -1)), pid=int(e.get("pid", -1)),
        hold_mib=int(e.get("hold_mib", _DEFAULT_HOLD_MIB)),
        reserved_at=e.get("reserved_at", ""), expires_at=e.get("expires_at", ""),
    )
