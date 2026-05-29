"""Local GPU allocator for parallel paper reproductions on a multi-GPU host.

This module provides a file-backed, crash-safe, concurrency-safe allocator that
schedules parallel ``run_experiment`` calls across the host's NVIDIA GPUs without
double-allocating a card to two simultaneous reproductions.

Design principles
-----------------
* **Respects other users** — a GPU is considered occupied (and therefore
  excluded from allocation) if it has *any* active compute process on it
  (``ext_proc_count > 0``) or if its used-memory already exceeds
  ``free_mem_threshold_mb``.  This is intentionally conservative: we would
  rather skip a card than evict another user's training job.

* **Crash-safe leasing** — leases are persisted as JSON in a caller-supplied
  state file.  Stale leases (whose owning PID has exited) are reaped lazily on
  every mutating call so a crashed launcher never permanently strands GPUs.

* **Atomic writes** — all state mutations write to a sibling ``.tmp`` file then
  ``os.replace`` into place (POSIX rename-atomicity); partial writes are never
  observed by concurrent readers.

* **Exclusive OS-level locking** — every read-modify-write cycle acquires an
  exclusive ``fcntl.flock`` on a sibling ``.lock`` file so two launcher
  processes racing to ``acquire`` from the same runs directory cannot
  double-allocate.

* **Monkeypatch-friendly** — the class methods call the module-level
  ``discover_gpus`` and ``free_devices`` by their global names rather than
  binding them as default arguments, so tests can replace those module
  attributes and the class picks up the replacement transparently.

Typical usage
-------------
::

    from pathlib import Path
    from backend.services.runtime.local_gpu_allocator import LocalGpuAllocator

    allocator = LocalGpuAllocator(Path("runs/.gpu_leases.json"))
    lease = allocator.acquire("run-abc123", os.getpid(), count=1)
    if lease is None:
        raise RuntimeError("No free GPU available")
    try:
        env = {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in lease.gpu_indices)}
        # ... spawn experiment subprocess with env ...
    finally:
        allocator.release(lease.lease_id)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

try:
    import fcntl as _fcntl

    _FCNTL_AVAILABLE = True
except ImportError:  # non-POSIX (shouldn't happen on Linux; degrade gracefully)
    _fcntl = None  # type: ignore[assignment]
    _FCNTL_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    "GpuDevice",
    "GpuLease",
    "LocalGpuAllocator",
    "discover_gpus",
    "free_devices",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuDevice:
    """Snapshot of one NVIDIA GPU as reported by nvidia-smi.

    ``ext_proc_count`` is the number of compute processes currently running on
    this card.  Any non-zero value means someone else (or a foreign process) is
    using the card and it should be treated as occupied.
    """

    index: int  # nvidia-smi index (under CUDA_DEVICE_ORDER=PCI_BUS_ID)
    uuid: str  # stable "GPU-xxxxxxxx-..." identity
    memory_total_mb: int
    memory_used_mb: int
    ext_proc_count: int  # number of compute processes on this GPU (any = occupied)


@dataclass(frozen=True)
class GpuLease:
    """An exclusive allocation of one or more GPUs held by a specific PID.

    ``lease_id`` is a caller-supplied opaque string (e.g. a run project_id).
    ``gpu_uuids`` / ``gpu_indices`` are parallel tuples: index ``i`` in each
    corresponds to the same physical card.
    """

    lease_id: str
    pid: int
    gpu_uuids: tuple[str, ...]
    gpu_indices: tuple[int, ...]
    created_at: str  # ISO-8601 UTC


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _run_nvidia_smi(*args: str, timeout: int = 30) -> str | None:
    """Run nvidia-smi with the given arguments; return stdout or None on any failure.

    Mirrors the ``_probe_nvidia_smi`` idiom in ``gpu_resolution.py``:
    ``capture_output=True, text=True, check=False``, safe-default to None on
    FileNotFoundError / TimeoutExpired / OSError.

    The default timeout is 30 s rather than the 3 s used by the lightweight
    ``-L`` probe in ``gpu_resolution.py``.  The heavier ``--query-gpu`` and
    ``--query-compute-apps`` CSV queries can take 10–15 s on a busy 8-GPU host
    where the NVIDIA management fabric is under contention.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("local_gpu_allocator: nvidia-smi %s failed (%s)", args[0] if args else "", exc)
        return None
    if result.returncode != 0:
        logger.debug(
            "local_gpu_allocator: nvidia-smi exit=%s stderr=%r",
            result.returncode,
            (result.stderr or "")[:200],
        )
        return None
    return result.stdout or ""


def discover_gpus() -> list[GpuDevice]:
    """Return a list of :class:`GpuDevice` objects, one per NVIDIA GPU on the host.

    Runs two nvidia-smi queries:

    1. ``--query-gpu=index,uuid,memory.total,memory.used --format=csv,noheader,nounits``
       to enumerate GPUs and their memory state.
    2. ``--query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits`` to
       find active compute processes; any process on a given UUID contributes to
       ``ext_proc_count``.

    Returns ``[]`` on any nvidia-smi failure (safe default — callers treat an
    empty list as "no GPUs available").  Devices are sorted by ``index``.

    Note on ``ext_proc_count`` semantics: we do not attempt to identify which
    PIDs belong to *our* process tree; any compute process at all counts as
    occupancy.  This is intentionally conservative — we would rather skip a
    card than compete with another user's training job.
    """
    # --- Step 1: enumerate GPUs ---
    gpu_output = _run_nvidia_smi(
        "--query-gpu=index,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    )
    if gpu_output is None:
        return []

    devices_raw: list[tuple[int, str, int, int]] = []
    for line in gpu_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            logger.debug("local_gpu_allocator: unexpected gpu query row: %r", line)
            continue
        try:
            idx = int(parts[0])
            uuid = parts[1]
            mem_total = int(parts[2])
            mem_used = int(parts[3])
        except ValueError:
            logger.debug("local_gpu_allocator: failed to parse gpu row: %r", line)
            continue
        devices_raw.append((idx, uuid, mem_total, mem_used))

    if not devices_raw:
        return []

    # --- Step 2: compute-app process counts per UUID ---
    uuid_to_proc_count: dict[str, int] = {}
    app_output = _run_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    )
    if app_output is not None:
        for line in app_output.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            gpu_uuid = parts[0]
            uuid_to_proc_count[gpu_uuid] = uuid_to_proc_count.get(gpu_uuid, 0) + 1

    # --- Assemble GpuDevice list ---
    devices = [
        GpuDevice(
            index=idx,
            uuid=uuid,
            memory_total_mb=mem_total,
            memory_used_mb=mem_used,
            ext_proc_count=uuid_to_proc_count.get(uuid, 0),
        )
        for idx, uuid, mem_total, mem_used in devices_raw
    ]
    devices.sort(key=lambda d: d.index)
    return devices


def free_devices(
    devices: list[GpuDevice] | None = None,
    *,
    free_mem_threshold_mb: int = 1024,
) -> list[GpuDevice]:
    """Filter ``devices`` to those that are unoccupied and have sufficient free memory.

    A device is considered free iff **both** conditions hold:

    * ``memory_used_mb < free_mem_threshold_mb`` — the card has less used
      memory than the threshold (default 1 GiB), indicating no heavyweight
      workload is loaded.
    * ``ext_proc_count == 0`` — no compute processes are currently running on
      the card.

    If ``devices`` is ``None``, :func:`discover_gpus` is called to obtain a
    fresh snapshot.

    This two-pronged check ensures that a card held by another user (high
    used-memory *or* an active process) is excluded, even if one of the two
    indicators temporarily lags the other.
    """
    if devices is None:
        devices = discover_gpus()
    return [
        d
        for d in devices
        if d.memory_used_mb < free_mem_threshold_mb and d.ext_proc_count == 0
    ]


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


class LocalGpuAllocator:
    """File-backed, crash-safe, concurrency-safe local GPU lease manager.

    One instance per state file; multiple processes may share the same state
    file safely because all mutations are protected by an exclusive ``fcntl``
    lock on a sibling ``.lock`` file.

    Lease lifecycle::

        lease = allocator.acquire("my-run-id", os.getpid(), count=1)
        # ... use lease.gpu_indices / lease.gpu_uuids ...
        allocator.release("my-run-id")

    Stale leases (PIDs that no longer exist) are reaped automatically on every
    ``acquire``, ``release``, and ``reclaim_stale`` call.

    Args:
        state_path: Path to the JSON lease-state file.  The parent directory
            is created on first write.  A sibling ``<state_path>.lock`` file
            is created for OS-level mutual exclusion.
    """

    def __init__(self, state_path: Path) -> None:
        """Initialise with the path to the JSON state file.

        The file and its parent directory need not exist yet; they are created
        on the first mutating call.
        """
        self._state_path = Path(state_path)
        self._lock_path = self._state_path.with_suffix(self._state_path.suffix + ".lock")
        if not _FCNTL_AVAILABLE:
            logger.warning(
                "local_gpu_allocator: fcntl not available on this platform; "
                "concurrent acquire() calls are NOT safe.  "
                "This module is intended for Linux only."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Generator[None, None, None]:
        """Context manager that holds an exclusive OS-level flock for the duration.

        Creates the lock file if it does not exist.  On platforms where
        ``fcntl`` is unavailable the context manager is a no-op (with a
        one-time warning already emitted in ``__init__``).
        """
        if not _FCNTL_AVAILABLE:
            yield
            return

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self._lock_path, "a")  # noqa: WPS515  (open in contextmanager)
        try:
            _fcntl.flock(fd.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(fd.fileno(), _fcntl.LOCK_UN)
        finally:
            fd.close()

    def _load_state(self) -> dict[str, dict]:
        """Load lease state from disk.  Returns ``{}`` if the file is absent or corrupt."""
        if not self._state_path.exists():
            return {}
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("local_gpu_allocator: state file is not a JSON object; resetting")
                return {}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("local_gpu_allocator: failed to read state file (%s); resetting", exc)
            return {}

    def _save_state(self, state: dict[str, dict]) -> None:
        """Write lease state atomically via a temp file + os.replace."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            logger.error("local_gpu_allocator: failed to persist state (%s)", exc)
            raise

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Return True if the PID is alive on this system.

        Uses ``os.kill(pid, 0)``:
        * ``ProcessLookupError`` (ESRCH) — process does not exist → dead.
        * ``PermissionError`` (EPERM) — process exists but is owned by another
          user → alive, keep the lease.
        """
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, just not ours

    @staticmethod
    def _lease_from_dict(lease_id: str, d: dict) -> GpuLease:
        return GpuLease(
            lease_id=lease_id,
            pid=int(d["pid"]),
            gpu_uuids=tuple(d["gpu_uuids"]),
            gpu_indices=tuple(int(i) for i in d["gpu_indices"]),
            created_at=d["created_at"],
        )

    @staticmethod
    def _lease_to_dict(lease: GpuLease) -> dict:
        return {
            "pid": lease.pid,
            "gpu_uuids": list(lease.gpu_uuids),
            "gpu_indices": list(lease.gpu_indices),
            "created_at": lease.created_at,
        }

    def _reclaim_stale_inplace(self, state: dict[str, dict]) -> list[str]:
        """Reap dead leases from ``state`` in-place; return reclaimed lease_ids."""
        dead: list[str] = []
        for lid, entry in list(state.items()):
            pid = int(entry.get("pid", -1))
            if not self._is_pid_alive(pid):
                dead.append(lid)
                del state[lid]
                logger.info(
                    "local_gpu_allocator: reclaimed stale lease %r (pid %d exited)",
                    lid,
                    pid,
                )
        return dead

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(
        self,
        lease_id: str,
        pid: int,
        *,
        count: int,
        free_mem_threshold_mb: int = 1024,
    ) -> GpuLease | None:
        """Attempt to exclusively allocate ``count`` free GPUs for ``lease_id``.

        All allocation logic runs under an exclusive OS lock to prevent
        concurrent launcher processes from double-allocating the same card.

        Steps:

        1. Reap stale leases (dead PIDs).
        2. If ``lease_id`` already exists in state, return the existing lease
           (idempotent — safe for retry after a transient failure).
        3. Discover GPUs, find free ones, subtract already-leased UUIDs.
        4. If ``< count`` free-and-unleased GPUs remain, return ``None``
           (no partial allocation).
        5. Take the first ``count`` by index (deterministic), write the lease,
           return the :class:`GpuLease`.

        Args:
            lease_id: Stable identifier for this allocation (e.g. run project_id).
            pid: PID of the process that should hold the lease.
            count: Number of GPUs requested.
            free_mem_threshold_mb: Upper bound on ``memory_used_mb`` for a GPU
                to be considered free (default 1 GiB).

        Returns:
            A :class:`GpuLease` on success, or ``None`` when fewer than
            ``count`` GPUs are available.
        """
        with self._locked():
            state = self._load_state()
            # 1. Reap stale leases
            self._reclaim_stale_inplace(state)

            # 2. Idempotent: return existing lease
            if lease_id in state:
                logger.debug("local_gpu_allocator: lease %r already exists; returning existing", lease_id)
                return self._lease_from_dict(lease_id, state[lease_id])

            # 3. Discover + filter free GPUs, subtract already-leased
            leased_uuids: set[str] = set()
            for entry in state.values():
                leased_uuids.update(entry.get("gpu_uuids", []))

            all_devices = discover_gpus()  # module-global — monkeypatchable
            candidates = [
                d
                for d in free_devices(all_devices, free_mem_threshold_mb=free_mem_threshold_mb)
                if d.uuid not in leased_uuids
            ]
            # candidates already sorted by index via discover_gpus

            # 4. Not enough GPUs?
            if len(candidates) < count:
                logger.debug(
                    "local_gpu_allocator: acquire %r count=%d — only %d free-and-unleased GPUs available",
                    lease_id,
                    count,
                    len(candidates),
                )
                return None

            # 5. Take first `count`, write lease
            chosen = candidates[:count]
            now_utc = datetime.now(timezone.utc).isoformat()
            lease = GpuLease(
                lease_id=lease_id,
                pid=pid,
                gpu_uuids=tuple(d.uuid for d in chosen),
                gpu_indices=tuple(d.index for d in chosen),
                created_at=now_utc,
            )
            state[lease_id] = self._lease_to_dict(lease)
            self._save_state(state)
            logger.info(
                "local_gpu_allocator: acquired lease %r → GPU indices %s (pid %d)",
                lease_id,
                list(lease.gpu_indices),
                pid,
            )
            return lease

    def release(self, lease_id: str) -> None:
        """Release the lease identified by ``lease_id``.

        A no-op if the lease does not exist.  Runs under the exclusive lock and
        persists atomically.
        """
        with self._locked():
            state = self._load_state()
            if lease_id not in state:
                logger.debug("local_gpu_allocator: release %r — not found (already released?)", lease_id)
                return
            del state[lease_id]
            self._save_state(state)
            logger.info("local_gpu_allocator: released lease %r", lease_id)

    def reclaim_stale(self) -> list[str]:
        """Reap all leases whose owning PID has exited.

        Returns the list of reclaimed ``lease_id`` strings.  A no-op (returns
        ``[]``) when all leases are held by live processes.

        This is called implicitly on every ``acquire`` and ``release``, but may
        also be called explicitly by a monitoring loop.
        """
        with self._locked():
            state = self._load_state()
            reclaimed = self._reclaim_stale_inplace(state)
            if reclaimed:
                self._save_state(state)
            return reclaimed

    def capacity(
        self,
        *,
        count: int,
        free_mem_threshold_mb: int = 1024,
    ) -> int:
        """Return the number of non-overlapping ``count``-GPU allocations currently possible.

        Reads the current lease state and queries nvidia-smi to find free,
        unleased GPUs, then returns ``max(0, floor(available / count))``.

        This is a read-only snapshot; it does NOT reserve any GPUs.

        Args:
            count: The size of each hypothetical allocation.
            free_mem_threshold_mb: Memory threshold passed to :func:`free_devices`.

        Returns:
            Maximum number of simultaneous ``count``-GPU runs that could be
            started right now.
        """
        with self._locked():
            state = self._load_state()
            self._reclaim_stale_inplace(state)

            leased_uuids: set[str] = set()
            for entry in state.values():
                leased_uuids.update(entry.get("gpu_uuids", []))

            all_devices = discover_gpus()
            available = [
                d
                for d in free_devices(all_devices, free_mem_threshold_mb=free_mem_threshold_mb)
                if d.uuid not in leased_uuids
            ]
        return max(0, len(available) // count)

    def active_leases(self) -> list[GpuLease]:
        """Return all currently recorded leases (including potentially stale ones).

        This is a pure read; it does NOT reap stale leases.  Call
        :meth:`reclaim_stale` first if you want a clean view.
        """
        state = self._load_state()
        return [self._lease_from_dict(lid, entry) for lid, entry in state.items()]
