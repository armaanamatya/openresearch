"""GPU cell scheduler — harness-owned, OOM-proof multi-GPU training matrix.

The "GPU-stacking" problem (Lane I).  Agent-written code almost always
hard-codes ``device="cuda:0"`` and iterates the full training matrix in a
single process.  When the matrix has 3 models × 5 baselines × 3 envs, all
cells pile onto one 24 GB card while the other reserved cards sit idle —
guaranteed CUDA OOM for the 7B model cells.

This module makes that class of failure **structurally impossible** by owning
GPU placement at the harness level:

* Each cell is launched as a **subprocess** whose ``CUDA_VISIBLE_DEVICES`` is
  set to exactly **one** physical GPU id.  The child process sees only
  ``cuda:0``; it physically cannot address another card.
* A pool scheduler keeps at most ``len(gpus)`` cells running concurrently.
  As each subprocess exits, the freed GPU immediately accepts the next cell
  from the queue.
* On CUDA-OOM the cell is **retried** on a free GPU with reduced footprint
  (``REPROLAB_CELL_BATCH_SCALE=0.5`` then ``0.25``, plus
  ``REPROLAB_CELL_GRAD_CHECKPOINT=1``) before being marked ``oom_failed``.

Copyable helper — mirror of the ``rubric_guard.py`` pattern.
The ``implement_baseline`` prompt instructs the agent to copy this file into
``code/gpu_cell_runner.py``; the agent's orchestrator script then does::

    from gpu_cell_runner import run_matrix, discover_visible_gpus
    results = run_matrix(cells, "train_cell.py", output_root="outputs/")

Zero non-stdlib dependencies, so the copy-and-paste route always works inside
an agent sandbox.

Auth-agnostic by construction (no provider branching, no LLM calls).
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)

# Signatures that indicate a CUDA out-of-memory event in subprocess output.
_OOM_SIGNATURES: tuple[str, ...] = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "OutOfMemoryError",
    "out of memory",      # torch: "RuntimeError: CUDA out of memory."
)

# Batch-scale values tried on successive OOM retries (after the original attempt).
_OOM_BATCH_SCALES: tuple[float, ...] = (0.5, 0.25)

__all__ = ["CellResult", "discover_visible_gpus", "run_matrix"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CellResult:
    """Result record for a single training cell.

    Attributes:
        cell_id:  Identifier matching the input ``cells`` entry.
        status:   ``"ok"`` | ``"oom_failed"`` | ``"error"``.
        metrics:  Dict loaded from the cell's ``metrics.json``, or ``None``.
        gpu:      Physical GPU id the cell ran on (last attempt).
        retries:  Number of OOM retries attempted (0 = first attempt succeeded
                  or failed with a non-OOM error).
        error:    Stderr snippet / exception message, or ``None`` on success.
    """

    __slots__ = ("cell_id", "status", "metrics", "gpu", "retries", "error")

    def __init__(
        self,
        *,
        cell_id: str,
        status: str,
        metrics: dict[str, Any] | None,
        gpu: str,
        retries: int,
        error: str | None,
    ) -> None:
        self.cell_id = cell_id
        self.status = status
        self.metrics = metrics
        self.gpu = gpu
        self.retries = retries
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "gpu": self.gpu,
            "retries": self.retries,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def discover_visible_gpus() -> list[str]:
    """Return the list of physical GPU ids visible to this process.

    Resolution order:

    1. ``CUDA_VISIBLE_DEVICES`` — parsed if set and non-empty.  Handles both
       integer indices (``"0,1,2,3"``) and UUID forms
       (``"GPU-xxxxxxxx-xxxx-..."``) transparently.
    2. ``nvidia-smi --query-gpu=index`` — falls back to the host device count
       when the env var is absent or set to ``""``.
    3. ``["0"]`` — last-resort single-card assumption.

    Returns:
        List of GPU id strings (e.g. ``["0", "1", "2", "3"]`` or
        ``["GPU-abc123...", "GPU-def456..."]``).  Never empty.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if cvd and cvd.upper() not in ("", "NODEVFILES", "-1"):
        # Split on comma; each token is either an integer index or a UUID.
        ids = [tok.strip() for tok in cvd.split(",") if tok.strip()]
        if ids:
            return ids

    # Fall back to nvidia-smi — count visible cards from the system.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=10,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            return ids
    except Exception:  # nvidia-smi absent or timed out
        pass

    return ["0"]


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------

def _is_oom(output: str) -> bool:
    """Return True iff ``output`` (stderr + stdout combined) contains an OOM signature."""
    return any(sig in output for sig in _OOM_SIGNATURES)


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

def _load_metrics(output_dir: Path) -> dict[str, Any] | None:
    """Load ``metrics.json`` from ``output_dir``, returning None on any failure."""
    mf = output_dir / "metrics.json"
    if not mf.is_file():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-cell subprocess launcher
# ---------------------------------------------------------------------------

def _run_cell_subprocess(
    *,
    cell: dict[str, Any],
    cell_script: str | Path,
    gpu_id: str,
    output_dir: Path,
    batch_scale: float | None,
    grad_checkpoint: bool,
    timeout_s: float | None,
    log_path: Path,
) -> tuple[int, str]:
    """Launch the cell script as a subprocess pinned to ``gpu_id``.

    Returns ``(returncode, captured_output)``.  The captured output is the
    combined stdout+stderr, written to ``log_path`` and also returned for OOM
    detection.

    The child environment inherits the current environment with overrides:

    ``CUDA_VISIBLE_DEVICES=<gpu_id>``
        The child sees exactly one GPU ("cuda:0").  Physical cannot use others.

    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
        Reduces allocator fragmentation on long training loops.

    ``REPROLAB_CELL_BATCH_SCALE=<batch_scale>``
        Set on OOM retries so the cell script can reduce its batch size.

    ``REPROLAB_CELL_GRAD_CHECKPOINT=1``
        Set on OOM retries to enable gradient checkpointing in the cell script.

    ``REPROLAB_CELL_OUTPUT_DIR=<output_dir>``
        Canonical output directory for ``metrics.json`` and artifacts.

    ``REPROLAB_CELL_PARAMS=<json>``
        Full cell dict serialised as JSON, so the cell script can read its
        own parameters without parsing argv.
    """
    child_env = {**os.environ}
    child_env["CUDA_VISIBLE_DEVICES"] = gpu_id
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    child_env["REPROLAB_CELL_OUTPUT_DIR"] = str(output_dir)
    child_env["REPROLAB_CELL_PARAMS"] = json.dumps(cell)

    if batch_scale is not None:
        child_env["REPROLAB_CELL_BATCH_SCALE"] = str(batch_scale)
        child_env["REPROLAB_CELL_GRAD_CHECKPOINT"] = "1" if grad_checkpoint else "0"
    else:
        # Clear any inherited values from a parent run.
        child_env.pop("REPROLAB_CELL_BATCH_SCALE", None)
        child_env.pop("REPROLAB_CELL_GRAD_CHECKPOINT", None)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", str(cell_script)] + [
        f"--cell-id={cell.get('id', '')}",
        f"--output-dir={output_dir}",
    ]

    captured_chunks: list[str] = []

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
            proc = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                # New process group so we can kill the whole tree on timeout.
                start_new_session=True,
            )

            def _reader() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    captured_chunks.append(line)
                    log_fh.write(line)
                    log_fh.flush()

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()

            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                # Kill the process group to clean up child processes.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                timeout_msg = f"\n[gpu_cell_runner] TIMEOUT after {timeout_s}s\n"
                captured_chunks.append(timeout_msg)
                log_fh.write(timeout_msg)

            reader_thread.join(timeout=5.0)

        return proc.returncode, "".join(captured_chunks)

    except Exception as exc:
        msg = f"[gpu_cell_runner] failed to launch subprocess: {exc}\n"
        captured_chunks.append(msg)
        try:
            log_path.write_text(msg, encoding="utf-8")
        except OSError:
            pass
        return -1, "".join(captured_chunks)


# ---------------------------------------------------------------------------
# Pool scheduler
# ---------------------------------------------------------------------------

def run_matrix(
    cells: list[dict[str, Any]],
    cell_script: str | Path,
    *,
    output_root: str | Path,
    gpus: list[str] | None = None,
    max_parallel: int | None = None,
    max_oom_retries: int = 2,
    per_cell_timeout_s: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Schedule and run all cells across the GPU pool.

    Each cell is an independent training job that runs on **exactly one** GPU.
    The scheduler keeps up to ``max_parallel`` (default: ``len(gpus)``) cells
    running concurrently.  As each subprocess exits, the freed GPU immediately
    accepts the next queued cell.

    Args:
        cells:               List of cell-description dicts.  Each must have an
                             ``"id"`` key (used as directory name and result key).
        cell_script:         Path to the single-cell trainer script.  The script
                             trains exactly ONE cell on ``cuda:0`` (its only
                             visible GPU), reads parameters from env vars
                             ``REPROLAB_CELL_PARAMS`` / ``REPROLAB_CELL_OUTPUT_DIR``
                             and argv ``--cell-id`` / ``--output-dir``, and writes
                             ``metrics.json`` to its output directory.
        output_root:         Root directory; each cell writes to
                             ``output_root/<cell_id>/``.
        gpus:                Physical GPU ids to use.  Defaults to
                             :func:`discover_visible_gpus`.
        max_parallel:        Maximum simultaneous subprocesses.  Defaults to
                             ``len(gpus)``.
        max_oom_retries:     How many OOM retries to attempt before marking a
                             cell ``oom_failed`` (default 2 — tries batch_scale
                             0.5 then 0.25 before giving up).
        per_cell_timeout_s:  Per-cell wall-clock timeout in seconds.  ``None``
                             means no limit.

    Returns:
        Dict mapping ``cell["id"]`` → :meth:`CellResult.to_dict`.  Every input
        cell is represented regardless of outcome — never raises on individual
        cell failure.

    Design invariants:

    * Each subprocess starts with ``CUDA_VISIBLE_DEVICES=<one id>`` so it
      physically cannot access a second card.
    * Concurrency never exceeds ``min(max_parallel, len(gpus))``.
    * GPU ids in ``gpus`` are the only ids ever assigned.
    * OOM retries pass ``REPROLAB_CELL_BATCH_SCALE`` and
      ``REPROLAB_CELL_GRAD_CHECKPOINT=1`` as env vars; the cell script is
      responsible for honouring them.
    """
    if not cells:
        return {}

    resolved_gpus: list[str] = gpus if gpus is not None else discover_visible_gpus()
    if not resolved_gpus:
        resolved_gpus = ["0"]

    parallelism = max_parallel if max_parallel is not None else len(resolved_gpus)
    parallelism = max(1, min(parallelism, len(resolved_gpus)))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cell_script = Path(cell_script)

    results: dict[str, CellResult] = {}
    results_lock = threading.Lock()

    # Queue of (cell, retry_index) — retry_index 0 = first attempt.
    work_queue: Queue[tuple[dict[str, Any], int]] = Queue()
    for cell in cells:
        work_queue.put((cell, 0))

    # GPU pool — a queue of free GPU ids.
    gpu_pool: Queue[str] = Queue()
    for gid in resolved_gpus[:parallelism]:
        gpu_pool.put(gid)

    active_threads: list[threading.Thread] = []
    active_lock = threading.Lock()

    def _worker() -> None:
        """One worker thread: grab a GPU, run one cell attempt, release the GPU."""
        while True:
            # Acquire a free GPU (blocks until one is available).
            try:
                gpu_id = gpu_pool.get(timeout=120)
            except Empty:
                # No GPU became available within 2 minutes — something is stuck.
                logger.warning("gpu_cell_runner: timed out waiting for a free GPU")
                break

            try:
                # Get next work item — non-blocking (workers only run when there is work).
                cell, retry_idx = work_queue.get_nowait()
            except Empty:
                # Queue drained; return the GPU and exit this worker.
                gpu_pool.put(gpu_id)
                break

            cell_id: str = cell.get("id", f"cell_{id(cell)}")
            output_dir = output_root / cell_id
            log_path = output_root / f"{cell_id}.log"

            # Determine OOM-mitigation parameters for this attempt.
            batch_scale: float | None = None
            grad_checkpoint = False
            if retry_idx > 0:
                scale_idx = retry_idx - 1  # 0 → 0.5, 1 → 0.25
                if scale_idx < len(_OOM_BATCH_SCALES):
                    batch_scale = _OOM_BATCH_SCALES[scale_idx]
                grad_checkpoint = True

            logger.info(
                "gpu_cell_runner: starting cell=%s gpu=%s retry=%d batch_scale=%s",
                cell_id, gpu_id, retry_idx, batch_scale,
            )

            returncode, output = _run_cell_subprocess(
                cell=cell,
                cell_script=cell_script,
                gpu_id=gpu_id,
                output_dir=output_dir,
                batch_scale=batch_scale,
                grad_checkpoint=grad_checkpoint,
                timeout_s=per_cell_timeout_s,
                log_path=log_path,
            )

            # Return GPU to pool immediately after subprocess exits.
            gpu_pool.put(gpu_id)

            if returncode == 0:
                metrics = _load_metrics(output_dir)
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="ok",
                        metrics=metrics,
                        gpu=gpu_id,
                        retries=retry_idx,
                        error=None,
                    )
                logger.info("gpu_cell_runner: cell=%s DONE ok (gpu=%s)", cell_id, gpu_id)
            elif _is_oom(output) and retry_idx < max_oom_retries:
                next_retry = retry_idx + 1
                logger.warning(
                    "gpu_cell_runner: cell=%s OOM on gpu=%s, scheduling retry %d/%d",
                    cell_id, gpu_id, next_retry, max_oom_retries,
                )
                work_queue.put((cell, next_retry))
                # Spawn a new worker to handle the queued retry (the current
                # worker exits; the retry will be picked up by the next idle
                # worker or a freshly spawned one).
                _spawn_worker()
            else:
                status = "oom_failed" if _is_oom(output) else "error"
                error_snippet = output[-2000:] if output else f"exit code {returncode}"
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status=status,
                        metrics=_load_metrics(output_dir),
                        gpu=gpu_id,
                        retries=retry_idx,
                        error=error_snippet,
                    )
                logger.warning(
                    "gpu_cell_runner: cell=%s FAILED status=%s gpu=%s retries=%d",
                    cell_id, status, gpu_id, retry_idx,
                )

    def _spawn_worker() -> None:
        t = threading.Thread(target=_worker, daemon=True)
        with active_lock:
            active_threads.append(t)
        t.start()

    # Spawn initial pool of workers — one per parallel slot.
    for _ in range(parallelism):
        _spawn_worker()

    # Wait for all workers to finish.
    while True:
        with active_lock:
            live = [t for t in active_threads if t.is_alive()]
        if not live:
            break
        for t in live:
            t.join(timeout=1.0)

    # Ensure every input cell has a result entry (defensive — all paths should
    # write to results, but guard against unexpected worker exits).
    for cell in cells:
        cell_id = cell.get("id", f"cell_{id(cell)}")
        if cell_id not in results:
            results[cell_id] = CellResult(
                cell_id=cell_id,
                status="error",
                metrics=None,
                gpu="unknown",
                retries=0,
                error="worker exited without recording a result",
            )

    return {cid: r.to_dict() for cid, r in results.items()}
