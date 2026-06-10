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
  (``OPENRESEARCH_CELL_BATCH_SCALE=0.5`` then ``0.25``, plus
  ``OPENRESEARCH_CELL_GRAD_CHECKPOINT=1``) before being marked ``oom_failed``.

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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

try:  # sandbox-flat: cell_scheduler.py is copied next to this file (see _HARNESS_CODE_HELPERS).
    from cell_scheduler import (
        CELL_MANIFEST_NAME,
        CellResult,
        clamp_cell_timeout,
        deadline_from_timeout,
        is_resume_armed,
        load_cell_manifest,
        should_skip_cell,
        write_cell_manifest,
    )
except ImportError:  # in-repo import path (running inside the harness package).
    from backend.agents.rlm.cell_scheduler import (
        CELL_MANIFEST_NAME,
        CellResult,
        clamp_cell_timeout,
        deadline_from_timeout,
        is_resume_armed,
        load_cell_manifest,
        should_skip_cell,
        write_cell_manifest,
    )

try:  # sandbox-flat: dead_training_guard.py is copied next to this file.
    import dead_training_guard
except ImportError:  # in-repo import path.
    from backend.agents.rlm import dead_training_guard

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

__all__ = ["CELL_MANIFEST_NAME", "CellResult", "discover_visible_gpus", "run_matrix"]


# ---------------------------------------------------------------------------
# Backward-compat aliases for any code that imported private names directly.
# (Tests import _headline_metric, _load_cell_manifest, _should_skip_cell,
# _write_cell_manifest only from test_gpu_cell_runner_resume.py — those tests
# reference public-facing run_matrix, not the private helpers, so no alias
# needed.  These are kept as thin aliases only for internal use in this module.)
# ---------------------------------------------------------------------------

# (No external importers of these private names were found in the test suite;
# the helpers are simply replaced by the cell_scheduler imports above.)

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


# (CELL_MANIFEST_NAME, CellResult, load_cell_manifest, should_skip_cell,
# write_cell_manifest, and the headline_metric helper are now provided by
# cell_scheduler — imported at the top of this module.)


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

    ``OPENRESEARCH_CELL_BATCH_SCALE=<batch_scale>``
        Set on OOM retries so the cell script can reduce its batch size.

    ``OPENRESEARCH_CELL_GRAD_CHECKPOINT=1``
        Set on OOM retries to enable gradient checkpointing in the cell script.

    ``OPENRESEARCH_CELL_OUTPUT_DIR=<output_dir>``
        Canonical output directory for ``metrics.json`` and artifacts.

    ``OPENRESEARCH_CELL_PARAMS=<json>``
        Full cell dict serialised as JSON, so the cell script can read its
        own parameters without parsing argv.
    """
    child_env = {**os.environ}
    # Put the running interpreter's bin/ on PATH so console scripts a cell may shell
    # out to (e.g. ``alfworld-download``) resolve. A venv invoked by path (not
    # "activated") does NOT place its bin/ on PATH, so a bare-name ``subprocess``
    # call FileNotFounds — exactly how the 2026-06-01 SDAR ALFWorld cells died
    # (``alfworld-download failed: [Errno 2] No such file or directory``).
    _interp_bin = os.path.dirname(os.path.abspath(sys.executable))
    if _interp_bin:
        child_env["PATH"] = _interp_bin + os.pathsep + child_env.get("PATH", "")
    child_env["CUDA_VISIBLE_DEVICES"] = gpu_id
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    child_env["OPENRESEARCH_CELL_OUTPUT_DIR"] = str(output_dir)
    child_env["OPENRESEARCH_CELL_PARAMS"] = json.dumps(cell)

    if batch_scale is not None:
        child_env["OPENRESEARCH_CELL_BATCH_SCALE"] = str(batch_scale)
        child_env["OPENRESEARCH_CELL_GRAD_CHECKPOINT"] = "1" if grad_checkpoint else "0"
    else:
        # Clear any inherited values from a parent run.
        child_env.pop("OPENRESEARCH_CELL_BATCH_SCALE", None)
        child_env.pop("OPENRESEARCH_CELL_GRAD_CHECKPOINT", None)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(cell_script)] + [
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

            # Dead-training early-stop (2026-06-09, flag-gated default OFF): watch the
            # streamed loss for the provably-stuck signature and kill the cell early as
            # a ``training_diverged`` signal rather than burning the full epoch budget on
            # a network that never learns. When disabled the detector is never built and
            # the reader is byte-for-byte the original.
            _detector = (
                dead_training_guard.DeadTrainingDetector()
                if dead_training_guard.is_enabled()
                else None
            )

            def _reader() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    captured_chunks.append(line)
                    log_fh.write(line)
                    log_fh.flush()
                    if _detector is not None:
                        diag = _detector.observe(line)
                        if diag:
                            marker = (
                                f"\n{dead_training_guard.MARKER} "
                                f"cell={cell.get('id', '')}: {diag}\n"
                            )
                            captured_chunks.append(marker)
                            log_fh.write(marker)
                            log_fh.flush()
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except (ProcessLookupError, OSError):
                                pass
                            return  # stop reading; proc.wait() in the parent returns

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
    overall_timeout_s: float | None = None,
    gpus_per_cell: int = 1,
    fingerprints: dict[str, str] | None = None,
    force_cells: set[str] | None = None,
    now_iso: str | None = None,
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
                             ``OPENRESEARCH_CELL_PARAMS`` / ``OPENRESEARCH_CELL_OUTPUT_DIR``
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
        overall_timeout_s:   Wall-clock budget for the WHOLE matrix in seconds.
                             ``None`` means no overall limit. When set, workers
                             stop launching new cells once the deadline passes
                             (recording them ``"timeout"``) and each in-flight
                             cell's effective timeout is clamped to the time
                             remaining, so the matrix can never run materially
                             past the budget — the 2026-06-01 fix for an
                             unbounded run_experiment matrix hanging for hours.
        fingerprints:        Optional ``{cell_id: fingerprint}`` (computed by the
                             caller, e.g. ``cell_fingerprint.compute_fingerprint``).
                             Two uses: (1) every terminal cell's authoritative
                             ``cell_manifest.json`` records its fingerprint, and
                             (2) when resume is armed (see below) a cell is
                             SKIPPED only if its prior manifest's fingerprint
                             still matches.  ``None`` → no fingerprints recorded,
                             no fingerprint-gated skips.
        force_cells:         Cell ids that must ALWAYS re-run even when resume is
                             armed and their manifest still matches — the wiring
                             for ``--rerun-env`` / ``--rerun-cell``.  ``None`` →
                             empty (force nothing).
        now_iso:             Optional ISO-8601 timestamp STRING stamped into each
                             cell's ``cell_manifest.json`` as ``completed_at``.
                             ``None`` → the field is omitted (this module is
                             stdlib-pure and intentionally does not call
                             ``datetime.now`` itself; the caller supplies the
                             clock).

    Resume (cell-level checkpoint, Track B):

    When the env var ``OPENRESEARCH_RESUME_CELLS`` is truthy, ``run_matrix`` reads
    each cell's ``output_dir/cell_manifest.json`` BEFORE launching it.  If the
    manifest's ``status == "ok"`` AND its stored ``fingerprint`` equals
    ``fingerprints[cell_id]`` AND the cell is not in ``force_cells``, the cell is
    recorded ``status="skipped"`` (carrying its prior ``metrics.json``) WITHOUT
    launching a subprocess — a $0 no-op that reuses the prior result.  With the
    env var unset (the default) every cell always runs, exactly as before.

    Returns:
        Dict mapping ``cell["id"]`` → :meth:`CellResult.to_dict`.  Every input
        cell is represented regardless of outcome — never raises on individual
        cell failure.

    Design invariants:

    * Each subprocess starts with ``CUDA_VISIBLE_DEVICES=<one id>`` so it
      physically cannot access a second card.
    * Concurrency never exceeds ``min(max_parallel, len(gpus))``.
    * GPU ids in ``gpus`` are the only ids ever assigned.
    * OOM retries pass ``OPENRESEARCH_CELL_BATCH_SCALE`` and
      ``OPENRESEARCH_CELL_GRAD_CHECKPOINT=1`` as env vars; the cell script is
      responsible for honouring them.
    """
    if not cells:
        return {}

    # Resume (Track B): armed by OPENRESEARCH_RESUME_CELLS. When armed, a cell whose
    # prior manifest is status=ok + fingerprint-matched + not force-listed is
    # skipped without launching a subprocess. Unset → every cell always runs.
    _resume_armed = is_resume_armed()
    _fingerprints: dict[str, str] = fingerprints or {}
    _force_cells: set[str] = force_cells or set()

    # Overall-matrix deadline (monotonic), or None for no overall bound.
    overall_deadline: float | None = deadline_from_timeout(overall_timeout_s)

    resolved_gpus: list[str] = gpus if gpus is not None else discover_visible_gpus()
    if not resolved_gpus:
        resolved_gpus = ["0"]

    # Multi-GPU cells (2026-06-02): group GPUs into SLOTS of `gpus_per_cell` so a
    # cell that shards a large model (device_map='auto') sees several GPUs at once.
    # Each slot is a CSV of physical ids → CUDA_VISIBLE_DEVICES for that cell.
    n_per = max(1, int(gpus_per_cell))
    gpu_slots = [",".join(resolved_gpus[i:i + n_per])
                 for i in range(0, len(resolved_gpus) - n_per + 1, n_per)]
    if not gpu_slots:                       # fewer GPUs than n_per → one slot of all
        gpu_slots = [",".join(resolved_gpus)]

    parallelism = max_parallel if max_parallel is not None else len(gpu_slots)
    parallelism = max(1, min(parallelism, len(gpu_slots)))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cell_script = Path(cell_script)

    results: dict[str, CellResult] = {}
    results_lock = threading.Lock()

    # Queue of (cell, retry_index) — retry_index 0 = first attempt.
    work_queue: Queue[tuple[dict[str, Any], int]] = Queue()
    for cell in cells:
        work_queue.put((cell, 0))

    # GPU pool — a queue of free GPU SLOTS (each a CSV of `gpus_per_cell` ids,
    # set as CUDA_VISIBLE_DEVICES so a cell can device_map-shard across them).
    gpu_pool: Queue[str] = Queue()
    for slot in gpu_slots[:parallelism]:
        gpu_pool.put(slot)

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

            # Resume skip pre-filter (Track B): on the FIRST attempt of a cell,
            # when resume is armed, a prior ok+fingerprint-matched+not-forced cell
            # is reused WITHOUT launching a subprocess. retry_idx>0 means an OOM
            # retry already in flight — never skip those.
            if _resume_armed and retry_idx == 0 and should_skip_cell(
                cell_id, output_dir, _fingerprints, _force_cells
            ):
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="skipped",
                        metrics=_load_metrics(output_dir),
                        gpu=gpu_id,
                        retries=0,
                        error=None,
                    )
                logger.info(
                    "gpu_cell_runner: cell=%s SKIPPED (resume: prior ok + fingerprint match)",
                    cell_id,
                )
                gpu_pool.put(gpu_id)
                continue

            # Overall-matrix deadline (2026-06-01): once the matrix budget is
            # spent, do NOT launch further cells — record them as ``timeout`` and
            # keep draining the queue so every cell still gets a result entry.
            if overall_deadline is not None and time.monotonic() >= overall_deadline:
                _tmo_metrics = _load_metrics(output_dir)
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="timeout",
                        metrics=_tmo_metrics,
                        gpu=gpu_id,
                        retries=retry_idx,
                        error="overall matrix timeout — cell not launched",
                    )
                write_cell_manifest(
                    output_dir, caller="gpu_cell_runner", cell_id=cell_id, status="timeout",
                    fingerprint=_fingerprints.get(cell_id), metrics=_tmo_metrics,
                    retries=retry_idx, now_iso=now_iso,
                )
                gpu_pool.put(gpu_id)
                continue

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

            # Clamp this cell's timeout to the time left in the overall budget so a
            # single in-flight cell can't overrun the matrix deadline either.
            eff_timeout = clamp_cell_timeout(per_cell_timeout_s, overall_deadline)

            returncode, output = _run_cell_subprocess(
                cell=cell,
                cell_script=cell_script,
                gpu_id=gpu_id,
                output_dir=output_dir,
                batch_scale=batch_scale,
                grad_checkpoint=grad_checkpoint,
                timeout_s=eff_timeout,
                log_path=log_path,
            )

            # Return GPU to pool immediately after subprocess exits.
            gpu_pool.put(gpu_id)
            deadline_hit = (
                overall_deadline is not None and time.monotonic() >= overall_deadline
            )

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
                write_cell_manifest(
                    output_dir, caller="gpu_cell_runner", cell_id=cell_id, status="ok",
                    fingerprint=_fingerprints.get(cell_id), metrics=metrics,
                    retries=retry_idx, now_iso=now_iso,
                )
                logger.info("gpu_cell_runner: cell=%s DONE ok (gpu=%s)", cell_id, gpu_id)
            elif _is_oom(output) and retry_idx < max_oom_retries and not deadline_hit:
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
            elif dead_training_guard.is_dead_training(output):
                # Early-stopped as provably-stuck training. Deterministic (an
                # architecture/init bug, not a transient OOM) → do NOT retry; mark
                # ``training_diverged`` so the matrix surfaces a targeted, repairable
                # signal that drives the agent to fix its own trainer.
                _div_metrics = _load_metrics(output_dir)
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="training_diverged",
                        metrics=_div_metrics,
                        gpu=gpu_id,
                        retries=retry_idx,
                        error=(output[-2000:] if output else "training diverged"),
                    )
                write_cell_manifest(
                    output_dir, caller="gpu_cell_runner", cell_id=cell_id,
                    status="training_diverged",
                    fingerprint=_fingerprints.get(cell_id), metrics=_div_metrics,
                    retries=retry_idx, now_iso=now_iso,
                )
                logger.warning(
                    "gpu_cell_runner: cell=%s TRAINING_DIVERGED (early-stop) gpu=%s",
                    cell_id, gpu_id,
                )
            else:
                if deadline_hit:
                    status = "timeout"
                elif _is_oom(output):
                    status = "oom_failed"
                else:
                    status = "error"
                error_snippet = output[-2000:] if output else f"exit code {returncode}"
                _fail_metrics = _load_metrics(output_dir)
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status=status,
                        metrics=_fail_metrics,
                        gpu=gpu_id,
                        retries=retry_idx,
                        error=error_snippet,
                    )
                write_cell_manifest(
                    output_dir, caller="gpu_cell_runner", cell_id=cell_id, status=status,
                    fingerprint=_fingerprints.get(cell_id), metrics=_fail_metrics,
                    retries=retry_idx, now_iso=now_iso,
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
