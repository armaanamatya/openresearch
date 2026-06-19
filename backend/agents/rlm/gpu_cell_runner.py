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
import re
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

# C3: extra OOM stderr signatures, matched ONLY under OPENRESEARCH_OOM_ENFORCE. The
# base set above stays the default (byte-for-byte); these catch non-standard CUDA
# OOM messages that today misclassify a terminal OOM as a non-OOM failure.
_OOM_SIGNATURES_EXTRA: tuple[str, ...] = (
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
    "cudaErrorMemoryAllocation",
    "torch.cuda.OutOfMemoryError",
    "failed to allocate",
    "tried to allocate",          # torch OOM detail line
)


def _oom_enforce_enabled() -> bool:
    """C3: OPENRESEARCH_OOM_ENFORCE — on an OOM retry, ENFORCE a per-GPU memory cap a
    non-cooperating trainer cannot ignore (set_per_process_memory_fraction shim)
    + broaden OOM classification. Default OFF → only OPENRESEARCH_CELL_BATCH_SCALE is
    set (advisory, byte-for-byte today)."""
    return os.environ.get("OPENRESEARCH_OOM_ENFORCE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# C3 memory-cap shim. Injected as a sitecustomize on a dedicated PYTHONPATH dir on
# OOM retries so it runs at interpreter startup BEFORE the trainer allocates —
# enforcing the cap without trainer cooperation. Stdlib-only (this file is copied
# flat into the agent sandbox). Best-effort: any failure is swallowed so the cell
# still runs. (A pre-existing site-packages sitecustomize is shadowed for this one
# subprocess only — acceptable for a flag-gated OOM-retry path.)
_OOM_MEMCAP_SHIM = (
    "# Auto-injected by gpu_cell_runner on an OOM retry (C3, OPENRESEARCH_OOM_ENFORCE).\n"
    "# Caps this process's per-GPU memory fraction so a non-cooperating trainer\n"
    "# cannot re-OOM identically. Runs at interpreter startup.\n"
    "import os as _os\n"
    "_frac = _os.environ.get('OPENRESEARCH_CELL_MEM_FRACTION')\n"
    "if _frac:\n"
    "    try:\n"
    "        import torch as _torch\n"
    "        if _torch.cuda.is_available():\n"
    "            _f = max(0.05, min(1.0, float(_frac)))\n"
    "            for _d in range(_torch.cuda.device_count()):\n"
    "                _torch.cuda.set_per_process_memory_fraction(_f, _d)\n"
    "    except Exception:\n"
    "        pass\n"
)


def _inject_oom_memcap(child_env: dict, shim_root: "Path | str", fraction: float) -> None:
    """Write the memcap sitecustomize into ``shim_root/_oom_shim`` and wire
    ``child_env`` (OPENRESEARCH_CELL_MEM_FRACTION + PYTHONPATH prepend). Fail-soft: on
    any error the cell still runs (advisory batch-scale already set)."""
    try:
        shim_dir = Path(shim_root) / "_oom_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        (shim_dir / "sitecustomize.py").write_text(_OOM_MEMCAP_SHIM, encoding="utf-8")
        child_env["OPENRESEARCH_CELL_MEM_FRACTION"] = str(max(0.05, min(1.0, float(fraction))))
        child_env["PYTHONPATH"] = str(shim_dir) + os.pathsep + child_env.get("PYTHONPATH", "")
    except Exception:  # noqa: BLE001 — enforcement is best-effort; never block the cell
        pass


# C2 orphan-guard hooks (soft): register each spawned cell's process group with the
# harness so binding's per-primitive timeout can SIGKILL it on abandonment. Lazy +
# fail-soft so this file stays zero-non-stdlib-dep when copied FLAT into the agent
# sandbox (where there is no backend package — and no binding handler either, so
# skipping registration there is correct). No-op kill unless OPENRESEARCH_ORPHAN_GUARD.
def _orphan_register(pid: int) -> None:
    try:
        from backend.agents.rlm import orphan_guard as _og
        _og.register(pid)
    except Exception:  # noqa: BLE001 — sandbox-flat (no backend pkg) or any error: skip
        pass


def _orphan_deregister(pid: int) -> None:
    try:
        from backend.agents.rlm import orphan_guard as _og
        _og.deregister(pid)
    except Exception:  # noqa: BLE001
        pass

__all__ = ["CELL_MANIFEST_NAME", "CellResult", "discover_visible_gpus", "run_matrix",
           "_cell_gpu_count", "vram_evidence_verdict", "metrics_claim_gpu_training"]


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

def _cell_gpu_count(cell: Any, default_per_cell: int, total_gpus: int) -> int:
    """How many GPUs a cell wants: cell['gpus'] or the uniform default, clamped to [1, total_gpus].

    When ``cell`` is not a dict, or ``cell['gpus']`` is absent/non-numeric, falls
    back to ``default_per_cell``.  The result is clamped so a cell declaring more
    GPUs than exist still runs (degraded to all available cards) and never hangs.
    """
    raw = cell.get("gpus", default_per_cell) if isinstance(cell, dict) else default_per_cell
    try:
        k = int(raw)
    except (TypeError, ValueError):
        k = default_per_cell
    return max(1, min(k, max(1, total_gpus)))


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
    """Return True iff ``output`` (stderr + stdout combined) contains an OOM signature.

    C3: under OPENRESEARCH_OOM_ENFORCE the broadened ``_OOM_SIGNATURES_EXTRA`` set is
    also matched (catches non-standard CUDA OOM messages). Default → base set only
    (byte-for-byte today).
    """
    if any(sig in output for sig in _OOM_SIGNATURES):
        return True
    if _oom_enforce_enabled() and any(sig in output for sig in _OOM_SIGNATURES_EXTRA):
        return True
    return False


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

def _antifab_guard_enabled() -> bool:
    """Part-2 gate: OPENRESEARCH_ANTIFAB_GUARD (default on). Mirrors Part-1 gate in preflight_ast."""
    return os.environ.get("OPENRESEARCH_ANTIFAB_GUARD", "1").strip() not in ("0", "false", "off", "no")


def _antifab_min_vram_gb() -> float:
    """Return the minimum VRAM threshold (GiB) below which a successful cell is suspected fabrication."""
    try:
        return float(os.environ.get("OPENRESEARCH_ANTIFAB_MIN_VRAM_GB", "1.5"))
    except (ValueError, TypeError):
        return 1.5


def _sample_vram_mib(gpu_id: str) -> float | None:
    """Return a single nvidia-smi memory.used reading in MiB, or None on any failure."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i", gpu_id,
    ]
    try:
        out = subprocess.check_output(cmd, timeout=5, stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            stripped = line.strip()
            if stripped:
                return float(stripped)
    except Exception:  # noqa: BLE001
        pass
    return None


def vram_evidence_verdict(
    peak_vram_gb: float | None, *, claims_gpu_training: bool
) -> bool:
    """Shared VRAM-evidence decision — True iff fabrication is suspected.

    The single decision function for BOTH the cells route (``run_matrix``) and the
    monolithic ``run_experiment`` path, so the two never drift. Returns True ONLY
    when ALL of these hold:

      * the antifab guard is enabled (``OPENRESEARCH_ANTIFAB_GUARD``, default on),
      * ``peak_vram_gb is not None`` — a measurement was actually taken
        (None ⇒ nvidia-smi absent / sampling failed ⇒ fail-soft, never flag),
      * ``claims_gpu_training`` — the run's own evidence claims a GPU/training
        result (a CPU-only run with no such claim is never flagged), AND
      * the measured net-peak is below the floor (``OPENRESEARCH_ANTIFAB_MIN_VRAM_GB``,
        default 1.5 GiB) — even a tiny real GPU run (Qwen-1.7B ≈ 3.4 GB) clears it.

    Fail-soft + conservative by construction: any "I don't know" input
    (peak=None, or no GPU-training claim, or guard off) returns False.
    """
    if not _antifab_guard_enabled():
        return False
    if peak_vram_gb is None:
        return False
    if not claims_gpu_training:
        return False
    return peak_vram_gb < _antifab_min_vram_gb()


# device values that claim GPU execution.
_GPU_DEVICE_VALUES: frozenset[str] = frozenset({"cuda", "gpu"})
# A HF-style model id as a metric VALUE ("Qwen/Qwen2.5-3B-Instruct") claims a real
# model load. Strict org/model shape (>=2 chars each side) so a bare "/" path, a
# date "2026/06/18", or a fraction "3/4" never matches.
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9._-]{2,}/[A-Za-z0-9._-]{2,}")


def metrics_claim_gpu_training(metrics: object) -> bool:
    """Return True if ``metrics`` carries a GPU/training signal — i.e. the run's
    OWN evidence claims it trained/evaluated a model on a GPU.

    Two precise, GPU-SPECIFIC signals (either is enough), read from the run's own
    emitted metrics so a genuinely CPU-only paper is NEVER matched. Generic
    training-metric keys (accuracy/loss/reward/…) are deliberately NOT used — they
    appear in CPU runs too and would false-positive a CPU-only paper that happens to
    run on a GPU host:

      * a ``device`` field equal to ``"cuda"`` / ``"gpu"`` (case-insensitive), OR
      * a value under a model-ish key (``model`` / ``model_id`` / ``model_name`` …)
        matching a strict HF model-id shape (``org/model``, >=2 chars each side —
        e.g. ``"Qwen/Qwen2.5-3B-Instruct"``). Anchoring to a model key means an
        output path (``output_dir="outputs/run_123"``), a date, or a fraction is
        never misread as a model claim.

    Fail-soft: a non-dict or any traversal error returns False (never flag).
    """
    try:
        return _scan_metrics_for_gpu_claim(metrics, _depth=0)
    except Exception:  # noqa: BLE001 — predicate must never raise into the run
        return False


def _scan_metrics_for_gpu_claim(obj: object, *, _depth: int) -> bool:
    if _depth > 6:  # cheap recursion guard for pathological nesting
        return False
    if isinstance(obj, dict):
        for key, val in obj.items():
            ks = str(key).strip().lower()
            # device == cuda/gpu — an explicit, unambiguous GPU claim.
            if ks == "device" and isinstance(val, str) and val.strip().lower() in _GPU_DEVICE_VALUES:
                return True
            # A model-id (org/model) ONLY when it sits under a model-ish key, so an
            # output path like ``output_dir="outputs/run_123"`` is never misread as a
            # model claim.
            if "model" in ks and isinstance(val, str) and _MODEL_ID_RE.search(val):
                return True
            if _scan_metrics_for_gpu_claim(val, _depth=_depth + 1):
                return True
        return False
    if isinstance(obj, (list, tuple)):
        return any(_scan_metrics_for_gpu_claim(v, _depth=_depth + 1) for v in obj)
    return False


def _poll_peak_vram_daemon(
    gpu_id: str, interval_s: float, stop_flag: threading.Event, readings: list
) -> None:
    """Daemon thread body: poll nvidia-smi every ``interval_s`` seconds while ``stop_flag`` is clear.

    Appends observed MiB values into ``readings`` (a mutable list passed in).
    The physical ``gpu_id`` here is a single id string (e.g. "0", "2").

    Fail-soft: any nvidia-smi error → just skip that reading. If nvidia-smi is absent,
    ``readings`` stays empty and the caller treats ``peak_vram_gb`` as None (no flagging).
    """
    while not stop_flag.is_set():
        val = _sample_vram_mib(gpu_id)
        if val is not None:
            readings.append(val)
        stop_flag.wait(timeout=interval_s)


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

    ``OPENRESEARCH_CELL_CHECKPOINT_DIR=<output_dir>/checkpoints``
        Stable per-cell directory on the persistent output disk.  The
        trainer creates and writes it; the harness only names it.  Keyed
        by ``output_dir`` (which is keyed by ``cell_id``), so the path
        is identical across OOM retries and VM stop/restart — enabling
        spot-preemption resume.

    ``OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S=<seconds>``
        Advisory checkpoint interval in seconds.  Defaults to 600 (10 min).
        Override with the same env-var in the parent environment.
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
    # T2 intra-cell checkpoint (2026-06-18): a STABLE per-cell dir on the
    # persistent output disk so a spot preemption mid-cell resumes from the
    # latest checkpoint instead of restarting the cell. Stable across retries and
    # VM stop/restart (keyed by cell_id, which keys output_dir). The trainer
    # creates/writes it; the harness only names it.
    child_env["OPENRESEARCH_CELL_CHECKPOINT_DIR"] = str(output_dir / "checkpoints")
    child_env["OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S"] = os.environ.get(
        "OPENRESEARCH_CELL_CHECKPOINT_INTERVAL_S", "600"
    )

    if batch_scale is not None:
        child_env["OPENRESEARCH_CELL_BATCH_SCALE"] = str(batch_scale)
        child_env["OPENRESEARCH_CELL_GRAD_CHECKPOINT"] = "1" if grad_checkpoint else "0"
        # C3: advisory batch-scale alone lets a non-cooperating trainer OOM
        # identically. When OPENRESEARCH_OOM_ENFORCE is on, ALSO enforce a hard per-GPU
        # memory cap (sitecustomize shim, no trainer cooperation) tracking the
        # batch-scale ladder (0.5 → 0.25). Off → byte-for-byte today.
        if _oom_enforce_enabled():
            _inject_oom_memcap(child_env, output_dir, batch_scale)
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
    _cell_proc_pid: int | None = None  # C2: process group, deregistered in finally

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
            # C2: register this cell's process group so binding's per-primitive
            # timeout handler can SIGKILL it if the OUTER timeout abandons us before
            # this cell's own timeout fires. Soft + flag-gated (no-op when off).
            _cell_proc_pid = proc.pid
            _orphan_register(proc.pid)

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
    finally:
        # C2: cell reaped (normal exit, timeout-kill, or launch failure) → drop it
        # from the orphan registry so a later primitive timeout can't target a
        # since-recycled PID. Soft no-op when the guard never registered it.
        if _cell_proc_pid is not None:
            _orphan_deregister(_cell_proc_pid)


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

    Each cell is an independent training job.  By default each cell runs on
    **exactly one** GPU; a cell may declare ``"gpus": k`` to claim k cards for
    model-parallel sharding (e.g. ``device_map="auto"`` across 2 A100s for a 7B
    model).  When no cell declares ``"gpus"``, scheduling is byte-equivalent to
    the prior uniform-slot design.

    The scheduler keeps up to ``max_parallel`` (default: ``len(gpus)``) workers
    active.  Each worker waits for enough free GPUs (atomically k-acquires), runs
    one cell attempt, then releases the GPUs immediately so the next waiter can
    proceed.  This is deadlock-free: a worker holds zero GPUs while waiting.

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

    # Heterogeneous per-cell GPU counts (2026-06-18): each cell may carry an optional
    # integer "gpus" field declaring how many cards it needs (e.g. a 7B model sharded
    # over 2 cards).  When no cell uses "gpus", `gpus_per_cell` is the uniform default
    # and scheduling is byte-equivalent to the prior slot-pool design.
    #
    # Implementation: individual-GPU free-pool + Condition-guarded atomic k-acquire.
    # A worker holds ZERO GPUs while waiting, then takes all k at once (no hold-and-wait
    # → deadlock-free).  GPUs are always released in a `finally` block (no leak on crash).
    # A cell that asks for more GPUs than exist is clamped to total_gpus and still runs.
    total_gpus = len(resolved_gpus)
    n_per = max(1, int(gpus_per_cell))  # default per-cell GPU count

    free_gpus: list[str] = list(resolved_gpus)
    gpu_cond = threading.Condition()

    def _acquire_gpus(k: int) -> list[str]:
        """Block until k GPUs are free, then take them atomically."""
        with gpu_cond:
            while len(free_gpus) < k:
                gpu_cond.wait()
            return [free_gpus.pop() for _ in range(k)]  # take any k (cards are identical)

    def _release_gpus(assigned: list[str]) -> None:
        """Return GPUs to the pool and wake waiting workers."""
        with gpu_cond:
            free_gpus.extend(assigned)
            gpu_cond.notify_all()

    # Worker count: cap at total_gpus (can't run more cells than we have GPUs).
    worker_count = max(1, min(max_parallel if max_parallel is not None else total_gpus, total_gpus))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cell_script = Path(cell_script)

    results: dict[str, CellResult] = {}
    results_lock = threading.Lock()
    # Extras keyed by cell_id: {peak_vram_gb, fabrication_suspected} — merged into
    # to_dict() output at the end.  Written only by _worker; read only in the final
    # comprehension after all workers finish (no lock needed there).
    _cell_extras: dict[str, dict[str, Any]] = {}
    _extras_lock = threading.Lock()

    # Queue of (cell, retry_index) — retry_index 0 = first attempt.
    work_queue: Queue[tuple[dict[str, Any], int]] = Queue()
    for cell in cells:
        work_queue.put((cell, 0))

    active_threads: list[threading.Thread] = []
    active_lock = threading.Lock()

    def _worker() -> None:
        """One worker thread: cell-first loop — grab work, acquire GPUs, run, release."""
        while True:
            # Get the next work item first — hold ZERO GPUs while waiting for work.
            try:
                cell, retry_idx = work_queue.get_nowait()
            except Empty:
                break  # no work; exit without holding any GPU

            cell_id: str = cell.get("id", f"cell_{id(cell)}")
            output_dir = output_root / cell_id
            log_path = output_root / f"{cell_id}.log"

            # Resume skip pre-filter (Track B): on the FIRST attempt, when resume is
            # armed, a prior ok+fingerprint-matched+not-forced cell is reused WITHOUT
            # launching a subprocess — BEFORE acquiring any GPUs.
            if _resume_armed and retry_idx == 0 and should_skip_cell(
                cell_id, output_dir, _fingerprints, _force_cells
            ):
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="skipped",
                        metrics=_load_metrics(output_dir),
                        gpu="skipped",
                        retries=0,
                        error=None,
                    )
                logger.info(
                    "gpu_cell_runner: cell=%s SKIPPED (resume: prior ok + fingerprint match)",
                    cell_id,
                )
                continue

            # Overall-matrix deadline — checked BEFORE acquiring GPUs so we don't
            # starve the pool for a cell we'd immediately time-out anyway.
            if overall_deadline is not None and time.monotonic() >= overall_deadline:
                _tmo_metrics = _load_metrics(output_dir)
                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status="timeout",
                        metrics=_tmo_metrics,
                        gpu="unassigned",
                        retries=retry_idx,
                        error="overall matrix timeout — cell not launched",
                    )
                write_cell_manifest(
                    output_dir, caller="gpu_cell_runner", cell_id=cell_id, status="timeout",
                    fingerprint=_fingerprints.get(cell_id), metrics=_tmo_metrics,
                    retries=retry_idx, now_iso=now_iso,
                )
                continue

            # Determine how many GPUs this cell needs (per-cell "gpus" field, or default).
            k = _cell_gpu_count(cell, n_per, total_gpus)
            assigned = _acquire_gpus(k)  # blocks until k cards are free
            slot = ",".join(assigned)    # CUDA_VISIBLE_DEVICES value for this cell

            try:
                # Determine OOM-mitigation parameters for this attempt.
                batch_scale: float | None = None
                grad_checkpoint = False
                if retry_idx > 0:
                    scale_idx = retry_idx - 1  # 0 → 0.5, 1 → 0.25
                    if scale_idx < len(_OOM_BATCH_SCALES):
                        batch_scale = _OOM_BATCH_SCALES[scale_idx]
                    grad_checkpoint = True

                logger.info(
                    "gpu_cell_runner: starting cell=%s gpus=%s retry=%d batch_scale=%s",
                    cell_id, slot, retry_idx, batch_scale,
                )

                # Clamp this cell's timeout to the time left in the overall budget.
                eff_timeout = clamp_cell_timeout(per_cell_timeout_s, overall_deadline)

                # Part-2 VRAM poller: measure the cell's PEAK net GPU memory use as
                # (peak_during - baseline_before) so background GPU processes don't
                # falsely trigger the threshold.  Active only when the antifab guard
                # is on; fail-soft (any measurement failure → peak_vram_gb=None → no flag).
                _vram_stop_flag = threading.Event()
                _vram_mib: list[float] = []
                _vram_poll_thread: threading.Thread | None = None
                _vram_baseline_mib: float | None = None
                if _antifab_guard_enabled():
                    # When slot = "0,1" (multi-GPU), poll only the first card — a stub
                    # (Linear(1,1)) uses near-zero VRAM on every card, so one is enough.
                    _phys_gpu = slot.split(",")[0].strip()
                    # Sample baseline BEFORE launching (fail-soft, one-shot).
                    _vram_baseline_mib = _sample_vram_mib(_phys_gpu)
                    _vram_poll_thread = threading.Thread(
                        target=_poll_peak_vram_daemon,
                        args=(_phys_gpu, 2.0, _vram_stop_flag, _vram_mib),
                        daemon=True,
                    )
                    _vram_poll_thread.start()

                try:
                    returncode, output = _run_cell_subprocess(
                        cell=cell,
                        cell_script=cell_script,
                        gpu_id=slot,
                        output_dir=output_dir,
                        batch_scale=batch_scale,
                        grad_checkpoint=grad_checkpoint,
                        timeout_s=eff_timeout,
                        log_path=log_path,
                    )
                finally:
                    _vram_stop_flag.set()
                    if _vram_poll_thread is not None:
                        _vram_poll_thread.join(timeout=5.0)

                # Net cell VRAM = peak_during - baseline_before.
                # Requirements for reliable measurement:
                #   * baseline_before must be known (nvidia-smi present)
                #   * at least 2 during-run samples (so the subprocess was long enough
                #     to poll twice — prevents a zero-delta false positive for short
                #     mock/synthetic runs where the only sample equals the baseline)
                # If any requirement is unmet, peak_vram_gb=None → no flagging (fail-soft).
                if len(_vram_mib) >= 2 and _vram_baseline_mib is not None:
                    _peak_mib = max(_vram_mib)
                    peak_vram_gb: float | None = max(0.0, _peak_mib - _vram_baseline_mib) / 1024.0
                else:
                    peak_vram_gb = None
            finally:
                # ALWAYS release GPUs — even if _run_cell_subprocess raises.
                _release_gpus(assigned)

            deadline_hit = (
                overall_deadline is not None and time.monotonic() >= overall_deadline
            )

            if returncode == 0:
                metrics = _load_metrics(output_dir)

                # Part-2 VRAM fabrication check: a successful cell that used
                # near-zero GPU memory cannot have trained a real LLM — flag it.
                # A returncode-0 cell IS claiming it ran its training (the cells route
                # only ever trains real models), so claims_gpu_training=True here.
                # Routed through the SHARED vram_evidence_verdict so the cells route and
                # the monolithic run_experiment path can never drift apart.
                # Gate + fail-soft live inside the helper.
                _min_vram = _antifab_min_vram_gb()
                _fab_suspected = vram_evidence_verdict(
                    peak_vram_gb, claims_gpu_training=True
                )
                if _fab_suspected:
                    logger.warning(
                        "gpu_cell_runner: cell=%s fabrication_suspected: "
                        "peak_vram_gb=%.3f < threshold=%.3f — degrading to 'failed'",
                        cell_id, peak_vram_gb, _min_vram,
                    )

                _cell_status = "failed" if _fab_suspected else "ok"
                _cell_error = (
                    f"fabrication_suspected: peak VRAM {peak_vram_gb:.3f} GiB < "
                    f"threshold {_min_vram:.3f} GiB — stub model detected, not a "
                    f"real LLM training run"
                    if _fab_suspected else None
                )

                with results_lock:
                    results[cell_id] = CellResult(
                        cell_id=cell_id,
                        status=_cell_status,
                        metrics=metrics,
                        gpu=slot,
                        retries=retry_idx,
                        error=_cell_error,
                    )
                with _extras_lock:
                    _cell_extras[cell_id] = {
                        "peak_vram_gb": peak_vram_gb,
                        "fabrication_suspected": _fab_suspected,
                    }
                _manifest_status = _cell_status
                write_cell_manifest(
                    output_dir, caller="gpu_cell_runner", cell_id=cell_id, status=_manifest_status,
                    fingerprint=_fingerprints.get(cell_id), metrics=metrics,
                    retries=retry_idx, now_iso=now_iso,
                )
                if _fab_suspected:
                    logger.warning(
                        "gpu_cell_runner: cell=%s FABRICATION SUSPECTED — "
                        "status degraded to 'failed' (peak_vram_gb=%.3f)",
                        cell_id, peak_vram_gb,
                    )
                else:
                    logger.info("gpu_cell_runner: cell=%s DONE ok (gpu=%s)", cell_id, slot)
            elif _is_oom(output) and retry_idx < max_oom_retries and not deadline_hit:
                next_retry = retry_idx + 1
                logger.warning(
                    "gpu_cell_runner: cell=%s OOM on gpu=%s, scheduling retry %d/%d",
                    cell_id, slot, next_retry, max_oom_retries,
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
                        gpu=slot,
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
                    cell_id, slot,
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
                        gpu=slot,
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
                    cell_id, status, slot, retry_idx,
                )

    def _spawn_worker() -> None:
        t = threading.Thread(target=_worker, daemon=True)
        with active_lock:
            active_threads.append(t)
        t.start()

    # Spawn initial pool of workers.
    for _ in range(worker_count):
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

    out: dict[str, dict[str, Any]] = {}
    for cid, r in results.items():
        d = r.to_dict()
        extras = _cell_extras.get(cid)
        if extras:
            d.update(extras)
        out[cid] = d
    return out
