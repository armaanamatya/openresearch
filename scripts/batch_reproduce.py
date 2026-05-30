"""batch_reproduce.py — dynamic batch scheduler for parallel paper reproductions.

Runs one or more arXiv / PDF / bundle reproductions in parallel, each pinned to
a disjoint set of local NVIDIA GPUs, using the ``local`` sandbox.  GPU leases
are managed via :class:`LocalGpuAllocator` so no two runs share a card.

Usage example::

    python scripts/batch_reproduce.py 2605.15155 2605.15155 \\
        --gpus-per-run 4 --model claude-oauth

    # Two copies of SDAR, each on 4 GPUs, auto-capped to however many fit.

    # Let each run grab ALL currently free GPUs (useful for 1-paper deep runs):
    python scripts/batch_reproduce.py 2605.15155 --gpus-per-run auto \\
        --model claude-oauth

    # Three papers at once, 2 GPUs each, extra time budget:
    python scripts/batch_reproduce.py 2501.00001 2501.00002 2501.00003 \\
        --gpus-per-run 2 --max-parallel 3 \\
        --extra "--max-wall-clock 7200"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# When invoked as `python scripts/batch_reproduce.py`, Python puts the
# script's own directory (scripts/) on sys.path[0] — NOT the repo root — so
# `import backend` fails unless the repo was pip-installed editable. Insert the
# repo root explicitly so the script is self-contained either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

# Load .env so API keys and REPROLAB_* settings reach os.environ before
# anything else runs (mirrors rlm_paperbench.py idiom).
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpx", "httpcore", "openai", "urllib3", "docker"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACQUIRE_TIMEOUT_S = 1800      # max time to wait for a GPU lease
_ACQUIRE_SLEEP_BASE_S = 5      # initial sleep between acquire retries
_ACQUIRE_SLEEP_MAX_S = 60      # cap on backoff sleep
_SIGTERM_WAIT_S = 10           # seconds to wait for graceful child exit

# ---------------------------------------------------------------------------
# Global state for signal cleanup
# ---------------------------------------------------------------------------

_shutting_down: bool = False
live_procs: dict[str, subprocess.Popen[bytes]] = {}   # lease_id -> Popen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(s: str) -> str:
    """Replace non-alphanumeric characters with underscores (safe dir/lease names)."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def _extract_score(report_path: Path) -> float | None:
    """Defensively read ``final_report.json`` and return the overall rubric score.

    The canonical location written by ``backend.agents.rlm.report.build_final_report``
    is ``report["rubric"]["overall_score"]``. The flattened projection/preserved
    shape uses a top-level ``rubric_overall_score``. We try the nested form first,
    then the flat fallbacks, so the summary works for either schema.
    """
    if not report_path.exists():
        return None
    try:
        data: Any = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        # Canonical: report["rubric"]["overall_score"]
        rubric = data.get("rubric")
        if isinstance(rubric, dict) and rubric.get("overall_score") is not None:
            return float(rubric["overall_score"])
        # Flat fallbacks (projection / preserved / legacy shapes)
        for key in ("rubric_overall_score", "overall_score", "rubric_score"):
            if data.get(key) is not None:
                return float(data[key])
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("batch_reproduce: could not parse score from %s: %s", report_path, exc)
        return None


def _resolve_interpreter(explicit: str | None, repo_root: Path) -> str:
    """Pick the Python used to launch CLI children.

    Order: an explicit ``--python`` wins; otherwise prefer the project venv at
    ``<repo>/.venv/bin/python`` (the backend needs 3.12+ and its installed
    deps); otherwise fall back to ``sys.executable``. A warning is logged when
    the chosen interpreter is older than 3.12, since ``backend.cli`` will not
    import under it.
    """
    if explicit:
        chosen = explicit
    else:
        venv_py = repo_root / ".venv" / "bin" / "python"
        chosen = str(venv_py) if venv_py.exists() else sys.executable
    # Best-effort version sanity check — never fatal.
    try:
        out = subprocess.run(
            [chosen, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        ver = (out.stdout or "").strip()
        major_minor = tuple(int(x) for x in ver.split(".")[:2]) if ver else (0, 0)
        if major_minor < (3, 12):
            logger.warning(
                "batch_reproduce: interpreter %s is Python %s (< 3.12). The backend "
                "requires 3.12+; CLI children will likely fail to import. Create a "
                "project venv (e.g. `uv venv --python 3.12 .venv`) or pass --python.",
                chosen, ver or "unknown",
            )
    except Exception:  # noqa: BLE001 — version probe is advisory only
        pass
    return chosen


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _install_signal_handlers(alloc: Any) -> None:
    """Install SIGINT/SIGTERM handlers that tear down live children and leases."""

    def _handler(signum: int, _frame: Any) -> None:
        global _shutting_down
        if _shutting_down:
            return
        _shutting_down = True
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.warning("batch_reproduce: %s received — terminating %d child(ren)", sig_name, len(live_procs))

        # SIGTERM all live children
        for lid, proc in list(live_procs.items()):
            try:
                proc.terminate()
                logger.info("batch_reproduce: sent SIGTERM to pid %d (lease %s)", proc.pid, lid)
            except OSError:
                pass

        # Wait up to _SIGTERM_WAIT_S then SIGKILL survivors
        deadline = time.monotonic() + _SIGTERM_WAIT_S
        for lid, proc in list(live_procs.items()):
            remaining = deadline - time.monotonic()
            try:
                proc.wait(timeout=max(0.1, remaining))
            except subprocess.TimeoutExpired:
                logger.warning("batch_reproduce: pid %d did not exit in time; sending SIGKILL", proc.pid)
                try:
                    proc.kill()
                except OSError:
                    pass

        # Release all known active leases
        try:
            for lease in alloc.active_leases():
                try:
                    alloc.release(lease.lease_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("batch_reproduce: could not release lease %s: %s", lease.lease_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_reproduce: active_leases() failed during shutdown: %s", exc)

        logger.info("batch_reproduce: shutdown complete")
        sys.exit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Per-run worker
# ---------------------------------------------------------------------------


def _canonical_project_id(paper: str) -> str:
    """Compute the deterministic source-derived project_id the CLI registers under.

    The arXiv/PDF/DOI ingest path in ``cmd_reproduce`` calls
    ``register_project(source)`` which keys the event-store aggregate by
    ``project_id_for(source)``. A ``--project-id`` that differs from that id is
    NOT honored for ingest — ``fetch_paper`` looks up the override id, finds no
    aggregate, and raises ``UnknownProject``. So we mirror the CLI's own
    derivation here: pass this id as ``--project-id`` (an identity override that
    ``fetch_paper`` can resolve) and use it as the run directory so per-run venv
    + caches + the final_report.json lookup all point at the real run dir.

    NOTE: this id is deterministic per source, so two CONCURRENT runs of the
    SAME paper would collide on the run dir (distinct papers are fine). Per-paper
    concurrency on the arXiv path is therefore unsupported by design.
    """
    from backend.cli import _source_from_cli
    from backend.services.paths import normalize_path_input
    from backend.services.ingestion.intake.service import project_id_for

    source = _source_from_cli(normalize_path_input(paper), "auto")
    return project_id_for(source)


def _run_one(
    *,
    paper: str,
    idx: int,
    args: argparse.Namespace,
    alloc: Any,
    runs_root: Path,
    interpreter: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Acquire GPUs, spawn one ``backend.cli reproduce`` child, wait for it.

    Returns a result dict with keys: paper, project_id, gpu_indices, exit_code,
    score, log_path.
    """
    from backend.services.runtime.local_gpu_allocator import free_devices  # local import keeps startup fast
    from backend.services.runtime.gpu_reservation import GpuReservationManager

    # Reservation-aware: discount our own GPU-reservation holders so a reserved
    # card (parked to keep other users off it) is still leasable by this run.
    _res_mgr = GpuReservationManager(runs_root / ".gpu_reservations.json", repo_root=repo_root)

    lease_id = f"{_safe(paper)}-{idx}-{int(time.time())}"
    auto_mode = args.gpus_per_run == "auto"

    # --- 6b: Acquire with bounded retry/backoff ---
    lease = None
    acquire_deadline = time.monotonic() + _ACQUIRE_TIMEOUT_S
    sleep_s = _ACQUIRE_SLEEP_BASE_S
    while time.monotonic() < acquire_deadline:
        if _shutting_down:
            raise RuntimeError("Shutting down — aborting acquire for paper %s" % paper)

        own_pids = _res_mgr.holder_pids()
        if auto_mode:
            count = max(1, len(free_devices(
                free_mem_threshold_mb=args.free_mem_threshold_mb, own_pids=own_pids)))
        else:
            count = args._gpus_per_run_int  # type: ignore[attr-defined]

        attempt = alloc.acquire(
            lease_id,
            os.getpid(),
            count=count,
            free_mem_threshold_mb=args.free_mem_threshold_mb,
            own_pids=own_pids,
        )
        if attempt is not None:
            lease = attempt
            break
        logger.info(
            "batch_reproduce: [%s] waiting for %d GPU(s) — retrying in %ds",
            paper, count, sleep_s,
        )
        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 2, _ACQUIRE_SLEEP_MAX_S)

    if lease is None:
        raise RuntimeError(
            "batch_reproduce: [%s] timed out waiting for GPU lease after %ds" % (paper, _ACQUIRE_TIMEOUT_S)
        )

    # Everything after a successful acquire is wrapped in try/finally so the
    # lease is ALWAYS released — even if venv creation, env setup, or launch
    # raises before the child process starts. reclaim_stale() only reaps DEAD
    # PIDs, and this batch process stays alive, so a lease leaked here would
    # otherwise be stranded for the whole batch run.
    try:
        # --- 6c: Compute project_id (canonical source-derived id; see helper) ---
        project_id = _canonical_project_id(paper)
        run_dir = runs_root / project_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- 6d: Create per-run venv (optional isolation) ---
        venv_dir = run_dir / ".venv"
        venv_ok = False
        if not venv_dir.exists():
            logger.info("batch_reproduce: [%s] creating venv at %s", paper, venv_dir)
            try:
                result = subprocess.run(
                    [interpreter, "-m", "venv", str(venv_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    venv_ok = True
                    logger.info("batch_reproduce: [%s] venv created OK", paper)
                else:
                    logger.warning(
                        "batch_reproduce: [%s] venv creation failed (rc=%d); continuing without isolation: %s",
                        paper, result.returncode, result.stderr[:400],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("batch_reproduce: [%s] venv creation exception: %s; continuing", paper, exc)
        else:
            venv_ok = True

        # --- 6e: Build child environment ---
        gpu_uuid_csv = ",".join(lease.gpu_uuids)
        shared_hf = (runs_root / ".cache" / "hf").resolve()
        shared_hf.mkdir(parents=True, exist_ok=True)
        # Shared pip wheel cache across ALL runs (NOT per-project): wheels are immutable
        # and reusable, so a new paper cache-hits instead of re-downloading ~5.5 GB of
        # torch/deps and recompiling flash-attn every time. The per-run venv stays
        # isolated (deps); only the download/build cache is shared → dep install becomes
        # fast + offline-resilient. PIP_CACHE_DIR overrides the per-run XDG-derived pip
        # cache, so the per-run torch/triton/mpl runtime caches below are unaffected.
        shared_pip = (runs_root / ".cache" / "pip").resolve()
        shared_pip.mkdir(parents=True, exist_ok=True)

        for cache_subdir in ("torch", "triton", "xdg", "mpl"):
            (runs_root / ".cache" / project_id / cache_subdir).mkdir(parents=True, exist_ok=True)

        per_run_env: dict[str, str] = {
            "CUDA_VISIBLE_DEVICES": gpu_uuid_csv,
            "REPROLAB_GPU_DEVICE_IDS": gpu_uuid_csv,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "HF_HOME": str(shared_hf),
            "PIP_CACHE_DIR": str(shared_pip),  # shared wheel cache — fast, resilient re-installs (see above)
            # NOTE: HF_HUB_ENABLE_HF_TRANSFER is intentionally NOT set — modern
            # huggingface_hub raises at download time if the flag is on but the
            # `hf_transfer` package isn't installed, and the fresh per-run venv
            # won't have it. Leave default (off) for robustness.
            "TORCH_HOME": str((runs_root / ".cache" / project_id / "torch").resolve()),
            "TRITON_CACHE_DIR": str((runs_root / ".cache" / project_id / "triton").resolve()),
            "XDG_CACHE_HOME": str((runs_root / ".cache" / project_id / "xdg").resolve()),
            "MPLCONFIGDIR": str((runs_root / ".cache" / project_id / "mpl").resolve()),
            "REPROLAB_RLM_ROOT_MODEL": args.model,
            "PYTHONUNBUFFERED": "1",
        }
        if venv_ok:
            per_run_env["REPROLAB_EXPERIMENT_VENV"] = str(venv_dir)

        child_env = {**os.environ, **per_run_env}

        # --- 6f: Build command ---
        # NOTE: --runs-root and --database-url are GLOBAL args on backend.cli's
        # main parser (added before add_subparsers), so they MUST precede the
        # `reproduce` subcommand. Everything else (--mode/--model/--sandbox/
        # --gpu-mode/--project-id/--paper-hint/--max-wall-clock) is a reproduce
        # subcommand arg and goes AFTER `reproduce <paper>`.
        extra_argv = shlex.split(args.extra) if args.extra.strip() else []
        cmd = [
            interpreter, "-m", "backend.cli",
            "--runs-root", str(runs_root),   # global — before the subcommand
            "reproduce", paper,
            "--mode", args.mode,
            "--model", args.model,
            "--sandbox", args.sandbox,
            "--gpu-mode", args.gpu_mode,
            "--project-id", project_id,
            *extra_argv,
        ]
        if args.accelerator != "off":
            cmd.extend(["--accelerator", args.accelerator])

        # --- 6g: Launch child ---
        log_path = run_dir / "batch_child.log"
        logger.info(
            "batch_reproduce: launching [%s] project_id=%s gpus=%s -> %s",
            paper, project_id, list(lease.gpu_indices), log_path,
        )

        with open(log_path, "wb") as logf:
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=child_env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    cwd=str(repo_root),
                )
            except Exception as exc:
                # Lease release is handled by the outer finally.
                raise RuntimeError(
                    "batch_reproduce: [%s] failed to launch subprocess: %s" % (paper, exc)
                ) from exc

        live_procs[lease_id] = proc
        logger.info(
            "batch_reproduce: launched [%s] project_id=%s pid=%d gpus=%s -> %s",
            paper, project_id, proc.pid, list(lease.gpu_indices), log_path,
        )

        # --- 6h: Wait for completion ---
        rc = proc.wait()

        score = _extract_score(run_dir / "final_report.json")
        logger.info(
            "batch_reproduce: [%s] project_id=%s exit=%d score=%s",
            paper, project_id, rc, score,
        )
        return {
            "paper": paper,
            "project_id": project_id,
            "gpu_indices": list(lease.gpu_indices),
            "exit_code": rc,
            "score": score,
            "log_path": str(log_path),
        }
    finally:
        # --- 6i: Always release lease + deregister (idempotent) ---
        live_procs.pop(lease_id, None)
        alloc.release(lease_id)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Print an aligned table of completed run outcomes."""
    if not results:
        print("\n(no runs completed)")
        return

    # Column widths
    w_paper = max(len("PAPER"), *(len(r["paper"]) for r in results))
    w_pid   = max(len("PROJECT_ID"), *(len(r["project_id"]) for r in results))
    w_gpus  = max(len("GPUS"), *(len(str(r["gpu_indices"])) for r in results))
    w_exit  = len("EXIT")
    w_score = len("SCORE")

    sep = "  "
    header = (
        f"{'PAPER':<{w_paper}}{sep}"
        f"{'PROJECT_ID':<{w_pid}}{sep}"
        f"{'GPUS':<{w_gpus}}{sep}"
        f"{'EXIT':<{w_exit}}{sep}"
        f"{'SCORE':<{w_score}}{sep}"
        f"LOG"
    )
    divider = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("BATCH REPRODUCE SUMMARY")
    print(divider)
    print(header)
    print(divider)
    for r in results:
        score_str = f"{r['score']:.4f}" if r["score"] is not None else "n/a"
        print(
            f"{r['paper']:<{w_paper}}{sep}"
            f"{r['project_id']:<{w_pid}}{sep}"
            f"{str(r['gpu_indices']):<{w_gpus}}{sep}"
            f"{str(r['exit_code']):<{w_exit}}{sep}"
            f"{score_str:<{w_score}}{sep}"
            f"{r['log_path']}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Accelerator auto-start
# ---------------------------------------------------------------------------

# Readiness poll parameters for the accelerator auto-start path.
_ACCEL_PROBE_TIMEOUT_S = 3      # per-probe urllib timeout
_ACCEL_LOCAL_WAIT_S = 120       # max seconds to wait when --accelerator=local
_ACCEL_POLL_INTERVAL_S = 5      # seconds between readiness probes


def _probe_accelerator(host: str, port: int) -> bool:
    """Return True if GET http://{host}:{port}/v1/models indicates a live server.

    Sends ``Authorization: Bearer <key>`` when ``REPROLAB_ACCELERATOR_API_KEY`` is
    set (default ``"local"``), because a vLLM server started with ``--api-key``
    returns 401 to an unauthenticated probe — which means the server IS up.
    HTTP 401/403 are therefore treated as reachable (mirrors
    ``serve_local_llm._probe_server`` and ``accelerator.probe_endpoint``).
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/v1/models"
    api_key = os.environ.get("REPROLAB_ACCELERATOR_API_KEY", "local")
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=_ACCEL_PROBE_TIMEOUT_S) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403)
    except (urllib.error.URLError, OSError):
        return False


def _maybe_start_accelerator(
    args: argparse.Namespace,
    runs_root: Path,
    repo_root: Path,
) -> None:
    """Best-effort: start serve_local_llm.py in the background if not already up.

    - For ``--accelerator local``:  wait up to _ACCEL_LOCAL_WAIT_S seconds for
      readiness and warn (but do NOT abort) if it doesn't come up.
    - For ``--accelerator auto``:   fire-and-forget — do not block the batch;
      each run's resolve_accelerator will fall back to Sonnet if not ready.

    Any exception is swallowed so a broken accelerator start never kills a batch.
    """
    accel_mode: str = args.accelerator  # "local" or "auto"
    # Default to the same port/host that serve_local_llm defaults to.
    accel_host = "127.0.0.1"
    accel_port = 8001

    try:
        # Idempotent: already up?
        if _probe_accelerator(accel_host, accel_port):
            logger.info(
                "batch_reproduce: accelerator already serving at "
                "http://%s:%d/v1 — skipping auto-start",
                accel_host, accel_port,
            )
            return

        serve_script = repo_root / "scripts" / "serve_local_llm.py"
        if not serve_script.exists():
            logger.warning(
                "batch_reproduce: %s not found — cannot auto-start accelerator",
                serve_script,
            )
            return

        # Launch in --no-foreground mode so it exits once ready and doesn't
        # compete for the same Python interpreter thread.
        cmd = [
            sys.executable, str(serve_script),
            "--no-foreground",
            "--runs-root", str(runs_root),
        ]
        logger.info(
            "batch_reproduce: auto-starting accelerator (mode=%s): %s",
            accel_mode, " ".join(cmd),
        )
        accel_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("batch_reproduce: accelerator launcher pid=%d", accel_proc.pid)

        if accel_mode == "local":
            # Wait up to _ACCEL_LOCAL_WAIT_S for readiness.
            deadline = time.monotonic() + _ACCEL_LOCAL_WAIT_S
            ready = False
            while time.monotonic() < deadline:
                if _probe_accelerator(accel_host, accel_port):
                    ready = True
                    break
                time.sleep(_ACCEL_POLL_INTERVAL_S)
            if ready:
                logger.info(
                    "batch_reproduce: accelerator ready at http://%s:%d/v1",
                    accel_host, accel_port,
                )
            else:
                logger.warning(
                    "batch_reproduce: accelerator did not become ready within %ds "
                    "(mode=local) — batch will proceed; runs may fall back to Sonnet",
                    _ACCEL_LOCAL_WAIT_S,
                )
        else:
            # auto: fire-and-forget — don't block
            logger.info(
                "batch_reproduce: accelerator auto-start fired (mode=auto); "
                "not waiting for readiness — runs will fall back to Sonnet if not ready"
            )

    except Exception as exc:  # noqa: BLE001 — defensive: never abort the batch
        logger.warning(
            "batch_reproduce: accelerator auto-start failed (mode=%s): %s — continuing without accelerator",
            accel_mode, exc,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic batch scheduler: run one or more paper reproductions in parallel, "
            "each pinned to a disjoint set of local NVIDIA GPUs via LocalGpuAllocator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "papers",
        nargs="+",
        metavar="PAPER",
        help=(
            "arXiv id, PDF path, or bundle id — passed straight to "
            "``python -m backend.cli reproduce <PAPER> ...``."
        ),
    )
    parser.add_argument(
        "--gpus-per-run",
        default="4",
        metavar="STR",
        help=(
            "Integer number of GPUs to allocate per run, OR the literal 'auto' "
            "to grab ALL currently-free GPUs for one run at a time."
        ),
    )
    parser.add_argument(
        "--sandbox",
        default="local",
        help="Sandbox backend: local | docker | runpod.",
    )
    parser.add_argument(
        "--gpu-mode",
        default="prefer",
        help="GPU scheduling hint passed to the CLI: auto | prefer | max | none.",
    )
    parser.add_argument(
        "--model",
        default="claude-oauth",
        help="Root model key (e.g. 'claude-oauth', 'gpt-5', 'claude').",
    )
    parser.add_argument(
        "--mode",
        default="rlm",
        help="Orchestration mode: rlm | rdr | rlm-pure.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        metavar="STR",
        help="Root directory where run artefacts are written.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Cap on concurrent runs.  None = derive automatically from free GPU "
            "capacity and gpus-per-run."
        ),
    )
    parser.add_argument(
        "--free-mem-threshold-mb",
        type=int,
        default=1024,
        metavar="INT",
        help="A GPU is considered free only if used memory < this value (MiB).",
    )
    parser.add_argument(
        "--python",
        default=None,
        metavar="STR",
        help=(
            "Python interpreter used to launch CLI child processes. "
            "Default: <repo>/.venv/bin/python if present, else sys.executable."
        ),
    )
    parser.add_argument(
        "--extra",
        default="",
        metavar="STR",
        help="Extra args appended verbatim to each CLI invocation (shlex.split).",
    )
    parser.add_argument(
        "--accelerator",
        choices=["off", "auto", "local", "runpod", "azure", "endpoint"],
        default="off",
        help=(
            "Accelerator tier for cheap RLM calls. "
            "'local'/'auto' trigger best-effort auto-start of scripts/serve_local_llm.py "
            "before the batch runs begin; the CLI flag is forwarded to each reproduce child."
        ),
    )

    args = parser.parse_args()

    # ---- 1. Validate gpus-per-run -------------------------------------------
    auto_mode = args.gpus_per_run == "auto"
    if not auto_mode:
        try:
            gpus_int = int(args.gpus_per_run)
            if gpus_int < 1:
                raise ValueError
        except ValueError:
            logger.error(
                "batch_reproduce: --gpus-per-run must be a positive integer or 'auto', got %r",
                args.gpus_per_run,
            )
            return 2
        args._gpus_per_run_int = gpus_int  # type: ignore[attr-defined]
    else:
        args._gpus_per_run_int = 0  # type: ignore[attr-defined]  # placeholder; computed at acquire-time

    # ---- 2. Resolve paths and interpreter -----------------------------------
    runs_root = Path(args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    interpreter = _resolve_interpreter(args.python, repo_root)

    logger.info("batch_reproduce: repo_root=%s  runs_root=%s  interpreter=%s", repo_root, runs_root, interpreter)

    # ---- 3. Initialise allocator and reclaim stale leases -------------------
    from backend.services.runtime.local_gpu_allocator import (
        LocalGpuAllocator,
        free_devices,
    )

    alloc = LocalGpuAllocator(runs_root / ".gpu_leases.json")
    reclaimed = alloc.reclaim_stale()
    if reclaimed:
        logger.info("batch_reproduce: reclaimed %d stale lease(s): %s", len(reclaimed), reclaimed)
    else:
        logger.info("batch_reproduce: no stale leases to reclaim")

    # ---- 4. Install signal handlers (main thread only) ----------------------
    _install_signal_handlers(alloc)

    # ---- 4b. Best-effort accelerator auto-start (local/auto only) -----------
    if args.accelerator in {"auto", "local"}:
        _maybe_start_accelerator(args, runs_root, repo_root)

    # ---- 5. Determine parallelism -------------------------------------------
    papers: list[str] = args.papers
    free_now = free_devices(free_mem_threshold_mb=args.free_mem_threshold_mb)

    if len(free_now) == 0:
        logger.error(
            "batch_reproduce: no free GPUs available (threshold=%dMiB) — cannot schedule any runs",
            args.free_mem_threshold_mb,
        )
        return 2

    if auto_mode:
        # One run at a time, each grabs everything free
        derived_parallel = 1
    else:
        slots = max(1, len(free_now) // args._gpus_per_run_int)  # type: ignore[attr-defined]
        derived_parallel = min(slots, len(papers))

    if args.max_parallel is not None:
        max_parallel = min(derived_parallel, args.max_parallel, len(papers))
    else:
        max_parallel = derived_parallel

    max_parallel = max(1, max_parallel)

    logger.info(
        "batch_reproduce: papers=%d  free_gpus=%d  gpus_per_run=%s  max_parallel=%d",
        len(papers),
        len(free_now),
        args.gpus_per_run,
        max_parallel,
    )
    for i, p in enumerate(papers):
        logger.info("batch_reproduce:   [%d] %s", i, p)

    # ---- 6. Run papers through worker pool ----------------------------------
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_map = {
                executor.submit(
                    _run_one,
                    paper=paper,
                    idx=idx,
                    args=args,
                    alloc=alloc,
                    runs_root=runs_root,
                    interpreter=interpreter,
                    repo_root=repo_root,
                ): (paper, idx)
                for idx, paper in enumerate(papers)
            }
            for future in as_completed(future_map):
                paper, idx = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error("batch_reproduce: [%s] worker raised: %s", paper, exc, exc_info=True)
                    errors.append(f"{paper}[{idx}]: {exc}")
    finally:
        # Belt-and-suspenders: release any leases still registered
        for lid in list(live_procs.keys()):
            try:
                alloc.release(lid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("batch_reproduce: could not release lease %s in finally: %s", lid, exc)

    # ---- 8. Print summary ---------------------------------------------------
    _print_summary(results)

    if errors:
        print(f"\n{len(errors)} run(s) failed with exceptions:")
        for msg in errors:
            print(f"  ERROR: {msg}")

    all_ok = (
        not errors
        and all(r["exit_code"] == 0 for r in results)
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
