"""serve_local_llm.py — stand up a local vLLM OpenAI-compatible server for
the RLM "accelerator" cheap-call tier.

Leases GPUs from the shared :class:`LocalGpuAllocator` (same allocator used by
``batch_reproduce.py``), so paper-reproduction runs automatically get the
remaining cards while the accelerator server holds its own exclusive lease.

The OpenAI-compatible endpoint emitted here is consumed by ReproLab when the
environment variable ``REPROLAB_ACCELERATOR=local`` (or ``auto``) is set.
Upstream code constructs an ``openai.OpenAI(base_url="http://{host}:{port}/v1",
api_key=<api-key>)`` client and routes cheap accelerator calls through it
instead of burning Sonnet/GPT-5 tokens.

Typical invocations::

    # Foreground — holds lease until Ctrl-C:
    python scripts/serve_local_llm.py --model Qwen/Qwen2.5-Coder-32B-Instruct --tp 4

    # Background / best-effort (called by batch_reproduce.py --accelerator local):
    python scripts/serve_local_llm.py --no-foreground --model Qwen/Qwen2.5-Coder-7B-Instruct --tp 2

Check-ordering:
    1. Idempotent: GET /v1/models — if already 200, exit 0 (no double-start).
    2. vLLM presence: importlib.util.find_spec("vllm") — exit 3 if missing.
    3. GPU lease acquisition — exit 4 if not enough free GPUs.
    4. Launch + health-wait + lifecycle management.

Exit codes:
    0  — server already running (idempotent) or clean foreground exit
    3  — vLLM not installed
    4  — not enough free GPUs to satisfy the lease request
    5  — server health-check timed out (startup failed)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path bootstrap — identical to batch_reproduce.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402 (after sys.path insert)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEASE_ID = "local-llm-server"
_HEALTH_POLL_INTERVAL_S = 5     # seconds between /v1/models probes
_SIGTERM_GRACE_S = 15           # wait before escalating to SIGKILL on shutdown
_PROBE_TIMEOUT_S = 5            # urllib timeout for idempotent + health probes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_server(host: str, port: int, *, api_key: str | None = None) -> bool:
    """Return True if GET http://{host}:{port}/v1/models indicates a live server.

    Sends the bearer token when *api_key* is set (vLLM is launched with
    ``--api-key``, so an unauthenticated probe gets 401 and the server would
    never be seen as ready — observed 2026-05-29). HTTP 200 means ready; 401/403
    means the server is up and enforcing auth, which still counts as reachable
    for readiness because vLLM binds the port only after the model has loaded.
    Mirrors the auth-aware probe in ``backend/agents/rlm/accelerator.py``.
    """
    url = f"http://{host}:{port}/v1/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403)
    except (urllib.error.URLError, OSError):
        return False


def _check_vllm() -> bool:
    """Return True if the vllm package is importable."""
    return importlib.util.find_spec("vllm") is not None


def _terminate_child(proc: subprocess.Popen[bytes], label: str = "vLLM") -> None:
    """Terminate *proc* AND its whole process group: SIGTERM, then SIGKILL after grace.

    vLLM's tensor-parallel workers are 'spawn' multiprocessing subprocesses. Signalling
    only the parent Popen leaves those workers orphaned (reparented to init) and spinning
    at 100% CPU, which then starves the next serve attempt (observed 2026-05-29: a
    timed-out 32B boot left 3 orphans that crippled the following boot). vLLM is launched
    with start_new_session=True (see _launch_vllm) so the parent leads its own process
    group; we signal the whole group here so no worker survives.
    """
    if proc.poll() is not None:
        # Already exited.
        return
    try:
        pgid: int | None = os.getpgid(proc.pid)
    except OSError:
        pgid = None

    def _signal_group(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except OSError:
                pass
        try:  # fallback: at least signal the direct child
            proc.send_signal(sig)
        except OSError:
            pass

    logger.info("serve_local_llm: SIGTERM %s process group (pid %d)", label, proc.pid)
    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=_SIGTERM_GRACE_S)
    except subprocess.TimeoutExpired:
        logger.warning(
            "serve_local_llm: %s pid %d did not exit after %ds — SIGKILL group",
            label, proc.pid, _SIGTERM_GRACE_S,
        )
        _signal_group(signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _model_is_cached(model: str) -> bool:
    """Return True if *model* is already present in the local HuggingFace cache.

    Probes for ``config.json`` — the first file vLLM resolves on boot; if it is
    cached the rest of the snapshot is too. A local directory path counts as
    cached by definition. Used to decide whether to force HF offline mode for
    the vLLM subprocess (see ``_build_child_env``).
    """
    if not model:
        return False
    if os.path.isdir(model):  # a local path, not a hub id
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model, "config.json")
        return isinstance(path, str) and os.path.isfile(path)
    except Exception:  # noqa: BLE001 — any failure → treat as not cached (stay online)
        return False


def _build_child_env(lease: Any, *, model: str | None = None) -> dict[str, str]:
    """Build the environment dict for the vLLM subprocess.

    Sets ``CUDA_VISIBLE_DEVICES`` to the leased GPU indices and
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (mirrors batch_reproduce.py idiom). When
    *model* is already in the local HF cache, forces HF offline mode so vLLM
    boots from disk without any huggingface.co round-trip.
    """
    env = dict(os.environ)
    # Use GPU *indices*, not UUIDs: vLLM 0.7.x parses CUDA_VISIBLE_DEVICES as
    # integers (int(device)), so a UUID like "GPU-75757272-..." raises
    # ValueError and the engine fails to start. CUDA_DEVICE_ORDER=PCI_BUS_ID
    # (below) makes the index→physical-GPU mapping deterministic, so indices are
    # safe here. (torch in the experiment sandbox accepts UUIDs; vLLM does not.)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in lease.gpu_indices)
    # Ensure vLLM sees unbuffered output so logs stream in real time.
    env["PYTHONUNBUFFERED"] = "1"
    # Use 'spawn' for vLLM worker multiprocessing. The default 'fork' deadlocks
    # when CUDA is already initialized in the parent — the engine starts but the
    # model never loads onto the GPU (verified 2026-05-29: fork wedged >6min with
    # the GPU idle; spawn loaded the engine in 3.6s). Caller-set value wins.
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # If the model is already cached locally, force HF offline so vLLM never
    # blocks on a huggingface.co lookup. Observed 2026-05-29: vLLM looped on
    # NameResolutionError / MaxRetryError fetching config.json for an
    # already-cached model when DNS to huggingface.co was intermittently down,
    # so the engine never finished booting. If NOT cached, stay online so vLLM
    # can fetch the weights.
    if _model_is_cached(model or ""):
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info(
            "serve_local_llm: model %r found in local HF cache — forcing "
            "HF_HUB_OFFLINE=1 (no network round-trip on boot)", model,
        )
    return env


def _launch_vllm(
    *,
    model: str,
    tp: int,
    host: str,
    port: int,
    api_key: str,
    child_env: dict[str, str],
) -> subprocess.Popen[bytes]:
    """Spawn the vLLM OpenAI-compatible API server and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp),
        "--host", host,
        "--port", str(port),
        "--api-key", api_key,
        "--served-model-name", model,
        # Skip CUDA-graph capture: this accelerator serves short cheap-tier
        # completions where graphs add little, and capture is slow/flaky on first
        # boot. Fast, reliable startup; pair with the 'spawn' worker method above.
        "--enforce-eager",
    ]
    logger.info("serve_local_llm: launching vLLM: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        env=child_env,
        stdout=None,   # inherit — let logs flow to this process's stdout
        stderr=None,
        # New session → the parent leads its own process group, so _terminate_child
        # can killpg the whole tree (parent + all tensor-parallel spawn workers).
        # Without this, a SIGKILLed parent orphans its workers (100%-CPU zombies that
        # starve the next boot — the 2026-05-29 32B failure).
        start_new_session=True,
    )


def _wait_for_readiness(
    *,
    proc: subprocess.Popen[bytes],
    host: str,
    port: int,
    max_wait: int,
    api_key: str | None = None,
) -> bool:
    """Poll GET /v1/models until 200 or *max_wait* seconds elapse.

    Returns True on success, False on timeout.  If the child process exits
    early it is treated as a failure.
    """
    deadline = time.monotonic() + max_wait
    attempt = 0
    while time.monotonic() < deadline:
        # Child exited prematurely?
        if proc.poll() is not None:
            logger.error(
                "serve_local_llm: vLLM process exited prematurely (rc=%d)",
                proc.returncode,
            )
            return False

        if _probe_server(host, port, api_key=api_key):
            return True

        attempt += 1
        remaining = deadline - time.monotonic()
        wait = min(_HEALTH_POLL_INTERVAL_S, remaining)
        if wait > 0:
            logger.info(
                "serve_local_llm: waiting for vLLM to become ready "
                "(attempt %d, %.0fs remaining) ...",
                attempt, remaining,
            )
            time.sleep(wait)

    return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stand up a local vLLM OpenAI-compatible server for the ReproLab "
            "accelerator tier. Leases GPUs via LocalGpuAllocator so paper runs "
            "get the remaining cards. Set REPROLAB_ACCELERATOR=local (or auto) "
            "to route cheap RLM calls through this endpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model id to serve.",
    )
    parser.add_argument(
        "--tp",
        "--tensor-parallel",
        dest="tp",
        type=int,
        default=4,
        metavar="INT",
        help="Tensor-parallel degree (number of GPUs vLLM shards across).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="TCP port for the vLLM server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for the vLLM server.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Number of GPUs to lease from the allocator. "
            "Defaults to --tp when not specified."
        ),
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        metavar="STR",
        help=(
            "Root directory used by LocalGpuAllocator for the lease state file "
            "(<runs-root>/.gpu_leases.json)."
        ),
    )
    parser.add_argument(
        "--api-key",
        default="local",
        help="API key vLLM requires on incoming requests.",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=600,
        metavar="SEC",
        help="Seconds to wait for the server to become ready before giving up.",
    )
    parser.add_argument(
        "--foreground",
        dest="foreground",
        action="store_true",
        default=True,
        help=(
            "Run vLLM in the foreground and hold the GPU lease until this "
            "process is killed (default)."
        ),
    )
    parser.add_argument(
        "--no-foreground",
        dest="foreground",
        action="store_false",
        help=(
            "Return as soon as the server is ready (lease is released on exit "
            "of THIS script, not when vLLM exits). Use for background/auto-start."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901 (complexity is inherent to lifecycle management)
    parser = _build_parser()
    args = parser.parse_args()

    host: str = args.host
    port: int = args.port
    model: str = args.model
    tp: int = args.tp
    api_key: str = args.api_key
    max_wait: int = args.max_wait
    foreground: bool = args.foreground
    runs_root = Path(args.runs_root).resolve()
    gpu_count: int = args.gpus if args.gpus is not None else tp

    # ------------------------------------------------------------------
    # Step 1: Idempotent probe — exit 0 if already serving
    # ------------------------------------------------------------------
    if _probe_server(host, port, api_key=api_key):
        logger.info(
            "serve_local_llm: server already serving at http://%s:%d/v1 — nothing to do",
            host, port,
        )
        return 0

    # ------------------------------------------------------------------
    # Step 2: vLLM presence check — exit 3 if not installed
    # ------------------------------------------------------------------
    if not _check_vllm():
        print(
            "vLLM not installed: run `uv pip install vllm` (or "
            "`pip install vllm`) and re-try.",
            file=sys.stderr,
        )
        return 3

    # ------------------------------------------------------------------
    # Step 3: Lease GPUs
    # ------------------------------------------------------------------
    from backend.services.runtime.local_gpu_allocator import LocalGpuAllocator  # noqa: E402

    leases_path = runs_root / ".gpu_leases.json"
    alloc = LocalGpuAllocator(leases_path)

    reclaimed = alloc.reclaim_stale()
    if reclaimed:
        logger.info(
            "serve_local_llm: reclaimed %d stale lease(s) before acquire: %s",
            len(reclaimed), reclaimed,
        )

    lease = alloc.acquire(
        _LEASE_ID,
        pid=os.getpid(),
        count=gpu_count,
    )
    if lease is None:
        logger.error(
            "serve_local_llm: not enough free GPUs for the accelerator "
            "(need %d) — cannot start server",
            gpu_count,
        )
        return 4

    logger.info(
        "serve_local_llm: leased %d GPU(s): indices=%s uuids=%s",
        gpu_count, list(lease.gpu_indices), list(lease.gpu_uuids),
    )

    # ------------------------------------------------------------------
    # Step 4: Launch vLLM subprocess + lifecycle management
    # ------------------------------------------------------------------
    child_env = _build_child_env(lease, model=model)
    proc: subprocess.Popen[bytes] | None = None

    def _cleanup(release_lease: bool = True) -> None:
        """Terminate child and release lease — called from finally and signal handler."""
        if proc is not None:
            _terminate_child(proc)
        if release_lease:
            try:
                alloc.release(_LEASE_ID)
                logger.info("serve_local_llm: GPU lease %r released", _LEASE_ID)
            except Exception as exc:  # noqa: BLE001
                logger.warning("serve_local_llm: could not release lease: %s", exc)

    # Install SIGINT / SIGTERM handlers so Ctrl-C or systemd STOP release the lease.
    def _signal_handler(signum: int, _frame: object) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info("serve_local_llm: received %s — shutting down", sig_name)
        _cleanup(release_lease=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        proc = _launch_vllm(
            model=model,
            tp=tp,
            host=host,
            port=port,
            api_key=api_key,
            child_env=child_env,
        )
        logger.info("serve_local_llm: vLLM pid=%d; waiting for readiness ...", proc.pid)

        # ------------------------------------------------------------------
        # Step 5: Health-wait
        # ------------------------------------------------------------------
        ready = _wait_for_readiness(proc=proc, host=host, port=port, max_wait=max_wait, api_key=api_key)

        if not ready:
            logger.error(
                "serve_local_llm: vLLM did not become ready within %ds — aborting",
                max_wait,
            )
            _cleanup(release_lease=True)
            return 5

        endpoint = f"http://{host}:{port}/v1"
        logger.info(
            "serve_local_llm: vLLM ready at %s (model=%s, tp=%d, gpus=%s)",
            endpoint, model, tp, list(lease.gpu_indices),
        )
        print(f"REPROLAB_ACCELERATOR_ENDPOINT={endpoint}", flush=True)

        # ------------------------------------------------------------------
        # Step 6: Foreground vs background exit
        # ------------------------------------------------------------------
        if foreground:
            # Block until the child exits (e.g. OOM, SIGKILL from outside).
            logger.info("serve_local_llm: foreground mode — waiting on vLLM process")
            ret = proc.wait()
            logger.info("serve_local_llm: vLLM exited with code %d", ret)
            # Lease released in finally.
        else:
            # --no-foreground: release lease NOW and return.  The vLLM process
            # is deliberately left running; the caller (batch_reproduce.py) is
            # responsible for any later teardown if needed.  We release the lease
            # here so the allocator doesn't hold the GPUs locked to a PID that
            # is about to exit — the vLLM child process itself keeps CUDA
            # visibility via the env vars already set.
            logger.info(
                "serve_local_llm: --no-foreground: detaching from vLLM (pid=%d); "
                "releasing GPU lease from this launcher process",
                proc.pid,
            )
            alloc.release(_LEASE_ID)
            logger.info("serve_local_llm: GPU lease released (no-foreground mode)")
            # Don't call _cleanup again in finally for the lease.
            return 0

    finally:
        # Belt-and-suspenders: always attempt lease release on any exit path
        # that did not already call _cleanup().  release() is idempotent.
        try:
            alloc.release(_LEASE_ID)
        except Exception as exc:  # noqa: BLE001
            logger.debug("serve_local_llm: finally-block release (may be already released): %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
