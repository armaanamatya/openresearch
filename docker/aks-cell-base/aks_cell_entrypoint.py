"""AKS Job in-container entrypoint for a single SDAR training cell.

Flow
----
1. Pull project code from Azure Blob (``runs/<run_id>/code/``).
2. Set HF_HOME / TRANSFORMERS_CACHE / PIP_CACHE_DIR under the Azure Files mount
   (``/mnt/reprolab-cache``) so expensive model weights are shared across runs.
3. ``pip install`` project requirements through that cache (delta only; wheels
   already cached on the Files share make this ~instant after the first cell).
4. Execute ``train_cell.py`` **unchanged**, inheriting ``REPROLAB_CELL_*`` env
   vars the orchestrator injected.
5. On CUDA OOM: self-re-exec with the shrink ladder:
   attempt 0 — original params
   attempt 1 — REPROLAB_CELL_BATCH_SCALE=0.5, REPROLAB_CELL_GRAD_CHECKPOINT=1
   attempt 2 — REPROLAB_CELL_BATCH_SCALE=0.25, REPROLAB_CELL_GRAD_CHECKPOINT=1
   floor (max_oom_retries>2) — repeat 0.25+grad-ckpt.
6. Push ``metrics.json`` + per-attempt logs + ``status.json`` to Blob.
7. Exit with the exact code the orchestrator maps:
   0  → ok
   40 → bootstrap_error
   41 → error
   42 → oom_shrink_exhausted
   43 → metrics_invalid
   44 → artifact_upload_error

Testability contract
--------------------
All business logic lives in PURE functions (no azure import at module top):
  plan_attempts(max_oom_retries)          -> list[dict]
  is_oom(stderr)                          -> bool
  classify_outcome(exit_code, stderr, ...) -> tuple[int, str]
  build_sentinel(...)                     -> dict

Blob I/O is behind an injectable ``client`` param; when ``client is None`` the
real ``azure.storage.blob`` is imported lazily.  Tests pass a ``FakeBlobClient``
and run without any azure package or GPU.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel version
# ---------------------------------------------------------------------------
SENTINEL_VERSION = "1"

# ---------------------------------------------------------------------------
# Exit codes — the contract between this wrapper and k8s_job_cell_runner.py
# ---------------------------------------------------------------------------
EXIT_OK: int = 0
EXIT_BOOTSTRAP_ERROR: int = 40
EXIT_ERROR: int = 41
EXIT_OOM_SHRINK_EXHAUSTED: int = 42
EXIT_METRICS_INVALID: int = 43
EXIT_ARTIFACT_UPLOAD_ERROR: int = 44

# Outcome strings that land in status.json.outcome
_OUTCOME_OK = "ok"
_OUTCOME_BOOTSTRAP_ERROR = "bootstrap_error"
_OUTCOME_ERROR = "error"
_OUTCOME_OOM_SHRINK_EXHAUSTED = "oom_shrink_exhausted"
_OUTCOME_METRICS_INVALID = "metrics_invalid"
_OUTCOME_ARTIFACT_UPLOAD_ERROR = "artifact_upload_error"

# ---------------------------------------------------------------------------
# OOM recognition patterns (mirrors gpu_cell_runner._OOM_SIGNATURES)
# ---------------------------------------------------------------------------
_OOM_PATTERNS: tuple[str, ...] = (
    r"CUDA out of memory",
    r"cuda.*OutOfMemoryError",
    r"OutOfMemoryError",
    r"out of memory",
    r"CUDA error: out of memory",
)

# Signal numbers that, when received as ``returncode == -<sig>``, indicate the
# kernel killed the child for exceeding its memory cgroup (OOM killer).
_OOM_KILL_SIGNALS: frozenset[int] = frozenset({9, signal.SIGKILL})


# ---------------------------------------------------------------------------
# Pure functions (no azure import, no subprocess — fully unit-testable)
# ---------------------------------------------------------------------------

def plan_attempts(
    max_oom_retries: int,
    *,
    # P1-fix-9: configurable OOM batch-shrink ratios read from env (runner injects these).
    batch_scale_step1: float | None = None,
    batch_scale_floor: float | None = None,
) -> list[dict[str, Any]]:
    """Return the ordered list of attempt configs for the shrink ladder.

    Attempt 0 is always the original (no overrides).
    Attempt 1: BATCH_SCALE=<step1> (default 0.5), GRAD_CHECKPOINT=1.
    Attempt 2: BATCH_SCALE=<floor> (default 0.25), GRAD_CHECKPOINT=1.
    Attempts 3+: repeat the floor only when max_oom_retries > 2.

    The caller passes the *number of OOM retries allowed after the original*
    (i.e. the value of REPROLAB_CELL_MAX_OOM_RETRIES, default 2).  Total
    attempts = max_oom_retries + 1.

    ``batch_scale_step1`` and ``batch_scale_floor`` override the defaults when
    provided; when None they are read from
    ``REPROLAB_CELL_OOM_BATCH_SCALE_STEP1`` / ``REPROLAB_CELL_OOM_BATCH_SCALE_FLOOR``
    (defaults 0.5 / 0.25) so the orchestrator can tune them per-run.

    Args:
        max_oom_retries:  Non-negative integer (0 means no retry after OOM).
        batch_scale_step1: Override for attempt-1 scale (or None → read env).
        batch_scale_floor: Override for attempt-2+ scale (or None → read env).

    Returns:
        List of dicts, each with keys:
          - ``attempt``:        0-based attempt index.
          - ``batch_scale``:    float override for REPROLAB_CELL_BATCH_SCALE.
          - ``grad_checkpoint``:str "0" or "1" for REPROLAB_CELL_GRAD_CHECKPOINT.
    """
    # P1-fix-9: read from env when not supplied by caller.
    if batch_scale_step1 is None:
        try:
            batch_scale_step1 = float(
                os.environ.get("REPROLAB_CELL_OOM_BATCH_SCALE_STEP1", "0.5")
            )
        except (ValueError, TypeError):
            batch_scale_step1 = 0.5
    if batch_scale_floor is None:
        try:
            batch_scale_floor = float(
                os.environ.get("REPROLAB_CELL_OOM_BATCH_SCALE_FLOOR", "0.25")
            )
        except (ValueError, TypeError):
            batch_scale_floor = 0.25

    attempts: list[dict[str, Any]] = []
    for i in range(max_oom_retries + 1):
        if i == 0:
            cfg = {"attempt": 0, "batch_scale": 1.0, "grad_checkpoint": "0"}
        elif i == 1:
            cfg = {"attempt": 1, "batch_scale": batch_scale_step1, "grad_checkpoint": "1"}
        else:
            # i >= 2: floor at batch_scale_floor
            cfg = {"attempt": i, "batch_scale": batch_scale_floor, "grad_checkpoint": "1"}
        attempts.append(cfg)
    return attempts


def is_oom(stderr: str) -> bool:
    """Return True iff *stderr* contains a recognized CUDA OOM signal.

    Matches against both text patterns and the set of OOM-kill-like substrings.
    Case-insensitive for the cuda patterns.

    Args:
        stderr: Combined stderr (and optionally stdout) text from the child.

    Returns:
        True if the output indicates a CUDA out-of-memory condition.
    """
    if not stderr:
        return False
    for pat in _OOM_PATTERNS:
        if re.search(pat, stderr, re.IGNORECASE):
            return True
    return False


def is_oom_kill(returncode: int) -> bool:
    """Return True iff *returncode* looks like an OOM-killer termination.

    The Linux OOM killer delivers SIGKILL (9).  Python subprocess exposes
    signal-terminated processes as ``returncode = -<signal>``.

    Args:
        returncode: The subprocess return code.

    Returns:
        True when returncode == -9 (SIGKILL) or -signal.SIGKILL.
    """
    return returncode in {-s for s in _OOM_KILL_SIGNALS}


def classify_outcome(
    *,
    exit_code: int,
    stderr: str,
    metrics: dict[str, Any] | None,
    is_last_attempt: bool,
) -> tuple[int, str]:
    """Map a single attempt's result to a wrapper exit code and outcome string.

    This function is pure and does NOT decide whether to retry; the caller
    (``_run_with_ladder``) handles retry logic.  It answers the question:
    "If this were the *last* attempt, what should we exit with?"

    Args:
        exit_code:      Return code from the trainer subprocess (0 = success).
        stderr:         Combined output text for OOM detection.
        metrics:        Parsed metrics.json dict, or None if absent/unparseable.
        is_last_attempt:True when no further attempts will be made.

    Returns:
        (wrapper_exit_code, outcome_string) — values from the exit-code table.
    """
    if exit_code == 0:
        # Trainer exited cleanly — check metrics
        if metrics is None:
            return EXIT_METRICS_INVALID, _OUTCOME_METRICS_INVALID
        if not isinstance(metrics, dict):
            return EXIT_METRICS_INVALID, _OUTCOME_METRICS_INVALID
        # Minimally require "status" or "final_training_loss" to prove it ran
        if "final_training_loss" not in metrics and "status" not in metrics:
            return EXIT_METRICS_INVALID, _OUTCOME_METRICS_INVALID
        return EXIT_OK, _OUTCOME_OK

    # Non-zero exit
    if is_oom(stderr) or is_oom_kill(exit_code):
        if is_last_attempt:
            return EXIT_OOM_SHRINK_EXHAUSTED, _OUTCOME_OOM_SHRINK_EXHAUSTED
        # Intermediate OOM — caller will retry
        return EXIT_OOM_SHRINK_EXHAUSTED, _OUTCOME_OOM_SHRINK_EXHAUSTED

    # Non-OOM failure
    return EXIT_ERROR, _OUTCOME_ERROR


def build_sentinel(
    *,
    cell_id: str,
    outcome: str,
    exit_code: int,
    attempts: int,
    retries: int,
    error: str | None,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    """Build the status.json payload uploaded to Blob.

    Args:
        cell_id:     Cell identifier string.
        outcome:     One of the OUTCOME_* constants.
        exit_code:   Wrapper exit code (EXIT_* constant).
        attempts:    Total number of launch attempts (1 = first attempt only).
        retries:     Number of OOM retries that actually ran (attempts - 1).
        error:       Error summary string, or None on success.
        started_at:  ISO-8601 timestamp of wrapper start.
        finished_at: ISO-8601 timestamp of wrapper finish.

    Returns:
        Dict suitable for serialising to JSON as status.json.
    """
    return {
        "version": SENTINEL_VERSION,
        "cell_id": cell_id,
        "outcome": outcome,
        "exit_code": exit_code,
        "attempts": attempts,
        "retries": retries,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def validate_metrics(metrics: dict[str, Any] | None) -> bool:
    """Return True iff metrics is a non-empty dict with at least a status key.

    Args:
        metrics: Parsed metrics.json content, or None.

    Returns:
        True when the metrics dict satisfies the minimal contract.
    """
    if not isinstance(metrics, dict) or not metrics:
        return False
    return "final_training_loss" in metrics or "status" in metrics


# ---------------------------------------------------------------------------
# Blob helpers (injectable client; lazy azure import)
# ---------------------------------------------------------------------------

def _blob_container_client(
    *,
    account_name: str,
    container_name: str,
    client: Any | None,
) -> Any:
    """Return a ContainerClient, creating one lazily when *client* is None.

    The lazy creation imports ``DefaultAzureCredential`` and
    ``ContainerClient`` only when needed — so tests that inject a fake client
    never touch the azure package.

    Args:
        account_name:   Azure storage account name.
        container_name: Blob container name.
        client:         Pre-built ContainerClient, or None to construct one.

    Returns:
        A ContainerClient (real or injected fake).
    """
    if client is not None:
        return client
    from azure.identity import DefaultAzureCredential  # type: ignore[import]
    from azure.storage.blob import ContainerClient as _CC  # type: ignore[import]
    url = f"https://{account_name}.blob.core.windows.net/{container_name}"
    return _CC(url, credential=DefaultAzureCredential())


def upload_bytes_to_blob(
    data: bytes,
    *,
    blob_name: str,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> None:
    """Upload raw bytes to a single Blob, overwriting if it exists.

    Args:
        data:           Raw bytes to upload.
        blob_name:      Destination blob name (path within the container).
        account_name:   Azure storage account name.
        container_name: Blob container name.
        client:         Injectable ContainerClient (None = build lazily).
    """
    cc = _blob_container_client(
        account_name=account_name,
        container_name=container_name,
        client=client,
    )
    cc.upload_blob(blob_name, data, overwrite=True)


def download_prefix_to_dir(
    blob_prefix: str,
    local_dir: Path,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> list[str]:
    """Download all blobs under *blob_prefix* into *local_dir*.

    Preserves the relative path beneath the prefix.  Creates missing
    intermediate directories.  Returns the list of downloaded blob names.

    Args:
        blob_prefix:    Blob name prefix to list and download.
        local_dir:      Local filesystem root to materialise blobs under.
        account_name:   Azure storage account name.
        container_name: Blob container name.
        client:         Injectable ContainerClient (None = build lazily).

    Returns:
        List of blob names that were downloaded.
    """
    cc = _blob_container_client(
        account_name=account_name,
        container_name=container_name,
        client=client,
    )
    downloaded: list[str] = []
    for blob in cc.list_blobs(name_starts_with=blob_prefix):
        name: str = blob.name
        relative = name[len(blob_prefix):].lstrip("/")
        # Defense-in-depth: a blob name is an external boundary — never let it
        # escape local_dir via ".." or an absolute path. The upload side
        # (azure_blob._validate_blob_name) already rejects these; the download
        # side must validate too, or a poisoned blob name could write outside the
        # cell's code dir. Skip the unsafe entry (don't fail the whole pull).
        rel = PurePosixPath(relative)
        if not relative or rel.is_absolute() or ".." in rel.parts:
            logger.warning("download_prefix_to_dir: skipping unsafe blob name %r", name)
            continue
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        data: bytes = cc.download_blob(name).readall()
        dest.write_bytes(data)
        downloaded.append(name)
    return downloaded


def upload_file_to_blob(
    local_path: Path,
    blob_name: str,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> None:
    """Upload a local file to a single Blob.

    Args:
        local_path:     Path to the file on disk.
        blob_name:      Destination blob name.
        account_name:   Azure storage account name.
        container_name: Blob container name.
        client:         Injectable ContainerClient (None = build lazily).
    """
    upload_bytes_to_blob(
        local_path.read_bytes(),
        blob_name=blob_name,
        account_name=account_name,
        container_name=container_name,
        client=client,
    )


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_metrics_from_disk(output_dir: Path) -> dict[str, Any] | None:
    """Load metrics.json from *output_dir*, returning None on any failure."""
    mf = output_dir / "metrics.json"
    if not mf.is_file():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None


def _remove_stale_metrics(output_dir: Path) -> None:
    """Remove metrics.json before each attempt to avoid carrying stale data."""
    mf = output_dir / "metrics.json"
    try:
        mf.unlink()
    except FileNotFoundError:
        pass


def _run_trainer_subprocess(
    train_cell_path: Path,
    output_dir: Path,
    env_overrides: dict[str, str],
    attempt_log_path: Path,
) -> tuple[int, str]:
    """Execute train_cell.py as a subprocess; return (returncode, combined_output).

    The child inherits the current process environment, with *env_overrides*
    merged in.  Stdout and stderr are written to *attempt_log_path* as they
    arrive and returned together as a string.

    Args:
        train_cell_path:  Path to the train_cell.py script.
        output_dir:       Value for REPROLAB_CELL_OUTPUT_DIR.
        env_overrides:    Dict of env-var overrides to inject (batch scale etc).
        attempt_log_path: File path where combined output will be tee'd.

    Returns:
        (returncode, combined_output_text)
    """
    env = os.environ.copy()
    env.update(env_overrides)
    env["REPROLAB_CELL_OUTPUT_DIR"] = str(output_dir)

    output_lines: list[str] = []
    attempt_log_path.parent.mkdir(parents=True, exist_ok=True)

    with attempt_log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, str(train_cell_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            output_lines.append(line)
            # Echo to wrapper stdout so K8s log stream captures it
            print(line, end="", flush=True)
        proc.wait()

    return proc.returncode, "".join(output_lines)


def _upload_attempt_log(
    log_path: Path,
    attempt: int,
    *,
    output_blob_prefix: str,
    cell_id: str,
    account_name: str,
    container_name: str,
    client: Any | None,
) -> None:
    """Best-effort upload of a per-attempt log file to Blob; never raises."""
    try:
        blob_name = f"{output_blob_prefix}/{cell_id}/logs/attempt-{attempt}.log"
        upload_file_to_blob(
            log_path,
            blob_name,
            account_name=account_name,
            container_name=container_name,
            client=client,
        )
    except Exception as exc:
        logger.warning("Could not upload attempt-%d log: %s", attempt, exc)


def _upload_metrics(
    output_dir: Path,
    *,
    output_blob_prefix: str,
    cell_id: str,
    account_name: str,
    container_name: str,
    client: Any | None,
) -> bool:
    """Upload metrics.json to Blob if it exists; return True on success."""
    mf = output_dir / "metrics.json"
    if not mf.is_file():
        return False
    try:
        blob_name = f"{output_blob_prefix}/{cell_id}/metrics.json"
        upload_file_to_blob(
            mf,
            blob_name,
            account_name=account_name,
            container_name=container_name,
            client=client,
        )
        return True
    except Exception as exc:
        logger.warning("Could not upload metrics.json: %s", exc)
        return False


def _upload_sentinel(
    sentinel: dict[str, Any],
    *,
    output_blob_prefix: str,
    cell_id: str,
    account_name: str,
    container_name: str,
    client: Any | None,
) -> bool:
    """Upload status.json sentinel to Blob; return True on success."""
    try:
        blob_name = f"{output_blob_prefix}/{cell_id}/status.json"
        upload_bytes_to_blob(
            json.dumps(sentinel, indent=2).encode(),
            blob_name=blob_name,
            account_name=account_name,
            container_name=container_name,
            client=client,
        )
        return True
    except Exception as exc:
        logger.warning("Could not upload status.json: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Core orchestration (injectable subprocess runner + blob client for tests)
# ---------------------------------------------------------------------------

def _run_with_ladder(
    *,
    train_cell_path: Path,
    output_dir: Path,
    max_oom_retries: int,
    log_dir: Path,
    output_blob_prefix: str,
    cell_id: str,
    account_name: str,
    container_name: str,
    blob_client: Any | None,
    subprocess_runner: Any | None = None,
) -> tuple[int, str, dict[str, Any] | None, int, int]:
    """Run train_cell.py through the OOM shrink ladder.

    Args:
        train_cell_path:    Path to train_cell.py inside the downloaded code.
        output_dir:         Where train_cell.py writes metrics.json.
        max_oom_retries:    Number of retries after the original attempt.
        log_dir:            Directory to store per-attempt log files.
        output_blob_prefix: Blob prefix for the cell's output artifacts.
        cell_id:            Cell identifier string.
        account_name:       Azure storage account name.
        container_name:     Blob container name.
        blob_client:        Injectable ContainerClient (None = build lazily).
        subprocess_runner:  Optional callable replacement for
                            _run_trainer_subprocess (used in tests).  Must
                            accept the same positional/keyword signature and
                            return (returncode: int, output: str).

    Returns:
        (final_exit_code, final_outcome, metrics_dict_or_None,
         total_attempts, total_retries)
    """
    runner = subprocess_runner if subprocess_runner is not None else _run_trainer_subprocess
    attempts_cfg = plan_attempts(max_oom_retries)

    final_exit_code = EXIT_ERROR
    final_outcome = _OUTCOME_ERROR
    final_metrics: dict[str, Any] | None = None
    total_attempts = 0
    total_retries = 0

    for i, cfg in enumerate(attempts_cfg):
        attempt_num: int = cfg["attempt"]
        is_last = (i == len(attempts_cfg) - 1)
        total_attempts += 1

        logger.info(
            "Cell %s: attempt %d/%d  batch_scale=%.2f  grad_ckpt=%s",
            cell_id, attempt_num + 1, len(attempts_cfg),
            cfg["batch_scale"], cfg["grad_checkpoint"],
        )

        _remove_stale_metrics(output_dir)

        env_overrides: dict[str, str] = {
            "REPROLAB_CELL_BATCH_SCALE": str(cfg["batch_scale"]),
            "REPROLAB_CELL_GRAD_CHECKPOINT": cfg["grad_checkpoint"],
        }

        log_path = log_dir / f"attempt-{attempt_num}.log"
        returncode, combined_output = runner(
            train_cell_path,
            output_dir,
            env_overrides,
            log_path,
        )

        # Upload log immediately (best-effort) before classification
        _upload_attempt_log(
            log_path,
            attempt_num,
            output_blob_prefix=output_blob_prefix,
            cell_id=cell_id,
            account_name=account_name,
            container_name=container_name,
            client=blob_client,
        )

        current_metrics = _load_metrics_from_disk(output_dir)

        exit_code, outcome = classify_outcome(
            exit_code=returncode,
            stderr=combined_output,
            metrics=current_metrics,
            is_last_attempt=is_last,
        )

        if exit_code == EXIT_OK:
            # Upload metrics now (before sentinel) so orchestrator gets them
            _upload_metrics(
                output_dir,
                output_blob_prefix=output_blob_prefix,
                cell_id=cell_id,
                account_name=account_name,
                container_name=container_name,
                client=blob_client,
            )
            return EXIT_OK, _OUTCOME_OK, current_metrics, total_attempts, total_retries

        oom = is_oom(combined_output) or is_oom_kill(returncode)

        if oom and not is_last:
            # Retry with a smaller footprint
            total_retries += 1
            logger.warning(
                "Cell %s: OOM on attempt %d, retrying (retry %d/%d)",
                cell_id, attempt_num, total_retries, max_oom_retries,
            )
            # Upload partial metrics before retry if any exist
            if current_metrics is not None:
                _upload_metrics(
                    output_dir,
                    output_blob_prefix=output_blob_prefix,
                    cell_id=cell_id,
                    account_name=account_name,
                    container_name=container_name,
                    client=blob_client,
                )
            continue

        # Terminal failure (last attempt OOM, or non-OOM error)
        final_exit_code = exit_code
        final_outcome = outcome
        final_metrics = current_metrics

        # Upload whatever partial metrics we have before returning
        if current_metrics is not None:
            _upload_metrics(
                output_dir,
                output_blob_prefix=output_blob_prefix,
                cell_id=cell_id,
                account_name=account_name,
                container_name=container_name,
                client=blob_client,
            )
        return final_exit_code, final_outcome, final_metrics, total_attempts, total_retries

    # Should not be reached, but be safe
    return final_exit_code, final_outcome, final_metrics, total_attempts, total_retries


# ---------------------------------------------------------------------------
# Bootstrap: pull code from Blob + pip install
# ---------------------------------------------------------------------------

def _bootstrap(
    *,
    code_blob_prefix: str,
    local_code_dir: Path,
    cache_root: Path,
    account_name: str,
    container_name: str,
    blob_client: Any | None,
) -> tuple[bool, str]:
    """Pull code from Blob and pip install requirements.

    Args:
        code_blob_prefix: Blob prefix for the uploaded code tree.
        local_code_dir:   Where to materialise the code on disk.
        cache_root:       Azure Files mount root used for all caches.
        account_name:     Azure storage account name.
        container_name:   Blob container name.
        blob_client:      Injectable ContainerClient.

    Returns:
        (success: bool, error_message: str)
    """
    try:
        logger.info("Downloading code from Blob prefix %s", code_blob_prefix)
        local_code_dir.mkdir(parents=True, exist_ok=True)
        blobs = download_prefix_to_dir(
            code_blob_prefix,
            local_code_dir,
            account_name=account_name,
            container_name=container_name,
            client=blob_client,
        )
        logger.info("Downloaded %d blobs to %s", len(blobs), local_code_dir)
    except Exception as exc:
        return False, f"Blob pull failed: {exc}"

    # Set up caches under the Azure Files mount
    hf_home = cache_root / "hf"
    pip_cache = cache_root / "pip"
    xdg_cache = cache_root / "xdg"
    for d in (hf_home, pip_cache, xdg_cache):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["PIP_CACHE_DIR"] = str(pip_cache)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    # pip install requirements if a requirements file is present
    req_path = local_code_dir / "requirements.txt"
    if req_path.is_file():
        # P1-fix-9: read pip timeout from env (runner injects REPROLAB_BOOTSTRAP_PIP_TIMEOUT_S).
        try:
            _pip_timeout = int(os.environ.get("REPROLAB_BOOTSTRAP_PIP_TIMEOUT_S", "600"))
        except (ValueError, TypeError):
            _pip_timeout = 600
        logger.info("pip installing requirements from %s (timeout=%ds)", req_path, _pip_timeout)
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "-r", str(req_path),
                    "--cache-dir", str(pip_cache),
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=_pip_timeout,
            )
            if result.returncode != 0:
                return False, f"pip install failed (rc={result.returncode}): {result.stderr[:500]}"
        except Exception as exc:
            return False, f"pip install exception: {exc}"

    return True, ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    *,
    blob_client: Any | None = None,
    subprocess_runner: Any | None = None,
) -> int:
    """Run the full cell entrypoint.

    Args:
        blob_client:       Injectable ContainerClient for Blob I/O (tests inject
                           a fake; production uses None → lazy construction).
        subprocess_runner: Replacement for _run_trainer_subprocess (tests inject
                           a fake; production uses None → real subprocess).

    Returns:
        The integer exit code to pass to sys.exit.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [aks-cell] %(levelname)s %(message)s",
    )

    started_at = _now_iso()

    # -----------------------------------------------------------------------
    # Config from environment (injected by k8s_job_cell_runner)
    # -----------------------------------------------------------------------
    cell_id = os.environ.get("REPROLAB_CELL_ID", "unknown")
    account_name = os.environ.get("REPROLAB_AZURE_STORAGE_ACCOUNT", "")
    container_name = os.environ.get("REPROLAB_AZURE_BLOB_CONTAINER", "")
    code_blob_prefix = os.environ.get("REPROLAB_BLOB_CODE_PREFIX", "")
    output_blob_prefix = os.environ.get("REPROLAB_BLOB_OUTPUT_PREFIX", "")
    cache_mount = Path(os.environ.get("REPROLAB_CACHE_MOUNT", "/mnt/reprolab-cache"))
    max_oom_retries = int(os.environ.get("REPROLAB_CELL_MAX_OOM_RETRIES", "2"))

    logger.info(
        "AKS cell entrypoint start: cell_id=%s account=%s container=%s",
        cell_id, account_name, container_name,
    )

    # -----------------------------------------------------------------------
    # Scratch directories (inside the container's writable layer)
    # -----------------------------------------------------------------------
    scratch = Path(tempfile.mkdtemp(prefix="aks_cell_"))
    local_code_dir = scratch / "code"
    output_dir = scratch / "output"
    log_dir = scratch / "logs"
    for d in (local_code_dir, output_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1 — Bootstrap
    # -----------------------------------------------------------------------
    ok, bootstrap_err = _bootstrap(
        code_blob_prefix=code_blob_prefix,
        local_code_dir=local_code_dir,
        cache_root=cache_mount,
        account_name=account_name,
        container_name=container_name,
        blob_client=blob_client,
    )

    if not ok:
        logger.error("Bootstrap failed: %s", bootstrap_err)
        finished_at = _now_iso()
        sentinel = build_sentinel(
            cell_id=cell_id,
            outcome=_OUTCOME_BOOTSTRAP_ERROR,
            exit_code=EXIT_BOOTSTRAP_ERROR,
            attempts=0,
            retries=0,
            error=bootstrap_err,
            started_at=started_at,
            finished_at=finished_at,
        )
        _upload_sentinel(
            sentinel,
            output_blob_prefix=output_blob_prefix,
            cell_id=cell_id,
            account_name=account_name,
            container_name=container_name,
            client=blob_client,
        )
        return EXIT_BOOTSTRAP_ERROR

    # -----------------------------------------------------------------------
    # Phase 2 — Training (shrink ladder)
    # -----------------------------------------------------------------------
    train_cell_path = local_code_dir / "train_cell.py"
    if not train_cell_path.is_file():
        logger.error("train_cell.py not found in downloaded code")
        finished_at = _now_iso()
        sentinel = build_sentinel(
            cell_id=cell_id,
            outcome=_OUTCOME_BOOTSTRAP_ERROR,
            exit_code=EXIT_BOOTSTRAP_ERROR,
            attempts=0,
            retries=0,
            error="train_cell.py not found in downloaded code prefix",
            started_at=started_at,
            finished_at=finished_at,
        )
        _upload_sentinel(
            sentinel,
            output_blob_prefix=output_blob_prefix,
            cell_id=cell_id,
            account_name=account_name,
            container_name=container_name,
            client=blob_client,
        )
        return EXIT_BOOTSTRAP_ERROR

    (
        final_exit_code,
        final_outcome,
        final_metrics,
        total_attempts,
        total_retries,
    ) = _run_with_ladder(
        train_cell_path=train_cell_path,
        output_dir=output_dir,
        max_oom_retries=max_oom_retries,
        log_dir=log_dir,
        output_blob_prefix=output_blob_prefix,
        cell_id=cell_id,
        account_name=account_name,
        container_name=container_name,
        blob_client=blob_client,
        subprocess_runner=subprocess_runner,
    )

    # -----------------------------------------------------------------------
    # Phase 3 — Upload sentinel (status.json)
    # -----------------------------------------------------------------------
    finished_at = _now_iso()

    error_summary: str | None = None
    if final_exit_code != EXIT_OK:
        if final_metrics is not None and "error" in final_metrics:
            error_summary = str(final_metrics["error"])[:500]
        else:
            error_summary = f"trainer exited with code {final_exit_code} (outcome={final_outcome})"

    sentinel = build_sentinel(
        cell_id=cell_id,
        outcome=final_outcome,
        exit_code=final_exit_code,
        attempts=total_attempts,
        retries=total_retries,
        error=error_summary,
        started_at=started_at,
        finished_at=finished_at,
    )

    uploaded_sentinel = _upload_sentinel(
        sentinel,
        output_blob_prefix=output_blob_prefix,
        cell_id=cell_id,
        account_name=account_name,
        container_name=container_name,
        client=blob_client,
    )

    if not uploaded_sentinel:
        # P1-fix-10: regardless of outcome — if the sentinel/status.json upload
        # fails the orchestrator cannot reliably score this cell.  On success path
        # this prevents a false-negative; on failure path it prevents the runner
        # from mis-scoring an oom_failed cell as ok because it sees "no status.json
        # → assume still running" and keeps polling.  Log ERROR so it is visible
        # in pod logs / K8s events.
        logger.error(
            "Cell %s: sentinel upload failed (outcome=%s, exit=%d) — "
            "returning EXIT_ARTIFACT_UPLOAD_ERROR so the orchestrator can "
            "detect the upload failure rather than silently mis-scoring.",
            cell_id, final_outcome, final_exit_code,
        )
        return EXIT_ARTIFACT_UPLOAD_ERROR

    logger.info(
        "Cell %s finished: outcome=%s exit=%d attempts=%d retries=%d",
        cell_id, final_outcome, final_exit_code, total_attempts, total_retries,
    )
    return final_exit_code


if __name__ == "__main__":
    sys.exit(main())
