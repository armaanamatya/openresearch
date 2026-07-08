"""Thin Cloud Build (gcloud) client for build-on-miss GKE
images — image-exists cache check + submit. The ONLY subprocess boundary goes
through an injected `runner` (default = a subprocess.run wrapper) so tests are
socket/subprocess-hermetic. No google-cloud SDK dependency; mirrors the operator's
manual `gcloud builds submit` (docker/gke-cell-verl/Dockerfile header).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

# A runner takes an argv list + a timeout and returns (returncode, stdout, stderr).
Runner = Callable[[Sequence[str], float], "tuple[int, str, str]"]


def _default_runner(argv: Sequence[str], timeout_s: float) -> "tuple[int, str, str]":
    proc = subprocess.run(  # noqa: S603 — argv is built from validated settings, never shell=True
        list(argv), capture_output=True, text=True, timeout=timeout_s,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    image_ref: str
    error: str = ""
    log_tail: str = ""


def image_exists(image_ref: str, *, runner: Runner = _default_runner, timeout_s: float = 120.0) -> bool:
    """True iff the Artifact Registry image tag already exists (cache hit). A
    non-zero exit (not found / auth) is treated as 'does not exist' — the caller
    then builds. Never raises (fail-open-to-build)."""
    argv = [
        "gcloud", "artifacts", "docker", "images", "describe", image_ref,
        "--format=value(image_summary.digest)", "--quiet",
    ]
    try:
        rc, _out, _err = runner(argv, timeout_s)
    except Exception:  # noqa: BLE001 — any probe failure => assume missing => build
        logger.warning("cloud_build.image_exists: probe failed for %s; assuming missing", image_ref, exc_info=True)
        return False
    return rc == 0


def submit_build(
    context_dir: str | Path,
    image_ref: str,
    *,
    project: str,
    machine_type: str = "E2_HIGHCPU_8",
    timeout_s: int = 3600,
    runner: Runner = _default_runner,
) -> BuildResult:
    """Submit a Cloud Build of `context_dir` (must contain a Dockerfile) tagged
    `image_ref`. Returns BuildResult(ok, image_ref, error, log_tail). Never raises
    — a failed build is a repairable BuildResult, not an exception."""
    ctx = Path(context_dir)
    if not (ctx / "Dockerfile").is_file():
        return BuildResult(ok=False, image_ref=image_ref, error=f"no Dockerfile in build context {ctx}")
    argv = [
        "gcloud", "builds", "submit",
        "--project", project,
        "--tag", image_ref,
        "--machine-type", machine_type,
        f"--timeout={timeout_s}s",
        "--quiet",
        str(ctx),
    ]
    # Cloud Build itself can take minutes; give the runner headroom over the build timeout.
    run_timeout = float(timeout_s) + 300.0
    try:
        rc, out, err = runner(argv, run_timeout)
    except Exception as exc:  # noqa: BLE001
        return BuildResult(ok=False, image_ref=image_ref, error=f"gcloud builds submit failed to launch: {exc}")
    if rc == 0:
        return BuildResult(ok=True, image_ref=image_ref)
    tail = "\n".join((out + "\n" + err).splitlines()[-40:])
    return BuildResult(ok=False, image_ref=image_ref, error=f"cloud build exited {rc}", log_tail=tail)
