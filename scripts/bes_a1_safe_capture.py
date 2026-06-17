#!/usr/bin/env python3
"""Safe A1 capture — ZERO paid GPU by construction.

Launches a BES candidate capture, then KILLS it the instant the graded candidate
pool exists on disk (``rlm_state/bes_candidates.json`` + all N ``candidates/rlm_impl_*``)
— which happens INSIDE ``implement_baseline``, BEFORE any ``run_experiment`` / GPU
cell ever runs. Then re-grades the captured candidates K times (CPU/LLM only) and
prints the SELECT-stability verdict.

Two independent guarantees that no paid GPU is spent:
  1. ``--sandbox local`` on a host with no NVIDIA GPU cannot spend paid GPU.
  2. We kill the run the moment the pool is graded — it never reaches run_experiment.

Cost: LLM tokens only (use ``--model claude-oauth`` → subscription). Zero GPU.

Usage:
    .venv/bin/python scripts/bes_a1_safe_capture.py --paper 1412.6980 --n 3 --k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bes_a1_capture import regrade_candidates, summarize_regrades  # noqa: E402

logger = logging.getLogger("bes_a1_safe_capture")


def pool_ready(run_dir: Path, n: int) -> bool:
    """True once the graded pool is fully on disk (safe-to-kill point).

    The graded pool exists when ``rlm_state/bes_candidates.json`` has been written
    AND all N candidate code snapshots exist. Both are produced inside
    ``implement_baseline`` BEFORE the winner's ``run_experiment`` — so killing here
    guarantees no GPU cell ran.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "rlm_state" / "bes_candidates.json").is_file():
        return False
    snaps = list((run_dir / "candidates").glob("rlm_impl_*"))
    return len(snaps) >= n


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the run's process group (graceful finalize), then SIGKILL if needed."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run_safe_capture(
    paper: str, *, n: int, k: int, sigma: float, timeout_s: int, project_id: str,
    model: str = "claude-oauth", poll_s: float = 5.0,
) -> dict:
    run_dir = REPO_ROOT / "runs" / project_id
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)  # never let a no-credit key shadow OAuth
    env.update({
        "OPENRESEARCH_BES_ENABLED": "1",
        "OPENRESEARCH_BES_CANDIDATES_PER_CLUSTER": str(n),
        "OPENRESEARCH_BES_ADAPTIVE": "0",
        # Fail fast if OAuth isn't detected rather than silently 400 on a no-credit
        # API key (factory.py honors oauth_only when the claude CLI is present).
        "OPENRESEARCH_LLM_AUTH_STRATEGY": "oauth_only",
    })
    cmd = [
        sys.executable, "-m", "backend.cli", "reproduce", paper,
        "--mode", "rlm", "--sandbox", "local", "--model", model,
        "--project-id", project_id,
    ]
    logger.info("launching capture: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, start_new_session=True)

    deadline = time.time() + timeout_s
    killed_reason = None
    try:
        while True:
            if pool_ready(run_dir, n):
                killed_reason = "pool_ready"
                logger.info("graded pool detected — killing run BEFORE any GPU cell")
                _kill_group(proc)
                break
            if proc.poll() is not None:
                killed_reason = "process_exited"
                logger.info("capture process exited on its own (rc=%s)", proc.returncode)
                break
            if time.time() > deadline:
                killed_reason = "timeout"
                logger.warning("timeout (%ss) before pool ready — killing run", timeout_s)
                _kill_group(proc)
                break
            time.sleep(poll_s)
    except KeyboardInterrupt:
        _kill_group(proc)
        killed_reason = "interrupted"

    if not pool_ready(run_dir, n):
        return {
            "ok": False, "killed_reason": killed_reason, "run_dir": str(run_dir),
            "error": "no graded candidate pool was produced (check LLM auth / BES engaged)",
        }

    regrades = regrade_candidates(run_dir, k=k)
    verdict = summarize_regrades(regrades, repeatability_sigma=sigma)
    (run_dir / "a1_result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return {"ok": True, "killed_reason": killed_reason, "run_dir": str(run_dir), **verdict}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Safe zero-GPU A1 SELECT-stability capture + verdict")
    p.add_argument("--paper", default="1412.6980", help="arXiv id (default Adam 1412.6980)")
    p.add_argument("--n", type=int, default=3, help="candidates to capture (>=3)")
    p.add_argument("--k", type=int, default=10, help="re-grades per candidate (temp=0)")
    p.add_argument("--sigma", type=float, default=0.02, help="grader repeatability sigma")
    p.add_argument("--timeout-s", type=int, default=2400, help="max wait for the pool")
    p.add_argument("--model", default="claude-oauth")
    p.add_argument("--project-id", default=None)
    a = p.parse_args(argv)
    pid = a.project_id or f"bes_a1_{a.paper.replace('.', '_')}_{int(time.time())}"
    out = run_safe_capture(
        a.paper, n=a.n, k=a.k, sigma=a.sigma, timeout_s=a.timeout_s,
        project_id=pid, model=a.model,
    )
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
