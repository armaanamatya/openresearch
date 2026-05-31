#!/usr/bin/env python
"""reserve_and_run_sdar.py — accumulate a hold on N GPUs, then run SDAR on them.

The box is a shared, scheduler-less 8xA5000 cluster (users zby22 / bgu9 also run
here), so cards free a few at a time. ``reserve_gpus.py`` only grabs what is free
*right now*; this wrapper LOOPS the reservation so each card is held the instant
zby22/bgu9 releases it, until N are accumulated — then it launches the SDAR
smallest-two run leasing exactly those cards.

How the hold works (uses the repo's tested gpu_reservation subsystem, NOT a new
mechanism): each reserved card gets a ~256 MiB CUDA "holder" process
(``start_new_session`` → it outlives this script and auto-releases on TTL). The
holder keeps other users off the card, and ``batch_reproduce`` recognises the
holder PIDs via ``own_pids`` so OUR run can still lease the reserved card. We only
ever reserve cards that are FREE right now — we NEVER evict zby22/bgu9.

Usage:
    .venv/bin/python scripts/reserve_and_run_sdar.py            # hold 4, run on 4
    RESERVE_N=2 .venv/bin/python scripts/reserve_and_run_sdar.py
Env knobs: RESERVE_N (default 4), RESERVE_TTL_HOURS (6), RESERVE_MAX_WAIT_HOURS (5),
RESERVE_POLL_SECONDS (30), RESERVE_RELEASE_ON_EXIT (1).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.runtime.gpu_reservation import GpuReservationManager  # noqa: E402

NEED = int(os.environ.get("RESERVE_N", "4"))
TTL_S = int(float(os.environ.get("RESERVE_TTL_HOURS", "6")) * 3600)
MAX_WAIT_S = int(float(os.environ.get("RESERVE_MAX_WAIT_HOURS", "5")) * 3600)
POLL_S = int(os.environ.get("RESERVE_POLL_SECONDS", "30"))
RELEASE_ON_EXIT = os.environ.get("RESERVE_RELEASE_ON_EXIT", "1") not in ("0", "false", "no")
LAUNCH_LOG = "/tmp/sdar_batch_launch3.log"
REGISTRY = REPO / "runs" / ".gpu_reservations.json"
VENV_PY = REPO / ".venv" / "bin" / "python"


def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _held(mgr: GpuReservationManager) -> list:
    return sorted(mgr.list_reservations(), key=lambda r: r.index)


def main() -> int:
    mgr = GpuReservationManager(REGISTRY, repo_root=REPO)

    # 1. Accumulate NEED reservations, grabbing free cards as they appear.
    _log(f"reserve: target={NEED} ttl={TTL_S // 3600}h max_wait={MAX_WAIT_S // 3600}h")
    deadline = time.time() + MAX_WAIT_S
    while len(_held(mgr)) < NEED:
        have = len(_held(mgr))
        created = mgr.reserve(count=NEED - have, ttl_seconds=TTL_S, hold_mib=256)
        for r in created:
            _log(f"reserve: grabbed GPU {r.index} pid={r.pid}  (held={len(_held(mgr))}/{NEED})")
        if len(_held(mgr)) >= NEED:
            break
        if time.time() > deadline:
            _log(f"reserve: TIMEOUT after {MAX_WAIT_S // 3600}h — holding {len(_held(mgr))}/{NEED}")
            break
        time.sleep(POLL_S)

    held = _held(mgr)
    idxs = [r.index for r in held]
    _log(f"reserve: holding {len(held)} GPU(s): {idxs}")
    if len(held) < 1:
        _log("reserve: no cards held — aborting (no run launched)")
        return 1

    run_n = min(len(held), NEED)

    # 2. Launch the SDAR run leasing the reserved cards. batch_reproduce reads the
    #    reservation registry → own_pids, so it leases exactly these held cards.
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)   # force clean claude-oauth (CLAUDE.md pitfall)
    env.pop("OPENAI_API_KEY", None)
    env["REPROLAB_DISABLE_TORCHRUN_WRAP"] = "1"
    env["REPROLAB_MIN_TRAIN_WALL_S"] = "120"
    env["REPROLAB_BASELINE_EXTRA_GUIDANCE"] = (REPO / "runs/.cache/extra_guidance_sdar.txt").read_text(encoding="utf-8")

    cmd = [
        str(VENV_PY), str(REPO / "scripts/batch_reproduce.py"), "2605.15155",
        "--gpus-per-run", str(run_n), "--sandbox", "local", "--model", "claude-oauth",
        "--mode", "rlm", "--runs-root", str(REPO / "runs"),
        "--extra",
        f"--paper-hint 2605.15155 --scope-spec {REPO}/runs/.cache/scope_sdar_smallest_two.json "
        f"--max-wall-clock 14400 --max-usd 25 --seed 42",
    ]
    _log(f"run: launching batch_reproduce --gpus-per-run {run_n} (log → {LAUNCH_LOG})")
    rc = 1
    try:
        with open(LAUNCH_LOG, "wb") as logf:
            rc = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT).returncode
        _log(f"run: batch_reproduce exited rc={rc}")
    finally:
        if RELEASE_ON_EXIT:
            released = mgr.release(all=True)
            _log(f"reserve: released {len(released)} hold(s) on exit")
        else:
            _log("reserve: leaving holds in place (RESERVE_RELEASE_ON_EXIT=0); TTL will reap them")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
