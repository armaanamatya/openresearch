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

import json
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
    # Parameterizable for the full-paper reproduction (defaults = smallest-two).
    # The smallest-two guidance is intentional: a 24GB A5000 fits Qwen3-1.7B +
    # Qwen2.5-3B only, never the 7B (the full-7B-matrix file regressed the run).
    guidance_file = os.environ.get("SDAR_GUIDANCE_FILE", "runs/.cache/extra_guidance_sdar.txt")
    scope_spec = os.environ.get("SDAR_SCOPE_SPEC", "runs/.cache/scope_sdar_smallest_two.json")
    max_usd = os.environ.get("SDAR_MAX_USD", "25")
    max_wall = os.environ.get("SDAR_MAX_WALL_CLOCK", "14400")
    max_iters = os.environ.get("SDAR_MAX_RLM_ITERATIONS", "20")
    exec_mode = os.environ.get("SDAR_EXECUTION_MODE", "max")

    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)   # force clean claude-oauth (CLAUDE.md pitfall)
    env.pop("OPENAI_API_KEY", None)
    # The cell runner (gpu_cell_runner.py) now owns GPU placement — one process
    # per cell pinned via CUDA_VISIBLE_DEVICES — so run_experiment skips the legacy
    # torchrun re-launch entirely. No torchrun-wrap reconciliation flag is needed.
    env["REPROLAB_MIN_TRAIN_WALL_S"] = "120"
    # GPU count is owned by the lease + cell runner (one GPU per cell,
    # min(free_gpus, cells) in parallel); no force-single/multi flag needed.
    env["REPROLAB_BASELINE_EXTRA_GUIDANCE"] = (REPO / guidance_file).read_text(encoding="utf-8")

    # Distinct-run identity (so each run is its own leaderboard row + readable
    # title). The leaderboard keys rows by run-dir NAME, but the project_id is
    # locked to the paper by ingest (project_id_for(source); a mismatched
    # --project-id raises UnknownProject). So before launching we preserve any
    # PRIOR completed run as its own timestamped dir (the new run then starts
    # fresh under the canonical id), and stamp REPROLAB_RUN_TITLE so the report
    # carries a human label. A run still actively writing is never clobbered.
    try:
        from backend.cli import _source_from_cli
        from backend.services.paths import normalize_path_input
        from backend.services.ingestion.intake.service import project_id_for

        _canonical = project_id_for(_source_from_cli(normalize_path_input("2605.15155"), "auto"))
        _prior = REPO / "runs" / _canonical
        if _prior.is_dir():
            _live = False
            _ds = _prior / "demo_status.json"
            if _ds.is_file():
                try:
                    _st = json.loads(_ds.read_text(encoding="utf-8")).get("status")
                    _live = (_st == "running") and (time.time() - _ds.stat().st_mtime < 180)
                except Exception:
                    _live = False
            if _live:
                _log(f"ABORT: runs/{_canonical} appears to be actively running — refusing to clobber it. Stop that run first.")
                return 1
            _stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            _archived = REPO / "runs" / f"{_canonical}__{_stamp}"
            _prior.rename(_archived)
            _log(f"preserved prior run → runs/{_archived.name} (its own leaderboard row)")
    except Exception as _e:  # never block a launch on the rename bookkeeping
        _log(f"warn: prior-run preservation skipped ({_e!r})")

    run_title = os.environ.get("SDAR_RUN_TITLE", "").strip() or (
        f"SDAR full · {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}"
    )
    env["REPROLAB_RUN_TITLE"] = run_title
    _log(f"run title: {run_title}")

    # Robust paper-text fidelity: point ingest at a pre-extracted full-text blob
    # so the RLM root gets the WHOLE paper even if the live arXiv fetch/parse
    # transiently degrades to lossy chunk-reassembly. parser/service.py honors
    # REPROLAB_PAPER_TEXT_PATH (>=1KB) with precedence over a degraded cascade.
    _paper_text = REPO / "runs" / ".cache" / "paper_text" / "2605.15155.txt"
    if _paper_text.is_file() and _paper_text.stat().st_size >= 1024:
        env["REPROLAB_PAPER_TEXT_PATH"] = str(_paper_text)
        _log(f"paper-text override → {_paper_text.name} ({_paper_text.stat().st_size} bytes)")
    else:
        _log(f"warn: no paper-text override at {_paper_text}; relying on live parse")

    # Cap flags are OMITTED when their env knob is a sentinel (none/off/0/empty) →
    # backend.cli defaults to None = UNLIMITED (budgets bind only when not-None;
    # cli.py:619). A plain launch keeps the historical 14400s/$25/20-iter caps; set
    # SDAR_MAX_WALL_CLOCK=none (etc.) for a truly uncapped run that can finish.
    def _cap(flag: str, raw: str) -> str:
        return "" if str(raw).strip().lower() in ("none", "off", "0", "") else f"{flag} {raw} "

    if str(max_iters).strip().lower() in ("none", "off", "0", ""):
        env.pop("REPROLAB_MAX_RLM_ITERATIONS", None)  # belt-and-suspenders: no inherited cap

    extra = (
        f"--paper-hint 2605.15155 --scope-spec {REPO / scope_spec} "
        + _cap("--max-wall-clock", max_wall)
        + _cap("--max-usd", max_usd)
        + _cap("--max-rlm-iterations", max_iters)
        + f"--execution-mode {exec_mode} --seed 42"
    )
    _caps_desc = ", ".join(
        f"{n}={v}" for n, v in (("wall", max_wall), ("usd", max_usd), ("iters", max_iters))
    )
    _log(f"caps: {_caps_desc}  (sentinel none/off/0 → uncapped)")

    cmd = [
        str(VENV_PY), str(REPO / "scripts/batch_reproduce.py"), "2605.15155",
        "--gpus-per-run", str(run_n), "--sandbox", "local", "--model", "claude-oauth",
        "--mode", "rlm", "--runs-root", str(REPO / "runs"),
        "--extra", extra,
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
