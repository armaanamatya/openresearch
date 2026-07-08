#!/usr/bin/env python3
"""Money-safe standalone dispatch of the hand-authored UCPO execute-mode cell to 1xA100 GKE.

Stages a clean code/ from the pristine cloned author repo + the hand-authored
train_cell.py / cells.json / verl_metrics_adapter, then dispatches EXACTLY ONE cell
through the validated GKE cell-matrix (k8s_job_cell_runner.run_matrix, gcp settings
prefix) — bypassing the LLM implement step entirely. This is the Track-A "prove now"
control: does the authors' pipeline, RUN verbatim, produce a real non-zero RLVR reward?

Watch `kubectl get nodes` for the single a2-ultragpu node; the a100-80-rw pool is
scale-to-zero, so it tears down (returns to $0) after the cell completes.

Usage:
  .venv/bin/python scripts/prove_ucpo_cell.py --stage-only   # stage + byte-compile, no GPU
  .venv/bin/python scripts/prove_ucpo_cell.py                # stage + dispatch to 1xA100
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/abheekp/openresearch")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

RUN = ROOT / "runs" / "prj_618445173e9ae4f2"
REPO = RUN / "repo"
SRC = ROOT / "scripts" / "ucpo_execute_cell"
OUT = RUN / "execute_proof"          # output_root.name -> GCS run_id "execute_proof"
CODE = OUT / "code"

_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.egg-info", "*.log", "wandb")


def stage() -> None:
    if CODE.exists():
        shutil.rmtree(CODE)
    CODE.mkdir(parents=True, exist_ok=True)
    for item in sorted(REPO.iterdir()):
        if item.name in (".git",):
            continue
        dst = CODE / item.name
        if item.is_dir():
            shutil.copytree(item, dst, symlinks=False, ignore=_IGNORE)
        else:
            shutil.copy2(item, dst)
    shutil.copy2(SRC / "train_cell.py", CODE / "train_cell.py")
    shutil.copy2(SRC / "cells.json", CODE / "cells.json")
    shutil.copy2(ROOT / "backend/agents/rlm/verl_metrics_adapter.py", CODE / "verl_metrics_adapter.py")

    # byte-compile the trainer so a syntax slip never wastes a GPU dispatch
    import py_compile
    py_compile.compile(str(CODE / "train_cell.py"), doraise=True)
    total = sum(1 for _ in CODE.rglob("*") if _.is_file())
    print(f"staged {total} files -> {CODE}")
    for must in ("train_cell.py", "cells.json", "verl_metrics_adapter.py",
                 "ucpo/main_run.py", "verl/setup.py", "dataset/train_data_10k.parquet"):
        print(f"  {'OK ' if (CODE / must).exists() else 'MISS'} {must}")


def dispatch() -> None:
    from backend.agents.rlm import k8s_job_cell_runner
    cells = json.loads((CODE / "cells.json").read_text())["cells"]
    print(f"dispatching {len(cells)} cell(s) to gcp GKE (1xA100)...", flush=True)
    with k8s_job_cell_runner._bind_settings_prefix("gcp"), \
         k8s_job_cell_runner.bind_run_context(run_budget=None, event_sink=None, gpu_plan=None):
        result = k8s_job_cell_runner.run_matrix(
            cells, str(CODE / "train_cell.py"),
            output_root=str(OUT),
            gpus=None,
            per_cell_timeout_s=float(os.environ.get("UCPO_CELL_TIMEOUT_S", "3600")),
            overall_timeout_s=float(os.environ.get("UCPO_MATRIX_TIMEOUT_S", "4200")),
            gpus_per_cell=1,
            now_iso=datetime.now(timezone.utc).isoformat(),
        )
    (OUT / "cell_result.json").write_text(json.dumps(result, indent=2))
    print("RESULT:", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    stage()
    if "--stage-only" in sys.argv:
        sys.exit(0)
    dispatch()
