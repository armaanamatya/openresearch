#!/usr/bin/env python3
"""
Single-cell SDAR trainer — harness entry point.

The harness calls this script as:
    python train_cell.py --cell-id <id> --output-dir <dir>

It reads REPROLAB_CELL_PARAMS (JSON of one cells.json entry),
trains on cuda:0 only, and writes a FLAT per-cell metrics.json.

The main coordinator (train.py) can also call this logic inline.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cell-id",    default=None)
parser.add_argument("--output-dir", default=None)
args, _ = parser.parse_known_args()

# ── Resolve cell params ────────────────────────────────────────────────────────
cell_params_raw = os.environ.get("REPROLAB_CELL_PARAMS", "")
cell_output_dir  = (
    args.output_dir
    or os.environ.get("REPROLAB_CELL_OUTPUT_DIR")
    or os.environ.get("OUTPUT_DIR", "/artifacts")
)
os.makedirs(cell_output_dir, exist_ok=True)

if cell_params_raw:
    cell = json.loads(cell_params_raw)
else:
    # Fall back: try to find cell by id in cells.json
    cell_id   = args.cell_id or os.environ.get("REPROLAB_CELL_ID", "")
    code_root = Path(__file__).parent
    cells_path = code_root / "cells.json"
    if cell_id and cells_path.exists():
        manifest = json.loads(cells_path.read_text())
        cell = next(
            (c for c in manifest["cells"] if c["id"] == cell_id),
            None,
        )
        if cell is None:
            print(f"[train_cell] ERROR: cell_id={cell_id!r} not found in cells.json", flush=True)
            sys.exit(1)
    else:
        print("[train_cell] ERROR: no REPROLAB_CELL_PARAMS and no --cell-id", flush=True)
        sys.exit(1)

model_id  = cell["model_id"]
model_key = cell["model_key"]
baseline  = cell["baseline"]
seed      = cell.get("seed", 42)

print(f"[train_cell] cell={cell['id']}", flush=True)
print(f"[train_cell] model_id={model_id}  baseline={baseline}  seed={seed}", flush=True)
print(f"[train_cell] output_dir={cell_output_dir}", flush=True)


def write_cell_metrics(d: dict) -> None:
    path = os.path.join(cell_output_dir, "metrics.json")
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


# ── Honor REPROLAB_CELL_BATCH_SCALE and REPROLAB_CELL_GRAD_CHECKPOINT ─────────
batch_scale      = float(os.environ.get("REPROLAB_CELL_BATCH_SCALE", "1.0"))
grad_checkpoint  = os.environ.get("REPROLAB_CELL_GRAD_CHECKPOINT", "0") == "1"

write_cell_metrics({"status": "starting", "cell_id": cell["id"]})

t0 = time.time()
try:
    # Redirect to train.py's train_one_run
    sys.path.insert(0, str(Path(__file__).parent))

    # Override OUTPUT_DIR so train.py's write_metrics goes to the right place
    os.environ["OUTPUT_DIR"] = cell_output_dir

    import train as sdar_train  # type: ignore

    # Apply batch scale
    if batch_scale < 1.0:
        sdar_train.GROUP_SIZE = max(2, int(sdar_train.GROUP_SIZE * batch_scale))
        print(f"[train_cell] batch_scale={batch_scale} → GROUP_SIZE={sdar_train.GROUP_SIZE}", flush=True)

    # Apply grad checkpoint flag
    if grad_checkpoint:
        print(f"[train_cell] grad checkpoint forced by harness", flush=True)

    # Load data
    train_data = sdar_train.load_qa_data(n_per_source=256)

    result = sdar_train.train_one_run(
        model_id=model_id,
        model_key=model_key,
        baseline=baseline,
        train_data=train_data,
        dev=sdar_train.device,
        seed=seed,
    )

    # Compute opsd_loss_mean over final 20 steps (for rubric leaf ddd54500)
    import numpy as _np
    opsd_curve = result.get("curves", {}).get("opsd_loss", [])
    opsd_loss_mean = float(_np.mean(opsd_curve[-20:])) if opsd_curve else 0.0

    write_cell_metrics({
        "status":         "ok",
        "metric":         result["final_f1"],
        "reward_mean":    result["final_reward"],
        "gate_mean":      result["gate_mean"],
        "gate_active":    result["gate_active_ratio"],
        "delta_t_mean":   result.get("delta_t_mean", 0.0),
        "opsd_loss_mean": opsd_loss_mean,
        "zero_shot_f1":   result["zero_shot_f1"],
        "steps_run":      sdar_train.STEPS,
        "wall_time_s":    time.time() - t0,
        "cell_id":        cell["id"],
    })

    # Write training curves for train.py's aggregate step
    curves = result.get("curves") or {}
    curves_path = os.path.join(cell_output_dir, "curves.json")
    try:
        with open(curves_path, "w") as f:
            import json as _json
            _json.dump(curves, f)
        print(f"[train_cell] curves.json written ({len(curves)} keys)", flush=True)
    except Exception as _exc:
        print(f"[train_cell] WARNING: failed to write curves.json: {_exc}", flush=True)

    print(f"[train_cell] Done. metric={result['final_f1']:.4f}", flush=True)
    sys.exit(0)

except Exception as exc:
    tb_str = traceback.format_exc()
    print(f"[train_cell] ERROR:\n{tb_str}", flush=True)
    write_cell_metrics({
        "status":   "error",
        "error":    str(exc)[:500],
        "wall_time_s": time.time() - t0,
        "cell_id":  cell.get("id", "unknown"),
    })
    sys.exit(1)
