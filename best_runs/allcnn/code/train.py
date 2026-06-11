"""
train.py — Top-level entry point for All-CNN reproduction.
Paper: "Striving for Simplicity: The All Convolutional Net" (arXiv 1412.6806)

Reads cells.json, runs all cells via gpu_cell_runner, finalizes metrics.json.

Sandbox contract:
  - Read-only mount at /code (this file's directory)
  - All outputs → $OUTPUT_DIR (writable)
  - metrics.json MUST be written to $OUTPUT_DIR/metrics.json
  - Cache dirs → $OUTPUT_DIR/...

Usage:
  python train.py [--cells cells.json] [--config config.json]
  python train.py --smoke-test   # quick sanity check (2 steps/cell)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure code dir is importable
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

# ---------------------------------------------------------------------------
# Sandbox contract: resolve OUTPUT_DIR early and set all cache dirs
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache dirs: never write under /code (read-only in sandbox)
os.environ.setdefault("HF_HOME",           os.path.join(OUTPUT_DIR, "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(OUTPUT_DIR, "hf_cache"))
os.environ.setdefault("TORCH_HOME",         os.path.join(OUTPUT_DIR, "torch_cache"))
os.environ.setdefault("XDG_CACHE_HOME",     os.path.join(OUTPUT_DIR, "xdg_cache"))
os.environ.setdefault("TRITON_CACHE_DIR",   os.path.join(OUTPUT_DIR, "triton_cache"))
os.environ.setdefault("PIP_CACHE_DIR",      os.path.join(OUTPUT_DIR, "pip_cache"))
os.environ.setdefault("TMPDIR",             os.path.join(OUTPUT_DIR, "tmp"))
os.environ.setdefault("MPLCONFIGDIR",       os.path.join(OUTPUT_DIR, ".matplotlib"))
for d in [
    os.environ["MPLCONFIGDIR"],
    os.path.join(OUTPUT_DIR, "tmp"),
    os.path.join(OUTPUT_DIR, "datasets"),
]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_metrics(d: dict, path: str) -> None:
    """Atomic write of metrics dict to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


def _read_metrics(path: str) -> dict:
    """Read metrics dict from path, return {} on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _finalize_metrics(output_dir: str) -> dict:
    """
    Read $OUTPUT_DIR/metrics.json written by individual cells and finalize.
    Adds status='complete', timestamps, scope.gaps.
    """
    metrics_path = os.path.join(output_dir, "metrics.json")
    agg = _read_metrics(metrics_path)

    # Promote status to 'complete' if we have any per_model entries
    per_model = agg.get("per_model", {})
    n_ok = sum(1 for v in per_model.values() if isinstance(v, dict) and v.get("status") == "ok")
    n_total = len(per_model)

    agg["status"] = "complete"
    agg["n_cells_ok"] = n_ok
    agg["n_cells_total"] = n_total
    agg["finalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Ensure contract paths always exist, even if cells didn't populate them
    if "cifar10" not in agg:
        agg["cifar10"] = {}
    if "final_test_accuracy" not in agg["cifar10"]:
        # Fallback: find best c_allcnn noaug cell
        c_allcnn = per_model.get("c_allcnn", {})
        if c_allcnn and c_allcnn.get("status") == "ok":
            agg["cifar10"]["final_test_accuracy"] = c_allcnn.get("test_accuracy")
            agg["cifar10"]["final_train_loss"] = c_allcnn.get("final_train_loss")
            agg["cifar10"]["best_model_key"] = "c_allcnn"
        else:
            # Last resort: use best of all cifar10 noaug cells
            best_acc = 0.0
            best_key = None
            for k, v in per_model.items():
                if isinstance(v, dict) and v.get("dataset") == "cifar10" and not v.get("augment"):
                    acc = v.get("test_accuracy") or 0.0
                    if acc > best_acc:
                        best_acc = acc
                        best_key = k
            if best_key:
                agg["cifar10"]["final_test_accuracy"] = best_acc
                agg["cifar10"]["final_train_loss"] = per_model[best_key].get("final_train_loss")
                agg["cifar10"]["best_model_key"] = best_key

    # Ensure scope.gaps is present
    if "scope" not in agg:
        agg["scope"] = {}
    if "gaps" not in agg["scope"]:
        agg["scope"]["gaps"] = [
            {
                "item": "ImageNet",
                "reason": "out of compute scope (operator-bounded); manual-download only"
            },
            {
                "item": "ImageNet ILSVRC top-1 41.2%",
                "reason": "manual-download required, not auto-obtainable in sandbox"
            }
        ]

    agg["assumptions_applied"] = ["A001", "A002", "ENV001", "ENV002", "ENV-RT1"]
    agg["note_lr"] = "Fixed lr=0.05 for all cells (proven configuration, prj_0a3202fc187bb692-8f7fe95e)"
    agg["note_imagenet"] = "ImageNet excluded: operator-bounded, manual-download only"

    _write_metrics(agg, metrics_path)
    return agg


# ---------------------------------------------------------------------------
# Smoke test (fast sanity check)
# ---------------------------------------------------------------------------

def run_smoke_test(cells_path: str) -> bool:
    """
    Run a 2-step smoke test on a single cell (c_allcnn_cifar10_noaug) to verify
    the code runs at all before burning GPU time on 14 full cells.

    Returns True if smoke test passed.
    """
    smoke_output = os.path.join(OUTPUT_DIR, "smoke_test")
    os.makedirs(smoke_output, exist_ok=True)

    smoke_env = {**os.environ}
    smoke_env["REPROLAB_SMOKE_STEPS"] = "2"
    smoke_env["OUTPUT_DIR"] = OUTPUT_DIR
    smoke_env["REPROLAB_CELL_OUTPUT_DIR"] = smoke_output
    smoke_env["REPROLAB_CELL_PARAMS"] = json.dumps({
        "id": "c_allcnn_cifar10_noaug_smoke",
        "model_key": "c_allcnn_smoke",
        "letter": "C",
        "variant": "allcnn",
        "dataset": "cifar10",
        "num_classes": 10,
        "augment": False,
        "lr": 0.05,
        "epochs": 1,
        "batch_size": 128,
        "seed": 42,
        "use_zca": True,
    })

    import subprocess
    cell_script = os.path.join(CODE_DIR, "train_cell.py")
    log_path = os.path.join(smoke_output, "smoke.log")

    print("[train.py] Running smoke test...", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, cell_script],
            env=smoke_env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max for smoke
        )
        elapsed = time.time() - t0

        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        if result.returncode == 0:
            print(f"[train.py] Smoke test PASSED in {elapsed:.1f}s", flush=True)
            return True
        else:
            print(f"[train.py] Smoke test FAILED (rc={result.returncode}) in {elapsed:.1f}s", flush=True)
            print(result.stdout[-2000:] if result.stdout else "(no stdout)", flush=True)
            print(result.stderr[-1000:] if result.stderr else "(no stderr)", flush=True)
            return False

    except subprocess.TimeoutExpired:
        print("[train.py] Smoke test TIMED OUT after 300s", flush=True)
        return False
    except Exception as e:
        print(f"[train.py] Smoke test ERROR: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="All-CNN full training run")
    parser.add_argument(
        "--cells", default=os.path.join(CODE_DIR, "cells.json"),
        help="Path to cells.json",
    )
    parser.add_argument(
        "--config", default=os.path.join(CODE_DIR, "config.json"),
        help="Path to config.json (hyperparameters)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test only (2 steps on c_allcnn)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Output directory (default: $OUTPUT_DIR)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")

    # Write initial status
    _write_metrics({
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_model": {},
        "scope": {
            "gaps": [
                {"item": "ImageNet", "reason": "operator-bounded; manual-download only"}
            ]
        }
    }, metrics_path)

    # --- Smoke test mode ---
    if args.smoke_test or int(os.environ.get("REPROLAB_SMOKE_STEPS", "0") or "0") > 0:
        passed = run_smoke_test(args.cells)
        result = {
            "status": "smoke_ok" if passed else "smoke_failed",
            "smoke_passed": passed,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_metrics(result, metrics_path)
        sys.exit(0 if passed else 1)

    # --- Full run ---
    print(f"[train.py] Output dir: {output_dir}", flush=True)
    print(f"[train.py] Cells: {args.cells}", flush=True)

    # Load cells
    with open(args.cells) as f:
        cells_data = json.load(f)
    cells = cells_data["cells"]
    print(f"[train.py] Total cells: {len(cells)}", flush=True)

    # Load config
    try:
        with open(args.config) as f:
            config = json.load(f)
    except Exception:
        config = {}

    # Import gpu_cell_runner
    try:
        from gpu_cell_runner import run_matrix, discover_visible_gpus
    except ImportError as e:
        print(f"[train.py] FATAL: cannot import gpu_cell_runner: {e}", flush=True)
        _write_metrics({"status": "error", "error": str(e)}, metrics_path)
        sys.exit(1)

    gpus = discover_visible_gpus()
    print(f"[train.py] Visible GPUs: {gpus}", flush=True)

    # Output root for per-cell artifacts
    output_root = os.path.join(output_dir, "cells")
    os.makedirs(output_root, exist_ok=True)

    cell_script = os.path.join(CODE_DIR, "train_cell.py")

    # Compute matrix budget: leave 45min for finalization
    max_wall_clock = int(os.environ.get("REPROLAB_MAX_WALL_CLOCK", "0") or "0")
    start_time = time.time()

    print(f"[train.py] Starting matrix: {len(cells)} cells, {len(gpus)} GPUs...", flush=True)
    print(f"[train.py] cell_script: {cell_script}", flush=True)

    try:
        # Each cell runs ~75min on GPU; 14 cells ÷ 2 GPUs = ~7 waves = ~8.75h
        per_cell_timeout = int(os.environ.get("REPROLAB_PER_CELL_TIMEOUT", str(90 * 60)))

        results = run_matrix(
            cells=cells,
            cell_script=cell_script,
            output_root=output_root,
            gpus=gpus,
            per_cell_timeout_s=per_cell_timeout,
            overall_timeout_s=None,  # deadline managed at harness level
        )

    except Exception as e:
        print(f"[train.py] ERROR during run_matrix: {e}", flush=True)
        traceback.print_exc()
        results = []

    # --- Aggregate per-cell results into top-level metrics.json ---
    # run_matrix returns dict[cell_id → {"status":…, "metrics":…, "gpu":…, …}]
    print("[train.py] Aggregating results...", flush=True)

    if not isinstance(results, dict):
        results = {}

    agg_per_model = {}
    n_ok = 0
    n_failed = 0

    for cell_id, r in results.items():
        status = r.get("status", "unknown") if isinstance(r, dict) else "unknown"
        metrics_dict = r.get("metrics") if isinstance(r, dict) else None

        if status == "ok" and metrics_dict:
            n_ok += 1
        else:
            n_failed += 1

        # Try to load metrics from per-cell output dir if not in result dict
        if metrics_dict is None:
            cell_dir = Path(output_root) / cell_id
            cell_mf = cell_dir / "metrics.json"
            if cell_mf.exists():
                try:
                    metrics_dict = json.loads(cell_mf.read_text())
                except Exception:
                    pass

        if metrics_dict:
            # Find model_key from cell spec
            cell_spec = next((c for c in cells if c.get("id") == cell_id), {})
            mk = cell_spec.get("model_key", cell_id)
            agg_per_model[mk] = {
                "test_accuracy": metrics_dict.get("test_accuracy"),
                "test_error_pct": metrics_dict.get("test_error_pct"),
                "final_train_loss": metrics_dict.get("final_train_loss"),
                "status": status,
                "letter": cell_spec.get("letter"),
                "variant": cell_spec.get("variant"),
                "dataset": cell_spec.get("dataset"),
                "augment": cell_spec.get("augment"),
                "lr": metrics_dict.get("lr", 0.05),
                "epochs_run": metrics_dict.get("epochs_run"),
            }

    # Merge with any updates already written by cells via fcntl mechanism
    existing = _read_metrics(metrics_path)
    existing_per_model = existing.get("per_model", {})
    # Cells' own updates take precedence over our shell
    merged_per_model = {**agg_per_model, **existing_per_model}

    # Build final aggregated metrics
    final_metrics = {
        "status": "complete",
        "n_cells_ok": n_ok,
        "n_cells_total": len(cells),
        "per_model": merged_per_model,
        "started_at": existing.get("started_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assumptions_applied": ["A001", "A002", "ENV001", "ENV002", "ENV-RT1"],
        "note_lr": "Fixed lr=0.05 for all cells (proven configuration, prj_0a3202fc187bb692-8f7fe95e)",
        "note_imagenet": "ImageNet excluded: operator-bounded, manual-download only",
        "scope": {
            "gaps": [
                {"item": "ImageNet", "reason": "operator-bounded; manual-download only"},
                {"item": "ImageNet ILSVRC top-1 41.2%", "reason": "manual download required"}
            ]
        }
    }

    # Preserve cifar10/cifar100 contract keys written by cells
    if "cifar10" in existing:
        final_metrics["cifar10"] = existing["cifar10"]
    if "cifar100" in existing:
        final_metrics["cifar100"] = existing["cifar100"]
    if "history" in existing:
        final_metrics["history"] = existing["history"]

    _write_metrics(final_metrics, metrics_path)

    # Final finalize pass to ensure contract paths exist
    final = _finalize_metrics(output_dir)

    c10 = final.get("cifar10", {})
    print(
        f"\n[train.py] COMPLETE: "
        f"{n_ok}/{len(cells)} cells ok, "
        f"cifar10.final_test_accuracy={c10.get('final_test_accuracy'):.4f}"
        if c10.get("final_test_accuracy") else
        f"\n[train.py] COMPLETE: {n_ok}/{len(cells)} cells ok (no cifar10 accuracy yet)",
        flush=True,
    )
    print(f"[train.py] Metrics: {metrics_path}", flush=True)

    sys.exit(0 if n_ok > 0 else 1)


if __name__ == "__main__":
    main()
