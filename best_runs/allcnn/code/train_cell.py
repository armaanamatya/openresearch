"""
train_cell.py — Training script for one cell of the All-CNN reproduction.
  Paper: "Striving for Simplicity: The All Convolutional Net" (arXiv 1412.6806)

Cell parameters arrive via:
  - env var REPROLAB_CELL_PARAMS (JSON)
  - env var REPROLAB_CELL_OUTPUT_DIR
  - CLI args --cell-id <str> --output-dir <path>

Outputs written to REPROLAB_CELL_OUTPUT_DIR:
  metrics.json, training_curves.json, config_used.json,
  fig_training.png, fig_training.json
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

# Ensure /code is importable when run as a subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from preprocess import preprocess_cifar, WhitenedDataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LR_CANDIDATES = [0.25, 0.1, 0.05, 0.01]
PROBE_EPOCHS = 15
PROBE_SUBSET_SIZE = 5000
TOTAL_EPOCHS = 350
LR_MILESTONES = [200, 250, 300]
LR_DECAY = 0.1
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
NESTEROV = False

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

HAS_GPU = torch.cuda.is_available()
device = "cuda" if HAS_GPU else "cpu"
num_workers = 4 if HAS_GPU else 0


# ---------------------------------------------------------------------------
# Top-level metrics aggregator (file-lock-safe multi-cell merge)
# ---------------------------------------------------------------------------

def _update_toplevel_metrics(
    top_output_dir: str,
    model_key: str,
    cell_metrics: dict,
    history: dict,
    variant: str,
    base_model: str,
    dataset: str,
    augment: bool,
) -> None:
    """
    Atomically update $OUTPUT_DIR/metrics.json with this cell's results.
    Uses a .lock sentinel file to coordinate between parallel cells.
    The c_allcnn + cifar10_noaug cell writes the contract headline key
    `cifar10_test_accuracy`.
    """
    import fcntl

    os.makedirs(top_output_dir, exist_ok=True)
    top_path = os.path.join(top_output_dir, "metrics.json")
    lock_path = top_path + ".lock"

    for attempt in range(20):
        try:
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Read current aggregated metrics
                if os.path.exists(top_path):
                    with open(top_path) as f:
                        agg = json.load(f)
                else:
                    agg = {"status": "running", "per_model": {}}

                if "per_model" not in agg:
                    agg["per_model"] = {}

                # Write this cell's flat metrics into per_model
                agg["per_model"][model_key] = {
                    "test_accuracy": cell_metrics.get("test_accuracy"),
                    "test_error_pct": cell_metrics.get("test_error_pct"),
                    "train_accuracy": cell_metrics.get("train_accuracy"),
                    "best_lr": cell_metrics.get("best_lr"),
                    "epochs_run": cell_metrics.get("epochs_run"),
                    "dataset": dataset,
                    "augment": augment,
                    "base_model": base_model,
                    "variant": variant,
                }

                # per_dataset: required when scope spans multiple datasets
                dataset_canonical = "CIFAR-10" if dataset == "cifar10" else "CIFAR-100"
                if "per_dataset" not in agg:
                    agg["per_dataset"] = {}
                if dataset_canonical not in agg["per_dataset"]:
                    agg["per_dataset"][dataset_canonical] = {}
                agg["per_dataset"][dataset_canonical][model_key] = {
                    "test_accuracy": cell_metrics.get("test_accuracy"),
                    "test_error_pct": cell_metrics.get("test_error_pct"),
                    "train_accuracy": cell_metrics.get("train_accuracy"),
                    "best_lr": cell_metrics.get("best_lr"),
                    "epochs_run": cell_metrics.get("epochs_run"),
                    "base_model": base_model,
                    "variant": variant,
                    "augment": augment,
                }

                # Headline contract metric: c_allcnn, cifar10, no augmentation
                if (base_model == "c" and variant == "allcnn"
                        and dataset == "cifar10" and not augment):
                    agg["cifar10_test_accuracy"] = cell_metrics.get("test_accuracy")

                # Also track per-epoch convergence at top level for the headline model
                if (base_model == "c" and variant == "allcnn"
                        and dataset == "cifar10" and not augment):
                    if "history" not in agg:
                        agg["history"] = {}
                    agg["history"]["c_allcnn_noaug"] = {
                        "epoch": history.get("epoch", []),
                        "test_acc": history.get("test_acc", []),
                        "train_loss": history.get("train_loss", []),
                        "train_acc": history.get("train_acc", []),
                        "lr": history.get("lr", []),
                    }
                    # Write top-level training_curves.json (contract artifact)
                    tc_path = os.path.join(top_output_dir, "training_curves.json")
                    tc_tmp = tc_path + ".tmp"
                    with open(tc_tmp, "w") as f:
                        json.dump(history, f, indent=2)
                    os.replace(tc_tmp, tc_path)

                # Atomic write
                tmp = top_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(agg, f, indent=2)
                os.replace(tmp, top_path)

                fcntl.flock(lf, fcntl.LOCK_UN)
                break
        except (OSError, IOError):
            time.sleep(0.3 * (attempt + 1))

    # Write README.md to top-level OUTPUT_DIR (contract artifact)
    try:
        code_dir = os.path.dirname(os.path.abspath(__file__))
        readme_src = os.path.join(code_dir, "README.md")
        readme_dst = os.path.join(top_output_dir, "README.md")
        if os.path.exists(readme_src) and not os.path.exists(readme_dst):
            import shutil
            shutil.copy2(readme_src, readme_dst)
    except Exception:
        pass

    print(f"[train_cell] Updated top-level metrics at {top_path}", flush=True)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Atomic metrics write
# ---------------------------------------------------------------------------

def write_metrics(d: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "metrics.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cifar(dataset: str, data_root: str, augment: bool):
    """Download and preprocess CIFAR-10 or CIFAR-100 with GCN+ZCA whitening."""
    from torchvision import datasets, transforms

    os.makedirs(data_root, exist_ok=True)

    if dataset == "cifar10":
        raw_train = datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
        raw_test = datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=transforms.ToTensor()
        )
        zca_cache_name = "cifar_zca_stats.npz"
    elif dataset == "cifar100":
        raw_train = datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
        raw_test = datasets.CIFAR100(
            root=data_root, train=False, download=True,
            transform=transforms.ToTensor()
        )
        zca_cache_name = "cifar100_zca_stats.npz"
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    zca_cache_dir = data_root
    # Temporarily patch the cache filename used inside preprocess_cifar by
    # passing cache_dir; preprocess_cifar always uses 'cifar_zca_stats.npz'.
    # For CIFAR-100 we need a different cache file — call the lower-level API.
    if dataset == "cifar100":
        # Replicate preprocess_cifar logic with a custom cache path.
        from preprocess import compute_gcn_zca_stats, apply_gcn_zca
        import time as _time

        os.makedirs(zca_cache_dir, exist_ok=True)
        cache_path = os.path.join(zca_cache_dir, zca_cache_name)

        print(f"[data] Extracting CIFAR-100 train ({len(raw_train)} samples) ...",
              flush=True)
        t0 = _time.time()
        _ldr = torch.utils.data.DataLoader(
            raw_train, batch_size=1024, shuffle=False, num_workers=0
        )
        tr_imgs, tr_lbls = [], []
        for imgs, lbls in _ldr:
            tr_imgs.append(imgs.numpy())
            tr_lbls.append(lbls.numpy())
        train_images = np.concatenate(tr_imgs, axis=0)
        train_labels = np.concatenate(tr_lbls, axis=0).astype(np.int32)
        print(f"[data] Extracted in {_time.time()-t0:.1f}s.", flush=True)

        N, C, H, W = train_images.shape
        D = C * H * W
        zca_mean, zca_W = compute_gcn_zca_stats(
            train_images.reshape(N, D), epsilon=0.1, cache_path=cache_path
        )

        print("[data] Whitening CIFAR-100 train ...", flush=True)
        train_white = apply_gcn_zca(train_images, zca_mean, zca_W)

        print(f"[data] Extracting CIFAR-100 test ({len(raw_test)} samples) ...",
              flush=True)
        _ldr_t = torch.utils.data.DataLoader(
            raw_test, batch_size=1024, shuffle=False, num_workers=0
        )
        te_imgs, te_lbls = [], []
        for imgs, lbls in _ldr_t:
            te_imgs.append(imgs.numpy())
            te_lbls.append(lbls.numpy())
        test_images = np.concatenate(te_imgs, axis=0)
        test_labels = np.concatenate(te_lbls, axis=0).astype(np.int32)

        print("[data] Whitening CIFAR-100 test ...", flush=True)
        test_white = apply_gcn_zca(test_images, zca_mean, zca_W)

        train_data = train_white.astype(np.float32)
        train_labels_arr = train_labels
        test_data = test_white.astype(np.float32)
        test_labels_arr = test_labels
    else:
        train_data, train_labels_arr, test_data, test_labels_arr = preprocess_cifar(
            raw_train, raw_test, cache_dir=zca_cache_dir
        )

    train_ds = WhitenedDataset(train_data, train_labels_arr, augment=augment)
    test_ds = WhitenedDataset(test_data, test_labels_arr, augment=False)
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Training and evaluation primitives
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, dev, smoke_steps=0):
    """Run one training epoch. Returns (avg_loss, accuracy).
    If smoke_steps > 0, stop after that many optimizer steps and return None
    to signal smoke completion.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    for inputs, targets in loader:
        inputs = inputs.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
        steps += 1

        if smoke_steps > 0 and steps >= smoke_steps:
            return None, None, steps  # signal smoke done

    avg_loss = total_loss / total if total > 0 else float("nan")
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc, steps


def eval_epoch(model, loader, dev):
    """Evaluate on loader, return accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# LR probe
# ---------------------------------------------------------------------------

def probe_lr(base_model: str, variant: str, num_classes: int,
             train_ds, seed: int, batch_size: int) -> float:
    """Run PROBE_EPOCHS epochs on a PROBE_SUBSET_SIZE subset for each LR candidate.
    Return the LR with the best final accuracy.
    """
    # Fixed subset indices (reproducible with seed)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=PROBE_SUBSET_SIZE, replace=False).tolist()
    probe_ds = torch.utils.data.Subset(train_ds, indices)
    probe_loader = torch.utils.data.DataLoader(
        probe_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=HAS_GPU, drop_last=False
    )

    criterion = nn.CrossEntropyLoss()
    best_lr = LR_CANDIDATES[0]
    best_acc = -1.0

    for lr_cand in LR_CANDIDATES:
        set_seeds(seed + hash(lr_cand) % 10000)
        model_p = get_model(base_model, variant, num_classes=num_classes)
        model_p = model_p.to(device)
        optimizer_p = torch.optim.SGD(
            model_p.parameters(), lr=lr_cand,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV
        )

        for ep in range(1, PROBE_EPOCHS + 1):
            model_p.train()
            for inputs, targets in probe_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer_p.zero_grad()
                loss = criterion(model_p(inputs), targets)
                loss.backward()
                optimizer_p.step()

        # Evaluate on same probe subset
        acc = eval_epoch(model_p, probe_loader, device)
        print(f"[probe] lr={lr_cand:.4f} -> probe_acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_lr = lr_cand

        del model_p, optimizer_p

    print(f"[probe] Selected lr={best_lr} (probe_acc={best_acc:.4f})", flush=True)
    return best_lr


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(history: dict, output_dir: str, cell_id: str) -> None:
    if not HAS_MPL:
        print("[fig] matplotlib not available, skipping figure.", flush=True)
        return

    try:
        epochs = history["epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax0 = axes[0]
        ax0.plot(epochs, history["train_loss"], label="train_loss")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.set_title(f"Training Loss — {cell_id}")
        ax0.legend()
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1]
        ax1.plot(epochs, history["train_acc"], label="train_acc")
        ax1.plot(epochs, history["test_acc"], label="test_acc")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"Accuracy — {cell_id}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = os.path.join(output_dir, "fig_training.png")
        fig.savefig(png_path, dpi=100)
        plt.close(fig)
        print(f"[fig] Saved {png_path}", flush=True)

        # Sidecar JSON for grader
        sidecar = {
            "cell_id": cell_id,
            "figure": "fig_training.png",
            "series": {
                "epoch": epochs,
                "train_loss": history["train_loss"],
                "train_acc": history["train_acc"],
                "test_acc": history["test_acc"],
                "lr": history["lr"],
            },
        }
        json_path = os.path.join(output_dir, "fig_training.json")
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"[fig] Saved {json_path}", flush=True)

    except Exception as exc:
        print(f"[fig] WARNING: figure save failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Parse CLI args
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="All-CNN cell trainer")
    parser.add_argument("--cell-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args, _unknown = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Load cell params from env (with CLI override for output-dir/cell-id)
    # ------------------------------------------------------------------
    cell_params_raw = os.environ.get("REPROLAB_CELL_PARAMS", "{}")
    try:
        cell_params = json.loads(cell_params_raw)
    except json.JSONDecodeError as exc:
        print(f"[train_cell] ERROR: invalid REPROLAB_CELL_PARAMS JSON: {exc}",
              flush=True)
        sys.exit(1)

    cell_id = args.cell_id or cell_params.get("id", "unknown_cell")
    output_dir = (
        args.output_dir
        or os.environ.get("REPROLAB_CELL_OUTPUT_DIR", "/artifacts")
    )
    os.makedirs(output_dir, exist_ok=True)

    base_model = cell_params.get("base_model", "c")
    variant = cell_params.get("variant", "allcnn")
    dataset = cell_params.get("dataset", "cifar10")
    num_classes = cell_params.get("num_classes", 10)
    augment = cell_params.get("augment", False)
    seed = cell_params.get("seed", 42)

    # ------------------------------------------------------------------
    # Smoke-test mode
    # ------------------------------------------------------------------
    smoke_steps_env = os.environ.get("REPROLAB_SMOKE_STEPS", "0")
    try:
        smoke_steps = int(smoke_steps_env)
    except ValueError:
        smoke_steps = 0

    # ------------------------------------------------------------------
    # Batch-size scaling
    # ------------------------------------------------------------------
    batch_size = BATCH_SIZE
    batch_scale_env = os.environ.get("REPROLAB_CELL_BATCH_SCALE", "")
    if batch_scale_env.strip():
        try:
            batch_size = int(batch_size * float(batch_scale_env))
            print(f"[train_cell] REPROLAB_CELL_BATCH_SCALE={batch_scale_env} "
                  f"-> batch_size={batch_size}", flush=True)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Cache directories
    # ------------------------------------------------------------------
    cache_base = os.path.join(output_dir, "cache")
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("TORCH_HOME", cache_base)
    os.environ.setdefault("XDG_CACHE_HOME", cache_base)

    data_root = os.path.join(
        os.environ.get("OUTPUT_DIR", output_dir), "data"
    )
    os.makedirs(data_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print(f"[train_cell] cell_id      = {cell_id}", flush=True)
    print(f"[train_cell] base_model   = {base_model}", flush=True)
    print(f"[train_cell] variant      = {variant}", flush=True)
    print(f"[train_cell] dataset      = {dataset}", flush=True)
    print(f"[train_cell] num_classes  = {num_classes}", flush=True)
    print(f"[train_cell] augment      = {augment}", flush=True)
    print(f"[train_cell] seed         = {seed}", flush=True)
    print(f"[train_cell] batch_size   = {batch_size}", flush=True)
    print(f"[train_cell] device       = {device}", flush=True)
    print(f"[train_cell] output_dir   = {output_dir}", flush=True)
    print(f"[train_cell] smoke_steps  = {smoke_steps}", flush=True)
    print("=" * 60, flush=True)

    # ------------------------------------------------------------------
    # Global seeds
    # ------------------------------------------------------------------
    set_seeds(seed)

    # ------------------------------------------------------------------
    # Load & preprocess data
    # ------------------------------------------------------------------
    print("[train_cell] Loading and preprocessing data ...", flush=True)
    t_data_start = time.time()
    train_ds, test_ds = load_cifar(dataset, data_root, augment=augment)
    print(f"[train_cell] Data ready in {time.time()-t_data_start:.1f}s. "
          f"Train={len(train_ds)} Test={len(test_ds)}", flush=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=HAS_GPU
    )

    # ------------------------------------------------------------------
    # Smoke test: short-circuit after 2 batches
    # ------------------------------------------------------------------
    if smoke_steps > 0:
        print(f"[train_cell] SMOKE MODE: will run {smoke_steps} optimizer steps "
              "then exit.", flush=True)
        set_seeds(seed)
        smoke_model = get_model(base_model, variant, num_classes=num_classes)
        smoke_model = smoke_model.to(device)
        smoke_opt = torch.optim.SGD(
            smoke_model.parameters(), lr=0.1,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV
        )
        smoke_crit = nn.CrossEntropyLoss()
        smoke_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=HAS_GPU
        )
        _, _, steps_done = train_epoch(
            smoke_model, smoke_loader, smoke_opt, smoke_crit,
            device, smoke_steps=smoke_steps
        )
        print(f"[train_cell] SMOKE complete after {steps_done} steps. Exiting 0.",
              flush=True)
        write_metrics(
            {"status": "smoke_ok", "steps_run": steps_done},
            output_dir
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # LR probe
    # ------------------------------------------------------------------
    print("[train_cell] Starting LR probe ...", flush=True)
    t_probe = time.time()
    best_lr = probe_lr(base_model, variant, num_classes, train_ds, seed, batch_size)
    print(f"[train_cell] LR probe done in {time.time()-t_probe:.1f}s. "
          f"best_lr={best_lr}", flush=True)

    # ------------------------------------------------------------------
    # Build full train loader
    # ------------------------------------------------------------------
    set_seeds(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=HAS_GPU, drop_last=False
    )

    # ------------------------------------------------------------------
    # Build model, move to device, THEN build optimizer
    # ------------------------------------------------------------------
    model = get_model(base_model, variant, num_classes=num_classes)
    model = model.to(device)  # MUST be before optimizer creation
    optimizer = torch.optim.SGD(
        model.parameters(), lr=best_lr,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY
    )

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    config_used = {
        "cell_id": cell_id,
        "base_model": base_model,
        "variant": variant,
        "dataset": dataset,
        "num_classes": num_classes,
        "augment": augment,
        "seed": seed,
        "batch_size": batch_size,
        "total_epochs": TOTAL_EPOCHS,
        "lr_milestones": LR_MILESTONES,
        "lr_decay": LR_DECAY,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": NESTEROV,
        "best_lr": best_lr,
        "probe_epochs": PROBE_EPOCHS,
        "probe_subset_size": PROBE_SUBSET_SIZE,
        "device": device,
    }
    with open(os.path.join(output_dir, "config_used.json"), "w") as f:
        json.dump(config_used, f, indent=2)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: dict = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "lr": [],
    }

    steps_total = 0
    t_train_start = time.time()

    print(f"[train_cell] Starting full training: {TOTAL_EPOCHS} epochs "
          f"| lr={best_lr} | batch={batch_size} | {device}", flush=True)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        t_ep = time.time()

        train_loss, train_acc, steps_ep = train_epoch(
            model, train_loader, optimizer, criterion, device, smoke_steps=0
        )
        test_acc = eval_epoch(model, test_loader, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        steps_total += steps_ep if steps_ep is not None else 0

        # Record
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(test_acc))
        history["lr"].append(float(current_lr))

        ep_time = time.time() - t_ep
        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"lr={current_lr:.6f} | "
            f"t={ep_time:.1f}s",
            flush=True
        )

        # NaN / Inf guard
        if not math.isfinite(train_loss):
            msg = f"train_loss={train_loss} at epoch {epoch} — aborting"
            print(f"[train_cell] ERROR: {msg}", flush=True)
            write_metrics({"status": "failed", "error": msg}, output_dir)
            raise RuntimeError(msg)

        # Atomic live metrics every 10 epochs
        if epoch % 10 == 0:
            write_metrics(
                {
                    "status": "running",
                    "epoch": epoch,
                    "test_acc": float(test_acc),
                    "train_acc": float(train_acc),
                    "train_loss": float(train_loss),
                    "steps_run": steps_total,
                },
                output_dir,
            )

    total_time = time.time() - t_train_start
    print(f"[train_cell] Training complete in {total_time:.1f}s.", flush=True)

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    final_test_acc = history["test_acc"][-1] if history["test_acc"] else 0.0
    final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    final_metrics = {
        "status": "ok",
        # "metric" is the primary scalar the harness reads from each cell leaf
        "metric": round(final_test_acc, 6),
        "test_error_pct": round((1.0 - final_test_acc) * 100.0, 4),
        "test_accuracy": round(final_test_acc, 6),
        "train_accuracy": round(final_train_acc, 6),
        "epochs_run": TOTAL_EPOCHS,
        "best_lr": best_lr,
        "steps_run": steps_total,
    }
    write_metrics(final_metrics, output_dir)
    print(f"[train_cell] Final metrics: {final_metrics}", flush=True)

    # ------------------------------------------------------------------
    # Update top-level OUTPUT_DIR/metrics.json (atomic, file-lock safe)
    # Each cell contributes per_model entry; c_allcnn cell writes the
    # headline cifar10_test_accuracy key required by the contract.
    # ------------------------------------------------------------------
    top_output_dir = os.environ.get("OUTPUT_DIR", output_dir)
    model_key = cell_params.get("model_key", cell_id)
    try:
        _update_toplevel_metrics(
            top_output_dir=top_output_dir,
            model_key=model_key,
            cell_metrics=final_metrics,
            history=history,
            variant=variant,
            base_model=base_model,
            dataset=dataset,
            augment=augment,
        )
    except Exception as exc:
        print(f"[train_cell] WARNING: top-level metrics update failed: {exc}",
              flush=True)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    try:
        from provenance import emit_provenance, emit_figure_sidecar
        emit_provenance(
            output_dir,
            experiments={
                cell_id: {
                    "model_key": model_key,
                    "baseline": variant,
                    "env": cell_params.get("env", dataset),
                    "seed": seed,
                    "epochs": TOTAL_EPOCHS,
                    "batch_size": batch_size,
                    "per_optimizer": {"sgd": {
                        "lr": best_lr,
                        "momentum": MOMENTUM,
                        "weight_decay": WEIGHT_DECAY,
                        "nesterov": NESTEROV,
                        "lr_milestones": LR_MILESTONES,
                        "lr_decay": LR_DECAY,
                    }},
                    "hardware": device,
                    "framework_versions": {"torch": torch.__version__},
                    "convergence": {
                        "epoch": history["epoch"],
                        "train_loss": history["train_loss"],
                        "train_acc": history["train_acc"],
                        "test_acc": history["test_acc"],
                        "lr": history["lr"],
                    },
                }
            },
        )
        print(f"[train_cell] Provenance emitted to {output_dir}", flush=True)
    except Exception as exc:
        print(f"[train_cell] WARNING: provenance emit failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Rubric guard (self-validation)
    # ------------------------------------------------------------------
    try:
        from rubric_guard import assert_metrics_schema
        assert_metrics_schema(
            final_metrics,
            required_keys=["test_accuracy", "test_error_pct", "status"],
            required_artifacts=["metrics.json", "training_curves.json"],
            artifact_dir=output_dir,
        )
        print("[train_cell] Rubric guard: PASSED", flush=True)
    except Exception as exc:
        print(f"[train_cell] WARNING: rubric guard: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Training curves JSON
    # ------------------------------------------------------------------
    curves_path = os.path.join(output_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train_cell] Training curves saved to {curves_path}", flush=True)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    save_figure(history, output_dir, cell_id)

    print("[train_cell] Done.", flush=True)


if __name__ == "__main__":
    main()
