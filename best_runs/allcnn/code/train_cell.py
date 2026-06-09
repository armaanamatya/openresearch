"""
Single-cell trainer — All-CNN paper (Springenberg et al. 2015)

One cell = one (arch, dataset, augment, lr, seed) configuration.
Reads cell parameters from REPROLAB_CELL_PARAMS env var (JSON).
Writes metrics.json to REPROLAB_CELL_OUTPUT_DIR.

Training recipe (Section 3.2):
  - SGD with fixed momentum 0.9
  - Total 350 epochs (full GPU run) or 5 epochs (CPU smoke)
  - LR schedule S=[200, 250, 300]: multiply γ by 0.1 at those epochs
  - Initial γ selected from {0.25, 0.1, 0.05, 0.01}
  - Weight decay λ=0.001 (CIFAR), λ=0.0005 (ImageNet)
  - Batch size 128 (CIFAR), 64 (ImageNet)
  - Dropout 20% on input, 50% after each pooling layer

ImageNet:
  - Not executable (manual download required); documented in scope.gaps.

Assumptions applied:
  A001: Initial LR γ=0.05 (middle of search set) unless lr_search=true
  A002: ZCA whitening may be skipped on CPU (use_zca=false) for speed
  A003: 350 epochs on GPU, 5 epochs on CPU
  A004: Batch size 128 for CIFAR on GPU, 64 on CPU
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def write_metrics(d: Dict, path: str):
    """Atomic write of metrics.json."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


def assert_lr_sane(lr: float):
    if not (1e-7 <= lr <= 1.0):
        raise ValueError(
            f"PREFLIGHT FAIL: lr={lr} outside sane range [1e-7, 1.0]. "
            "Check config — lr > 1.0 causes immediate NaN loss."
        )


def check_loss(loss_val: float, epoch: int, lr: float):
    if not math.isfinite(loss_val):
        raise RuntimeError(
            f"train_loss={loss_val} at epoch={epoch}, lr={lr} — aborting. "
            "Loss diverged. Reduce learning rate."
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(images)
        n += len(images)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(images)
    return {
        'loss': total_loss / max(n, 1),
        'accuracy': correct / max(n, 1),
        'error': 1.0 - correct / max(n, 1),
    }


# ---------------------------------------------------------------------------
# LR search — trains with all candidates, returns best
# ---------------------------------------------------------------------------

def lr_search(
    build_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    candidates: List[float],
    n_epochs_search: int = 5,
    momentum: float = 0.9,
    weight_decay: float = 0.001,
) -> float:
    """Train with each LR candidate for n_epochs_search epochs, return best."""
    print(f"[lr_search] Testing {candidates} over {n_epochs_search} epochs each ...", flush=True)
    best_lr = candidates[0]
    best_val_acc = -1.0
    for lr in candidates:
        model = build_fn().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                     momentum=momentum, weight_decay=weight_decay)
        for ep in range(n_epochs_search):
            train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"[lr_search] lr={lr:.4f} → val_acc={val_metrics['accuracy']:.4f}", flush=True)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_lr = lr
        del model, optimizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f"[lr_search] Best lr={best_lr:.4f} (val_acc={best_val_acc:.4f})", flush=True)
    return best_lr


# ---------------------------------------------------------------------------
# Main cell entry point
# ---------------------------------------------------------------------------

def run_cell(cell_params: Dict[str, Any], output_dir: str) -> Dict:
    # --- Parse cell config ---
    arch        = cell_params.get('arch', 'allcnn_c')
    dataset     = cell_params.get('dataset', 'cifar10')
    augment     = cell_params.get('augment', True)
    seed        = cell_params.get('seed', 42)
    lr          = float(cell_params.get('lr', 0.05))
    lr_search_  = bool(cell_params.get('lr_search', False))
    use_zca     = bool(cell_params.get('use_zca', False))  # expensive, skip for quick runs
    weight_decay = float(cell_params.get('weight_decay', 0.001))
    momentum    = float(cell_params.get('momentum', 0.9))
    # Canonical writable data root for this sandbox (shared across cells so CIFAR
    # is downloaded once).  Resolution order:
    #   1. explicit 'data_root' key in cell_params (set by train.py sequential path)
    #   2. DATA_ROOT env var (can be overridden by caller)
    #   3. OUTPUT_DIR/data if OUTPUT_DIR is set (harness top-level run)
    #   4. one level up from REPROLAB_CELL_OUTPUT_DIR → shared across all cells
    #   5. output_dir/data as final fallback (always a valid writable path)
    # NEVER default to /artifacts — that path may not exist or may be read-only.
    _cell_out = os.environ.get('REPROLAB_CELL_OUTPUT_DIR') or output_dir
    _default_data = (
        os.path.join(os.environ['OUTPUT_DIR'], 'data') if os.environ.get('OUTPUT_DIR')
        else os.path.normpath(os.path.join(_cell_out, '..', 'cifar_shared_data'))
    )
    data_root   = cell_params.get('data_root',
                    os.environ.get('DATA_ROOT', _default_data))

    # Compute detection
    HAS_GPU = torch.cuda.is_available()
    device = torch.device('cuda:0' if HAS_GPU else 'cpu')
    print(f"[cell] device={device} arch={arch} dataset={dataset} augment={augment} lr={lr}", flush=True)

    # Scale epochs and batch by device
    if HAS_GPU:
        n_epochs   = int(cell_params.get('n_epochs', 350))
        batch_size = int(cell_params.get('batch_size', 128))
    else:
        # CPU smoke: 5 epochs, smaller batch
        n_epochs   = int(cell_params.get('n_epochs_cpu', 5))
        batch_size = int(cell_params.get('batch_size_cpu', 64))
        use_zca    = False  # ZCA too slow for CPU smoke

    lr_schedule_epochs = [200, 250, 300]   # S from Section 3.2
    lr_gamma = 0.1                          # multiply LR by 0.1 at each milestone

    # Seeding
    torch.manual_seed(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

    assert_lr_sane(lr)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    # --- Dataset ---
    from data import prepare_cifar, get_cifar_loaders_simple

    max_train = None if HAS_GPU else int(cell_params.get('max_train_cpu', 5000))

    try:
        if use_zca:
            # Store ZCA cache in data_root (shared + writable) rather than output_dir/..
            zca_cache = os.path.join(data_root, f'zca_{dataset}.npy')
            train_ds, test_ds, num_classes = prepare_cifar(
                dataset, data_root, augment=augment,
                use_zca=True, zca_cache_path=zca_cache,
                max_train_samples=max_train,
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=HAS_GPU)
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                      num_workers=0, pin_memory=HAS_GPU)
        else:
            train_loader, test_loader, num_classes = get_cifar_loaders_simple(
                dataset, data_root, batch_size, augment=augment,
                max_train_samples=max_train, num_workers=0,
            )
    except Exception as e:
        print(f"[cell] Dataset load error: {e}", flush=True)
        traceback.print_exc()
        err_metrics = {
            'status': 'data_unavailable',
            'reason': f'{type(e).__name__}: {str(e)[:200]}',
        }
        write_metrics(err_metrics, os.path.join(output_dir, 'metrics.json'))
        return err_metrics

    # --- Model ---
    from models import build_cifar_model

    def build_fn():
        return build_cifar_model(arch, num_classes=num_classes)

    # LR search (optional — Section 3.2 "best γ from {0.25, 0.1, 0.05, 0.01}")
    if lr_search_ and HAS_GPU:
        # Use a validation split from training data for search
        from torch.utils.data import random_split
        base_ds = train_loader.dataset
        n_val = min(5000, len(base_ds) // 10)
        n_tr = len(base_ds) - n_val
        tr_split, val_split = random_split(base_ds, [n_tr, n_val],
                                           generator=torch.Generator().manual_seed(seed))
        val_loader_search = DataLoader(val_split, batch_size=batch_size, shuffle=False)
        tr_loader_search  = DataLoader(tr_split,  batch_size=batch_size, shuffle=True)
        criterion_search = nn.CrossEntropyLoss()
        lr = lr_search(
            build_fn,
            tr_loader_search, val_loader_search,
            criterion_search, device,
            candidates=[0.25, 0.1, 0.05, 0.01],
            n_epochs_search=10,
            momentum=momentum, weight_decay=weight_decay,
        )
        print(f"[cell] LR search selected: {lr:.4f}", flush=True)
        del tr_loader_search, val_loader_search

    # Build final model — BEFORE optimizer (device placement ordering rule)
    model = build_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                 momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_schedule_epochs, gamma=lr_gamma)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision on GPU
    scaler = torch.cuda.amp.GradScaler() if HAS_GPU and device.type == 'cuda' else None

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cell] Model {arch}: {param_count:,} parameters", flush=True)

    # --- Training ---
    training_curves = {'epoch': [], 'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    metrics_path = os.path.join(output_dir, 'metrics.json')
    t0 = time.time()

    # Eager initial write
    write_metrics({'status': 'running', 'arch': arch, 'dataset': dataset}, metrics_path)

    best_test_acc = 0.0
    best_test_err = 1.0

    for epoch in range(1, n_epochs + 1):
        ep_t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        check_loss(train_loss, epoch, lr)
        scheduler.step()

        # Evaluate every epoch
        test_metrics = evaluate(model, test_loader, criterion, device)
        ep_secs = time.time() - ep_t0

        training_curves['epoch'].append(epoch)
        training_curves['train_loss'].append(round(train_loss, 4))
        training_curves['test_loss'].append(round(test_metrics['loss'], 4))
        training_curves['test_accuracy'].append(round(test_metrics['accuracy'], 4))

        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_test_err = test_metrics['error']

        # Print every epoch for visibility
        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"[{arch}|{dataset}] epoch {epoch:3d}/{n_epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"test_err={test_metrics['error']:.4f} "
            f"lr={lr_now:.5f} ({ep_secs:.1f}s)",
            flush=True,
        )

        # Incremental atomic flush every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs:
            write_metrics({
                'status': 'running',
                'arch': arch,
                'dataset': dataset,
                'augmented': augment,
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'test_accuracy': round(test_metrics['accuracy'], 4),
                'test_error': round(test_metrics['error'], 4),
                'best_test_accuracy': round(best_test_acc, 4),
                'best_test_error': round(best_test_err, 4),
            }, metrics_path)

    wall_time = time.time() - t0
    print(f"[cell] Training complete: best_test_err={best_test_err:.4f} in {wall_time:.0f}s", flush=True)

    # Save training curves
    curves_path = os.path.join(output_dir, 'training_curves.json')
    with open(curves_path, 'w') as f:
        json.dump(training_curves, f, indent=2)

    # Final test evaluation
    final_test = evaluate(model, test_loader, criterion, device)

    # Config snapshot
    config_used = {
        'arch': arch,
        'dataset': dataset,
        'num_classes': num_classes,
        'augmented': augment,
        'use_zca': use_zca,
        'seed': seed,
        'lr': lr,
        'lr_search': lr_search_,
        'lr_schedule_epochs': lr_schedule_epochs,
        'lr_gamma': lr_gamma,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'param_count': param_count,
        'device': str(device),
        'has_gpu': HAS_GPU,
        'framework': f'torch=={torch.__version__}',
        'assumptions': ['A001', 'A002', 'A003', 'A004'],
    }
    with open(os.path.join(output_dir, 'config_used.json'), 'w') as f:
        json.dump(config_used, f, indent=2)

    # Guided backpropagation visualization
    try:
        from guided_backprop import visualize_class_saliency
        all_images, all_labels = [], []
        for imgs, lbls in test_loader:
            all_images.append(imgs)
            all_labels.append(lbls)
            if sum(len(x) for x in all_images) >= 8:
                break
        all_images = torch.cat(all_images)[:8]
        all_labels = torch.cat(all_labels)[:8]
        gb_results = visualize_class_saliency(
            model, all_images, all_labels,
            device=device, n_images=4, output_dir=output_dir,
        )
        print(f"[cell] Guided backprop complete: {gb_results['n_images']} images", flush=True)
    except Exception as e:
        print(f"[cell] Guided backprop skipped: {e}", flush=True)
        gb_results = {}

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = training_curves['epoch']
        ax1.plot(epochs, training_curves['train_loss'], label='train')
        ax1.plot(epochs, training_curves['test_loss'], label='test')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{arch} — {dataset} — Loss')
        ax1.legend()
        for m in lr_schedule_epochs:
            ax1.axvline(x=m, color='gray', linestyle='--', alpha=0.5)

        ax2.plot(epochs, training_curves['test_accuracy'], label='test accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{arch} — {dataset} — Test Accuracy')
        ax2.legend()
        for m in lr_schedule_epochs:
            ax2.axvline(x=m, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'fig_training_{arch}_{dataset}.png')
        plt.savefig(fig_path, dpi=100)
        plt.close()
        print(f"[cell] Saved training curves: {fig_path}", flush=True)
    except Exception as e:
        print(f"[cell] Plot skipped: {e}", flush=True)

    # Provenance
    try:
        from provenance import emit_provenance, emit_figure_sidecar
        emit_provenance(output_dir, experiments={
            arch: {
                'model_key': arch,
                'baseline': 'all_cnn',
                'seed': seed,
                'epochs': n_epochs,
                'batch_size': batch_size,
                'per_optimizer': {
                    'sgd': {
                        'lr': lr,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'lr_schedule_epochs': lr_schedule_epochs,
                        'lr_gamma': lr_gamma,
                    }
                },
                'hardware': str(device),
                'framework_versions': {'torch': torch.__version__},
                'convergence': {
                    'epoch': training_curves['epoch'],
                    'test_accuracy': training_curves['test_accuracy'],
                    'train_loss': training_curves['train_loss'],
                },
            }
        })
    except Exception as e:
        print(f"[cell] Provenance skipped: {e}", flush=True)

    # TERMINAL metrics
    final_metrics = {
        'status': 'ok',
        'arch': arch,
        'dataset': dataset,
        'augmented': augment,
        'test_accuracy': round(final_test['accuracy'], 4),
        'test_error': round(final_test['error'], 4),
        'test_loss': round(final_test['loss'], 4),
        'best_test_accuracy': round(best_test_acc, 4),
        'best_test_error': round(best_test_err, 4),
        'train_loss_final': round(training_curves['train_loss'][-1], 4) if training_curves['train_loss'] else None,
        'epochs_run': n_epochs,
        'lr': lr,
        'param_count': param_count,
        'wall_time_seconds': round(wall_time, 1),
        'steps_run': n_epochs * len(train_loader),
        'reward_mean': round(best_test_acc, 4),  # harness uses this as headline
        'metric': round(best_test_acc, 4),
    }
    write_metrics(final_metrics, metrics_path)
    print(f"[cell] Wrote final metrics: {metrics_path}", flush=True)
    return final_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Parse cell params
    cell_params_str = os.environ.get('REPROLAB_CELL_PARAMS', '')
    if cell_params_str:
        cell_params = json.loads(cell_params_str)
    else:
        # CLI fallback: accept --cell-id / --output-dir and a side-loaded cells.json
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cell-id', default=None)
        parser.add_argument('--output-dir', default=None)
        parser.add_argument('--cell-params', default='{}')
        args, _ = parser.parse_known_args()

        if args.cell_params and args.cell_params != '{}':
            cell_params = json.loads(args.cell_params)
        elif args.cell_id:
            # Load from cells.json
            cells_path = os.path.join(os.path.dirname(__file__), 'cells.json')
            with open(cells_path) as f:
                cells_data = json.load(f)
            cell_params = next(
                (c for c in cells_data['cells'] if c['id'] == args.cell_id), {}
            )
        else:
            cell_params = {}

    output_dir = os.environ.get('REPROLAB_CELL_OUTPUT_DIR') or \
                 os.environ.get('OUTPUT_DIR', '/artifacts')
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        output_dir = sys.argv[idx + 1]

    os.makedirs(output_dir, exist_ok=True)

    metrics = run_cell(cell_params, output_dir)
    print(f"[cell] Done. metrics={json.dumps(metrics, indent=2)}", flush=True)


if __name__ == '__main__':
    main()
