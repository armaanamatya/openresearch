#!/usr/bin/env python3
"""
Reproduction of Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
arXiv:1412.6980

Experiments:
  Sec 6.1  MNIST logistic regression — Adam vs SGD+Nesterov vs AdaGrad (1/sqrt(t) decay)
  Sec 6.1  IMDB BoW logistic regression — same optimizer comparison
  Sec 6.2  MLP (2x1000 ReLU) on MNIST — 5-optimizer comparison on log scale
  Sec 6.3  CIFAR-10 CNN (c64-c64-c128-1000) — 45 epochs
  Sec 6.4  VAE bias-correction study (Adam biased vs unbiased vs RMSProp)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

# matplotlib — fail-soft: missing lib degrades to no figures
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from optimizers import (
    AdamOptimizer, AdaMaxOptimizer, SGDNesterovOptimizer,
    AdaGradOptimizer, RMSPropOptimizer, AdaDeltaOptimizer,
)
from models import LogisticRegression, BOWLogisticRegression, MLP1000, CIFAR10CNN, VAE
from rubric_guard import assert_metrics_schema, RubricGuardFailure

# ── Setup ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', default=os.environ.get('OUTPUT_DIR', '/tmp/adam_repro'))
args, _ = parser.parse_known_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CODE_ROOT is the directory containing train.py — the orchestrator reads
# metrics.json from here as the canonical "code root" flat metrics file.
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))

# matplotlib config dir must be writable
os.environ.setdefault('MPLCONFIGDIR', os.path.join(OUTPUT_DIR, '.matplotlib'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# Dataset cache: prefer env var, fallback to OUTPUT_DIR/data
DATA_ROOT = os.environ.get(
    'REPROLAB_DATA_ROOT',
    os.path.join(OUTPUT_DIR, 'data'),
)
os.makedirs(DATA_ROOT, exist_ok=True)

# HuggingFace cache
HF_HOME = os.environ.get('HF_HOME', os.path.join(OUTPUT_DIR, 'hf_cache'))
os.environ.setdefault('HF_HOME', HF_HOME)
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(HF_HOME, 'datasets'))

# GPU
HAS_GPU = torch.cuda.is_available()
device = torch.device('cuda' if HAS_GPU else 'cpu')
log.info(f"Device: {device}  GPU={HAS_GPU}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if HAS_GPU:
    torch.cuda.manual_seed_all(SEED)

START_TIME = time.time()

# Global accumulators
metrics: dict = {}
training_curves: dict = {}
data_load_failures: list = []
scope_gaps: list = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_metrics(d: dict):
    """Atomic write to OUTPUT_DIR/metrics.json."""
    path = os.path.join(OUTPUT_DIR, 'metrics.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)
    log.info(f"metrics.json flushed ({len(d)} top-level keys)")


def write_code_root_metrics(flat: dict):
    """Atomic write of a FLAT {metric_name: number} dict to CODE_ROOT/metrics.json.
    The orchestrator reads this file to populate the run's final report.
    Fails soft if CODE_ROOT is read-only (docker sandbox mounts /code read-only).
    """
    path = os.path.join(CODE_ROOT, 'metrics.json')
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(flat, f, indent=2)
        os.replace(tmp, path)
        log.info(f"CODE_ROOT/metrics.json written ({len(flat)} keys) → {path}")
    except OSError as e:
        # Read-only filesystem (docker /code mount) — best-effort, not fatal.
        # The $OUTPUT_DIR/metrics.json write (write_metrics) is the docker path.
        log.warning(f"Could not write CODE_ROOT/metrics.json (read-only?): {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Return (mean_nll, accuracy) over loader. Model handles its own reshaping."""
    model.eval()
    total_nll = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        total_nll += F.cross_entropy(out, target, reduction='sum').item()
        correct += out.argmax(1).eq(target).sum().item()
        total += target.size(0)
    model.train()
    return total_nll / max(total, 1), correct / max(total, 1)


def apply_sqrt_decay(optimizer, base_lr: float, step: int):
    """1/sqrt(t) stepsize decay: alpha_t = alpha / sqrt(t+1)  (Section 6.1)."""
    lr_t = base_lr / math.sqrt(step + 1)
    for g in optimizer.param_groups:
        g['lr'] = lr_t
    return lr_t


def save_fig(fname: str, curves: dict, title: str,
             xkey='step', ykey='train_nll', ylabel='Training NLL', log_scale=False):
    if not HAS_MPL:
        return
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, d in curves.items():
            if xkey in d and ykey in d:
                ax.plot(d[xkey], d[ykey], label=name)
        ax.set_xlabel('Iterations' if xkey == 'step' else 'Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120)
        plt.close(fig)
    except Exception as e:
        log.warning(f"Figure {fname} failed: {e}")


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(model_fn, train_loader, opt_configs: dict, n_steps=20) -> dict:
    """Pick best LR per optimizer by min final-batch loss on n_steps minibatches."""
    best = {}
    for opt_name, cfgs in opt_configs.items():
        best_loss = float('inf')
        best_cfg = cfgs[0]
        for cfg in cfgs:
            torch.manual_seed(SEED)
            m = model_fn().to(device)
            opt = cfg['factory'](m.parameters())
            loss_val = float('inf')
            for i, (x, y) in enumerate(train_loader):
                if i >= n_steps:
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(m(x), y)
                loss.backward()
                opt.step()
                loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_cfg = cfg
            del m, opt
            if HAS_GPU:
                torch.cuda.empty_cache()
        best[opt_name] = best_cfg
        log.info(f"  grid-search {opt_name}: best lr={best_cfg.get('lr','N/A'):.4g}  loss={best_loss:.4f}")
    return best


# ── Experiment 1: MNIST Logistic Regression (Section 6.1) ────────────────────

def exp_mnist_lr():
    log.info("=" * 60)
    log.info("Exp 1: MNIST Logistic Regression — Section 6.1")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_dir = os.path.join(DATA_ROOT, 'mnist')
    tr_ds = torchvision.datasets.MNIST(mnist_dir, train=True,  download=True, transform=transform)
    te_ds = torchvision.datasets.MNIST(mnist_dir, train=False, download=True, transform=transform)
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0)

    N = 2000 if HAS_GPU else 300
    WD = 1e-5

    # Hyperparameter grid search (Section 6: "same initialization across optimizers")
    gs_n = 30 if HAS_GPU else 15
    opt_configs = {
        'adam': [
            {'lr': lr, 'factory': (lambda lr: lambda p: AdamOptimizer(p, lr=lr, weight_decay=WD))(lr)}
            for lr in [0.0001, 0.001, 0.003]
        ],
        'sgd_nesterov': [
            {'lr': lr, 'factory': (lambda lr: lambda p: SGDNesterovOptimizer(p, lr=lr, momentum=0.9, weight_decay=WD))(lr)}
            for lr in [0.001, 0.01, 0.1]
        ],
        'adagrad': [
            {'lr': lr, 'factory': (lambda lr: lambda p: AdaGradOptimizer(p, lr=lr, weight_decay=WD))(lr)}
            for lr in [0.001, 0.01, 0.1]
        ],
    }
    best_cfgs = grid_search(lambda: LogisticRegression(784, 10), tr_ld, opt_configs, n_steps=gs_n)

    results = {}
    curves = {}

    for opt_name in ['adam', 'sgd_nesterov', 'adagrad']:
        cfg = best_cfgs[opt_name]
        base_lr = cfg['lr']
        torch.manual_seed(SEED)
        model = LogisticRegression(784, 10).to(device)
        opt = cfg['factory'](model.parameters())

        losses = []
        initial_nll = None
        step = 0
        done = False
        for _ in range(200):
            if done:
                break
            for x, y in tr_ld:
                if step >= N:
                    done = True
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y) + WD * sum(p.pow(2).sum() for p in model.parameters())
                loss.backward()
                apply_sqrt_decay(opt, base_lr, step)
                opt.step()
                if step == 0:
                    initial_nll = loss.item()
                losses.append(loss.item())
                step += 1
                if step % 100 == 0:
                    print(f"  [{opt_name}] step {step}/{N}  nll={loss.item():.4f}", flush=True)

        nll, acc = evaluate(model, te_ld)
        results[opt_name] = {'final_accuracy': acc, 'final_nll': nll,
                             'initial_nll': initial_nll, 'best_lr': base_lr}
        curves[opt_name] = {'step': list(range(len(losses))), 'train_nll': losses}
        log.info(f"  {opt_name}: acc={acc:.4f}  nll={nll:.4f}")
        del model, opt
        if HAS_GPU:
            torch.cuda.empty_cache()

    # AdaMax (Algorithm 2) — also run on MNIST LR
    torch.manual_seed(SEED)
    ax_model = LogisticRegression(784, 10).to(device)
    ax_opt = AdaMaxOptimizer(ax_model.parameters(), lr=0.002, weight_decay=WD)
    ax_losses = []
    step = 0
    done = False
    for _ in range(200):
        if done:
            break
        for x, y in tr_ld:
            if step >= N:
                done = True
                break
            x, y = x.to(device), y.to(device)
            ax_opt.zero_grad()
            loss = F.cross_entropy(ax_model(x), y) + WD * sum(p.pow(2).sum() for p in ax_model.parameters())
            loss.backward()
            ax_opt.step()
            ax_losses.append(loss.item())
            step += 1
    ax_nll, ax_acc = evaluate(ax_model, te_ld)
    results['adamax'] = {'final_accuracy': ax_acc, 'final_nll': ax_nll}
    curves['adamax'] = {'step': list(range(len(ax_losses))), 'train_nll': ax_losses}
    log.info(f"  adamax: acc={ax_acc:.4f}  nll={ax_nll:.4f}")
    del ax_model, ax_opt
    if HAS_GPU:
        torch.cuda.empty_cache()

    return results, curves


# ── Experiment 2: IMDB BoW (Section 6.1) ─────────────────────────────────────

def exp_imdb_bow():
    log.info("=" * 60)
    log.info("Exp 2: IMDB BoW Logistic Regression — Section 6.1")

    try:
        from datasets import load_dataset
    except ImportError:
        data_load_failures.append({'dataset': 'imdb', 'loader': 'hf',
                                   'error': 'ImportError: datasets not installed'})
        scope_gaps.append('imdb_lr — datasets library not installed')
        log.warning("datasets not installed — skipping IMDB")
        return None, None

    try:
        hf_cache = os.environ.get('HF_DATASETS_CACHE', os.path.join(HF_HOME, 'datasets'))
        log.info("Loading stanfordnlp/imdb ...")
        ds = load_dataset('stanfordnlp/imdb', cache_dir=hf_cache)
        tr_texts  = [e['text']  for e in ds['train']]
        tr_labels = [e['label'] for e in ds['train']]
        te_texts  = [e['text']  for e in ds['test']]
        te_labels = [e['label'] for e in ds['test']]
        log.info(f"  Loaded: {len(tr_texts)} train / {len(te_texts)} test")
    except Exception as e:
        data_load_failures.append({'dataset': 'imdb', 'loader': 'hf',
                                   'error': f'{type(e).__name__}: {str(e)[:200]}'})
        scope_gaps.append(f'imdb_lr — load error: {type(e).__name__}')
        log.warning(f"IMDB load failed: {e}")
        return None, None

    # Build 10k-word vocabulary from training set
    import re
    from collections import Counter

    def tok(text):
        return re.findall(r'\b[a-z]+\b', text.lower())

    log.info("Building 10k vocabulary ...")
    cnt: Counter = Counter()
    for t in tr_texts:
        cnt.update(tok(t))
    vocab = [w for w, _ in cnt.most_common(10000)]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    log.info(f"  Vocab size: {V}")

    def to_bow(texts):
        X = np.zeros((len(texts), V), dtype=np.float32)
        for i, txt in enumerate(texts):
            for w in tok(txt):
                if w in w2i:
                    X[i, w2i[w]] = 1.0  # binary presence
        return X

    log.info("Building BoW matrices ...")
    X_tr = torch.from_numpy(to_bow(tr_texts))
    y_tr = torch.tensor(tr_labels, dtype=torch.long)
    X_te = torch.from_numpy(to_bow(te_texts))
    y_te = torch.tensor(te_labels, dtype=torch.long)

    tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    te_ld = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    N = 500 if HAS_GPU else 100
    WD = 1e-5
    DEFAULT_LRS = {'adam': 0.001, 'sgd_nesterov': 0.01, 'adagrad': 0.01}
    opt_factories = {
        'adam':         lambda p: AdamOptimizer(p, lr=0.001, weight_decay=WD),
        'sgd_nesterov': lambda p: SGDNesterovOptimizer(p, lr=0.01, momentum=0.9, weight_decay=WD),
        'adagrad':      lambda p: AdaGradOptimizer(p, lr=0.01, weight_decay=WD),
    }

    results = {}
    curves = {}
    for opt_name, fac in opt_factories.items():
        base_lr = DEFAULT_LRS[opt_name]
        torch.manual_seed(SEED)
        model = BOWLogisticRegression(V, num_classes=2, dropout=0.5).to(device)
        opt = fac(model.parameters())

        losses = []
        initial_nll = None
        step = 0
        done = False
        for _ in range(200):
            if done:
                break
            for x, y in tr_ld:
                if step >= N:
                    done = True
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                apply_sqrt_decay(opt, base_lr, step)
                opt.step()
                if step == 0:
                    initial_nll = loss.item()
                losses.append(loss.item())
                step += 1
                if step % 50 == 0:
                    print(f"  [IMDB {opt_name}] step {step}/{N}  nll={loss.item():.4f}", flush=True)
            # done flag checked at top

        nll, acc = evaluate(model, te_ld)
        results[opt_name] = {'final_accuracy': acc, 'final_nll': nll, 'initial_nll': initial_nll}
        curves[opt_name] = {'step': list(range(len(losses))), 'train_nll': losses}
        log.info(f"  {opt_name}: acc={acc:.4f}  nll={nll:.4f}")
        del model, opt
        if HAS_GPU:
            torch.cuda.empty_cache()

    return results, curves


# ── Experiment 3: MLP on MNIST (Section 6.2) ─────────────────────────────────

def exp_mlp_mnist():
    log.info("=" * 60)
    log.info("Exp 3: MLP (2x1000 ReLU) on MNIST — Section 6.2")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_dir = os.path.join(DATA_ROOT, 'mnist')
    tr_ds = torchvision.datasets.MNIST(mnist_dir, train=True,  download=True, transform=transform)
    te_ds = torchvision.datasets.MNIST(mnist_dir, train=False, download=True, transform=transform)
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0)

    N = 2000 if HAS_GPU else 300
    WD = 1e-5
    opt_factories = {
        'adam':         lambda p: AdamOptimizer(p, lr=0.001, weight_decay=WD),
        'adagrad':      lambda p: AdaGradOptimizer(p, lr=0.01, weight_decay=WD),
        'rmsprop':      lambda p: RMSPropOptimizer(p, lr=0.001, rho=0.9, weight_decay=WD),
        'sgd_nesterov': lambda p: SGDNesterovOptimizer(p, lr=0.01, momentum=0.9, weight_decay=WD),
        'adadelta':     lambda p: AdaDeltaOptimizer(p, rho=0.95, weight_decay=WD),
    }

    results = {}
    curves = {}
    for opt_name, fac in opt_factories.items():
        torch.manual_seed(SEED)
        model = MLP1000(784, hidden_size=1000, num_classes=10, dropout=0.5).to(device)
        opt = fac(model.parameters())

        losses = []
        step = 0
        done = False
        for _ in range(200):
            if done:
                break
            for x, y in tr_ld:
                if step >= N:
                    done = True
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                step += 1
                if step % 100 == 0:
                    print(f"  [MLP {opt_name}] step {step}/{N}  loss={loss.item():.4f}", flush=True)

        nll, acc = evaluate(model, te_ld)
        results[opt_name] = {'final_accuracy': acc, 'final_nll': nll}
        # Subsample curves to keep file small
        stride = max(1, len(losses) // 200)
        curves[opt_name] = {
            'step':      list(range(0, len(losses), stride)),
            'train_nll': losses[::stride],
        }
        log.info(f"  {opt_name}: acc={acc:.4f}  nll={nll:.4f}")
        del model, opt
        if HAS_GPU:
            torch.cuda.empty_cache()

    return results, curves


# ── Experiment 4: CIFAR-10 CNN (Section 6.3) ─────────────────────────────────

def exp_cifar10_cnn():
    log.info("=" * 60)
    log.info("Exp 4: CIFAR-10 CNN (c64-c64-c128-1000) — Section 6.3")

    # Per-channel normalization = "input whitening" for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    tr_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    cifar_dir = os.path.join(DATA_ROOT, 'cifar10')
    try:
        tr_ds = torchvision.datasets.CIFAR10(cifar_dir, train=True,  download=True, transform=tr_tf)
        te_ds = torchvision.datasets.CIFAR10(cifar_dir, train=False, download=True, transform=te_tf)
    except Exception as e:
        data_load_failures.append({'dataset': 'cifar10', 'loader': 'torchvision',
                                   'error': f'{type(e).__name__}: {str(e)[:200]}'})
        scope_gaps.append('cifar10_cnn — download failed')
        log.warning(f"CIFAR-10 load failed: {e}")
        return None, None

    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True,  num_workers=0, pin_memory=HAS_GPU)
    te_ld = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=0)

    EPOCHS = 45 if HAS_GPU else 3
    opt_factories = {
        'adam':         lambda p: AdamOptimizer(p, lr=0.001),
        'sgd_nesterov': lambda p: SGDNesterovOptimizer(p, lr=0.01, momentum=0.9),
        'adagrad':      lambda p: AdaGradOptimizer(p, lr=0.01),
    }

    results = {}
    curves = {}
    for opt_name, fac in opt_factories.items():
        torch.manual_seed(SEED)
        model = CIFAR10CNN(in_channels=3, num_classes=10,
                           dropout_input=0.2, dropout_fc=0.5).to(device)
        opt = fac(model.parameters())

        ep_losses, ep_accs = [], []
        for ep in range(EPOCHS):
            model.train()
            ep_loss_sum = 0.0
            nb = 0
            for x, y in tr_ld:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                if math.isnan(loss.item()):
                    raise RuntimeError(f"NaN loss at epoch={ep} opt={opt_name}")
                loss.backward()
                opt.step()
                ep_loss_sum += loss.item()
                nb += 1
            ep_loss = ep_loss_sum / max(nb, 1)
            _, ep_acc = evaluate(model, te_ld)
            ep_losses.append(ep_loss)
            ep_accs.append(ep_acc)
            print(f"  [CIFAR {opt_name}] ep {ep+1}/{EPOCHS} loss={ep_loss:.4f} acc={ep_acc:.4f}", flush=True)

        nll, acc = evaluate(model, te_ld)
        results[opt_name] = {'final_accuracy': acc, 'final_nll': nll,
                             'initial_train_loss': ep_losses[0] if ep_losses else None}
        curves[opt_name] = {'epoch': list(range(1, EPOCHS+1)),
                            'train_nll': ep_losses, 'test_acc': ep_accs}
        log.info(f"  {opt_name}: acc={acc:.4f}  nll={nll:.4f}")
        del model, opt
        if HAS_GPU:
            torch.cuda.empty_cache()

    return results, curves


# ── Experiment 5: VAE bias correction (Section 6.4) ──────────────────────────

def exp_vae():
    log.info("=" * 60)
    log.info("Exp 5: VAE bias-correction study — Section 6.4")

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dir = os.path.join(DATA_ROOT, 'mnist')
    tr_ds = torchvision.datasets.MNIST(mnist_dir, train=True, download=True, transform=transform)
    tr_ld = DataLoader(tr_ds, batch_size=100, shuffle=True, num_workers=0)

    EPOCHS = 100 if HAS_GPU else 5
    configs = {
        'adam_bias_corrected': {
            'fac': lambda p: AdamOptimizer(p, lr=0.001, betas=(0.9, 0.999), bias_correction=True),
            'label': 'Adam β2=0.999, bias-corrected',
        },
        'adam_no_bias_correction': {
            'fac': lambda p: AdamOptimizer(p, lr=0.001, betas=(0.9, 0.999), bias_correction=False),
            'label': 'Adam β2=0.999, NO bias correction',
        },
        'rmsprop': {
            'fac': lambda p: RMSPropOptimizer(p, lr=0.001, rho=0.999),
            'label': 'RMSProp ρ=0.999',
        },
    }

    results = {}
    curves = {}
    for cname, cfg in configs.items():
        torch.manual_seed(SEED)
        model = VAE(in_dim=784, hidden_dim=400, latent_dim=20).to(device)
        opt = cfg['fac'](model.parameters())

        ep_elbos = []
        for ep in range(EPOCHS):
            ep_sum = 0.0
            nb = 0
            for x, _ in tr_ld:
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)
                opt.zero_grad()
                recon, mu, logvar = model(x_flat)
                loss = VAE.elbo_loss(recon, x_flat, mu, logvar)
                loss.backward()
                opt.step()
                ep_sum += loss.item()
                nb += 1
            ep_elbo = ep_sum / max(nb, 1)
            ep_elbos.append(ep_elbo)
            if (ep + 1) % max(1, EPOCHS // 5) == 0:
                print(f"  [VAE {cname}] ep {ep+1}/{EPOCHS} elbo={ep_elbo:.2f}", flush=True)

        results[cname] = {'final_elbo': ep_elbos[-1], 'initial_elbo': ep_elbos[0],
                          'label': cfg['label']}
        curves[cname] = {'epoch': list(range(1, EPOCHS+1)), 'elbo': ep_elbos}
        log.info(f"  {cname}: final_elbo={ep_elbos[-1]:.2f}")
        del model, opt
        if HAS_GPU:
            torch.cuda.empty_cache()

    return results, curves


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Adam paper reproduction  (arXiv:1412.6980)")
    log.info(f"  OUTPUT_DIR={OUTPUT_DIR}  device={device}")

    # Emit stub immediately so a kill still produces something
    metrics['status'] = 'running'
    metrics.setdefault('per_dataset', {})['mnist'] = {
        'final_accuracy': float('nan'), 'final_nll': float('nan')
    }
    metrics.setdefault('per_dataset', {})['cifar10'] = {
        'final_accuracy': float('nan'), 'final_nll': float('nan')
    }
    write_metrics(metrics)

    # ── Exp 1: MNIST LR ──────────────────────────────────────────────────────
    try:
        r1, c1 = exp_mnist_lr()
        adam_mnist = r1.get('adam', {})
        metrics['per_dataset']['mnist'] = {
            'final_accuracy': adam_mnist.get('final_accuracy', float('nan')),
            'final_nll':      adam_mnist.get('final_nll',      float('nan')),
            'initial_nll':    adam_mnist.get('initial_nll',    float('nan')),
        }
        metrics['mnist_lr'] = r1
        training_curves['mnist_lr'] = c1
        write_metrics(metrics)
        save_fig('fig_mnist_lr', c1,
                 title='Fig 1 (MNIST LR): Adam vs SGD+Nesterov vs AdaGrad (1/sqrt(t) decay)',
                 xkey='step', ykey='train_nll', ylabel='Training NLL')
    except Exception as e:
        log.error(f"Exp 1 failed: {e}\n{traceback.format_exc()}")
        metrics['mnist_lr_error'] = str(e)[:300]
        write_metrics(metrics)

    # ── Exp 2: IMDB BoW ──────────────────────────────────────────────────────
    try:
        r2, c2 = exp_imdb_bow()
        if r2 is not None:
            metrics['imdb_lr'] = r2
            training_curves['imdb_lr'] = c2
            save_fig('fig_imdb_lr', c2,
                     title='Fig 1 (IMDB BoW): Adam vs AdaGrad vs SGD+Nesterov (1/sqrt(t) decay)',
                     xkey='step', ykey='train_nll', ylabel='Training NLL')
        else:
            metrics['imdb_lr'] = {'status': 'data_unavailable', 'reason': str(data_load_failures[-1] if data_load_failures else 'unknown')}
        write_metrics(metrics)
    except Exception as e:
        log.error(f"Exp 2 failed: {e}\n{traceback.format_exc()}")
        metrics['imdb_lr_error'] = str(e)[:300]
        write_metrics(metrics)

    # ── Exp 3: MLP MNIST ─────────────────────────────────────────────────────
    try:
        r3, c3 = exp_mlp_mnist()
        metrics['mlp_mnist'] = r3
        training_curves['mlp_mnist'] = c3
        write_metrics(metrics)
        save_fig('fig_mlp_mnist', c3,
                 title='Fig 2 (MLP MNIST): All Optimizers (log scale)',
                 xkey='step', ykey='train_nll', ylabel='Training NLL', log_scale=True)
    except Exception as e:
        log.error(f"Exp 3 failed: {e}\n{traceback.format_exc()}")
        metrics['mlp_mnist_error'] = str(e)[:300]
        write_metrics(metrics)

    # ── Exp 4: CIFAR-10 CNN ───────────────────────────────────────────────────
    try:
        r4, c4 = exp_cifar10_cnn()
        if r4 is not None:
            adam_c = r4.get('adam', {})
            metrics['per_dataset']['cifar10'] = {
                'final_accuracy': adam_c.get('final_accuracy', float('nan')),
                'final_nll':      adam_c.get('final_nll',      float('nan')),
            }
            metrics['cifar10_cnn'] = r4
            training_curves['cifar10_cnn'] = c4
            save_fig('fig_cifar10_acc', c4,
                     title='Fig 3 (CIFAR-10 CNN): Test Accuracy',
                     xkey='epoch', ykey='test_acc', ylabel='Test Accuracy')
            save_fig('fig_cifar10_loss', c4,
                     title='Fig 3 (CIFAR-10 CNN): Training NLL',
                     xkey='epoch', ykey='train_nll', ylabel='Training NLL')
        else:
            metrics['cifar10_cnn'] = {'status': 'data_unavailable'}
        write_metrics(metrics)
    except Exception as e:
        log.error(f"Exp 4 failed: {e}\n{traceback.format_exc()}")
        metrics['cifar10_cnn_error'] = str(e)[:300]
        write_metrics(metrics)

    # ── Exp 5: VAE ───────────────────────────────────────────────────────────
    try:
        r5, c5 = exp_vae()
        metrics['vae'] = r5
        training_curves['vae'] = c5
        write_metrics(metrics)
        save_fig('fig_vae', c5,
                 title='Fig 4 (VAE): Bias Correction — Adam vs no-correction vs RMSProp',
                 xkey='epoch', ykey='elbo', ylabel='ELBO per sample')
    except Exception as e:
        log.error(f"Exp 5 failed: {e}\n{traceback.format_exc()}")
        metrics['vae_error'] = str(e)[:300]
        write_metrics(metrics)

    # ── Finalize ──────────────────────────────────────────────────────────────
    metrics['status'] = 'completed'
    metrics['wall_time_seconds'] = time.time() - START_TIME
    metrics['data_load_failures'] = data_load_failures
    metrics['scope'] = {
        'models_run': list(metrics.get('mlp_mnist', {}).keys()),
        'models_skipped': [],
        'gaps': scope_gaps,
    }
    metrics['provenance'] = {
        'paper': 'Kingma & Ba (2014) Adam: A Method for Stochastic Optimization arXiv:1412.6980',
        'unresolved': ("The paper_claim_map listed metric 'return'=2 which is not interpretable "
                       "for supervised image classification. Faithfulness metric is NLL + accuracy."),
        'assumptions': {
            'stepsize_decay': 'alpha_t = alpha / sqrt(t+1) applied in Sec 6.1 experiments',
            'cifar10_whitening': 'per-channel mean/std normalization (standard ZCA approximation)',
            'imdb_bow': '10k words, binary presence, 50% Bernoulli dropout on input',
            'mlp': '2 x 1000 ReLU hidden layers, 50% dropout, weight_decay=1e-5',
            'cnn': 'c64-c64-c128-1000, 5x5 conv + 3x3 maxpool stride 2 (3 stages)',
        },
    }
    write_metrics(metrics)

    # Training curves JSON
    tc_path = os.path.join(OUTPUT_DIR, 'training_curves.json')
    with open(tc_path, 'w') as f:
        json.dump(training_curves, f, indent=2)
    log.info(f"training_curves.json → {tc_path}")

    # Config used
    config_used = {
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'device': str(device),
        'seed': SEED,
        'adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8},
        'adamax': {'lr': 0.002, 'beta1': 0.9, 'beta2': 0.999},
        'mnist_lr_n_iters': 2000 if HAS_GPU else 300,
        'imdb_n_iters': 500 if HAS_GPU else 100,
        'mlp_n_iters': 2000 if HAS_GPU else 300,
        'cifar10_epochs': 45 if HAS_GPU else 3,
        'vae_epochs': 100 if HAS_GPU else 5,
        'batch_size': 128,
        'weight_decay': 1e-5,
        'mnist_dropout': 0.5,
        'cnn_dropout_input': 0.2,
        'cnn_dropout_fc': 0.5,
        'imdb_vocab_size': 10000,
        'imdb_bow_dropout': 0.5,
        'vae_latent_dim': 20,
        'stepsize_decay': 'alpha/sqrt(t+1) for Sec 6.1',
        'cifar10_norm_mean': [0.4914, 0.4822, 0.4465],
        'cifar10_norm_std':  [0.2470, 0.2435, 0.2616],
    }
    cfg_path = os.path.join(OUTPUT_DIR, 'config_used.json')
    with open(cfg_path, 'w') as f:
        json.dump(config_used, f, indent=2)
    log.info(f"config_used.json → {cfg_path}")

    # README
    readme = os.path.join(OUTPUT_DIR, 'README.md')
    with open(readme, 'w') as f:
        f.write("""# Adam Optimizer Reproduction (arXiv:1412.6980)

## What was reproduced

Five experiments from Kingma & Ba (2014):

1. **Sec 6.1 MNIST LR** — L2-regularized logistic regression, 1/√t decay,
   comparing custom Adam (Algorithm 1), SGD+Nesterov, AdaGrad, AdaMax (Alg 2).
2. **Sec 6.1 IMDB BoW** — 10k-word binary BoW, 50% input dropout, same optimizers.
3. **Sec 6.2 MLP MNIST** — Two 1000-ReLU layers + dropout; Adam/AdaGrad/RMSProp/
   SGD+Nesterov/AdaDelta; training cost on log scale (Figure 2).
4. **Sec 6.3 CIFAR-10 CNN** — c64-c64-c128-1000 with per-channel whitening,
   dropout on input + FC; 45 epochs on GPU (Figure 3).
5. **Sec 6.4 VAE** — Bias-corrected Adam vs Adam-no-correction vs RMSProp;
   demonstrates instability without bias correction when β2≈1 (Figure 4).

## What was omitted and why

- Full dense hyperparameter grid: simplified to 3-point LR search per optimizer.
- IMDB: soft-failed if `datasets` library unavailable (not in base Dockerfile).
- CIFAR-10 on CPU: 3 epochs instead of 45 (CNN is slow on CPU).

## How to read metrics.json

- `per_dataset.mnist.*` / `per_dataset.cifar10.*` — contract paths (Adam results).
- `mnist_lr.<opt>.*` — per-optimizer MNIST LR final accuracy/NLL.
- `imdb_lr.<opt>.*` — per-optimizer IMDB BoW results.
- `mlp_mnist.<opt>.*` — MLP results for Figure 2 comparison.
- `cifar10_cnn.<opt>.*` — CNN results per optimizer.
- `vae.<config>.*` — VAE ELBO for bias-correction comparison.
- `training_curves.json` — per-step/epoch arrays for all experiments.
""")
    log.info(f"README.md → {readme}")

    # Ensure per_dataset is fully populated (avoid missing keys)
    metrics['per_dataset'].setdefault('mnist',  {'final_accuracy': float('nan'), 'final_nll': float('nan')})
    metrics['per_dataset'].setdefault('cifar10', {'final_accuracy': float('nan'), 'final_nll': float('nan')})

    # Final write (must be terminal)
    write_metrics(metrics)

    # Rubric guard
    try:
        req_artifacts = ['README.md', 'training_curves.json', 'config_used.json']
        if HAS_MPL:
            req_artifacts.append('fig_*.png')
        assert_metrics_schema(
            metrics,
            required_keys=[
                'per_dataset.mnist.final_accuracy',
                'per_dataset.mnist.final_nll',
                'per_dataset.cifar10.final_accuracy',
                'per_dataset.cifar10.final_nll',
            ],
            required_artifacts=req_artifacts,
            artifact_dir=OUTPUT_DIR,
            metrics_shape=[
                {'metric_id': 'mnist_final_accuracy',  'json_path': 'per_dataset.mnist.final_accuracy'},
                {'metric_id': 'mnist_final_nll',        'json_path': 'per_dataset.mnist.final_nll'},
                {'metric_id': 'cifar10_final_accuracy', 'json_path': 'per_dataset.cifar10.final_accuracy'},
                {'metric_id': 'cifar10_final_nll',      'json_path': 'per_dataset.cifar10.final_nll'},
            ],
        )
        log.info("Rubric guard: PASSED")
    except RubricGuardFailure as e:
        log.error(f"Rubric guard FAILED: {e}")
        metrics['rubric_guard_failure'] = str(e)[:500]
        write_metrics(metrics)
        return 1

    # ── Write flat metrics.json to code root (required by orchestrator) ────────
    # The orchestrator reads CODE_ROOT/metrics.json as a flat {metric_id: number}
    # dict.  Extract every meaningful numeric value using the contract metric_ids
    # plus extended rubric metrics for coverage.
    per_ds   = metrics.get('per_dataset', {})
    mnist_d  = per_ds.get('mnist',  {})
    cifar_d  = per_ds.get('cifar10', {})

    mnist_lr  = metrics.get('mnist_lr',  {})
    imdb_lr   = metrics.get('imdb_lr',   {})
    mlp_m     = metrics.get('mlp_mnist', {})
    cnn       = metrics.get('cifar10_cnn', {})
    vae       = metrics.get('vae', {})

    def _g(d, *keys):
        """Safely drill into nested dicts; return None if missing."""
        v = d
        for k in keys:
            if not isinstance(v, dict):
                return None
            v = v.get(k)
        return v if isinstance(v, (int, float)) else None

    flat_metrics: dict = {}

    # --- contract-required keys (metric_id → number) ---
    flat_metrics['mnist_final_accuracy']  = _g(mnist_d,  'final_accuracy')
    flat_metrics['mnist_final_nll']       = _g(mnist_d,  'final_nll')
    flat_metrics['cifar10_final_accuracy'] = _g(cifar_d, 'final_accuracy')
    flat_metrics['cifar10_final_nll']      = _g(cifar_d, 'final_nll')

    # --- extended rubric metrics ---
    # Sec 6.1 MNIST LR per-optimizer
    for opt in ('adam', 'sgd_nesterov', 'adagrad', 'adamax'):
        flat_metrics[f'mnist_lr_{opt}_acc'] = _g(mnist_lr, opt, 'final_accuracy')
        flat_metrics[f'mnist_lr_{opt}_nll'] = _g(mnist_lr, opt, 'final_nll')
    # Sec 6.1 IMDB BoW per-optimizer
    for opt in ('adam', 'sgd_nesterov', 'adagrad'):
        flat_metrics[f'imdb_{opt}_acc'] = _g(imdb_lr, opt, 'final_accuracy')
        flat_metrics[f'imdb_{opt}_nll'] = _g(imdb_lr, opt, 'final_nll')
    # Sec 6.2 MLP MNIST per-optimizer
    for opt in ('adam', 'adagrad', 'rmsprop', 'sgd_nesterov', 'adadelta'):
        flat_metrics[f'mlp_{opt}_acc'] = _g(mlp_m, opt, 'final_accuracy')
        flat_metrics[f'mlp_{opt}_nll'] = _g(mlp_m, opt, 'final_nll')
    # Sec 6.3 CIFAR-10 CNN per-optimizer
    for opt in ('adam', 'sgd_nesterov', 'adagrad'):
        flat_metrics[f'cifar10_cnn_{opt}_acc'] = _g(cnn, opt, 'final_accuracy')
        flat_metrics[f'cifar10_cnn_{opt}_nll'] = _g(cnn, opt, 'final_nll')
    # Sec 6.4 VAE
    flat_metrics['vae_adam_bc_elbo']       = _g(vae, 'adam_bias_corrected',    'final_elbo')
    flat_metrics['vae_adam_no_bc_elbo']    = _g(vae, 'adam_no_bias_correction','final_elbo')
    flat_metrics['vae_rmsprop_elbo']       = _g(vae, 'rmsprop',               'final_elbo')
    flat_metrics['wall_time_seconds']      = metrics.get('wall_time_seconds')

    # Drop None values so every key maps to a real number
    flat_metrics = {k: v for k, v in flat_metrics.items() if v is not None}

    write_code_root_metrics(flat_metrics)

    log.info(f"Finished in {time.time() - START_TIME:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
