"""
Dataset loading and preprocessing — All-CNN paper (Springenberg et al. 2015)

Section 3.2 preprocessing (CIFAR):
  1. Whitening and contrast normalization following Goodfellow et al. (2013):
       - Global Contrast Normalization (GCN): subtract per-image mean, divide
         by max(std, ε) — removes global contrast differences.
       - ZCA whitening on the flattened 3072-dim pixel vectors, computed from
         the training set.  Decorrelates across spatial positions AND channels.

  2. Data augmentation (Section 3.2):
       - Horizontally flipped copies of all images (double the training set)
       - Randomly translated versions with max 5-pixel shift in each dimension
         (implemented as RandomCrop after symmetric padding of 5 pixels).

Canonical CIFAR-10 loader from data_recipes:
  torchvision.datasets.CIFAR10(root, train=True, download=True,
      transform=Compose([ToTensor(), Normalize((0.4914,0.4822,0.4465),
                                               (0.2470,0.2435,0.2616))]))
"""

from __future__ import annotations

import os
import pickle
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Global Contrast Normalization (GCN) — Goodfellow et al. 2013 §2.3
# ---------------------------------------------------------------------------

def global_contrast_normalize(
    X: np.ndarray,
    scale: float = 55.0,
    min_divisor: float = 1e-8,
) -> np.ndarray:
    """
    Per-image Global Contrast Normalization.

    X shape: (N, C, H, W) or (N, H*W*C)  float32
    Returns same shape, normalized.
    """
    orig_shape = X.shape
    X = X.reshape(len(X), -1).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    norms = np.sqrt((X ** 2).mean(axis=1, keepdims=True))
    norms = np.maximum(norms, min_divisor)
    X = scale * X / norms
    return X.reshape(orig_shape)


# ---------------------------------------------------------------------------
# ZCA Whitening — Goodfellow et al. 2013
# ---------------------------------------------------------------------------

class ZCAWhitener:
    """
    ZCA (zero-phase component analysis) whitening.

    Fit on training data, then applied to train and test.
    Stores the whitening matrix W and mean vector for re-use.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps
        self.W: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'ZCAWhitener':
        """
        X: (N, d) float32, d = C*H*W = 3072 for CIFAR.
        Computes covariance, eigen-decomposes, forms whitening matrix W.
        """
        N, d = X.shape
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / N          # (d, d) — may be large!
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        # ZCA whitening matrix: W = U diag(1/sqrt(S+eps)) U^T
        D = np.diag(1.0 / np.sqrt(S + self.eps))
        self.W = (U @ D @ U.T).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d)"""
        assert self.W is not None and self.mean_ is not None, "Call fit() first"
        return (X - self.mean_) @ self.W.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        np.save(path, {'W': self.W, 'mean': self.mean_})

    @classmethod
    def load(cls, path: str) -> 'ZCAWhitener':
        d = np.load(path, allow_pickle=True).item()
        w = cls()
        w.W, w.mean_ = d['W'], d['mean']
        return w


# ---------------------------------------------------------------------------
# CIFAR preprocessing pipeline
# ---------------------------------------------------------------------------

def _load_cifar_raw(dataset_cls, data_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw uint8 CIFAR pixels for ZCA computation."""
    train_ds = dataset_cls(root=data_root, train=True, download=True, transform=None)
    test_ds  = dataset_cls(root=data_root, train=False, download=True, transform=None)

    # Convert PIL images to numpy
    train_imgs = np.array([np.array(img) for img, _ in train_ds])  # (N,32,32,3)
    train_labels = np.array([lbl for _, lbl in train_ds])
    test_imgs  = np.array([np.array(img) for img, _ in test_ds])
    test_labels = np.array([lbl for _, lbl in test_ds])

    # HWC → CHW, float32 in [0,1]
    train_imgs = train_imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    test_imgs  = test_imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return train_imgs, train_labels, test_imgs, test_labels


def prepare_cifar(
    dataset: str,               # 'cifar10' or 'cifar100'
    data_root: str,
    augment: bool = False,
    use_zca: bool = True,
    zca_cache_path: Optional[str] = None,
    max_train_samples: Optional[int] = None,  # for CPU smoke runs
) -> Tuple[Dataset, Dataset, int]:
    """
    Full CIFAR preprocessing pipeline.

    Returns (train_dataset, test_dataset, num_classes).

    Augmentation (Section 3.2):
      - Random horizontal flip
      - Random translation ≤5 pixels in each direction
        (implemented as RandomCrop(32, padding=5, pad_if_needed=True))

    Whitening (Section 3.2 / Goodfellow 2013):
      1. GCN per image
      2. ZCA on training set (fit on train, apply to both splits)
    """
    dataset_cls = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    num_classes = 10 if dataset == 'cifar10' else 100

    print(f"[data] Loading raw {dataset} from {data_root} ...", flush=True)
    train_imgs, train_labels, test_imgs, test_labels = _load_cifar_raw(dataset_cls, data_root)

    # 1. GCN per image
    print("[data] Applying Global Contrast Normalization ...", flush=True)
    N, C, H, W = train_imgs.shape
    train_flat = train_imgs.reshape(N, -1)
    test_flat  = test_imgs.reshape(len(test_imgs), -1)
    train_flat = global_contrast_normalize(train_flat)
    test_flat  = global_contrast_normalize(test_flat)

    # 2. ZCA whitening
    if use_zca:
        zca_loaded = False
        if zca_cache_path and os.path.exists(zca_cache_path):
            try:
                print(f"[data] Loading cached ZCA from {zca_cache_path}", flush=True)
                zca = ZCAWhitener.load(zca_cache_path)
                zca_loaded = True
            except Exception as e:
                print(f"[data] ZCA cache load failed ({e}), recomputing", flush=True)

        if not zca_loaded:
            print("[data] Computing ZCA whitening matrix (may take ~30s) ...", flush=True)
            zca = ZCAWhitener(eps=0.1)
            zca.fit(train_flat)
            if zca_cache_path:
                os.makedirs(os.path.dirname(zca_cache_path), exist_ok=True)
                zca.save(zca_cache_path)
                print(f"[data] Saved ZCA to {zca_cache_path}", flush=True)

        train_flat = zca.transform(train_flat)
        test_flat  = zca.transform(test_flat)

    train_imgs_proc = train_flat.reshape(N, C, H, W)
    test_imgs_proc  = test_flat.reshape(len(test_imgs), C, H, W)

    # Subsample for CPU smoke runs
    if max_train_samples and max_train_samples < N:
        idx = np.arange(max_train_samples)
        train_imgs_proc = train_imgs_proc[idx]
        train_labels    = train_labels[idx]
        print(f"[data] Subsampled train to {max_train_samples} samples (CPU mode)", flush=True)

    # Convert to tensors
    train_t = torch.from_numpy(train_imgs_proc)
    test_t  = torch.from_numpy(test_imgs_proc)
    train_y = torch.from_numpy(train_labels).long()
    test_y  = torch.from_numpy(test_labels).long()

    # Augmented dataset wraps tensors with on-the-fly transforms
    if augment:
        train_ds = AugmentedTensorDataset(
            train_t, train_y,
            flip_prob=0.5,       # random horizontal flip
            max_translate=5,     # max 5 pixels in each direction (Section 3.2)
        )
    else:
        train_ds = TensorDataset(train_t, train_y)

    test_ds = TensorDataset(test_t, test_y)
    return train_ds, test_ds, num_classes


# ---------------------------------------------------------------------------
# Augmented dataset — Section 3.2 augmentation
# ---------------------------------------------------------------------------

class AugmentedTensorDataset(Dataset):
    """
    On-the-fly augmentation for pre-whitened CIFAR tensors.

    Section 3.2:
      - "horizontally flipped copies of all images"
      - "randomly translated versions with a maximum translation of 5 pixels
         in each dimension"
    """

    def __init__(
        self,
        images: torch.Tensor,   # (N, C, H, W)
        labels: torch.Tensor,
        flip_prob: float = 0.5,
        max_translate: int = 5,
    ):
        self.images = images
        self.labels = labels
        self.flip_prob = flip_prob
        self.max_translate = max_translate

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].clone()
        lbl = self.labels[idx]

        # Random horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            img = torch.flip(img, dims=[-1])

        # Random translation ≤ max_translate pixels in each direction.
        # Pad then crop: equivalent to random shift.
        p = self.max_translate
        H, W = img.shape[-2], img.shape[-1]
        # Pad with zeros (or reflect — zeros following the paper's style)
        padded = F_pad(img, p)    # (C, H+2p, W+2p)
        dy = torch.randint(0, 2 * p + 1, (1,)).item()
        dx = torch.randint(0, 2 * p + 1, (1,)).item()
        img = padded[:, dy:dy + H, dx:dx + W]

        return img, lbl


def F_pad(img: torch.Tensor, p: int) -> torch.Tensor:
    """Zero-pad a CHW tensor by p pixels on each side."""
    return torch.nn.functional.pad(img, (p, p, p, p), mode='constant', value=0.0)


# ---------------------------------------------------------------------------
# Simple torchvision loader (fallback without ZCA for fast CPU smoke tests)
# ---------------------------------------------------------------------------

def get_cifar_loaders_simple(
    dataset: str,
    data_root: str,
    batch_size: int,
    augment: bool = False,
    max_train_samples: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Fast loader using standard torchvision normalization (no ZCA).
    Used when use_zca=False (e.g. quick smoke validation).
    """
    dataset_cls = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    num_classes = 10 if dataset == 'cifar10' else 100
    mean = (0.4914, 0.4822, 0.4465) if dataset == 'cifar10' else (0.5071, 0.4867, 0.4408)
    std  = (0.2470, 0.2435, 0.2616) if dataset == 'cifar10' else (0.2675, 0.2565, 0.2761)

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=5),         # ≤5 pixel translation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = dataset_cls(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = dataset_cls(data_root, train=False, download=True, transform=test_tf)

    if max_train_samples and max_train_samples < len(train_ds):
        indices = list(range(max_train_samples))
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader, num_classes
