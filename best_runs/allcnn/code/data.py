"""
Data loading, preprocessing, and augmentation for CIFAR-10/100.

Preprocessing following Goodfellow et al. (2013) as stated in Section 3.1:
  1. Global Contrast Normalization (GCN): per-sample mean subtraction + std division
  2. ZCA whitening: computed from training set statistics, cached to disk

Augmentation (Section 3.1):
  - Horizontal flips (add flipped copies of all images)
  - Random translations: max ±5 pixels in each dimension
  NOTE: Paper describes "adding horizontally flipped copies" (offline augmentation)
        and "randomly translated versions" at train time.
        We implement: hflip + random crop with padding=5 (equivalent to ±5px translation)

Reference: Goodfellow et al. (2013) "Maxout Networks" for the GCN+ZCA preprocessing.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

try:
    from scipy.linalg import eigh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# ZCA whitening
# ---------------------------------------------------------------------------

def compute_zca_matrix(X_flat: np.ndarray, epsilon: float = 0.1) -> tuple:
    """
    Compute ZCA whitening matrix from flattened training data.

    Args:
        X_flat: (N, D) array of training images, flattened and mean-subtracted
        epsilon: regularization constant (default 0.1 from Goodfellow 2013)

    Returns:
        (mean, zca_matrix) where zca_matrix is (D, D)
    """
    N, D = X_flat.shape
    mean = X_flat.mean(axis=0)
    X_centered = X_flat - mean

    # Covariance (unbiased)
    cov = np.dot(X_centered.T, X_centered) / (N - 1)  # (D, D)

    # Eigen-decomposition
    if HAS_SCIPY:
        eigenvalues, eigenvectors = eigh(cov)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # ZCA transform: W = U * diag(1/sqrt(D + eps)) * U^T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + epsilon))
    zca_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T  # (D, D)

    return mean, zca_matrix


def apply_zca(X_flat: np.ndarray, mean: np.ndarray, zca_matrix: np.ndarray) -> np.ndarray:
    """Apply ZCA whitening to flattened data."""
    return (X_flat - mean) @ zca_matrix


class ZCAPreprocessor:
    """
    Fits ZCA whitening on training data and transforms train/test sets.
    Caches the computed matrices to disk for reuse.
    """

    def __init__(self, cache_dir: str, epsilon: float = 0.1):
        self.cache_dir = cache_dir
        self.epsilon = epsilon
        self.mean = None
        self.zca_matrix = None

    def _cache_path(self, dataset_name: str):
        return os.path.join(self.cache_dir, f"zca_{dataset_name}_eps{self.epsilon}.npz")

    def fit_or_load(self, train_data: np.ndarray, dataset_name: str) -> "ZCAPreprocessor":
        """
        Fit ZCA on train_data, or load from cache if available.
        train_data: (N, C, H, W) float32 array
        """
        cache_path = self._cache_path(dataset_name)
        if os.path.exists(cache_path):
            print(f"[ZCA] Loading cached ZCA matrices from {cache_path}")
            cached = np.load(cache_path)
            self.mean = cached['mean']
            self.zca_matrix = cached['zca_matrix']
            return self

        print(f"[ZCA] Computing ZCA whitening for {dataset_name} (N={len(train_data)})...")
        N = len(train_data)
        D = int(np.prod(train_data.shape[1:]))
        X_flat = train_data.reshape(N, D).astype(np.float64)

        # Global contrast normalization first (per sample)
        X_flat = _gcn(X_flat)

        self.mean, self.zca_matrix = compute_zca_matrix(X_flat, self.epsilon)

        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(cache_path, mean=self.mean, zca_matrix=self.zca_matrix)
        print(f"[ZCA] Saved ZCA matrices to {cache_path}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply GCN + ZCA to data.
        data: (N, C, H, W) float32 array
        Returns: (N, C, H, W) float32 whitened array
        """
        assert self.mean is not None, "Call fit_or_load() first"
        N = len(data)
        D = int(np.prod(data.shape[1:]))
        orig_shape = data.shape
        X_flat = data.reshape(N, D).astype(np.float64)

        # Per-sample GCN
        X_flat = _gcn(X_flat)

        # ZCA whitening
        X_white = apply_zca(X_flat, self.mean, self.zca_matrix)

        return X_white.reshape(orig_shape).astype(np.float32)


def _gcn(X: np.ndarray, scale: float = 55.0) -> np.ndarray:
    """
    Global Contrast Normalization (GCN) per Goodfellow et al. (2013).
    Subtracts per-sample mean and divides by per-sample std.
    scale parameter makes output in a stable range for the ZCA step.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.maximum(std, 1.0 / np.sqrt(X.shape[1]))  # avoid div by zero
    return scale * (X - mean) / std


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class CIFARDataset(Dataset):
    """
    CIFAR-10 or CIFAR-100 dataset with:
    - Optional ZCA whitening (Goodfellow et al. 2013 preprocessing)
    - Optional augmentation (hflip + random translation ±5px)
    - Returns tensors suitable for AllCNNModel
    """

    def __init__(
        self,
        data: np.ndarray,         # (N, C, H, W) float32, [0,1] or ZCA-whitened
        targets: np.ndarray,      # (N,) int
        augment: bool = False,    # apply hflip + random ±5px translation
        device: str = 'cpu',
    ):
        # Keep on CPU as numpy; convert per-batch to avoid GPU memory overhead
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].copy()  # (C, H, W)
        label = self.targets[idx]

        if self.augment:
            img = _augment(img)

        return torch.from_numpy(img), int(label)


def _augment(img: np.ndarray) -> np.ndarray:
    """
    Training augmentation per Section 3.1:
    - Random horizontal flip
    - Random translation: max ±5 pixels in each dimension

    img: (C, H, W) numpy float32
    """
    C, H, W = img.shape
    pad = 5  # max translation

    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1].copy()

    # Random translation (pad then crop)
    padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    # Random offsets in [0, 2*pad]
    top = np.random.randint(0, 2 * pad + 1)
    left = np.random.randint(0, 2 * pad + 1)
    img = padded[:, top:top + H, left:left + W]

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading with preprocessing
# ---------------------------------------------------------------------------

def load_cifar(
    dataset: str,           # 'cifar10' or 'cifar100'
    data_root: str,
    augment_train: bool = True,
    use_zca: bool = True,
    zca_cache_dir: str = None,
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple:
    """
    Load CIFAR-10 or CIFAR-100 with ZCA preprocessing.

    Returns:
        (train_loader, test_loader, zca_preprocessor)
        where zca_preprocessor is None if use_zca=False
    """
    assert dataset in ('cifar10', 'cifar100'), f"Unknown dataset: {dataset}"

    # Download via torchvision (canonical loader per data recipe)
    if dataset == 'cifar10':
        # Canonical loader: torchvision.datasets.CIFAR10
        train_ds = datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
        test_ds = datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=transforms.ToTensor()
        )
    else:
        train_ds = datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
        test_ds = datasets.CIFAR100(
            root=data_root, train=False, download=True,
            transform=transforms.ToTensor()
        )

    # Extract numpy arrays — use direct .data access (17× faster than per-image iteration)
    # torchvision CIFAR stores data as (N, H, W, C) uint8; transpose to (N, C, H, W) float32 in [0,1]
    train_data = train_ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    train_targets = np.array(train_ds.targets, dtype=np.int64)
    test_data = test_ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    test_targets = np.array(test_ds.targets, dtype=np.int64)

    print(f"[Data] {dataset}: train={len(train_data)}, test={len(test_data)}")

    # ZCA preprocessing
    zca = None
    if use_zca:
        if zca_cache_dir is None:
            zca_cache_dir = os.path.join(data_root, 'zca_cache')
        zca = ZCAPreprocessor(zca_cache_dir)
        zca.fit_or_load(train_data, dataset)
        print(f"[Data] Applying ZCA whitening...")
        train_data = zca.transform(train_data)
        test_data = zca.transform(test_data)
        print(f"[Data] ZCA done. train range: [{train_data.min():.2f}, {train_data.max():.2f}]")
    else:
        # Standard normalization (CIFAR-10 channel stats)
        # Per-channel mean/std from the canonical recipe
        if dataset == 'cifar10':
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
        else:
            mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(1, 3, 1, 1)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

    # Create Dataset objects
    train_dataset = CIFARDataset(train_data, train_targets, augment=augment_train)
    test_dataset = CIFARDataset(test_data, test_targets, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader, zca


def load_cifar_fast(
    dataset: str,
    data_root: str,
    augment_train: bool = True,
    use_zca: bool = True,
    zca_cache_dir: str = None,
    batch_size: int = 128,
    num_workers: int = 0,
    subsample_n: int = None,  # optional subsample for smoke tests
) -> tuple:
    """
    Convenience loader that handles both smoke-test subsampling and full loading.
    """
    train_loader, test_loader, zca = load_cifar(
        dataset=dataset,
        data_root=data_root,
        augment_train=augment_train,
        use_zca=use_zca,
        zca_cache_dir=zca_cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if subsample_n is not None and subsample_n > 0:
        from torch.utils.data import Subset
        n_train = min(subsample_n, len(train_loader.dataset))
        n_test = min(subsample_n, len(test_loader.dataset))
        train_loader = DataLoader(
            Subset(train_loader.dataset, list(range(n_train))),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            Subset(test_loader.dataset, list(range(n_test))),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, test_loader, zca
