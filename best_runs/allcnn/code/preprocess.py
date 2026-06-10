"""
GCN + ZCA whitening for CIFAR-10/100 following Goodfellow et al. (2013),
as required by Striving for Simplicity: The All Convolutional Net (arXiv 1412.6806).
"""

import os
import time

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms


def compute_gcn_zca_stats(train_images_flat, epsilon=0.1, cache_path=None):
    """
    Compute ZCA whitening statistics from GCN-normalized training images.

    Args:
        train_images_flat: numpy array [N, D] where D = C*H*W, values in [0,1]
        epsilon: regularization term added to eigenvalues before sqrt
        cache_path: if provided, load stats from this .npz file if it exists,
                    otherwise compute and save to it

    Returns:
        (zca_mean [D], zca_W [D,D]) as numpy float32
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[preprocess] Loading cached ZCA stats from {cache_path}")
        npz = np.load(cache_path)
        return npz["zca_mean"].astype(np.float32), npz["zca_W"].astype(np.float32)

    N, D = train_images_flat.shape
    print(f"[preprocess] Computing GCN on {N} training images (D={D}) ...")
    t0 = time.time()

    # GCN: per-image zero mean + unit std (vectorized)
    X = train_images_flat.astype(np.float32)
    img_mean = X.mean(axis=1, keepdims=True)        # [N, 1]
    img_std  = X.std(axis=1, keepdims=True) + 1e-8  # [N, 1]
    X_gcn = (X - img_mean) / img_std                # [N, D]

    print(f"[preprocess] GCN done in {time.time()-t0:.1f}s. Computing ZCA covariance ...")
    t1 = time.time()

    # ZCA: dataset-level whitening
    zca_mean = X_gcn.mean(axis=0)          # [D]
    X_centered = X_gcn - zca_mean          # [N, D]

    cov = (X_centered.T @ X_centered) / N  # [D, D]

    print(f"[preprocess] Covariance computed in {time.time()-t1:.1f}s. Running SVD ...")
    t2 = time.time()

    U, S, _Vt = np.linalg.svd(cov)        # U: [D,D], S: [D]
    W = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T  # [D, D]

    zca_mean = zca_mean.astype(np.float32)
    W = W.astype(np.float32)

    print(f"[preprocess] SVD + W computed in {time.time()-t2:.1f}s.")

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez(cache_path, zca_mean=zca_mean, zca_W=W)
        print(f"[preprocess] ZCA stats saved to {cache_path}")

    return zca_mean, W


def apply_gcn_zca(images_chw, zca_mean, zca_W):
    """
    Apply GCN then ZCA whitening to a batch of images.

    Args:
        images_chw: numpy array [N, C, H, W], values in [0,1]
        zca_mean:   numpy array [D] (D = C*H*W), float32
        zca_W:      numpy array [D, D], float32

    Returns:
        numpy array [N, C, H, W] whitened float32
    """
    N, C, H, W = images_chw.shape
    D = C * H * W

    X = images_chw.reshape(N, D).astype(np.float32)  # [N, D]

    # GCN per-image (vectorized)
    img_mean = X.mean(axis=1, keepdims=True)        # [N, 1]
    img_std  = X.std(axis=1, keepdims=True) + 1e-8  # [N, 1]
    X_gcn = (X - img_mean) / img_std                # [N, D]

    # ZCA
    X_centered = X_gcn - zca_mean                   # [N, D]
    X_white = X_centered @ zca_W.T                  # [N, D]

    return X_white.reshape(N, C, H, W)


def preprocess_cifar(train_dataset, test_dataset, cache_dir, epsilon=0.1):
    """
    Apply GCN + ZCA whitening to CIFAR-10 or CIFAR-100 torchvision datasets.

    Args:
        train_dataset: torchvision dataset loaded with ToTensor() transform
        test_dataset:  torchvision dataset loaded with ToTensor() transform
        cache_dir:     directory where cifar_zca_stats.npz will be cached
        epsilon:       ZCA regularization (default 0.1 per Goodfellow et al.)

    Returns:
        (train_data [N,C,H,W] float32, train_labels [N] int32,
         test_data  [M,C,H,W] float32, test_labels  [M] int32)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "cifar_zca_stats.npz")

    print(f"[preprocess] Extracting training images from dataset ({len(train_dataset)} samples) ...")
    t0 = time.time()

    # Extract all training images into a contiguous array
    # torchvision CIFAR datasets store .data as [N,H,W,C] uint8; after ToTensor() each
    # sample is a float32 tensor [C,H,W] in [0,1].
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=False, num_workers=0
    )
    train_images_list = []
    train_labels_list = []
    for imgs, lbls in train_loader:
        train_images_list.append(imgs.numpy())
        train_labels_list.append(lbls.numpy())

    train_images = np.concatenate(train_images_list, axis=0)   # [N, C, H, W]
    train_labels = np.concatenate(train_labels_list, axis=0).astype(np.int32)
    print(f"[preprocess] Training images extracted in {time.time()-t0:.1f}s. "
          f"Shape: {train_images.shape}")

    N, C, H, W = train_images.shape
    D = C * H * W
    train_flat = train_images.reshape(N, D)  # [N, D]

    # Compute (or load) ZCA stats
    zca_mean, zca_W = compute_gcn_zca_stats(
        train_flat, epsilon=epsilon, cache_path=cache_path
    )

    # Apply GCN + ZCA to training set
    print("[preprocess] Applying GCN+ZCA to training set ...")
    t1 = time.time()
    train_white = apply_gcn_zca(train_images, zca_mean, zca_W)
    print(f"[preprocess] Training whitening done in {time.time()-t1:.1f}s.")

    # Extract and whiten test set
    print(f"[preprocess] Extracting test images ({len(test_dataset)} samples) ...")
    t2 = time.time()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=0
    )
    test_images_list = []
    test_labels_list = []
    for imgs, lbls in test_loader:
        test_images_list.append(imgs.numpy())
        test_labels_list.append(lbls.numpy())

    test_images = np.concatenate(test_images_list, axis=0)   # [M, C, H, W]
    test_labels = np.concatenate(test_labels_list, axis=0).astype(np.int32)
    print(f"[preprocess] Test images extracted in {time.time()-t2:.1f}s.")

    print("[preprocess] Applying GCN+ZCA to test set ...")
    t3 = time.time()
    test_white = apply_gcn_zca(test_images, zca_mean, zca_W)
    print(f"[preprocess] Test whitening done in {time.time()-t3:.1f}s.")

    print("[preprocess] Preprocessing complete.")
    return (
        train_white.astype(np.float32),
        train_labels,
        test_white.astype(np.float32),
        test_labels,
    )


class WhitenedDataset(torch.utils.data.Dataset):
    """Dataset wrapper for pre-whitened images with optional augmentation.

    Args:
        data:    numpy array [N, C, H, W] float32 (pre-whitened)
        labels:  numpy array [N] int
        augment: if True, apply RandomCrop(32, padding=5) + RandomHorizontalFlip
                 at __getitem__ time (suitable for training)
    """

    def __init__(self, data, labels, augment=False):
        self.data = torch.from_numpy(data)          # [N, C, H, W] float32
        self.labels = torch.from_numpy(labels.astype(np.int64))  # [N] long

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=5),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]    # [C, H, W] float32 tensor
        lbl = self.labels[idx]  # scalar long tensor

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl
