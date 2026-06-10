"""
All Convolutional Net architectures from:
  "Striving for Simplicity: The All Convolutional Net"
  Springenberg, Dosovitskiy, Brox, Riedmiller (arXiv 1412.6806)

12 architectures covering three base network designs (A, B, C) and four
variants (base, strided, convpool, allcnn).

Notation used in the paper (and here):
  D(p)   = nn.Dropout(p)
  CkxN   = Conv2d + ReLU, kernel k, N filters, pad=(k-1)//2, stride=1 unless noted
  Pool   = MaxPool2d(3, stride=2, padding=1)
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Building-block helpers
# ---------------------------------------------------------------------------

def _conv_relu(in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> nn.Sequential:
    """Single Conv2d + ReLU block with symmetric same-style padding."""
    pad = (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=True),
        nn.ReLU(inplace=True),
    )


def _classifier_head(num_classes: int) -> nn.Sequential:
    """
    Shared classifier head (Table 1, paper §3):
      D(0.5) → Conv(192,3x3,p=1)+ReLU → Conv(192,1x1)+ReLU
             → Conv(num_classes,1x1) → AdaptiveAvgPool2d(1) → flatten
    No ReLU on the final conv — outputs are raw logits for CrossEntropyLoss.
    """
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, 192, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, num_classes, kernel_size=1, bias=True),
        # NOTE: no ReLU here — outputs are raw logits for CrossEntropyLoss.
        # A ReLU here zeros negative logits → uniform softmax → loss stuck at
        # ln(num_classes) and zero gradient flow (dead training).
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _AllCNNBase(nn.Module):
    """
    Common forward: run self.features then self.classifier.
    Sub-classes populate self.features (nn.Sequential) and
    self.classifier (nn.Sequential via _classifier_head).
    """

    features: nn.Sequential
    classifier: nn.Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Model A  —  base
# ---------------------------------------------------------------------------

class ModelA(_AllCNNBase):
    """
    D(0.2) → C5x96 → Pool → D(0.5) → C5x192 → Pool → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# Model B  —  NIN variant of A (add 1×1 after each 5×5)
# ---------------------------------------------------------------------------

class ModelB(_AllCNNBase):
    """
    D(0.2) → C5x96 → C1x96 → Pool → D(0.5) → C5x192 → C1x192 → Pool
           → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# Model C  —  replace 5×5 with two 3×3, remove 1×1 NIN
# ---------------------------------------------------------------------------

class ModelC(_AllCNNBase):
    """
    D(0.2) → C3x96 → C3x96 → Pool → D(0.5) → C3x192 → C3x192 → Pool
           → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# Strided CNN A  —  stride=2 conv replaces MaxPool
# ---------------------------------------------------------------------------

class StridedCNN_A(_AllCNNBase):
    """
    D(0.2) → C5x96(s=2) → D(0.5) → C5x192(s=2) → D(0.5) → classifier_head
    (No MaxPool; preceding conv takes stride=2 to subsample)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1 — strided conv replaces Pool
            nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            # block 2 — strided conv replaces Pool
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            # Dropout after last pool-replacement is provided by _classifier_head's
            # first D(0.5); keeping it here would double the effective dropout rate.
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# Strided CNN B  —  1×1 NIN gets stride=2 instead of MaxPool
# ---------------------------------------------------------------------------

class StridedCNN_B(_AllCNNBase):
    """
    D(0.2) → C5x96 → C1x96(s=2) → D(0.5) → C5x192 → C1x192(s=2) → D(0.5)
           → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # D(0.5) after last pool-replacement supplied by _classifier_head.
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# Strided CNN C  —  second 3×3 gets stride=2
# ---------------------------------------------------------------------------

class StridedCNN_C(_AllCNNBase):
    """
    D(0.2) → C3x96 → C3x96(s=2) → D(0.5) → C3x192 → C3x192(s=2) → D(0.5)
           → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # D(0.5) after last pool-replacement supplied by _classifier_head.
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# ConvPool CNN A  —  add C3xN before Pool
# ---------------------------------------------------------------------------

class ConvPoolCNN_A(_AllCNNBase):
    """
    D(0.2) → C5x96 → C3x96 → Pool → D(0.5) → C5x192 → C3x192 → Pool
           → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# ConvPool CNN B  —  NIN then 3×3 before Pool
# ---------------------------------------------------------------------------

class ConvPoolCNN_B(_AllCNNBase):
    """
    D(0.2) → C5x96 → C1x96 → C3x96 → Pool → D(0.5) → C5x192 → C1x192
           → C3x192 → Pool → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# ConvPool CNN C  —  three 3×3s with Pool
# ---------------------------------------------------------------------------

class ConvPoolCNN_C(_AllCNNBase):
    """
    D(0.2) → C3x96 → C3x96 → C3x96 → Pool → D(0.5) → C3x192 → C3x192
           → C3x192 → Pool → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# All-CNN A  —  strided 3×3 replaces Pool in ConvPool A
# ---------------------------------------------------------------------------

class AllCNN_A(_AllCNNBase):
    """
    D(0.2) → C5x96 → C3x96(s=2) → D(0.5) → C5x192 → C3x192(s=2) → D(0.5)
           → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # D(0.5) after last pool-replacement supplied by _classifier_head.
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# All-CNN B  —  strided 3×3 replaces Pool in ConvPool B
# ---------------------------------------------------------------------------

class AllCNN_B(_AllCNNBase):
    """
    D(0.2) → C5x96 → C1x96 → C3x96(s=2) → D(0.5) → C5x192 → C1x192
           → C3x192(s=2) → D(0.5) → classifier_head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # D(0.5) after last pool-replacement supplied by _classifier_head.
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# All-CNN C  —  strided 3×3 replaces Pool in ConvPool C  (paper's star model)
# ---------------------------------------------------------------------------

class AllCNN_C(_AllCNNBase):
    """
    D(0.2) → C3x96 → C3x96 → C3x96(s=2) → D(0.5) → C3x192 → C3x192
           → C3x192(s=2) → D(0.5) → classifier_head

    This is the paper's headline result (Table 1, best CIFAR-10 accuracy).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Dropout(p=0.2),
            # block 1
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # block 2
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # D(0.5) after last pool-replacement supplied by _classifier_head.
        )
        self.classifier = _classifier_head(num_classes)


# ---------------------------------------------------------------------------
# Factory & utility
# ---------------------------------------------------------------------------

_MODEL_MAP = {
    ("a", "base"):     ModelA,
    ("b", "base"):     ModelB,
    ("c", "base"):     ModelC,
    ("a", "strided"):  StridedCNN_A,
    ("b", "strided"):  StridedCNN_B,
    ("c", "strided"):  StridedCNN_C,
    ("a", "convpool"): ConvPoolCNN_A,
    ("b", "convpool"): ConvPoolCNN_B,
    ("c", "convpool"): ConvPoolCNN_C,
    ("a", "allcnn"):   AllCNN_A,
    ("b", "allcnn"):   AllCNN_B,
    ("c", "allcnn"):   AllCNN_C,
}


def get_model(base_model: str, variant: str, num_classes: int = 10) -> nn.Module:
    """
    Instantiate one of the 12 All-CNN architectures.

    Parameters
    ----------
    base_model : str
        Network design family: 'a', 'b', or 'c'.
    variant : str
        Pooling strategy: 'base', 'strided', 'convpool', or 'allcnn'.
    num_classes : int
        Number of output classes (default 10 for CIFAR-10).

    Returns
    -------
    nn.Module
        The requested architecture, weight-initialised randomly.

    Examples
    --------
    >>> model = get_model('c', 'allcnn')          # All-CNN-C
    >>> model = get_model('b', 'convpool', 100)   # ConvPool-CNN-B on CIFAR-100
    """
    key = (base_model.lower(), variant.lower())
    if key not in _MODEL_MAP:
        valid = sorted(_MODEL_MAP.keys())
        raise ValueError(
            f"Unknown (base_model={base_model!r}, variant={variant!r}). "
            f"Valid combinations: {valid}"
        )
    return _MODEL_MAP[key](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(2, 3, 32, 32, device=device)   # CIFAR spatial size

    rows = []
    for (bm, var), cls in _MODEL_MAP.items():
        m = cls(num_classes=10).to(device).eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 10), f"Bad output shape {out.shape} for ({bm},{var})"
        params = count_parameters(m)
        rows.append((cls.__name__, params, out.shape))

    print(f"{'Model':<18}  {'Params':>10}  Output")
    print("-" * 45)
    for name, p, shape in rows:
        print(f"{name:<18}  {p:>10,}  {tuple(shape)}")

    print(f"\nAll {len(rows)} models passed forward-pass check on {device}.")
    sys.exit(0)
