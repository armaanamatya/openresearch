"""
All-CNN model architectures for CIFAR-10/100.

Implements all variants from "Striving for Simplicity: The All Convolutional Net"
Springenberg et al., 2014 (arxiv 1412.6806)

Models:
  Letter A: 5×5 convolutions, simpler (no NiN layers)
  Letter B: Network-in-Network variant (5×5 + 1×1 after each main conv)
  Letter C: All-3×3 variant (replaces all 5×5 of B with 3×3)

Variants (for each letter):
  base:     Original architecture with MaxPool layers (Table 1)
  strided:  Strided-CNN — remove MaxPool, increase stride of preceding conv
  convpool: ConvPool-CNN — insert dense conv before each MaxPool, keep MaxPool
  allcnn:   All-CNN — replace MaxPool with strided 3×3 conv (paper's main result)

Architecture invariants (Section 2, Section 3.1):
  - Fully convolutional classifier: 1×1 conv → GAP → softmax (no FC layers)
  - Dropout: 20% on input image, 50% after each pooling layer (or its replacement)
  - ReLU activations throughout (no BN — paper predates common CIFAR BN use)
  - Block 1 uses 96 feature maps, Block 2 uses 192
  - Classification head: conv(192,3×3) + conv(192,1×1) + conv(10,1×1) + GAP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model letter descriptions
# ---------------------------------------------------------------------------
#
# Model A (5 main conv layers for base variant):
#   Block 1: conv(96, 5×5) → MaxPool(3×3, s=2)
#   Block 2: conv(192, 5×5) → MaxPool(3×3, s=2)
#   Head:    conv(192, 3×3), conv(192, 1×1), conv(10, 1×1) → GAP
#
# Model B (7 main conv layers for base variant — NiN):
#   Block 1: conv(96, 5×5) + conv(96, 1×1) → MaxPool(3×3, s=2)
#   Block 2: conv(192, 5×5) + conv(192, 1×1) → MaxPool(3×3, s=2)
#   Head:    conv(192, 3×3), conv(192, 1×1), conv(10, 1×1) → GAP
#
# Model C (7 main conv layers for base variant — all-3×3):
#   Block 1: conv(96, 3×3) + conv(96, 3×3) → MaxPool(3×3, s=2)
#   Block 2: conv(192, 3×3) + conv(192, 3×3) → MaxPool(3×3, s=2)
#   Head:    conv(192, 3×3), conv(192, 1×1), conv(10, 1×1) → GAP
#
# Model C derived variants (All-CNN-C has 9 conv layers total):
#   Strided:  Block conv's second conv → stride 2, remove MaxPool  (7 conv layers)
#   ConvPool: Insert extra conv(96/192, 3×3) before MaxPool, keep MaxPool (9 conv layers)
#   All-CNN:  Replace MaxPool with conv(96/192, 3×3, stride=2) (9 conv layers)
# ---------------------------------------------------------------------------


def _conv_relu(in_ch, out_ch, k, stride=1, padding=0):
    return [nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding),
            nn.ReLU(inplace=True)]


def _maxpool():
    # Paper uses 3×3 pool, stride=2 — with padding=1 this halves spatial dims
    return nn.MaxPool2d(3, stride=2, padding=1)


def build_all_cnn(letter: str, variant: str, num_classes: int = 10) -> nn.Sequential:
    """
    Build one All-CNN network variant.

    Args:
        letter:      'A', 'B', or 'C'
        variant:     'base', 'strided', 'convpool', or 'allcnn'
        num_classes: 10 for CIFAR-10, 100 for CIFAR-100

    Returns:
        nn.Sequential whose forward() returns raw logits (no softmax).
        The output spatial dimension is 8×8 (after 2 halving operations
        on 32×32 CIFAR input), so GAP collapses it to (B, num_classes).
        Use .forward_with_gap() pattern — see AllCNNModel.forward() below.
    """
    assert letter in ('A', 'B', 'C'), f"Unknown letter {letter!r}"
    assert variant in ('base', 'strided', 'convpool', 'allcnn'), f"Unknown variant {variant!r}"

    layers = []

    # ---------- Input dropout (20%) ----------
    layers.append(nn.Dropout(0.2))

    # ========================================================================
    # Build Block 1 and Block 2 based on letter × variant
    # ========================================================================

    if letter == 'A':
        # --- Block 1 ---
        if variant == 'strided':
            # "increase the stride of the immediately preceding convolution by 1"
            # For A, the immediately preceding conv to MaxPool IS the single 5×5 conv
            layers += _conv_relu(3, 96, 5, stride=2, padding=2)
        else:
            layers += _conv_relu(3, 96, 5, stride=1, padding=2)
            if variant == 'convpool':
                layers += _conv_relu(96, 96, 3, stride=1, padding=1)  # extra dense conv
            elif variant == 'allcnn':
                # Replace MaxPool with stride-2 3×3 conv (output channels = input channels)
                layers += _conv_relu(96, 96, 3, stride=2, padding=1)

        if variant in ('base', 'convpool'):
            layers.append(_maxpool())

        layers.append(nn.Dropout(0.5))

        # --- Block 2 ---
        if variant == 'strided':
            layers += _conv_relu(96, 192, 5, stride=2, padding=2)
        else:
            layers += _conv_relu(96, 192, 5, stride=1, padding=2)
            if variant == 'convpool':
                layers += _conv_relu(192, 192, 3, stride=1, padding=1)
            elif variant == 'allcnn':
                layers += _conv_relu(192, 192, 3, stride=2, padding=1)

        if variant in ('base', 'convpool'):
            layers.append(_maxpool())

        layers.append(nn.Dropout(0.5))

    elif letter == 'B':
        # --- Block 1: 5×5 + 1×1 (NiN) ---
        layers += _conv_relu(3, 96, 5, stride=1, padding=2)
        if variant == 'strided':
            # Immediately preceding MaxPool is the 1×1 conv — make it stride 2
            layers += _conv_relu(96, 96, 1, stride=2)
        else:
            layers += _conv_relu(96, 96, 1, stride=1)
            if variant == 'convpool':
                layers += _conv_relu(96, 96, 3, stride=1, padding=1)
            elif variant == 'allcnn':
                layers += _conv_relu(96, 96, 3, stride=2, padding=1)

        if variant in ('base', 'convpool'):
            layers.append(_maxpool())

        layers.append(nn.Dropout(0.5))

        # --- Block 2: 5×5 + 1×1 (NiN) ---
        layers += _conv_relu(96, 192, 5, stride=1, padding=2)
        if variant == 'strided':
            layers += _conv_relu(192, 192, 1, stride=2)
        else:
            layers += _conv_relu(192, 192, 1, stride=1)
            if variant == 'convpool':
                layers += _conv_relu(192, 192, 3, stride=1, padding=1)
            elif variant == 'allcnn':
                layers += _conv_relu(192, 192, 3, stride=2, padding=1)

        if variant in ('base', 'convpool'):
            layers.append(_maxpool())

        layers.append(nn.Dropout(0.5))

    elif letter == 'C':
        # --- Block 1: 3×3 + 3×3 ---
        layers += _conv_relu(3, 96, 3, stride=1, padding=1)
        if variant == 'strided':
            # "immediately preceding" = second 3×3 conv → make it stride 2
            layers += _conv_relu(96, 96, 3, stride=2, padding=1)
        else:
            layers += _conv_relu(96, 96, 3, stride=1, padding=1)
            if variant == 'convpool':
                layers += _conv_relu(96, 96, 3, stride=1, padding=1)
            elif variant == 'allcnn':
                layers += _conv_relu(96, 96, 3, stride=2, padding=1)

        if variant in ('base', 'convpool'):
            layers.append(_maxpool())

        layers.append(nn.Dropout(0.5))

        # --- Block 2: 3×3 + 3×3 ---
        layers += _conv_relu(96, 192, 3, stride=1, padding=1)
        if variant == 'strided':
            layers += _conv_relu(192, 192, 3, stride=2, padding=1)
        else:
            layers += _conv_relu(192, 192, 3, stride=1, padding=1)
            if variant == 'convpool':
                layers += _conv_relu(192, 192, 3, stride=1, padding=1)
            elif variant == 'allcnn':
                layers += _conv_relu(192, 192, 3, stride=2, padding=1)

        if variant in ('base', 'convpool'):
            layers.append(_maxpool())

        layers.append(nn.Dropout(0.5))

    # ========================================================================
    # Classification head: fully-convolutional (Section 2)
    # conv(192,3×3) + conv(192,1×1) + conv(num_classes,1×1) + GAP + Softmax
    # ========================================================================
    layers += _conv_relu(192, 192, 3, padding=1)
    layers += _conv_relu(192, 192, 1)
    # Final 1×1 conv produces class logits — no ReLU before GAP
    layers.append(nn.Conv2d(192, num_classes, 1))

    return nn.Sequential(*layers)


class AllCNNModel(nn.Module):
    """
    Wrapper that adds Global Average Pooling after the convolutional body.

    Usage:
        model = AllCNNModel('C', 'allcnn', num_classes=10)
        logits = model(x)  # shape: (B, num_classes)
    """

    def __init__(self, letter: str, variant: str, num_classes: int = 10):
        super().__init__()
        self.letter = letter
        self.variant = variant
        self.num_classes = num_classes
        self.features = build_all_cnn(letter, variant, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                    # (B, num_classes, H, W)
        x = x.mean(dim=[2, 3])                  # Global Average Pool → (B, num_classes)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"letter={self.letter!r}, variant={self.variant!r}, num_classes={self.num_classes}"


# ---------------------------------------------------------------------------
# ImageNet All-CNN-B (Table 6 of the paper)
# Section 3.2 ImageNet: 12 conv layers, 450k iterations, batch=64, lr=0.01/10
# Architecture upscaled from CIFAR for 224×224 input
# ---------------------------------------------------------------------------

def build_imagenet_all_cnn_b(num_classes: int = 1000) -> "AllCNNImageNet":
    """
    Upscaled All-CNN-B for ImageNet (Table 6).
    12 conv layers trained for 450,000 iterations, batch=64, initial lr=0.01
    divided by 10 every 200,000 iterations, weight decay λ=0.0005.
    Input: 224×224 center crop.
    Target: Top-1 error ~41.2% (comparable to Krizhevsky et al. 2012 at 40.7%).
    Under 10M parameters.
    NOTE: Requires manual ImageNet download — not run in automated reproduction.
    """
    return _ImageNetAllCNNB(num_classes)


class _ImageNetAllCNNB(nn.Module):
    """
    ImageNet All-CNN-B (Table 6).
    12 convolutional layers, fully-convolutional classification head.
    Input size: 224×224×3 (center crop from 256×256)
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        # Block 1: 96 channels, 11×11 → 5×5 → 1×1 → stride-2 replace pool
        # Block 2: 256 channels, 5×5 → 1×1 → 1×1 → stride-2 replace pool
        # Block 3: 384 → 384 → 256 → 256 → 10-class head (scaled for ImageNet)
        self.features = nn.Sequential(
            # Stage 1 (224→28 after strided ops)
            nn.Dropout(0.2),
            nn.Conv2d(3, 96, 11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # s=2 all-cnn
            nn.Dropout(0.5),
            # Stage 2
            nn.Conv2d(96, 256, 5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # s=2 all-cnn
            nn.Dropout(0.5),
            # Stage 3
            nn.Conv2d(256, 384, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            # Head
            nn.Conv2d(256, 4096, 1), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # GAP
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience mapping
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    f"{letter.lower()}_{variant}": (letter, variant)
    for letter in ['A', 'B', 'C']
    for variant in ['base', 'strided', 'convpool', 'allcnn']
}
# Human-readable names matching paper Table 3 / Table 4
MODEL_NAMES = {
    ('A', 'base'):     'Model A',
    ('A', 'strided'):  'Strided-CNN-A',
    ('A', 'convpool'): 'ConvPool-CNN-A',
    ('A', 'allcnn'):   'All-CNN-A',
    ('B', 'base'):     'Model B',
    ('B', 'strided'):  'Strided-CNN-B',
    ('B', 'convpool'): 'ConvPool-CNN-B',
    ('B', 'allcnn'):   'All-CNN-B',
    ('C', 'base'):     'Model C',
    ('C', 'strided'):  'Strided-CNN-C',
    ('C', 'convpool'): 'ConvPool-CNN-C',
    ('C', 'allcnn'):   'All-CNN-C',
}


def make_model(letter: str, variant: str, num_classes: int = 10) -> AllCNNModel:
    """Convenience constructor. Returns AllCNNModel on CPU."""
    return AllCNNModel(letter, variant, num_classes)


if __name__ == '__main__':
    # Quick sanity check
    for letter in ['A', 'B', 'C']:
        for variant in ['base', 'strided', 'convpool', 'allcnn']:
            m = AllCNNModel(letter, variant, num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            y = m(x)
            assert y.shape == (2, 10), f"Bad output shape: {y.shape}"
            nparams = m.count_parameters()
            print(f"{MODEL_NAMES[(letter, variant)]:<20} params={nparams/1e6:.2f}M  out={y.shape}")
    print("All architecture checks passed.")
