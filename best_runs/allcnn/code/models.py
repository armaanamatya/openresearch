"""
All Convolutional Net architectures — Springenberg et al., ICLR 2015
arXiv:1412.6806  "Striving for Simplicity: The All Convolutional Net"

Implements:
  CIFAR models (Section 3.1, Tables 1-2):
    Base models A / B / C  ×  four variants
      Base        — original max-pooling
      Strided-CNN — last stage-conv strided (absorbs pool, no new params)
      ConvPool-CNN— extra 3×3 conv before pool (keeps pool, adds params)
      All-CNN     — extra 3×3 stride-2 conv replaces pool (adds params, no pool)

    Model A: uses 5×5 spatial convolutions
    Model B: NiN variant — 5×5 + 1×1 after each spatial conv
    Model C: replaces all 5×5 with stacked 3×3 + 3×3

  ImageNet All-CNN-B (Section 3.3, Table 6):
    12 convolutional layers, conv1–conv12
    Input 224×224 — max-pooling replaced by strided convolutions

Architecture details per Sections 3.1–3.3 + Tables 1, 2, 6.

Dropout:
  - 20% on input image (Dropout2d)
  - 50% after each pooling layer (or its replacement)
  weight decay λ=0.001 (CIFAR), λ=0.0005 (ImageNet) applied by optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# CIFAR helper: build one feature-extraction "stage"
# ---------------------------------------------------------------------------

def _cifar_stage(
    in_c: int,
    out_c: int,
    model_id: str,   # 'A', 'B', 'C'
    pool_mode: str,  # 'base', 'strided', 'convpool', 'allcnn'
) -> List[nn.Module]:
    """
    Return a list of modules for one CIFAR feature stage.

    Spatial dimensions (for 32×32 input):
      After stage 1: 16×16   After stage 2: 8×8

    Parameter counts (no bias counted):
      Base       — spatial_params + 0 (pool free)
      Strided    — spatial_params + 0 (stride absorbed into last spatial conv)
      ConvPool   — spatial_params + out_c*out_c*9 (extra 3×3) + 0 (pool free)
      All-CNN    — spatial_params + out_c*out_c*9 (extra strided 3×3, replaces pool)

    ConvPool ≈ All-CNN in parameters — the key parameter-matched comparison in
    Section 3.2 ruling out that gains stem solely from extra parameters.
    """
    layers: List[nn.Module] = []
    # inplace=False required for guided backpropagation compatibility
    ReLU = lambda: nn.ReLU(inplace=False)

    # ---------- feature extraction (differs by model_id) ----------
    if model_id == 'A':
        # Single 5×5 convolution
        if pool_mode == 'strided':
            # Absorb downsampling into the 5×5 (stride=2, no new layer)
            layers += [nn.Conv2d(in_c, out_c, 5, padding=2, stride=2), ReLU()]
        else:
            layers += [nn.Conv2d(in_c, out_c, 5, padding=2), ReLU()]

    elif model_id == 'B':
        # NiN: 5×5 + 1×1 (Section 3.1 — "one 1×1 convolution after each normal conv")
        if pool_mode == 'strided':
            layers += [nn.Conv2d(in_c, out_c, 5, padding=2, stride=2), ReLU()]
        else:
            layers += [nn.Conv2d(in_c, out_c, 5, padding=2), ReLU()]
        # 1×1 mixer always at stride=1
        layers += [nn.Conv2d(out_c, out_c, 1), ReLU()]

    elif model_id == 'C':
        # Two 3×3 stacked — replaces each 5×5 (Section 3.1, rubric item)
        if pool_mode == 'strided':
            # Second 3×3 becomes strided (absorbs pool, no new layer)
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1),   ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1, stride=2), ReLU(),
            ]
        else:
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1), ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), ReLU(),
            ]
    else:
        raise ValueError(f"Unknown model_id: {model_id!r}")

    # ---------- downsampling (for non-strided variants) ----------
    if pool_mode == 'base':
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))
    elif pool_mode == 'strided':
        pass  # stride already embedded in the last feature conv above
    elif pool_mode == 'convpool':
        # Extra 3×3 conv (stride=1) then max-pool
        layers += [
            nn.Conv2d(out_c, out_c, 3, padding=1), ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        ]
    elif pool_mode == 'allcnn':
        # Extra 3×3 stride=2 conv replaces pool — same params as convpool
        layers += [
            nn.Conv2d(out_c, out_c, 3, padding=1, stride=2), ReLU(),
        ]
    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

    return layers


class CIFARNet(nn.Module):
    """
    Unified CIFAR-10/100 network covering all 12 variants in Tables 1–2.

    Architecture (Section 3.1):
      Input: 3×32×32
      Dropout(0.2) on input
      Stage 1:  feature convs (A/B/C) → downsampling → Dropout(0.5)
      Stage 2:  feature convs (A/B/C) → downsampling → Dropout(0.5)
      Stage 3:  Conv(192, 3×3, valid) → Conv(192, 1×1) → Conv(num_classes, 1×1)
      GlobalAvgPool → logits

    All models terminate in a global average pooling layer (NiN convention),
    removing fully-connected layers entirely.
    """

    def __init__(
        self,
        model_id: str = 'C',     # 'A', 'B', 'C'
        pool_mode: str = 'allcnn',  # 'base', 'strided', 'convpool', 'allcnn'
        num_classes: int = 10,
        dropout_input: float = 0.2,   # Section 3.2: 20% on input
        dropout_pool: float = 0.5,    # Section 3.2: 50% after each pool/replacement
    ):
        super().__init__()
        self.model_id = model_id
        self.pool_mode = pool_mode
        self.num_classes = num_classes

        self.drop_input = nn.Dropout2d(dropout_input)

        # Stage 1: 3 → 96 channels
        stage1 = _cifar_stage(3, 96, model_id, pool_mode)
        self.stage1 = nn.Sequential(*stage1)
        self.drop1 = nn.Dropout2d(dropout_pool)

        # Stage 2: 96 → 192 channels
        stage2 = _cifar_stage(96, 192, model_id, pool_mode)
        self.stage2 = nn.Sequential(*stage2)
        self.drop2 = nn.Dropout2d(dropout_pool)

        # Stage 3: classifier (valid conv + 1×1s + global avg pool)
        # 8×8 → valid 3×3 → 6×6 → 1×1 → 1×1 → GAP → num_classes
        self.stage3 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=0),   # valid (no padding): 8→6
            nn.ReLU(inplace=False),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(192, num_classes, 1),
            nn.ReLU(inplace=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Weight init: Kaiming He uniform (common for ReLU networks)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_input(x)
        x = self.stage1(x)
        x = self.drop1(x)
        x = self.stage2(x)
        x = self.drop2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.flatten(1)
        return x  # raw logits — use cross-entropy loss

    def extra_repr(self) -> str:
        return f"model_id={self.model_id!r}, pool_mode={self.pool_mode!r}, num_classes={self.num_classes}"


# ---------------------------------------------------------------------------
# ImageNet All-CNN-B  (Table 6, Section 3.3)
# ---------------------------------------------------------------------------

class ImageNetAllCNNB(nn.Module):
    """
    All-CNN-B for ImageNet (ILSVRC-2012), Table 6.

    12 convolutional layers (conv1–conv12).  Model B style (NiN: 1×1 after
    each spatial conv).  All max-pooling replaced by stride-2 convolutions
    (All-CNN convention).

    Training spec (Section 3.3):
      - 450,000 iterations, batch size 64
      - SGD, momentum 0.9
      - Initial LR γ=0.01 divided by 10 every 200,000 iterations
      - Weight decay λ=0.0005
      - Center 224×224 crop for validation (no multi-crop)
      - Fewer than 10M parameters

    Spatial trace (224×224 input):
      conv1  (stride 4): 224 → 55
      conv3  (stride 2): 55  → 27
      conv6  (stride 2): 27  → 14
      conv9  (stride 2): 14  →  7
      conv10 (valid 3×3): 7  →  5
      GAP                5  →  1

    NOTE: ImageNet data requires manual download from image-net.org and
    is NOT auto-downloadable.  This architecture is provided for completeness;
    the model is not trained in this reproduction (scope.gaps).
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # conv1–conv3: first feature block + All-CNN strided replacement
        self.block1 = nn.Sequential(
            nn.Dropout2d(0.2),
            # conv1: large-stride entry — 224 → 55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=False),
            # conv2: NiN 1×1 mixer
            nn.Conv2d(96, 96, kernel_size=1),
            nn.ReLU(inplace=False),
            # conv3: All-CNN pool replacement — 55 → 27
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.5),
        )
        # conv4–conv6: second feature block
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            # conv5: NiN 1×1
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=False),
            # conv6: All-CNN pool replacement — 27 → 14
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.5),
        )
        # conv7–conv9: third feature block
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            # conv8: NiN 1×1
            nn.Conv2d(384, 384, kernel_size=1),
            nn.ReLU(inplace=False),
            # conv9: All-CNN pool replacement — 14 → 7
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.5),
        )
        # conv10–conv12: classifier head
        self.classifier = nn.Sequential(
            # conv10: valid 3×3 — 7 → 5
            nn.Conv2d(384, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=False),
            # conv11: 1×1
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=False),
            # conv12: 1×1 → num_classes
            nn.Conv2d(1024, num_classes, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        x = self.gap(x)
        return x.flatten(1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cifar_model(
    arch: str,           # e.g. 'allcnn_c', 'base_a', 'strided_b', ...
    num_classes: int = 10,
) -> CIFARNet:
    """
    Build a CIFAR model from a combined architecture string.

    Format: '<pool_mode>_<model_id>'  (all lowercase, underscore-separated)
    Examples: 'allcnn_c', 'base_a', 'strided_b', 'convpool_c', 'allcnn_a'

    Aliases also accepted: 'allcnn-c', 'all_cnn_c', 'base-a', etc.
    """
    arch_clean = arch.lower().replace('-', '_').replace(' ', '_')

    # Parse pool_mode and model_id
    for pool in ('allcnn', 'strided', 'convpool', 'base'):
        if arch_clean.startswith(pool):
            remainder = arch_clean[len(pool):].lstrip('_')
            model_id = remainder.upper()
            if model_id not in ('A', 'B', 'C'):
                raise ValueError(f"Unknown model_id {model_id!r} in arch={arch!r}")
            return CIFARNet(model_id=model_id, pool_mode=pool, num_classes=num_classes)
    raise ValueError(
        f"Cannot parse arch={arch!r}. Use format '<pool>_<id>', e.g. 'allcnn_c'."
    )


# ---------------------------------------------------------------------------
# Layer registry for guided-backpropagation visualization
# ---------------------------------------------------------------------------

CIFAR_LAYER_NAMES = {
    'stage1': 'conv1–conv_stage1',
    'stage2': 'conv_stage2',
    'stage3.0': 'conv_final_3x3',
    'stage3.2': 'conv_final_1x1',
    'stage3.4': 'conv_classifier',
}

IMAGENET_LAYER_NAMES = {
    'block1.1': 'conv1',
    'block1.3': 'conv2',
    'block1.5': 'conv3',
    'block2.0': 'conv4',
    'block2.2': 'conv5',
    'block2.4': 'conv6',
    'block3.0': 'conv7',
    'block3.2': 'conv8',
    'block3.4': 'conv9',
    'classifier.0': 'conv10',
    'classifier.2': 'conv11',
    'classifier.4': 'conv12',
}
