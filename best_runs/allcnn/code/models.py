"""
All-CNN model architectures for CIFAR-10/100.

Implements all variants from "Striving for Simplicity: The All Convolutional Net"
Springenberg et al., 2014 (arxiv 1412.6806)

Models:
  Letter A: 5×5 convolutions (simpler, single conv per block)
  Letter B: Network-in-Network variant (5×5 + 1×1 after each main conv)
  Letter C: All-3×3 variant (replaces all 5×5 of B with stacked 3×3+3×3)

Variants (for each letter):
  base:     Original architecture with MaxPool layers (Table 1)
  strided:  Strided-CNN — remove MaxPool, increase stride of preceding conv
  convpool: ConvPool-CNN — insert dense conv before each MaxPool, keep MaxPool
  allcnn:   All-CNN — replace MaxPool with strided 3×3 conv (paper's main result)

Architecture invariants (Section 2, Section 3.1):
  - Fully convolutional classifier: 1×1 conv → GAP → softmax (no FC layers)
  - Dropout: 20% on input image, 50% after each pooling layer (or its replacement)
  - ReLU activations throughout (no BN — paper predates common CIFAR BN use)
  - Block 1 (stage 1) uses 96 feature maps, Block 2 (stage 2) uses 192
  - Classification head (stage 3): conv(192,3×3) + conv(192,1×1) + conv(num_classes,1×1) + GAP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer primitives
# ---------------------------------------------------------------------------

def _conv_relu(in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int = 0):
    return [nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding),
            nn.ReLU(inplace=True)]


def _maxpool():
    # Paper uses 3×3 pool, stride=2 — with padding=1 this halves spatial dims
    return nn.MaxPool2d(3, stride=2, padding=1)


# ---------------------------------------------------------------------------
# _cifar_stage: build one downsampling stage (block) for a CIFAR model
# ---------------------------------------------------------------------------

def _cifar_stage(letter: str, variant: str, in_c: int, out_c: int) -> list:
    """
    Build one spatial downsampling stage (block) for a CIFAR All-CNN model.

    Args:
        letter:  'A', 'B', or 'C' — determines main kernel size and NiN usage
        variant: 'base', 'strided', 'convpool', or 'allcnn'
        in_c:    input channel count
        out_c:   output channel count

    Returns:
        list of nn.Module layers (to be extended into a Sequential)

    Structural rules per letter × variant (Section 3.1):
    Letter A (single 5×5 main conv per stage):
      base:     conv5(in→out) + MaxPool
      strided:  conv5(in→out, s=2)  [preceding conv gets stride 2]
      convpool: conv5(in→out) + conv3(out→out) + MaxPool
      allcnn:   conv5(in→out) + Conv2d(out,out,3,s=2)

    Letter B (5×5 + 1×1 NiN per stage):
      base:     conv5(in→out) + conv1(out→out) + MaxPool
      strided:  conv5(in→out) + Conv2d(out,out,1,s=2)  [1×1 NiN gets stride 2]
      convpool: conv5(in→out) + conv1(out→out) + conv3(out→out) + MaxPool
      allcnn:   conv5(in→out) + conv1(out→out) + Conv2d(out,out,3,s=2)

    Letter C (stacked 3×3+3×3 per stage):
      base:     conv3(in→out) + conv3(out→out) + MaxPool
      strided:  conv3(in→out) + Conv2d(out,out,3,padding=1,stride=2) [2nd 3×3 gets stride=2, no MaxPool]
      convpool: conv3(in→out) + conv3(out→out) + Conv2d(out,out,3,padding=1) + MaxPool
      allcnn:   conv3(in→out) + conv3(out→out) + Conv2d(out,out,3,padding=1,stride=2) [replaces MaxPool]
    """
    assert letter in ('A', 'B', 'C'), f"Unknown letter {letter!r}"
    assert variant in ('base', 'strided', 'convpool', 'allcnn'), f"Unknown variant {variant!r}"

    layers = []

    if letter == 'A':
        # --- Single 5×5 main conv ---
        if variant == 'strided':
            # Preceding conv (the only main conv) gets stride=2 — no MaxPool
            layers += _conv_relu(in_c, out_c, 5, stride=2, padding=2)
        else:
            layers += _conv_relu(in_c, out_c, 5, stride=1, padding=2)
            if variant == 'convpool':
                # Extra dense conv before MaxPool
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)]
            elif variant == 'allcnn':
                # Replace MaxPool with stride-2 3×3 conv (out_c==in_c for pool replacement)
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1, stride=2), nn.ReLU(inplace=True)]

    elif letter == 'B':
        # --- 5×5 main conv + 1×1 NiN conv ---
        layers += _conv_relu(in_c, out_c, 5, stride=1, padding=2)
        if variant == 'strided':
            # 1×1 NiN conv (immediately preceding MaxPool) gets stride=2 — no MaxPool
            layers += [nn.Conv2d(out_c, out_c, 1, stride=2), nn.ReLU(inplace=True)]
        else:
            # Standard 1×1 NiN after each spatial conv (model B adds conv1x1 after each spatial conv)
            layers += [nn.Conv2d(out_c, out_c, 1), nn.ReLU(inplace=True)]
            if variant == 'convpool':
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)]
            elif variant == 'allcnn':
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1, stride=2), nn.ReLU(inplace=True)]

    elif letter == 'C':
        # --- Stacked 3×3 + 3×3 (model C uses stacked 3×3+3×3) ---
        layers += _conv_relu(in_c, out_c, 3, stride=1, padding=1)
        if variant == 'strided':
            # Second 3×3 conv gets stride=2, no new layer and no MaxPool
            layers += [nn.Conv2d(out_c, out_c, 3, padding=1, stride=2), nn.ReLU(inplace=True)]
        else:
            layers += _conv_relu(out_c, out_c, 3, stride=1, padding=1)
            if variant == 'convpool':
                # Extra dense conv immediately before MaxPool — same 3×3 kernel, no stride
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)]
            elif variant == 'allcnn':
                # Replace MaxPool with stride-2 3×3 conv (out_c==in_c)
                layers += [nn.Conv2d(out_c, out_c, 3, padding=1, stride=2), nn.ReLU(inplace=True)]

    # MaxPool for base and convpool variants
    if variant in ('base', 'convpool'):
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

    return layers


# ---------------------------------------------------------------------------
# CIFARNet: the primary model class built on _cifar_stage
# ---------------------------------------------------------------------------

class CIFARNet(nn.Module):
    """
    Fully-convolutional CIFAR classification network.
    Implements all A/B/C letters and base/strided/convpool/allcnn variants.

    Architecture (stages 1-3):
      Stage 1: _cifar_stage(letter, variant, in_c=3,  out_c=96)  + Dropout(0.5)
      Stage 2: _cifar_stage(letter, variant, in_c=96, out_c=192) + Dropout(0.5)
      Stage 3 (head): conv3×3(192→192) + conv1×1(192→192) + conv1×1(192→num_classes)
      Output: GAP → (B, num_classes)

    Dropout: 20% on input, 50% after each stage's pooling (or its replacement).
    No FC layers — fully convolutional inference (Section 2).
    """

    def __init__(self, letter: str, variant: str, num_classes: int = 10):
        super().__init__()
        self.letter = letter
        self.variant = variant
        self.num_classes = num_classes

        layers = []

        # Input dropout (20%) — Section 3.2
        layers.append(nn.Dropout(0.2))

        # Stage 1 (3 → 96)
        layers += _cifar_stage(letter, variant, in_c=3, out_c=96)
        layers.append(nn.Dropout(0.5))

        # Stage 2 (96 → 192)
        layers += _cifar_stage(letter, variant, in_c=96, out_c=192)
        layers.append(nn.Dropout(0.5))

        # Stage 3 (classification head — fully convolutional)
        # stage3 terminates in nn.Conv2d(192, num_classes, 1) → GAP (Section 2)
        layers += _conv_relu(192, 192, 3, padding=1)
        layers += _conv_relu(192, 192, 1)
        layers.append(nn.Conv2d(192, num_classes, 1))  # stage3 final 1×1 → num_classes

        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, num_classes, H, W)
        x = x.mean(dim=[2, 3])        # Global Average Pool → (B, num_classes)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"letter={self.letter!r}, variant={self.variant!r}, num_classes={self.num_classes}"


# ---------------------------------------------------------------------------
# AllCNNModel: alias / backward-compat wrapper for CIFARNet
# ---------------------------------------------------------------------------

# AllCNNModel is identical to CIFARNet (backward compatibility with train_cell.py)
AllCNNModel = CIFARNet


def build_all_cnn(letter: str, variant: str, num_classes: int = 10) -> nn.Sequential:
    """
    Build one All-CNN network body (without GAP wrapper) for backward compat.
    Prefer CIFARNet / AllCNNModel for new code.
    """
    net = CIFARNet(letter, variant, num_classes)
    return net.features


def make_model(letter: str, variant: str, num_classes: int = 10) -> CIFARNet:
    """Convenience constructor. Returns CIFARNet on CPU."""
    return CIFARNet(letter, variant, num_classes)


# ---------------------------------------------------------------------------
# Model registry
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


# ---------------------------------------------------------------------------
# ImageNet All-CNN-B (Table 6 of the paper) — OUT OF SCOPE for this run
# ---------------------------------------------------------------------------

class _ImageNetAllCNNB(nn.Module):
    """
    ImageNet All-CNN-B (Table 6 Appendix).
    12 convolutional layers, fully-convolutional classification head.
    Input size: 224×224×3 (center crop from 256×256).
    Trained for 450,000 iterations, batch=64, initial lr=0.01 divided by 10
    every 200,000 iterations, weight decay λ=0.0005.
    Target: Top-1 error ~41.2% (comparable to Krizhevsky et al. 2012 at 40.7%).
    Under 10M parameters.
    NOTE: Requires manual ImageNet download — declared in scope.gaps.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        # 12 conv layers as described in the paper appendix (Table 6)
        self.features = nn.Sequential(
            nn.Dropout(0.2),
            # Stage 1
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


def build_imagenet_all_cnn_b(num_classes: int = 1000) -> _ImageNetAllCNNB:
    """Upscaled All-CNN-B for ImageNet (Table 6). OUT OF SCOPE — ImageNet needs manual download."""
    return _ImageNetAllCNNB(num_classes)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for letter in ['A', 'B', 'C']:
        for variant in ['base', 'strided', 'convpool', 'allcnn']:
            m = CIFARNet(letter, variant, num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            y = m(x)
            assert y.shape == (2, 10), f"Bad output shape: {y.shape}"
            nparams = m.count_parameters()
            print(f"{MODEL_NAMES[(letter, variant)]:<20} params={nparams/1e6:.3f}M  out={y.shape}")
    print("All architecture checks passed.")
