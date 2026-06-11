"""
guided_backprop.py — Guided Backpropagation visualization for All-CNN.

Implements the visualization technique from Section 4 of:
  "Striving for Simplicity: The All Convolutional Net" (arXiv 1412.6806)

Guided Backpropagation rule (Springenberg et al. 2014, citing Zeiler & Fergus 2013
+ Simonyan et al. 2013):
  "In guided backpropagation we additionally zero out the gradient
   in locations where the input to the ReLU during the forward pass was negative."

Combined rule at each ReLU:
  grad_out = grad_in * (input > 0).float() * (grad_in > 0).float()

where:
  - (input > 0) is the standard backprop ReLU mask
  - (grad_in > 0) is the deconvnet-style mask (only propagate positive gradients)

This produces "clean" saliency maps that highlight which input pixels
most positively contributed to a given class prediction.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Guided backprop hooks
# ---------------------------------------------------------------------------

class GuidedBackpropReLU(nn.Module):
    """Drop-in replacement for nn.ReLU that applies the guided backprop rule.

    Forward: standard ReLU (zero negatives).
    Backward: additionally zero gradient where either:
        (a) the forward input was negative, OR
        (b) the incoming gradient is negative.

    This is the 'guided backpropagation' rule from:
      Springenberg et al. 2014, Sec. 4 / Zeiler & Fergus 2013 / Simonyan 2013.
    """

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
        self._input_tensor: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for backward hook
        self._input_tensor = x.clone().detach()
        return torch.relu(x)

    def backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> tuple:
        """Custom backward hook implementing guided backpropagation."""
        grad = grad_output[0]  # gradient flowing back from above

        # Mask 1: standard ReLU — zero where forward input was negative
        fwd_mask = (self._input_tensor > 0).float()

        # Mask 2: guided — zero where incoming gradient is negative
        guided_mask = (grad > 0).float()

        # Combined: zero unless BOTH forward input > 0 AND incoming grad > 0
        guided_grad = grad * fwd_mask * guided_mask

        return (guided_grad,)


def _replace_relu_with_guided(model: nn.Module) -> list:
    """
    Recursively replace all nn.ReLU with GuidedBackpropReLU in-place.

    Returns a list of registered hook handles for cleanup.
    """
    handles = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU):
            gbp_relu = GuidedBackpropReLU(inplace=module.inplace)
            # Walk down to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], gbp_relu)
            # Register backward hook on the replaced module
            h = gbp_relu.register_backward_hook(gbp_relu.backward_hook)
            handles.append(h)
    return handles


def compute_guided_backprop(
    model: nn.Module,
    image_batch: torch.Tensor,
    target_class: Optional[int],
    device: str,
) -> torch.Tensor:
    """
    Compute guided backpropagation saliency for a batch of images.

    Args:
        model:        Trained model with nn.ReLU activations (will be modified).
        image_batch:  [N, C, H, W] float tensor on CPU.
        target_class: Class index to backprop towards (None = predicted class).
        device:       'cuda' or 'cpu'.

    Returns:
        saliency: [N, C, H, W] float32 tensor on CPU with gradient magnitudes.
    """
    model.eval()

    # Replace ReLUs with guided versions
    handles = _replace_relu_with_guided(model)

    try:
        imgs = image_batch.to(device)
        imgs.requires_grad_(True)

        with torch.enable_grad():
            logits = model(imgs)  # [N, num_classes]
            if target_class is None:
                # Use predicted classes
                target = logits.argmax(dim=1)  # [N]
            else:
                target = torch.full(
                    (imgs.shape[0],), target_class, dtype=torch.long, device=device
                )

            # Sum of logits for the target classes
            score = logits[torch.arange(imgs.shape[0], device=device), target].sum()
            score.backward()

        saliency = imgs.grad.detach().cpu().abs()  # [N, C, H, W]

    finally:
        # Remove hooks and restore original ReLUs (best-effort)
        for h in handles:
            h.remove()

    return saliency


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _normalize_saliency(s: np.ndarray) -> np.ndarray:
    """Normalize saliency map to [0, 1] for display.

    Args:
        s: [C, H, W] or [H, W] float array.

    Returns:
        [H, W] float array in [0, 1].
    """
    # Collapse channels: take max across channels
    if s.ndim == 3:
        s = s.max(axis=0)
    s = s - s.min()
    denom = s.max()
    if denom > 1e-8:
        s = s / denom
    return s


def _unnormalize_image(img: np.ndarray) -> np.ndarray:
    """Convert whitened image to displayable [H, W, 3] uint8.

    ZCA-whitened images have arbitrary value range; clip to [-3σ, +3σ] and
    rescale to [0, 255].
    """
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
    std = img.std()
    mean = img.mean()
    img = np.clip(img, mean - 3 * std, mean + 3 * std)
    img = img - img.min()
    denom = img.max()
    if denom > 1e-8:
        img = img / denom
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point called from train_cell.py
# ---------------------------------------------------------------------------

def run_guided_backprop(
    model: nn.Module,
    test_loader,
    device: str,
    output_dir: str,
    num_images: int = 8,
    target_class: Optional[int] = None,
) -> dict:
    """
    Run guided backpropagation on ``num_images`` test images, save figure.

    Produces:
        $output_dir/fig_relu_masking.png   — paper-style visualization
        $output_dir/guided_backprop.json   — saliency JSON sidecar

    Returns:
        dict with keys: images_processed, output_dir, fig_path, json_path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect a batch of images from test loader
    images_list = []
    labels_list = []
    for imgs, lbls in test_loader:
        images_list.append(imgs)
        labels_list.append(lbls)
        if sum(x.shape[0] for x in images_list) >= num_images:
            break

    if not images_list:
        print("[guided_backprop] WARNING: empty test loader", flush=True)
        return {"images_processed": 0, "output_dir": output_dir}

    all_images = torch.cat(images_list, dim=0)[:num_images]    # [N, C, H, W]
    all_labels = torch.cat(labels_list, dim=0)[:num_images]    # [N]

    print(f"[guided_backprop] Computing saliency for {len(all_images)} images...", flush=True)

    try:
        saliency = compute_guided_backprop(model, all_images, target_class, device)
    except Exception as e:
        print(f"[guided_backprop] Saliency computation failed: {e}", flush=True)
        return {"images_processed": 0, "output_dir": output_dir, "error": str(e)}

    # Save JSON sidecar: per-image saliency statistics
    saliency_stats = []
    for i in range(len(all_images)):
        s = saliency[i].numpy()
        saliency_stats.append({
            "image_idx": i,
            "true_label": int(all_labels[i].item()),
            "saliency_mean": float(s.mean()),
            "saliency_max": float(s.max()),
            "saliency_min": float(s.min()),
        })

    json_path = os.path.join(output_dir, "guided_backprop.json")
    import json
    with open(json_path, "w") as f:
        json.dump({
            "method": "guided_backpropagation",
            "paper_section": "4",
            "rule": "zero grad where input<0 OR incoming_grad<0",
            "num_images": len(all_images),
            "target_class": target_class,
            "saliency_stats": saliency_stats,
        }, f, indent=2)

    # Plot
    fig_path = os.path.join(output_dir, "fig_relu_masking.png")
    if HAS_MPL:
        try:
            n = len(all_images)
            fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
            if n == 1:
                axes = axes.reshape(2, 1)

            for i in range(n):
                img_np = all_images[i].numpy()
                sal_np = saliency[i].numpy()

                # Top row: original (whitened) image
                axes[0, i].imshow(_unnormalize_image(img_np))
                axes[0, i].set_title(f"cls={int(all_labels[i])}", fontsize=7)
                axes[0, i].axis("off")

                # Bottom row: saliency map
                sal_disp = _normalize_saliency(sal_np)
                axes[1, i].imshow(sal_disp, cmap="hot", vmin=0, vmax=1)
                axes[1, i].set_title("saliency", fontsize=7)
                axes[1, i].axis("off")

            axes[0, 0].set_ylabel("Input", fontsize=8)
            axes[1, 0].set_ylabel("Guided\nBackprop", fontsize=8)

            fig.suptitle(
                "Guided Backpropagation (All-CNN-C, CIFAR-10)\n"
                "Zero gradient where input<0 OR incoming_grad<0 [Sec. 4]",
                fontsize=9,
            )
            fig.tight_layout()
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[guided_backprop] Figure saved: {fig_path}", flush=True)
        except Exception as e:
            print(f"[guided_backprop] Figure save failed: {e}", flush=True)
            fig_path = None
    else:
        print("[guided_backprop] matplotlib not available, skipping figure", flush=True)
        fig_path = None

    print(f"[guided_backprop] Done. JSON: {json_path}", flush=True)
    return {
        "images_processed": len(all_images),
        "output_dir": output_dir,
        "fig_path": fig_path,
        "json_path": json_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point (for standalone testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import AllCNNModel
    from data import load_cifar_fast

    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/guided_backprop_test")
    data_root = os.path.join(output_dir, "datasets")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[guided_backprop CLI] device={device}, output_dir={output_dir}")

    model = AllCNNModel("C", "allcnn", num_classes=10)
    model = model.to(device)

    _, test_loader, _ = load_cifar_fast(
        dataset="cifar10",
        data_root=data_root,
        augment_train=False,
        use_zca=True,
        zca_cache_dir=os.path.join(data_root, "zca_cache"),
        batch_size=16,
        num_workers=0,
        subsample_n=32,
    )

    result = run_guided_backprop(model, test_loader, device, output_dir, num_images=4)
    print(_json.dumps(result, indent=2, default=str))
