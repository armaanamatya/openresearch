# All-CNN Reproduction — Springenberg et al. 2014 (arXiv 1412.6806)

## What was reproduced

Reproduces the core claim: max-pooling in CNNs can be replaced by strided
convolutions without loss of accuracy (Section 2 / Table 3-4).

Implements all 12 CIFAR model variants × letter (A/B/C) × variant
(base/strided/convpool/allcnn):
- **Model A**: single 5×5 conv per block
- **Model B**: 5×5 + 1×1 NiN per block
- **Model C**: stacked 3×3+3×3 per block
- **base**: standard MaxPool(3×3, stride=2) downsampling
- **strided**: preceding conv stride raised to 2 (Strided-CNN)
- **convpool**: extra 3×3 conv before MaxPool (ConvPool-CNN)
- **allcnn**: MaxPool replaced by 3×3 stride-2 conv (All-CNN)

Additional cells: All-CNN-C on CIFAR-10 with augmentation and CIFAR-100
with augmentation (identical hyperparameters, no re-tuning).

Training recipe (Section 3.2): SGD momentum=0.9, weight_decay=0.001,
350 epochs, lr=0.05 × 0.1 at epochs 200/250/300, batch_size=128.
Preprocessing: GCN (scale=55, Goodfellow et al. 2013) + ZCA whitening
(ε=0.1). Augmentation: horizontal flip + random ±5px translation.
Dropout: 20% on input, 50% after each downsampling stage.

Guided backpropagation visualization (Section 4) produced for All-CNN-C.

## What was omitted and why

**ImageNet experiment**: Requires manual download from image-net.org with
license agreement — infeasible in sandboxed environment. Declared in
`scope.gaps`. Paper reports All-CNN-B: Top-1 41.2%.

## How to read metrics.json

Top-level contract paths (flat floats, 0-100 accuracy scale):
- `cifar10_allcnn_c_test_accuracy`: All-CNN-C CIFAR-10 test accuracy %
- `cifar10_maxpool_baseline_test_accuracy`: Model C (MaxPool) test accuracy %
- `cifar10_allcnn_c_final_train_loss`: All-CNN-C final epoch training loss
- `cifar10_accuracy_gap_allcnn_minus_maxpool`: accuracy difference (All-CNN minus MaxPool)

`per_model`: per-cell results keyed by model_key (e.g. `c_allcnn`, `c_base`)
`per_dataset.CIFAR-10`: full table for paper's Table 3 comparison
`history.c_allcnn_noaug`: per-epoch training curves for All-CNN-C CIFAR-10 noaug
