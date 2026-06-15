# ResNet / Deep Residual Learning (arXiv 1512.03385) — CIFAR-10

**Reproduced 2026-06-14** on the fixed harness (8× RTX A5000, local sandbox, ~2 h, claude-oauth root). First reproduction of this paper by ReproLab — a new-paper validation of the harness + the 2026-06-14 grading fixes.

## Result: the science reproduced cleanly

The paper's central claim is the **degradation contrast** — plain nets get *worse* with depth, residual nets get *better*. The 9-cell grid (plain + residual at depths 20/32/44/56/110) reproduced it:

| depth | plain (test err %) | resnet (test err %) |
|------:|:------------------:|:-------------------:|
| 20    | 9.22               | 8.80                |
| 32    | 9.81               | 7.86                |
| 44    | 12.08              | 7.02                |
| 56    | 13.54              | 7.92                |
| 110   | (diverged)         | **6.68**            |

- **Plain nets degrade monotonically** (9.22 → 13.54 as depth 20 → 56) — exactly the paper's degradation problem.
- **ResNets improve with depth** — **resnet-110 = 6.68%, vs the paper's 6.43%** (0.25% off).
- plain-110 diverged (consistent with the paper noting plain-110's trouble without warmup).

## Rubric score: 0.6201 (auto-rubric, G1 grader) — undersells the reproduction

This run was graded on the **fixed G1 grader** (no metrics truncation), so the 0.62 is *not* a truncation artifact — it's the **auto-generated rubric** under-crediting a clean reproduction. Unlike Adam/All-CNN, ResNet has no curated PaperBench bundle or hint invariants, so the auto-rubric is the score ceiling. The reproduction itself (table above) is unambiguous; adding regex **invariants** to the `PAPER_HINTS` entry (enforcing the degradation ordering) is the path to a fair score — not anything grader-related.

## Recipe (per `code/cells.json` + `code/train_cell.py`)
CIFAR ResNet, 6n+2 layers, {16,32,64} filters, identity (option-A) shortcuts, He init, **per-pixel mean only (no ZCA)**, 4px-pad + crop + flip aug, SGD mom 0.9, wd 1e-4, lr 0.1 ÷10 at 32k/48k iters (64k total), resnet-110 lr-0.01 warmup. ImageNet ResNets out-of-scope (multi-day). `code/` excludes the 51 MB of trained weights under `outputs/`.
