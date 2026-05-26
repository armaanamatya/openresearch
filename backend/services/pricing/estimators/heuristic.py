"""Closed-form heuristic wall-clock estimator.

Always available (no data, no LLM call).  Uses a per-category × model-size
table seeded from public reproduction runtimes.  Conservative sigma = 50% of
mean — a wide safety net for the cold-start regime.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §heuristic
"""

from __future__ import annotations

import logging

from backend.services.pricing.ensemble import PointEstimate
from backend.services.pricing.paper_features import PaperFeatures

logger = logging.getLogger(__name__)

# Hours per experiment for category × model_size_class combinations.
# "experiment" = one full training run to convergence.
# Seeds from public reproductions and Weights & Biases run history.
#
# Rows: category ("vae", "optimizer", "cnn", "rl", "nlp", "other")
# Cols: model_size_class ("tiny", "small", "medium", "large", "unknown")
_HOURS_PER_EXPERIMENT: dict[tuple[str, str], float] = {
    # VAE / diffusion / GAN
    ("vae", "tiny"):    0.5,   # MNIST VAE on RTX 4090
    ("vae", "small"):   1.5,   # CIFAR-10 DCGAN
    ("vae", "medium"):  6.0,   # CelebA HQ diffusion
    ("vae", "large"):   40.0,  # Stable Diffusion fine-tune
    ("vae", "unknown"): 3.0,
    # Optimizer papers (often train to convergence on standard benchmarks)
    ("optimizer", "tiny"):    0.2,
    ("optimizer", "small"):   0.5,
    ("optimizer", "medium"):  2.0,
    ("optimizer", "large"):   12.0,
    ("optimizer", "unknown"): 1.0,
    # CNN / vision classification
    ("cnn", "tiny"):    0.3,   # LeNet-5 on MNIST
    ("cnn", "small"):   2.0,   # ResNet-18 on CIFAR-10
    ("cnn", "medium"):  8.0,   # ResNet-50 on ImageNet (subset)
    ("cnn", "large"):   50.0,  # ViT-L on ImageNet full
    ("cnn", "unknown"): 4.0,
    # RL / policy gradient
    ("rl", "tiny"):    0.5,
    ("rl", "small"):   2.0,
    ("rl", "medium"):  10.0,
    ("rl", "large"):   48.0,
    ("rl", "unknown"): 4.0,
    # NLP (fine-tuning / training from scratch)
    ("nlp", "tiny"):    0.5,   # DistilBERT on SST-2
    ("nlp", "small"):   2.0,   # BERT-base fine-tune
    ("nlp", "medium"):  12.0,  # GPT-2 medium fine-tune
    ("nlp", "large"):   80.0,  # GPT-3 fine-tune (proxy)
    ("nlp", "unknown"): 6.0,
    # Other / unknown category
    ("other", "tiny"):    1.0,
    ("other", "small"):   3.0,
    ("other", "medium"):  10.0,
    ("other", "large"):   50.0,
    ("other", "unknown"): 4.0,
}

_COLD_START_SIGMA_RATIO: float = 0.5  # σ = 50% of μ — spec requirement


def _hours_per_experiment(category: str, model_size_class: str) -> float:
    key = (category, model_size_class)
    if key in _HOURS_PER_EXPERIMENT:
        return _HOURS_PER_EXPERIMENT[key]
    # Fallback: use "other" row with the same size class, then "other/unknown"
    fallback_key = ("other", model_size_class)
    return _HOURS_PER_EXPERIMENT.get(fallback_key, _HOURS_PER_EXPERIMENT[("other", "unknown")])


def estimate_heuristic(features: PaperFeatures) -> PointEstimate:
    """Compute a closed-form wall-clock estimate in hours.

    The estimate is deliberately coarse — σ = 50% of μ signals to the
    inverse-variance combiner that this source has moderate uncertainty.
    When k-NN and LLM sources are available, their lower σ will dominate.

    Returns:
        PointEstimate with mean = wall_clock_hours, sigma = 0.5 * mean.
    """
    h_per_exp = _hours_per_experiment(features.category, features.model_size_class)
    mean_hours = features.num_experiments * h_per_exp

    # Guard: if num_experiments is 0, return a minimal estimate
    if mean_hours <= 0:
        mean_hours = h_per_exp

    sigma_hours = _COLD_START_SIGMA_RATIO * mean_hours

    logger.debug(
        "heuristic: category=%s size=%s n_exp=%d → %.2fh ± %.2fh",
        features.category,
        features.model_size_class,
        features.num_experiments,
        mean_hours,
        sigma_hours,
    )
    return PointEstimate(
        mean=mean_hours,
        sigma=sigma_hours,
        source="heuristic",
        n_samples=0,
        detail={
            "category": features.category,
            "model_size_class": features.model_size_class,
            "hours_per_experiment": h_per_exp,
            "num_experiments": features.num_experiments,
        },
    )
