"""Tests for paper_features.py — category classification and feature extraction.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §paper_features
"""

from __future__ import annotations

import pytest

from backend.services.pricing.paper_features import (
    PaperFeatures,
    _classify_category,
    _classify_model_size,
    extract_features,
)


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_category", [
    (
        "We propose a policy gradient method using PPO to train an agent on Atari games.",
        "rl",
    ),
    (
        "We fine-tune BERT on GLUE benchmarks using attention and transformer layers.",
        "nlp",
    ),
    (
        "We train a ResNet-50 on ImageNet using convolutional layers.",
        "cnn",
    ),
    (
        "We train a variational autoencoder (VAE) on CIFAR-10.",
        "vae",
    ),
    (
        "We propose a new optimizer based on Adam with momentum and weight decay.",
        "optimizer",
    ),
    (
        "This is a paper about algebraic topology with no ML keywords.",
        "other",
    ),
])
def test_classify_category(text, expected_category):
    assert _classify_category(text) == expected_category


# ---------------------------------------------------------------------------
# Model-size classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_size", [
    # 110M > 100M → medium (spec: small = 1M-100M, medium = 100M-1B)
    ("Our model has 110M parameters and was trained on...", "medium"),
    ("We use a 7B parameter language model.", "large"),
    ("The tiny 45K parameter network converges in seconds.", "tiny"),
    ("This 500M parameter model requires 40GB VRAM.", "medium"),
    ("No parameter count mentioned in this abstract.", "unknown"),
    ("1.5B parameters were used for this experiment.", "large"),
    # 50M is in the 1M-100M range → small
    ("We use a 50M parameter model on MNIST.", "small"),
])
def test_classify_model_size(text, expected_size):
    assert _classify_model_size(text) == expected_size


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

def test_extract_features_returns_paper_features():
    text = (
        "We train a ResNet-50 on ImageNet convolutional benchmark. "
        "The model has 25M parameters. We run 3 experiments."
    )
    features = extract_features(
        text,
        sha8="abcd1234",
        estimated_vram_gb=24,
        gpu_hints=("RTX 4090",),
        num_experiments=3,
        datasets=("ImageNet",),
    )
    assert isinstance(features, PaperFeatures)
    assert features.sha8 == "abcd1234"
    assert features.category == "cnn"
    assert features.model_size_class == "small"
    assert features.estimated_vram_gb == 24
    assert features.num_experiments == 3
    assert features.datasets == ("ImageNet",)
    assert features.gpu_hints == ("RTX 4090",)
    assert len(features.feature_vector) == 10
    # All feature vector values should be in [0, 1]
    for v in features.feature_vector:
        assert 0.0 <= v <= 1.0, f"feature_vector value {v} out of [0,1]"


def test_extract_features_rl_paper():
    text = (
        "We apply GRPO policy gradient to train agents on Atari. "
        "Reward-based training with PPO baseline. 3B parameter model."
    )
    features = extract_features(
        text,
        sha8="rl000001",
        estimated_vram_gb=40,
        gpu_hints=("A100",),
        num_experiments=5,
    )
    assert features.category == "rl"
    assert features.model_size_class == "large"


def test_extract_features_feature_vector_dimensionality():
    """Feature vector must always be exactly 10-dimensional."""
    text = "Some paper text."
    features = extract_features(
        text,
        sha8="00000000",
        estimated_vram_gb=8,
        gpu_hints=(),
        num_experiments=1,
    )
    assert len(features.feature_vector) == 10


def test_extract_features_frozen_dataclass():
    """PaperFeatures is frozen; mutation must raise."""
    text = "variational autoencoder with dropout"
    features = extract_features(
        text, sha8="00000001", estimated_vram_gb=16, gpu_hints=()
    )
    with pytest.raises((AttributeError, TypeError)):
        features.category = "other"  # type: ignore[misc]
