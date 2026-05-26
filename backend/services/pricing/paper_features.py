"""Single canonical paper-feature extractor for the three-source estimator.

All three estimators (heuristic, k-NN, LLM) consume a `PaperFeatures`
instance produced by `extract_features`.  The extraction does one LLM call
(shared with the LLM estimator) plus a regex pass — no second round-trip.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Category keyword sets — evaluated in priority order.  The first match wins.
# Priority rationale: more specific/unambiguous terms rank higher.
#   - rl:        "reinforcement" / "policy gradient" / "reward" are unambiguous
#   - vae:       "variational" / "autoencoder" / "diffusion" / "gan" are specific;
#                ranked before cnn because a VAE paper often trains on CIFAR/ImageNet
#   - cnn:       "convolutional" / "imagenet" / "cifar" — rank after vae so a
#                "variational autoencoder on CIFAR-10" → vae, not cnn
#   - nlp:       rank after vae/cnn; "transformer" can appear in vision papers
#   - optimizer: catch-all for papers whose primary contribution is training dynamics
#   - other:     default (no keyword required)
_CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("rl",        ["reinforcement", "policy gradient", "reward", "ppo", "grpo", "actor-critic", "q-learning", "mdp"]),
    ("vae",       ["variational autoencoder", "variational auto-encoder", "autoencoder", " vae ", "(vae)", "diffusion model", "denoising diffusion", "generative adversarial", " gan ", "(gan)"]),
    ("cnn",       ["convolutional", "imagenet", "cifar", "resnet", "vit", "vision transformer", "segmentation", "object detection"]),
    ("nlp",       ["language model", "bert", "gpt", " llm ", "seq2seq", "summarization", "translation", "tokenizer"]),
    ("optimizer", ["optimizer", "sgd", "adam", "momentum", "learning rate schedule", "weight decay"]),
]

# Model-size keywords → class.  Evaluated in order; first numeric match wins.
# Pattern: number + scale suffix + optional space + "param" or "model" or "network"
# (prefix match, not word boundary, so "parameters" also matches "param")
_SIZE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("large",  re.compile(r"\b(\d[\d,\.]*)\s*B\s*(?:param|model|network)", re.IGNORECASE)),
    ("medium", re.compile(r"\b(\d[\d,\.]*)\s*M\s*(?:param|model|network)", re.IGNORECASE)),
    ("small",  re.compile(r"\b(\d[\d,\.]*)\s*[Kk]\s*(?:param|model|network)", re.IGNORECASE)),
]

_BILLION  = 1_000_000_000
_MILLION  = 1_000_000
_THOUSAND = 1_000


def _classify_category(text: str) -> str:
    """Return the first matching category from text[:8000], or 'other'."""
    lower = text[:8000].lower()
    for category, keywords in _CATEGORY_PATTERNS:
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def _classify_model_size(text: str) -> str:
    """Parse explicit parameter-count mentions → 'tiny'|'small'|'medium'|'large'|'unknown'."""
    # Try B-scale first (1B+ → large, <1B is medium if expressed in billions)
    for m in _SIZE_PATTERNS[0][1].finditer(text[:10000]):
        try:
            n = float(m.group(1).replace(",", "")) * _BILLION
            if n >= _BILLION:
                return "large"
            if n >= 100 * _MILLION:
                return "medium"
            if n >= _MILLION:
                return "small"
            return "tiny"
        except ValueError:
            continue
    # M-scale
    for m in _SIZE_PATTERNS[1][1].finditer(text[:10000]):
        try:
            n = float(m.group(1).replace(",", "")) * _MILLION
            if n >= _BILLION:
                return "large"
            if n >= 100 * _MILLION:
                return "medium"
            if n >= _MILLION:
                return "small"
            return "tiny"
        except ValueError:
            continue
    # K-scale → tiny
    if _SIZE_PATTERNS[2][1].search(text[:10000]):
        return "tiny"
    return "unknown"


@dataclass(frozen=True)
class PaperFeatures:
    """Compact numeric + categorical description of a paper for estimation.

    `feature_vector` is a 10-dimensional normalized float tuple suitable for
    Euclidean k-NN distance.  Dimensions:
      0 — estimated_vram_gb / 80.0  (normalised to [0, 1] at A100-class)
      1 — num_experiments / 10.0
      2 — category_one_hot[0] = (category == "rl")
      3 — category_one_hot[1] = (category == "nlp")
      4 — category_one_hot[2] = (category == "cnn")
      5 — category_one_hot[3] = (category == "vae")
      6 — category_one_hot[4] = (category == "optimizer")
      7 — model_size_class ordinal / 4.0  (tiny=0, small=1, medium=2, large=3, unknown=2)
      8 — len(datasets) / 5.0
      9 — len(gpu_hints) / 3.0
    """
    sha8: str
    category: str          # "rl" | "nlp" | "cnn" | "vae" | "optimizer" | "other"
    model_size_class: str  # "tiny" | "small" | "medium" | "large" | "unknown"
    estimated_vram_gb: int
    num_experiments: int
    datasets: tuple[str, ...]
    gpu_hints: tuple[str, ...]
    feature_vector: tuple[float, ...]  # 10-dim


_SIZE_ORDINAL: dict[str, float] = {
    "tiny": 0.0,
    "small": 1.0,
    "medium": 2.0,
    "large": 3.0,
    "unknown": 2.0,  # assume medium when unknown
}

_CATEGORY_INDEX: list[str] = ["rl", "nlp", "cnn", "vae", "optimizer"]


def _make_feature_vector(
    estimated_vram_gb: int,
    num_experiments: int,
    category: str,
    model_size_class: str,
    datasets: tuple[str, ...],
    gpu_hints: tuple[str, ...],
) -> tuple[float, ...]:
    cat_oh = [1.0 if category == c else 0.0 for c in _CATEGORY_INDEX]
    return (
        min(estimated_vram_gb / 80.0, 1.0),
        min(num_experiments / 10.0, 1.0),
        *cat_oh,
        _SIZE_ORDINAL.get(model_size_class, 2.0) / 4.0,
        min(len(datasets) / 5.0, 1.0),
        min(len(gpu_hints) / 3.0, 1.0),
    )


def extract_features(
    paper_text: str,
    sha8: str,
    *,
    estimated_vram_gb: int,
    gpu_hints: tuple[str, ...],
    num_experiments: int = 1,
    datasets: tuple[str, ...] = (),
) -> PaperFeatures:
    """Build a PaperFeatures from pre-extracted fields + paper text.

    The caller (estimator.py) already ran `_extract_gpu_clues` and an LLM
    workload call.  This function only adds category + model-size
    classification (cheap regex) and assembles the feature vector.
    No LLM call happens here.

    Args:
        paper_text: Full (possibly truncated) paper text.
        sha8: First 8 hex chars of sha256(pdf_bytes) — for cache identity.
        estimated_vram_gb: From `_extract_gpu_clues` or LLM result.
        gpu_hints: GPU label strings from `_extract_gpu_clues`.
        num_experiments: From LLM workload estimate.
        datasets: Dataset names (empty tuple when unavailable).
    """
    category = _classify_category(paper_text)
    model_size_class = _classify_model_size(paper_text)

    fv = _make_feature_vector(
        estimated_vram_gb,
        num_experiments,
        category,
        model_size_class,
        datasets,
        gpu_hints,
    )
    return PaperFeatures(
        sha8=sha8,
        category=category,
        model_size_class=model_size_class,
        estimated_vram_gb=estimated_vram_gb,
        num_experiments=num_experiments,
        datasets=datasets,
        gpu_hints=gpu_hints,
        feature_vector=fv,
    )
