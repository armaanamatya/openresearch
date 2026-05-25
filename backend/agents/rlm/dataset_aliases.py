"""Canonical HuggingFace dataset-ID registry — single source of truth for
the pre-flight check + the agent's prompt + any future runtime helper.

The 2026-05-25 Adam IMDB regression was a `load_dataset("imdb")` call that
produced ``hf://datasets/imdb@.../plain_text/train-*`` — a URI the modern
huggingface_hub library refuses to parse because it requires the
``namespace/name`` form (``stanfordnlp/imdb``). The bare short names were
valid years ago and remain in countless tutorials, so every agent that
reads pre-2024 examples falls into the same trap.

Two maps own this surface:

  * ``HF_SHORT_NAME_REMAP`` — old short name → canonical ``owner/name``.
    Reachable via ``load_dataset(canonical, **kwargs)``.
  * ``USE_NATIVE_LIB_INSTEAD`` — names that exist as HF datasets but the
    agent should NOT use HF for them. Vision datasets (MNIST, CIFAR,
    SVHN, ImageNet) load faster + more reliably via ``torchvision.datasets``
    directly. The pre-flight check blocks the HF route and the agent
    rewrites the call.

Keep these alphabetised inside each group. Refresh quarterly when HF's
canonical-owner taxonomy shifts.
"""

from __future__ import annotations

from typing import Final


# Bare short names → canonical owner/name on the modern HF Hub.
# Verified against https://huggingface.co/datasets as of 2026-05-25.
HF_SHORT_NAME_REMAP: Final[dict[str, str]] = {
    "ag_news":           "fancyzhx/ag_news",
    "amazon_polarity":   "fancyzhx/amazon_polarity",
    "amazon_reviews_multi": "mteb/amazon_reviews_multi",
    "anli":              "facebook/anli",
    "arxiv_dataset":     "Cohere/arxiv-dataset",
    "boolq":             "google/boolq",
    "cnn_dailymail":     "abisee/cnn_dailymail",
    "cola":              "nyu-mll/glue",  # subset under glue
    "common_voice":      "mozilla-foundation/common_voice_17_0",
    "commonsense_qa":    "tau/commonsense_qa",
    "conll2003":         "eriktks/conll2003",
    "copa":              "pkavumba/balanced-copa",
    "dbpedia_14":        "fancyzhx/dbpedia_14",
    "emotion":           "dair-ai/emotion",
    "fever":             "pminervini/fever",
    "glue":              "nyu-mll/glue",
    "imdb":              "stanfordnlp/imdb",
    "lambada":           "EleutherAI/lambada_openai",
    "mnli":              "nyu-mll/multi_nli",
    "mrpc":              "nyu-mll/glue",  # subset under glue
    "newsroom":          "newsroom",  # still works
    "openwebtext":       "Skylion007/openwebtext",
    "paws":              "google-research-datasets/paws",
    "piqa":              "ybisk/piqa",
    "qnli":              "nyu-mll/glue",  # subset
    "qqp":               "nyu-mll/glue",  # subset
    "quora":             "toughdata/quora-question-answer-dataset",
    "race":              "ehovy/race",
    "rotten_tomatoes":   "cornell-movie-review-data/rotten_tomatoes",
    "rte":               "nyu-mll/glue",
    "snli":              "stanfordnlp/snli",
    "squad":             "rajpurkar/squad",
    "squad_v2":          "rajpurkar/squad_v2",
    "sst":               "stanfordnlp/sst",
    "sst2":              "stanfordnlp/sst2",
    "stsb":              "nyu-mll/glue",
    "trec":              "CogComp/trec",
    "tweet_eval":        "cardiffnlp/tweet_eval",
    "wikitext":          "Salesforce/wikitext",
    "wmt14":             "wmt/wmt14",
    "wmt16":             "wmt/wmt16",
    "wmt19":             "wmt/wmt19",
    "wnli":              "nyu-mll/glue",  # subset
    "xnli":              "facebook/xnli",
    "yahoo_answers_topics": "yahoo_answers_topics",  # still works
    "yelp_polarity":     "fancyzhx/yelp_polarity",
    "yelp_review_full":  "Yelp/yelp_review_full",
}


# Names that exist on HF but the agent should load via the native lib —
# faster, more reliable, no mid-stream URI breakage.
USE_NATIVE_LIB_INSTEAD: Final[dict[str, str]] = {
    "cifar10":           "torchvision.datasets.CIFAR10(root='/artifacts/datasets', train=True, download=True, transform=transform)",
    "cifar100":          "torchvision.datasets.CIFAR100(root='/artifacts/datasets', train=True, download=True, transform=transform)",
    "fashion_mnist":     "torchvision.datasets.FashionMNIST(root='/artifacts/datasets', train=True, download=True, transform=transform)",
    "imagenet":          "torchvision.datasets.ImageNet(root='/artifacts/datasets', split='train', transform=transform)  # ImageNet requires manual download — see torchvision docs",
    "kmnist":            "torchvision.datasets.KMNIST(root='/artifacts/datasets', train=True, download=True, transform=transform)",
    "mnist":             "torchvision.datasets.MNIST(root='/artifacts/datasets', train=True, download=True, transform=transform)",
    "qmnist":            "torchvision.datasets.QMNIST(root='/artifacts/datasets', what='train', download=True, transform=transform)",
    "stl10":             "torchvision.datasets.STL10(root='/artifacts/datasets', split='train', download=True, transform=transform)",
    "svhn":              "torchvision.datasets.SVHN(root='/artifacts/datasets', split='train', download=True, transform=transform)",
}


def canonicalize_hf_id(name: str) -> tuple[str | None, str]:
    """Return ``(canonical_name, hint)`` for a dataset short name.

    * If ``name`` is in ``USE_NATIVE_LIB_INSTEAD``: returns
      ``(None, "<torchvision suggestion>")`` — caller must NOT use
      ``load_dataset`` at all.
    * If ``name`` is in ``HF_SHORT_NAME_REMAP``: returns
      ``(canonical, "")`` — caller should use ``load_dataset(canonical)``.
    * Otherwise: returns ``(name, "")`` — assume the name is already
      canonical, pass through.
    """
    lower = name.lower().strip()
    if lower in USE_NATIVE_LIB_INSTEAD:
        return (None, USE_NATIVE_LIB_INSTEAD[lower])
    if lower in HF_SHORT_NAME_REMAP:
        return (HF_SHORT_NAME_REMAP[lower], "")
    return (name, "")


__all__ = ["HF_SHORT_NAME_REMAP", "USE_NATIVE_LIB_INSTEAD", "canonicalize_hf_id"]
