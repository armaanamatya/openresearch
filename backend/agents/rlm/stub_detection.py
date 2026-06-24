"""
G2 route-agnostic stub-metrics guard.

Pure stdlib module — no third-party imports.  Detects a run_experiment
result whose metrics dict contains only placeholder / non-metric keys and
no real paper metric, which is a strong signal that the agent wrote a stub
rather than running real training.  Conservative: favors False (no-fire)
to avoid blocking legitimate runs.  Flag-gated; default OFF.
"""

from __future__ import annotations


def stub_metrics_guard_enabled() -> bool:
    import os
    return os.environ.get("OPENRESEARCH_STUB_METRICS_GUARD", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# Generic non-metric / known-placeholder keys (compared lowercased).
_PLACEHOLDER_METRIC_KEYS = frozenset({
    "total_length",
    "chunk_count",
    "placeholder_metric",
    "placeholder",
    "dummy",
    "todo",
    "n_chunks",
    "num_chunks",
    "chunk",
    "length",
    "count",
    "n_items",
    "n_examples",
})

# Substrings (lowercased) that indicate a REAL metric.
_REAL_METRIC_HINTS = (
    "acc",
    "loss",
    "success",
    "reward",
    "f1",
    "bleu",
    "rouge",
    "perplex",
    "ppl",
    "score",
    "err",
    "rate",
    "auc",
    "map",
    "recall",
    "precision",
    "psnr",
    "ssim",
    "mae",
    "mse",
    "rmse",
    "exact",
    "_em",
    "win",
    "pass@",
    "return",
    "throughput",
    "latency",
)


def _collect_leaf_keys(obj, _out: list) -> None:
    """Recursively collect all dict keys at any depth (leaf key = a key whose
    value is not a dict, OR any key at any level — we collect ALL dict keys,
    not just those whose values are scalars, so a key like 'per_model' that
    nests real metrics is not mistaken for a placeholder)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _out.append(k)
            _collect_leaf_keys(v, _out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_leaf_keys(item, _out)


def looks_like_stub_metrics(metrics) -> bool:
    """Conservative stub signal.  True iff metrics is a non-empty dict, NO
    leaf key matches a real-metric hint, AND >=1 leaf key is a known
    placeholder key.  Recurses into nested dicts/lists (collects leaf keys).
    Fail-soft: any error -> False."""
    try:
        if not isinstance(metrics, dict) or not metrics:
            return False

        leaf_keys: list[str] = []
        _collect_leaf_keys(metrics, leaf_keys)

        lowered = [k.lower() for k in leaf_keys]

        # If ANY collected key contains ANY real-metric hint → not a stub.
        for lk in lowered:
            for hint in _REAL_METRIC_HINTS:
                if hint in lk:
                    return False

        # If ANY collected key is a known placeholder key → stub.
        for lk in lowered:
            if lk in _PLACEHOLDER_METRIC_KEYS:
                return True

        # Unknown, non-placeholder keys: do not fire (conservative).
        return False
    except Exception:  # noqa: BLE001
        return False


def stub_repair_message(metrics) -> str:
    """Actionable directive naming the offending placeholder keys."""
    try:
        offending: list[str] = []
        if isinstance(metrics, dict):
            leaf_keys: list[str] = []
            _collect_leaf_keys(metrics, leaf_keys)
            lowered = [k.lower() for k in leaf_keys]
            for lk in lowered:
                if lk in _PLACEHOLDER_METRIC_KEYS and lk not in offending:
                    offending.append(lk)
                    if len(offending) >= 6:
                        break
        keys_str = ", ".join(offending) if offending else "(unknown)"
    except Exception:  # noqa: BLE001
        keys_str = "(unknown)"
    return (
        f"fabrication_suspected: run reported success but metrics contain only "
        f"placeholder keys ({keys_str}) and no real paper metric "
        f"(accuracy/loss/success_rate/...). This is a stub, not a real run. "
        f"Re-implement: compute the paper's actual headline metric from real model "
        f"outputs on real data, and write it to metrics.json."
    )
