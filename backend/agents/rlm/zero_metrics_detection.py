"""
P0 deterministic anti-fabrication guard — zero/constant result-claiming metrics veto.

Pure stdlib module — no third-party imports.  Detects run_experiment results
whose every RESULT-CLAIMING metric value is 0.0 (all-zero) or bit-identical
across all cells (constant), which is a strong signal that training/evaluation
is not wired to real model outputs.  Mirrors stub_detection.py in shape.

Key design:
  - Normalizes BOTH the flat-scalar shape (the common case) and the nested
    per_model[model][env][baseline] shape to a flat list of result-claiming floats.
  - Structural/denominator/size keys are excluded (see EXCLUDED_KEY_PATTERNS).
  - Hyperparameter/config keys are also excluded (see _CONFIG_TERMS).
  - Pure shape signal only — provenance/GPU discriminators are applied by the
    CALLER (primitives.py:6465, W2-1 wire), not here.
  - Fail-soft everywhere; any exception -> safe fallback ([] or False).
  - Flag default-OFF (OPENRESEARCH_ZERO_METRICS_GUARD).
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def zero_metrics_guard_enabled() -> bool:
    """True iff OPENRESEARCH_ZERO_METRICS_GUARD is in {'1','true','yes','on'}."""
    return os.environ.get("OPENRESEARCH_ZERO_METRICS_GUARD", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# ---------------------------------------------------------------------------
# Exclusion rules for structural / denominator / size keys
# ---------------------------------------------------------------------------

# Exact lowercased key names that are always excluded.
_EXCLUDED_EXACT: frozenset[str] = frozenset({
    "status",
    "scope",
    "cell_id",
    "n",
    "count",
    "steps",
    "steps_run",
    "epochs",
    "seed",
    "wall_time_s",
    "len",
    "retries",
})

# Prefixes (lowercased) that mark a key as excluded.
_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "batch",   # batch_size, batch_scale, etc.
    "elapsed", # elapsed_s, elapsed_ms, etc.
)

# Suffix pattern: any key ending in "_n" (denominator / sample count).
_SUFFIX_N_RE = re.compile(r"_n$")

# Token-split pattern: split a key into words on underscores and non-alphanumeric chars.
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Hyperparameter / config tokens whose presence in a key's token decomposition (or
# whose compound form matches the full lowercased key) marks the key as a config
# value rather than a result-claiming metric.
#
# Spec: §4.0 of 2026-06-20-pre-gpu-code-review-and-report-validation-design.md.
# Fix for codex-1 hparam-masking bug: {"loss":0.0,"accuracy":0.0,"learning_rate":1e-5}
# was NOT flagged as all-zero because the nonzero hparam survived normalization.
# The compound term "learning_rate" is checked against the full lowercased key (not just
# its individual tokens) so that "learning" and "rate" are not independently excluded —
# which would wrongly drop "success_rate" whose last token is also "rate".
_CONFIG_TERMS: frozenset[str] = frozenset({
    "learning_rate",  # compound — matched against full key, not split tokens
    "lr",
    "beta",
    "lambda",
    "weight_decay",
    "momentum",
    "gamma",
    "eps",
    "epsilon",
    "clip",
    "clip_ratio",
    "warmup",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "gpu",
    "vram",
    "num_gpus",
    "hidden",
    "dim",
    "embed",
    "max_len",
    "max_prompt",
    "max_tokens",
    "num_layers",
    "vocab",
    "hour",
    "gb",
})


def _is_config_key(key: str) -> bool:
    """Return True iff this key is a hyperparameter/config value, not a result metric.

    Checks (a) whether the full lowercased key appears in _CONFIG_TERMS (handles
    compound terms like "learning_rate"), then (b) whether any individual token
    (split on underscores/non-alphanumeric) appears in _CONFIG_TERMS (handles
    short aliases like "lr", embedded terms like "clip" in "grad_clip_norm").
    Token-boundary match only — NOT naive substring — so "accuracy" is never
    excluded even though it contains the letters of no config token.
    """
    lk = key.lower()
    # (a) Full-key match for compound terms (e.g. "learning_rate").
    if lk in _CONFIG_TERMS:
        return True
    # (b) Token-level match.
    tokens = _TOKEN_SPLIT_RE.split(lk)
    return any(tok in _CONFIG_TERMS for tok in tokens if tok)


def _is_excluded_key(key: str) -> bool:
    """Return True iff this key represents a structural, denominator, size, or config value."""
    lk = key.lower()
    if lk in _EXCLUDED_EXACT:
        return True
    for prefix in _EXCLUDED_PREFIXES:
        if lk.startswith(prefix):
            return True
    if _SUFFIX_N_RE.search(lk):
        return True
    if _is_config_key(key):
        return True
    return False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _coerce_float(v: object) -> float | None:
    """Return v as float if it is numeric or a numeric-coercible str, else None."""
    if isinstance(v, bool):
        # booleans are a subclass of int but are structural (True/False flags)
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    return None


def _flatten(obj: object, parent_key: str | None, out: list[float]) -> None:
    """Recursively flatten obj, collecting result-claiming numeric leaves."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_excluded_key(k):
                continue
            _flatten(v, k, out)
    elif isinstance(obj, list):
        for item in obj:
            _flatten(item, parent_key, out)
    else:
        # Leaf value — try to coerce to float, but only when the parent key
        # (if known) is not excluded.  If parent_key is None we are at the
        # top level inside a list; include it.
        if parent_key is not None and _is_excluded_key(parent_key):
            return
        fv = _coerce_float(obj)
        if fv is not None:
            out.append(fv)


def normalize_metric_values(metrics: object) -> list[float]:
    """Flatten metrics to a list of result-claiming numeric leaf values.

    Handles both shapes:
      flat scalar: {"loss": 0.0, "return": 31.1, ...}
      nested:      {"per_model": {m: {e: {b: {"metric": 0.086}}}}, "status": ..., ...}

    Excluded keys (structural/denominator/size): see module docstring.
    Non-dict top-level / any error -> [].
    """
    try:
        if not isinstance(metrics, dict):
            return []
        result: list[float] = []
        _flatten(metrics, None, result)
        return result
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Detection predicate
# ---------------------------------------------------------------------------

def looks_like_zero_metrics(metrics: object) -> bool:
    """True iff normalize_metric_values is non-empty AND either:
      (a) all values are exactly 0.0, OR
      (b) all values are bit-identical to each other (constant across cells).

    Pure shape signal — the GPU-claim and provenance discriminators are applied
    by the caller.  Fail-soft -> False.
    """
    try:
        values = normalize_metric_values(metrics)
        if not values:
            return False
        # All-zero check (a single 0.0 is legitimately suspect)
        if all(v == 0.0 for v in values):
            return True
        # Constant-across-cells check — only meaningful with >= 2 values.
        # A single non-zero metric is a normal partial result, NOT "constant
        # across cells"; requiring >= 2 avoids vetoing a lone legitimate value.
        if len(values) >= 2 and all(v == values[0] for v in values):
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Composed veto decision (consumed by the run_experiment wire, P0.2)
# ---------------------------------------------------------------------------

def zero_metrics_should_veto(
    metrics: object,
    *,
    gpu_claim: bool,
    provenance_present: bool,
) -> bool:
    """Composed Tier-1 veto decision for the run_experiment wire (spec §6.1).

    Returns True (the caller degrades to fabrication_suspected) iff ALL hold:
      * the guard is enabled (``OPENRESEARCH_ZERO_METRICS_GUARD``),
      * the result-claiming metrics are all-zero / constant (``looks_like_zero_metrics``),
      * the metrics CLAIM gpu training (caller passes ``metrics_claim_gpu_training(...)``),
      * NO ``provenance.json`` links the metric to a real output (caller checks disk).

    Provenance presence is the fake-0-vs-real-0 discriminator: a legitimately
    failing baseline that scored 0 emits a provenance manifest and is NOT vetoed.
    Pure — the inputs are pre-computed by the caller; never raises.
    """
    if not zero_metrics_guard_enabled():
        return False
    if not looks_like_zero_metrics(metrics):
        return False
    return bool(gpu_claim) and not bool(provenance_present)


# ---------------------------------------------------------------------------
# Repair message
# ---------------------------------------------------------------------------

def zero_metrics_repair_message(metrics: object) -> str:
    """Actionable repair directive naming the offending pattern (all-zero vs constant)
    and the first ≤6 result-claiming keys.  Mirrors stub_repair_message tone."""
    try:
        # Collect the first ≤6 result-claiming keys (not excluded).
        result_keys: list[str] = []
        if isinstance(metrics, dict):
            for k in metrics:
                if not _is_excluded_key(k):
                    # For flat dicts grab top-level; for nested, the top-level
                    # structural keys (per_model/status/scope) are already excluded —
                    # so any surviving key IS a result key.
                    result_keys.append(k)
                    if len(result_keys) >= 6:
                        break

        keys_str = ", ".join(result_keys) if result_keys else "(unknown)"

        # Determine pattern
        values = normalize_metric_values(metrics)
        if values and all(v == 0.0 for v in values):
            pattern = "all-zero"
        elif values and len(set(values)) == 1:
            pattern = f"constant ({values[0]})"
        else:
            pattern = "all-zero or constant"

        return (
            f"fabrication_suspected: run_experiment reported success but the result-claiming "
            f"metrics are {pattern} (keys: {keys_str}). "
            f"This strongly indicates that training/reward/evaluation is NOT wired to real "
            f"model outputs — the metrics were never updated or were always zero-initialized. "
            f"Re-implement: ensure the training loop, reward function, and evaluation code "
            f"compute values from real model outputs on real data, and write non-trivial "
            f"numeric results to metrics.json. Do not return placeholder or zero-initialized "
            f"dictionaries."
        )
    except Exception:  # noqa: BLE001
        return (
            "fabrication_suspected: all result-claiming metrics are zero or constant. "
            "Re-implement training/evaluation to produce real non-trivial metric values."
        )
