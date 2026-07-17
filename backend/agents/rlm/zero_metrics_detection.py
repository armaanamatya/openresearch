"""
P0 deterministic anti-fabrication guard — zero/constant/non-finite result-claiming
metrics veto.

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
    CALLER (primitives.py, W2-1 wire), not here.
  - Fail-soft everywhere; any exception -> safe fallback ([] or False).
  - Flag default-OFF (OPENRESEARCH_ZERO_METRICS_GUARD).

NON-FINITE (NaN / ±Inf) — the strictly-worse-than-zero class (P0 fix):
  A diverged run whose loss/reward went NaN used to EVADE this guard entirely:
  IEEE-754 self-inequality makes BOTH veto predicates false for NaN
  (``nan == 0.0`` is False; ``nan == values[0]`` is False even against itself),
  so the guard built to catch degenerate results waved through the most
  degenerate result there is.  A non-finite value is not a measurement at all —
  it is therefore an AUTOMATIC, UNCONDITIONAL veto:
    * never silently dropped from the normalized value list,
    * never exempted by ``provenance.json`` (a manifest cannot launder a NaN —
      unlike a legitimate 0.0, there is no honest reading of "the result is NaN"),
    * never exempted by the absence of a GPU claim.
  The conservative key exclusions still apply FIRST, so a NaN parked in a
  denominator/size/config key (or a bool status flag) is not a result claim and
  does not veto.  ``OPENRESEARCH_ZERO_METRICS_GUARD`` still gates the composed
  decision, so the OFF state remains byte-identical.
"""

from __future__ import annotations

import math
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
    """Return v as float if it is numeric or a numeric-coercible str, else None.

    NaN / ±Inf are deliberately RETURNED, never filtered: a non-finite reading is
    the single most degenerate result a run can claim, and dropping it here is how
    it used to become invisible to the veto (it read as "no data").  ``is_finite``
    classification is the CALLER's job — see :func:`non_finite_metric_keys`.
    Note ``json.loads`` parses bare ``NaN``/``Infinity`` (Python's json extension)
    straight to floats, and ``float("nan")``/``float("inf")`` parses the string
    forms, so both reach this function from a real ``metrics.json``.
    """
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


def _flatten(obj: object, parent_key: str | None, out: list[tuple[str | None, float]]) -> None:
    """Recursively flatten obj, collecting (key, value) result-claiming numeric leaves.

    The single walker behind BOTH projections (:func:`normalize_metric_values` for
    the values and :func:`non_finite_metric_keys` for the offending keys) — one
    source of truth for what counts as a result-claiming leaf.
    """
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
            out.append((parent_key, fv))


def _normalize_metric_items(metrics: object) -> list[tuple[str | None, float]]:
    """(key, value) pairs for every result-claiming numeric leaf.  Fail-soft -> []."""
    try:
        if not isinstance(metrics, dict):
            return []
        items: list[tuple[str | None, float]] = []
        _flatten(metrics, None, items)
        return items
    except Exception:  # noqa: BLE001
        return []


def normalize_metric_values(metrics: object) -> list[float]:
    """Flatten metrics to a list of result-claiming numeric leaf values.

    Handles both shapes:
      flat scalar: {"loss": 0.0, "return": 31.1, ...}
      nested:      {"per_model": {m: {e: {b: {"metric": 0.086}}}}, "status": ..., ...}

    Excluded keys (structural/denominator/size): see module docstring.
    Non-finite (NaN/±Inf) values ARE included — they are result claims like any
    other, and the veto predicates below treat them as strictly worse than zero.
    Non-dict top-level / any error -> [].
    """
    try:
        return [v for _k, v in _normalize_metric_items(metrics)]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Non-finite (NaN / ±Inf) detection — the automatic veto
# ---------------------------------------------------------------------------

def non_finite_metric_keys(metrics: object) -> list[str]:
    """The RESULT-CLAIMING keys whose value is NaN or ±Inf (deduped, ordered).

    Empty when there are none.  Excluded keys (denominator/size/config/bool
    status flags) are already gone by the time we get here, so every key this
    returns is a metric the run is CLAIMING as a result.  Fail-soft -> [].
    """
    try:
        seen: list[str] = []
        for key, value in _normalize_metric_items(metrics):
            if math.isfinite(value):
                continue
            name = key if key is not None else "(unnamed)"
            if name not in seen:
                seen.append(name)
        return seen
    except Exception:  # noqa: BLE001
        return []


def has_non_finite_metric(metrics: object) -> bool:
    """True iff ANY result-claiming metric value is NaN or ±Inf.

    This is the diverged-training signal the zero/constant predicates structurally
    cannot see (``nan == 0.0`` and ``nan == nan`` are both False under IEEE-754).
    Fail-soft -> False.
    """
    try:
        return bool(non_finite_metric_keys(metrics))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Detection predicate
# ---------------------------------------------------------------------------

def looks_like_zero_metrics(metrics: object) -> bool:
    """True iff normalize_metric_values is non-empty AND any of:
      (a) ANY value is non-finite (NaN/±Inf) — diverged/degenerate, strictly
          worse than zero (checked FIRST: the equality predicates below are both
          False for NaN, which is exactly how a diverged run used to slip past),
      (b) all values are exactly 0.0, OR
      (c) all values are bit-identical to each other (constant across cells).

    Pure shape signal — the GPU-claim and provenance discriminators are applied
    by the caller.  Fail-soft -> False.
    """
    try:
        values = normalize_metric_values(metrics)
        if not values:
            return False
        # (a) Non-finite check FIRST — a NaN/Inf result is not a measurement.
        # Unlike the all-zero/constant branches this needs no minimum count: one
        # NaN result-claiming metric is already degenerate.
        if any(not math.isfinite(v) for v in values):
            return True
        # (b) All-zero check (a single 0.0 is legitimately suspect)
        if all(v == 0.0 for v in values):
            return True
        # (c) Constant-across-cells check — only meaningful with >= 2 values.
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

    NON-FINITE OVERRIDE (P0): a NaN/±Inf result-claiming metric vetoes
    UNCONDITIONALLY — no gpu_claim requirement, no provenance exemption.  The
    provenance discriminator exists to separate a HONEST zero (a baseline really
    scored 0 on real outputs) from a FAKE zero; there is no honest reading of a
    NaN result, so a manifest pointing at a NaN cannot launder it.  A diverged
    run must repair, not ship.
    """
    if not zero_metrics_guard_enabled():
        return False
    if has_non_finite_metric(metrics):
        return True
    if not looks_like_zero_metrics(metrics):
        return False
    return bool(gpu_claim) and not bool(provenance_present)


# ---------------------------------------------------------------------------
# Repair message
# ---------------------------------------------------------------------------

def zero_metrics_repair_message(metrics: object) -> str:
    """Actionable repair directive naming the offending pattern (non-finite vs
    all-zero vs constant) and the first ≤6 result-claiming keys.  Mirrors
    stub_repair_message tone."""
    try:
        # Non-finite takes precedence — it is a DIFFERENT bug (numerical divergence)
        # with a different repair (stabilize training), so it gets its own directive
        # rather than being mislabelled "all-zero or constant".
        nf_keys = non_finite_metric_keys(metrics)
        if nf_keys:
            nf_str = ", ".join(nf_keys[:6])
            return (
                f"fabrication_suspected: run_experiment reported success but the "
                f"result-claiming metrics are non-finite (NaN/Inf) — keys: {nf_str}. "
                f"A NaN/Inf metric is not a measurement: training DIVERGED (or the "
                f"metric was never assigned a real value), so this result cannot back "
                f"any claim. This is strictly worse than a zero result and is never "
                f"exempted by a provenance manifest. "
                f"Re-implement: find where the loss/reward becomes non-finite (typical "
                f"causes: learning rate too high, missing gradient clipping, log/div of "
                f"zero, fp16 overflow, an un-normalized reward), fix it, and re-run "
                f"until every reported metric is a finite number computed from real "
                f"model outputs. Do NOT write NaN/Inf (or placeholder) values to "
                f"metrics.json."
            )

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
