"""metric_binding.py — bind a claim's prose ``metric_name`` to a concrete
measurement path inside a parsed ``code/metrics.json`` (Track A, closes D2).

``repro_spec_extractor.py`` mines a paper's claims into prose fields
(``metric_name``, ``scope={model,dataset,split}``), but the deterministic
readers (``repro_spec_extractor._extract_metric_value`` /
``seed_bundle_from_metrics``) need a concrete key path — ``metric_key`` /
``model_key`` / ``env_key`` / ``baseline_key`` — into the parsed metrics dict.
Nothing binds prose to path today; this module is that binder. Output keys
match those two functions' parameter names exactly, so a caller can pass a
claim's ``metric_binding`` straight through as kwargs.

Design (locked — see
``docs/superpowers/specs/2026-07-09-eval-integrity-track-a-design.md`` §4.1):
deterministic tokenized candidate search over the ``metrics`` key tree,
reusing the tokenizers already in ``leaf_scorer`` (no new tokenization is
invented here), followed by a scope+unit ACCEPTANCE GATE. "The path exists in
metrics.json" is necessary but NOT sufficient — a candidate is accepted only
when the resolved model/env/baseline(split) keys deterministically match the
claim's declared ``scope`` (unconstrained axes are read permissively, but any
DECLARED axis must resolve unambiguously) and the candidate value is a real
finite numeric scalar (the "unit" check: metrics.json carries no separate
unit/direction metadata to compare against, so a bool/string/None/collection
masquerading as a measurement is rejected here). Ambiguous or scope-mismatched
candidates stay ``bound=False`` — asymmetric by design: a claim that can't be
verified stays unmeasured, it is never bound to the wrong number.

Deferred (YAGNI): an LLM-proposed fallback bind for claims this deterministic
matcher cannot resolve is a documented future seam, not built here — the
acceptance gate below is what would validate that proposal too, so nothing
here is blocked on it.
"""

from __future__ import annotations

import math
from typing import Any

from backend.evals.paperbench.leaf_scorer import (
    _normalise_dataset_name,
    _normalise_model_name,
)

# A resolved candidate: (metric_key, model_key, env_key, baseline_key).
_Candidate = tuple[str, str, str, str]


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _is_scalar_number(value: Any) -> bool:
    """True iff ``value`` is a finite, non-bool int/float (a real measurement)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _tokens_match(declared: frozenset[str], candidate: frozenset[str]) -> bool:
    """Symmetric containment over token sets produced by ``_normalise_dataset_name``.

    Either direction counts as a match (a short declared name like "mnist"
    matching a more decorated key like "mnist_v2", or vice versa) — this is
    exactly the containment convention ``leaf_scorer`` already uses for
    dataset/leaf fuzzy matching; no new tokenizer or matching rule is added.
    """
    if not declared or not candidate:
        return False
    return declared <= candidate or candidate <= declared


def _scope_str(scope: dict[str, Any], key: str) -> str:
    value = scope.get(key)
    return value.strip() if isinstance(value, str) else ""


def _unbound(reason: str) -> dict[str, Any]:
    return {
        "metric_key": "",
        "model_key": "",
        "env_key": "",
        "baseline_key": "",
        # Reserved for a future multi-value aggregation directive (mean/latest/...);
        # the deterministic binder never populates it today.
        "agg": None,
        "bound": False,
        "reason": reason,
    }


def _bound(candidate: _Candidate) -> dict[str, Any]:
    metric_key, model_key, env_key, baseline_key = candidate
    return {
        "metric_key": metric_key,
        "model_key": model_key,
        "env_key": env_key,
        "baseline_key": baseline_key,
        "agg": None,
        "bound": True,
        "reason": "",
    }


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------


def _flat_candidates(metrics: dict[str, Any], metric_name: str) -> list[_Candidate]:
    """Top-level ``data[metric_key]`` candidates — only meaningful when the
    claim declares NO scope at all (nothing to verify a nested path against)."""
    declared = _normalise_dataset_name(metric_name)
    out: list[_Candidate] = []
    for key, value in metrics.items():
        if not _is_scalar_number(value):
            continue
        if _tokens_match(declared, _normalise_dataset_name(str(key))):
            out.append((str(key), "", "", ""))
    return out


def _walk_per_model(
    per_model: dict[str, Any],
    model_scope: str,
    dataset_scope: str,
    split_scope: str,
    metric_name: str,
) -> list[_Candidate]:
    """Scope-consistent, metric-matching leaves anywhere in the ``per_model`` tree.

    Handles both the 3-level ``per_model[model][env][metric]`` shape (no
    baseline/split axis) and the 4-level cell-matrix shape
    ``per_model[model][env][baseline][metric]`` — the claim's declared
    ``split`` is checked against whatever literal key occupies that third
    nesting level (metrics.json has no dedicated split axis; a run's split
    value and its baseline/variant value share the same slot).  Axes the
    claim does NOT declare are left unconstrained (any literal key passes);
    axes it DOES declare must match or the branch is pruned — never silently
    substituted.
    """
    declared_metric = _normalise_dataset_name(metric_name)
    declared_dataset = _normalise_dataset_name(dataset_scope) if dataset_scope else frozenset()
    declared_split = _normalise_dataset_name(split_scope) if split_scope else frozenset()
    model_norm = _normalise_model_name(model_scope) if model_scope else ""

    out: list[_Candidate] = []
    for model_key, envs in per_model.items():
        if not isinstance(envs, dict):
            continue
        if model_scope and _normalise_model_name(str(model_key)) != model_norm:
            continue
        for env_key, sub in envs.items():
            if not isinstance(sub, dict):
                continue
            if dataset_scope and not _tokens_match(
                declared_dataset, _normalise_dataset_name(str(env_key))
            ):
                continue
            # Case A: metric is a direct scalar under (model, env) — no baseline/split axis.
            for k, v in sub.items():
                if _is_scalar_number(v) and _tokens_match(
                    declared_metric, _normalise_dataset_name(str(k))
                ):
                    out.append((str(k), str(model_key), str(env_key), ""))
            # Case B: one more nesting level (baseline / split / variant axis).
            for baseline_key, cell in sub.items():
                if not isinstance(cell, dict):
                    continue
                if split_scope and not _tokens_match(
                    declared_split, _normalise_dataset_name(str(baseline_key))
                ):
                    continue
                for k, v in cell.items():
                    if _is_scalar_number(v) and _tokens_match(
                        declared_metric, _normalise_dataset_name(str(k))
                    ):
                        out.append((str(k), str(model_key), str(env_key), str(baseline_key)))
    return out


def _bind_one(metric_name: str, scope: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    if not metric_name:
        return _unbound("missing_metric_name")
    if not isinstance(metrics, dict):
        return _unbound("metrics_not_a_dict")

    model_scope = _scope_str(scope, "model")
    dataset_scope = _scope_str(scope, "dataset")
    split_scope = _scope_str(scope, "split")
    scope_declared = bool(model_scope or dataset_scope or split_scope)

    candidates: list[_Candidate] = []
    per_model = metrics.get("per_model")
    if isinstance(per_model, dict):
        candidates.extend(
            _walk_per_model(per_model, model_scope, dataset_scope, split_scope, metric_name)
        )
    if not scope_declared:
        # No declared scope to verify a nested path against — a scope-blind
        # top-level flat key is the only thing that CAN be checked, so it's
        # only ever considered in this branch (never as a fallback once a
        # scope axis is declared and fails to resolve — see the module
        # docstring's asymmetry rule).
        candidates.extend(_flat_candidates(metrics, metric_name))

    unique = list(dict.fromkeys(candidates))
    if not unique:
        return _unbound("scope_mismatch" if scope_declared else "no_candidate_match")
    if len(unique) > 1:
        return _unbound("ambiguous_match")
    return _bound(unique[0])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bind_claims(repro_spec: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Attach a scope-verified ``metric_binding`` to every claim in ``repro_spec``.

    Pure, deterministic, stdlib-only — no LLM call, no network, no filesystem
    access. Never raises: any claim whose metric_name/scope can't be
    deterministically and unambiguously resolved against ``metrics`` just gets
    ``metric_binding.bound = False`` with a machine-readable ``reason``; it is
    never bound to the wrong number.

    Returns a NEW dict (does not mutate ``repro_spec`` or its claims in
    place); non-claim top-level keys of ``repro_spec`` are preserved.
    """
    if not isinstance(repro_spec, dict):
        return repro_spec
    claims = repro_spec.get("claims")
    if not isinstance(claims, list):
        return repro_spec

    out_claims: list[Any] = []
    for claim in claims:
        if not isinstance(claim, dict):
            out_claims.append(claim)
            continue
        new_claim = dict(claim)
        try:
            metric_name = str(claim.get("metric_name") or "").strip()
            scope = claim.get("scope")
            if not isinstance(scope, dict):
                scope = {}
            new_claim["metric_binding"] = _bind_one(metric_name, scope, metrics)
        except Exception as exc:  # noqa: BLE001 — a claim that can't be bound just stays unbound.
            new_claim["metric_binding"] = _unbound(f"bind_error:{type(exc).__name__}")
        out_claims.append(new_claim)

    out = dict(repro_spec)
    out["claims"] = out_claims
    return out
