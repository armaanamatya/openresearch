"""Pre-flight invariants for the pricing catalog.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md
§Pre-flight invariants 1, 2, 3, 4, 5, 6
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.models import ROOT_MODELS
from backend.services.pricing.catalog import (
    CATALOG_SCHEMA_VERSION,
    GPU_PRICING,
    MODEL_PRICING,
    check_audit_freshness,
)
from backend.services.runtime.gpu_catalog import CATALOG as GPU_CATALOG

# ---------------------------------------------------------------------------
# Invariant 1: MODEL_PRICING covers every model in ROOT_MODELS
# ---------------------------------------------------------------------------

# Mapping from ROOT_MODELS key → pricing key used in MODEL_PRICING.
# Each root model uses one provider-qualified pricing entry for its root model
# and (implicitly) one for its sub-agent.  The estimator uses the root-model
# entry to drive the per-provider cross-cost table.
_ROOT_MODEL_TO_PRICING_KEY: dict[str, str] = {
    "gpt-5": "openai.gpt-5",
    "qwen3-coder": "featherless.qwen3-coder-480b",   # OpenRouter, but Featherless is the ref impl
    "kimi-k2.5": "moonshot.kimi-k2-5",
    "claude": "anthropic.claude-opus-4-7",
    "claude-oauth": "anthropic.claude-oauth",
    "qwen3-coder-featherless": "featherless.qwen3-coder-480b",
    "azure-gpt-4o": "azure.gpt-4o",
}


def test_all_root_models_have_pricing_entry():
    missing = [
        key
        for key in ROOT_MODELS
        if _ROOT_MODEL_TO_PRICING_KEY.get(key) not in MODEL_PRICING
    ]
    assert not missing, f"ROOT_MODELS keys without MODEL_PRICING entries: {missing}"


# ---------------------------------------------------------------------------
# Invariant 2: GPU_PRICING keys are exactly the SKU short_names from gpu_catalog
# ---------------------------------------------------------------------------

def test_gpu_pricing_keys_match_catalog_short_names():
    catalog_short_names = {sku.short_name for sku in GPU_CATALOG}
    pricing_keys = set(GPU_PRICING.keys())
    orphaned = pricing_keys - catalog_short_names
    missing = catalog_short_names - pricing_keys
    assert not orphaned, f"GPU_PRICING has orphan keys not in catalog: {orphaned}"
    assert not missing, f"GPU catalog SKUs missing from GPU_PRICING: {missing}"


# ---------------------------------------------------------------------------
# Invariant 3: last_audited_utc staleness is warned but not crashed
# ---------------------------------------------------------------------------

def test_check_audit_freshness_returns_list():
    result = check_audit_freshness(max_age_days=90)
    assert isinstance(result, list)


def test_check_audit_freshness_does_not_raise():
    check_audit_freshness(max_age_days=0)  # force everything stale


# ---------------------------------------------------------------------------
# Invariant 4: CATALOG_SCHEMA_VERSION bump verified by unit test (see test_cache.py)
# ---------------------------------------------------------------------------

def test_catalog_schema_version_is_int():
    assert isinstance(CATALOG_SCHEMA_VERSION, int)
    assert CATALOG_SCHEMA_VERSION >= 1


# ---------------------------------------------------------------------------
# Invariant 5: OAuth entries have usd=0 and non-empty subscription_note
# ---------------------------------------------------------------------------

def test_oauth_entries_zero_price_and_note():
    for key, entry in MODEL_PRICING.items():
        if "oauth" in key.lower():
            assert entry.usd_per_1m_input == 0.0, f"{key}: input price must be 0"
            assert entry.usd_per_1m_output == 0.0, f"{key}: output price must be 0"
            assert entry.subscription_note, f"{key}: subscription_note must be non-empty"


# ---------------------------------------------------------------------------
# Invariant 6: Featherless / subscription entries have amortization fields
# ---------------------------------------------------------------------------

def test_subscription_entries_have_amortization_fields():
    for key, entry in MODEL_PRICING.items():
        if entry.subscription_usd_per_month is not None:
            assert entry.assumed_runs_per_month is not None, (
                f"{key}: subscription_usd_per_month requires assumed_runs_per_month"
            )
            assert entry.assumed_runs_per_month > 0, (
                f"{key}: assumed_runs_per_month must be positive"
            )
            assert entry.subscription_note, (
                f"{key}: subscription entries must have subscription_note"
            )


def test_featherless_entry_present():
    assert "featherless.qwen3-coder-480b" in MODEL_PRICING
    entry = MODEL_PRICING["featherless.qwen3-coder-480b"]
    assert entry.subscription_usd_per_month == 9.0
    assert entry.assumed_runs_per_month == 30


# ---------------------------------------------------------------------------
# Entry shape sanity — every entry is a valid Pydantic model
# ---------------------------------------------------------------------------

def test_all_model_entries_have_last_audited_utc():
    for key, entry in MODEL_PRICING.items():
        assert entry.last_audited_utc, f"{key}: last_audited_utc is empty"


def test_all_gpu_entries_have_last_audited_utc():
    for key, entry in GPU_PRICING.items():
        assert entry.last_audited_utc, f"{key}: last_audited_utc is empty"
