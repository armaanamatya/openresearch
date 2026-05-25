"""MODEL_PRICING + GPU_PRICING tables.

Prices are approximate snapshots audited quarterly. Bump CATALOG_SCHEMA_VERSION
on every audit pass to invalidate stale cached estimates. Each audit pass writes
docs/runbooks/pricing-audit-YYYY-QN.md documenting every changed entry.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §catalog.py
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone

from backend.services.pricing.schemas import GpuPriceEntry, ModelPriceEntry

logger = logging.getLogger(__name__)

CATALOG_SCHEMA_VERSION: int = 1

_AUDITED = "2026-05-25T00:00:00+00:00"

MODEL_PRICING: dict[str, ModelPriceEntry] = {
    "anthropic.claude-opus-4-7": ModelPriceEntry(
        usd_per_1m_input=15.0,
        usd_per_1m_output=75.0,
        last_audited_utc=_AUDITED,
    ),
    "anthropic.claude-sonnet-4-6": ModelPriceEntry(
        usd_per_1m_input=3.0,
        usd_per_1m_output=15.0,
        last_audited_utc=_AUDITED,
    ),
    "anthropic.claude-haiku-4-5": ModelPriceEntry(
        usd_per_1m_input=0.80,
        usd_per_1m_output=4.0,
        last_audited_utc=_AUDITED,
    ),
    "anthropic.claude-oauth": ModelPriceEntry(
        usd_per_1m_input=0.0,
        usd_per_1m_output=0.0,
        last_audited_utc=_AUDITED,
        subscription_note="Covered by your Claude Code subscription — rate limits apply.",
    ),
    "openai.gpt-5": ModelPriceEntry(
        usd_per_1m_input=1.25,
        usd_per_1m_output=10.0,
        last_audited_utc=_AUDITED,
    ),
    "openai.gpt-5-mini": ModelPriceEntry(
        usd_per_1m_input=0.25,
        usd_per_1m_output=2.0,
        last_audited_utc=_AUDITED,
    ),
    "openai.gpt-5-nano": ModelPriceEntry(
        usd_per_1m_input=0.05,
        usd_per_1m_output=0.40,
        last_audited_utc=_AUDITED,
    ),
    "google.gemini-2-5-pro": ModelPriceEntry(
        usd_per_1m_input=1.25,
        usd_per_1m_output=10.0,
        last_audited_utc=_AUDITED,
    ),
    "google.gemini-2-5-flash": ModelPriceEntry(
        usd_per_1m_input=0.075,
        usd_per_1m_output=0.30,
        last_audited_utc=_AUDITED,
    ),
    "featherless.qwen3-coder-480b": ModelPriceEntry(
        usd_per_1m_input=0.0,
        usd_per_1m_output=0.0,
        last_audited_utc=_AUDITED,
        subscription_note="Featherless flat-rate subscription — $9/mo amortized across runs.",
        subscription_usd_per_month=9.0,
        assumed_runs_per_month=30,
    ),
    "moonshot.kimi-k2-5": ModelPriceEntry(
        usd_per_1m_input=0.15,
        usd_per_1m_output=2.50,
        last_audited_utc=_AUDITED,
    ),
    # Azure GPT-4o — maps to the azure-gpt-4o registry key.
    # Priced at Azure's standard gpt-4o rate (May 2026 snapshot).
    "azure.gpt-4o": ModelPriceEntry(
        usd_per_1m_input=5.0,
        usd_per_1m_output=15.0,
        last_audited_utc=_AUDITED,
    ),
}

# Keys mirror GpuSku.short_name from gpu_catalog.py so GpuPlan.short_name can
# be used as a direct lookup key without a translation layer.
GPU_PRICING: dict[str, GpuPriceEntry] = {
    "rtx4090": GpuPriceEntry(
        usd_per_hour=0.34,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "a5000": GpuPriceEntry(
        usd_per_hour=0.36,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "a100_40": GpuPriceEntry(
        usd_per_hour=1.19,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "a6000": GpuPriceEntry(
        usd_per_hour=0.49,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "l40s": GpuPriceEntry(
        usd_per_hour=0.86,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "a100_80": GpuPriceEntry(
        usd_per_hour=1.89,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "h100_80": GpuPriceEntry(
        usd_per_hour=4.39,
        last_audited_utc=_AUDITED,
        cloud_type="COMMUNITY",
    ),
    "h200": GpuPriceEntry(
        usd_per_hour=7.99,
        last_audited_utc=_AUDITED,
        cloud_type="SECURE",
    ),
}


def check_audit_freshness(*, max_age_days: int = 90) -> list[str]:
    """Return keys whose last_audited_utc is older than max_age_days.

    Logs a warning at startup for each stale entry but never raises — callers
    must still compute estimates with stale prices (production resilience).
    """
    now = datetime.now(timezone.utc)
    stale: list[str] = []
    for key, entry in {**MODEL_PRICING, **GPU_PRICING}.items():
        try:
            audited = datetime.fromisoformat(entry.last_audited_utc)
        except ValueError:
            stale.append(key)
            continue
        age_days = (now - audited).days
        if age_days > max_age_days:
            stale.append(key)
            logger.warning(
                "pricing: catalog entry %r last audited %d days ago (>%d) — "
                "prices may be stale",
                key,
                age_days,
                max_age_days,
            )
    return stale
