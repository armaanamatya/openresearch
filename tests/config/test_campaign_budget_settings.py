"""Typed settings for durable campaign-wide money meters."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.config import Settings


_VARS = (
    "OPENRESEARCH_CAMPAIGN_MAX_LLM_USD",
    "OPENRESEARCH_CAMPAIGN_MAX_GPU_USD",
    "OPENRESEARCH_CAMPAIGN_MAX_GPU_HOURS",
)


@pytest.fixture
def clean_campaign_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_campaign_budgets_default_to_unset(clean_campaign_budgets: None) -> None:
    settings = Settings(_env_file=None)
    assert settings.campaign_max_llm_usd is None
    assert settings.campaign_max_gpu_usd is None
    assert settings.campaign_max_gpu_hours is None


def test_campaign_budgets_load_from_environment(
    clean_campaign_budgets: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MAX_LLM_USD", "20")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MAX_GPU_USD", "40.5")
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MAX_GPU_HOURS", "10")
    settings = Settings(_env_file=None)
    assert settings.campaign_max_llm_usd == 20
    assert settings.campaign_max_gpu_usd == 40.5
    assert settings.campaign_max_gpu_hours == 10


def test_campaign_budgets_reject_nonpositive_values(
    clean_campaign_budgets: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_CAMPAIGN_MAX_GPU_USD", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
