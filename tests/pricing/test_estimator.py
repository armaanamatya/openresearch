"""Happy-path + resilience tests for estimate_paper_budget.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §estimator.py
Invariant 7: estimate_paper_budget never spawns a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pdf_bytes() -> bytes:
    """A minimal valid-ish PDF payload (enough for PyMuPDF not to crash)."""
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000068 00000 n\n"
        b"0000000125 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
    )


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    r.mkdir()
    return r


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_arxiv_id(runs_root: Path, monkeypatch):
    """estimate_paper_budget returns a dict with required keys."""
    import backend.services.pricing.estimator as est_mod

    monkeypatch.setattr(
        est_mod,
        "_fetch_pdf_bytes",
        AsyncMock(return_value=(_minimal_pdf_bytes(), "1412.6980")),
    )
    monkeypatch.setattr(
        est_mod,
        "_extract_text_from_pdf",
        lambda pdf_bytes, **kw: "reinforcement learning policy gradient reward ppo grpo",
    )
    monkeypatch.setattr(
        est_mod,
        "_llm_estimate_workload",
        AsyncMock(return_value={
            "experiment_count": 2,
            "total_epochs_across_all_experiments": 50,
            "avg_epoch_seconds_on_target_gpu": 20.0,
            "confidence": "high",
        }),
    )

    result = await est_mod.estimate_paper_budget(
        "1412.6980",
        source_kind="arxiv_id",
        recipe_mode="strict",
        runs_root=runs_root,
    )

    assert "paper" in result
    assert result["paper"]["id"] == "1412.6980"
    assert "gpu" in result
    assert "api" in result
    assert isinstance(result["api"], list)
    assert len(result["api"]) > 0
    assert "recipes" in result
    assert "strict" in result["recipes"]
    assert "calibration_metadata" in result
    assert "estimate_id" in result


@pytest.mark.asyncio
async def test_both_recipe_modes(runs_root: Path, monkeypatch):
    import backend.services.pricing.estimator as est_mod

    monkeypatch.setattr(
        est_mod, "_fetch_pdf_bytes",
        AsyncMock(return_value=(_minimal_pdf_bytes(), "test-paper")),
    )
    monkeypatch.setattr(
        est_mod, "_extract_text_from_pdf",
        lambda b, **kw: "transformer language model attention token",
    )
    monkeypatch.setattr(
        est_mod, "_llm_estimate_workload",
        AsyncMock(return_value={
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 100,
            "avg_epoch_seconds_on_target_gpu": 10.0,
            "confidence": "medium",
        }),
    )

    result = await est_mod.estimate_paper_budget(
        "test-paper",
        source_kind="arxiv_id",
        recipe_mode="both",
        runs_root=runs_root,
    )

    assert "strict" in result["recipes"]
    assert "compressed" in result["recipes"]
    strict_hours = result["recipes"]["strict"]["wall_clock_hours_p50"]
    compressed_hours = result["recipes"]["compressed"]["wall_clock_hours_p50"]
    assert compressed_hours < strict_hours, "compressed must be cheaper than strict"


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_call(runs_root: Path, monkeypatch):
    """Second call must return from cache without making the LLM call."""
    import backend.services.pricing.estimator as est_mod

    call_count = {"n": 0}

    async def _mock_llm(*args, **kw):
        call_count["n"] += 1
        return {
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 10,
            "avg_epoch_seconds_on_target_gpu": 5.0,
            "confidence": "high",
        }

    monkeypatch.setattr(est_mod, "_fetch_pdf_bytes",
        AsyncMock(return_value=(_minimal_pdf_bytes(), "cached-paper")))
    monkeypatch.setattr(est_mod, "_extract_text_from_pdf",
        lambda b, **kw: "some paper text")
    monkeypatch.setattr(est_mod, "_llm_estimate_workload", _mock_llm)

    await est_mod.estimate_paper_budget(
        "cached-paper", source_kind="arxiv_id", recipe_mode="strict", runs_root=runs_root,
    )
    assert call_count["n"] == 1

    await est_mod.estimate_paper_budget(
        "cached-paper", source_kind="arxiv_id", recipe_mode="strict", runs_root=runs_root,
    )
    # LLM must NOT be called again — cache hit
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Invariant 7: never spawns a subprocess
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_subprocess_spawned(runs_root: Path, monkeypatch):
    import backend.services.pricing.estimator as est_mod

    monkeypatch.setattr(est_mod, "_fetch_pdf_bytes",
        AsyncMock(return_value=(_minimal_pdf_bytes(), "no-subprocess-paper")))
    monkeypatch.setattr(est_mod, "_extract_text_from_pdf",
        lambda b, **kw: "transformer language model")
    monkeypatch.setattr(est_mod, "_llm_estimate_workload",
        AsyncMock(return_value={
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 10,
            "avg_epoch_seconds_on_target_gpu": 5.0,
            "confidence": "low",
        }),
    )

    launched: list[str] = []

    def _no_subprocess(*args, **kw):
        launched.append(str(args))
        raise AssertionError("estimate_paper_budget must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "Popen", _no_subprocess)
    monkeypatch.setattr(subprocess, "run", _no_subprocess)

    # Should complete without triggering the assertion
    result = await est_mod.estimate_paper_budget(
        "no-subprocess-paper",
        source_kind="arxiv_id",
        recipe_mode="strict",
        runs_root=runs_root,
    )
    assert not launched
    assert "estimate_id" in result


# ---------------------------------------------------------------------------
# LLM call failure is handled gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_uses_defaults(runs_root: Path, monkeypatch):
    import backend.services.pricing.estimator as est_mod

    monkeypatch.setattr(est_mod, "_fetch_pdf_bytes",
        AsyncMock(return_value=(_minimal_pdf_bytes(), "llm-fail-paper")))
    monkeypatch.setattr(est_mod, "_extract_text_from_pdf",
        lambda b, **kw: "paper text about things")

    async def _fail_llm(*args, **kw):
        raise RuntimeError("Anthropic API error")

    monkeypatch.setattr(est_mod, "_llm_estimate_workload", _fail_llm)

    # Should not raise — falls back to defaults
    result = await est_mod.estimate_paper_budget(
        "llm-fail-paper",
        source_kind="arxiv_id",
        recipe_mode="strict",
        runs_root=runs_root,
    )
    assert result["calibration_metadata"]["catalog_schema_version"] >= 1
    assert result["recipes"]["strict"]["fidelity_label"] == "high"


# ---------------------------------------------------------------------------
# API cost table coverage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_table_covers_all_pricing_entries(runs_root: Path, monkeypatch):
    import backend.services.pricing.estimator as est_mod
    from backend.services.pricing.catalog import MODEL_PRICING

    monkeypatch.setattr(est_mod, "_fetch_pdf_bytes",
        AsyncMock(return_value=(_minimal_pdf_bytes(), "api-table-paper")))
    monkeypatch.setattr(est_mod, "_extract_text_from_pdf",
        lambda b, **kw: "deep learning")
    monkeypatch.setattr(est_mod, "_llm_estimate_workload",
        AsyncMock(return_value={
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 10,
            "avg_epoch_seconds_on_target_gpu": 5.0,
            "confidence": "low",
        }),
    )

    result = await est_mod.estimate_paper_budget(
        "api-table-paper",
        source_kind="arxiv_id",
        recipe_mode="strict",
        runs_root=runs_root,
    )

    returned_model_ids = {
        f"{r['provider']}.{r['model_id']}" for r in result["api"]
    }
    for key in MODEL_PRICING:
        assert key in returned_model_ids, f"{key} missing from API cost table"
