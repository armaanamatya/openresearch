"""Happy-path + resilience tests for estimate_paper_budget.

Spec: docs/history/specs/2026-05-25-budget-estimation-design.md §estimator.py
Invariant 7: estimate_paper_budget never spawns a subprocess.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


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


# ---------------------------------------------------------------------------
# Security fix (2026-07-13): pdf_path containment
#
# _fetch_pdf_bytes used to call Path(source).read_bytes() directly with zero
# containment check -- an unauthenticated caller (backend/routes/estimate.py
# had no demo-secret gate either) could read any file the server process
# could see. resolve_allowed_pdf_path / PdfPathNotAllowedError close that at
# the estimator layer too, as defense in depth behind the route's own check.
# ---------------------------------------------------------------------------

def test_resolve_allowed_pdf_path_allows_path_inside_root(tmp_path):
    from backend.services.pricing.estimator import resolve_allowed_pdf_path

    root = tmp_path / "runs"
    (root / "prj").mkdir(parents=True)
    pdf_path = root / "prj" / "paper.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes())

    resolved = resolve_allowed_pdf_path(str(pdf_path), (root,))
    assert resolved == pdf_path.resolve()


def test_resolve_allowed_pdf_path_rejects_path_outside_root(tmp_path):
    from backend.services.pricing.estimator import (
        PdfPathNotAllowedError,
        resolve_allowed_pdf_path,
    )

    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside" / "secret.env"
    outside.parent.mkdir()
    outside.write_text("API_KEY=leaked\n")

    with pytest.raises(PdfPathNotAllowedError):
        resolve_allowed_pdf_path(str(outside), (root,))


def test_resolve_allowed_pdf_path_rejects_dotdot_traversal(tmp_path):
    """A '..'-based escape must be caught by resolve(), not a naive prefix
    string comparison."""
    from backend.services.pricing.estimator import (
        PdfPathNotAllowedError,
        resolve_allowed_pdf_path,
    )

    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside" / "secret.pdf"
    outside.parent.mkdir()
    outside.write_bytes(b"%PDF-1.4\n%%EOF")

    traversal_source = str(root / ".." / "outside" / "secret.pdf")
    with pytest.raises(PdfPathNotAllowedError):
        resolve_allowed_pdf_path(traversal_source, (root,))


def test_resolve_allowed_pdf_path_rejects_symlink_escape(tmp_path):
    """A symlink planted *inside* the allowed root that points *outside* it
    must not be usable to escape -- resolve() follows the symlink before the
    containment comparison, mirroring live_runs.py's _is_relative_to guard."""
    from backend.services.pricing.estimator import (
        PdfPathNotAllowedError,
        resolve_allowed_pdf_path,
    )

    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside" / "secret.pdf"
    outside.parent.mkdir()
    outside.write_bytes(b"%PDF-1.4\n%%EOF")

    symlink_path = root / "escape.pdf"
    symlink_path.symlink_to(outside)

    with pytest.raises(PdfPathNotAllowedError):
        resolve_allowed_pdf_path(str(symlink_path), (root,))


@pytest.mark.asyncio
async def test_fetch_pdf_bytes_rejects_pdf_path_outside_allowed_root(tmp_path):
    import backend.services.pricing.estimator as est_mod

    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside" / "secret.env"
    outside.parent.mkdir()
    outside.write_text("API_KEY=leaked\n")

    with pytest.raises(est_mod.PdfPathNotAllowedError):
        await est_mod._fetch_pdf_bytes(
            "pdf_path", str(outside), allowed_pdf_roots=(root,)
        )


@pytest.mark.asyncio
async def test_fetch_pdf_bytes_allows_pdf_path_inside_allowed_root(tmp_path):
    import backend.services.pricing.estimator as est_mod

    root = tmp_path / "runs"
    (root / "prj").mkdir(parents=True)
    pdf_path = root / "prj" / "paper.pdf"
    pdf_bytes = _minimal_pdf_bytes()
    pdf_path.write_bytes(pdf_bytes)

    result_bytes, paper_id = await est_mod._fetch_pdf_bytes(
        "pdf_path", str(pdf_path), allowed_pdf_roots=(root,)
    )
    assert result_bytes == pdf_bytes
    assert paper_id == "paper"


@pytest.mark.asyncio
async def test_estimate_paper_budget_rejects_pdf_path_outside_runs_root(runs_root: Path, tmp_path, monkeypatch):
    """End-to-end (no mocking of _fetch_pdf_bytes): estimate_paper_budget
    itself must refuse a pdf_path outside runs_root, and never reach the
    LLM/GPU-resolution steps that follow the read."""
    import backend.services.pricing.estimator as est_mod

    outside = tmp_path / "outside" / "secret.env"
    outside.parent.mkdir()
    outside.write_text("API_KEY=leaked\n")

    llm_called = {"n": 0}

    async def _count_llm(*args, **kw):
        llm_called["n"] += 1
        return {
            "experiment_count": 1,
            "total_epochs_across_all_experiments": 1,
            "avg_epoch_seconds_on_target_gpu": 1.0,
            "confidence": "low",
        }

    monkeypatch.setattr(est_mod, "_llm_estimate_workload", _count_llm)

    with pytest.raises(est_mod.PdfPathNotAllowedError):
        await est_mod.estimate_paper_budget(
            str(outside),
            source_kind="pdf_path",
            recipe_mode="strict",
            runs_root=runs_root,
        )
    assert llm_called["n"] == 0, "LLM must not be called once containment rejects the path"
