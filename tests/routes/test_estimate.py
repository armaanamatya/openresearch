"""HTTP-level tests for POST /paper/estimate.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §HTTP API
Invariant 7: handler never spawns a subprocess.
Invariant 10: on failure returns 200 + error field (not 500).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    _reset_settings_cache()
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(tmp_path / "runs"))
    yield
    _reset_settings_cache()


def _fresh_app(monkeypatch, runs_root: Path):
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(runs_root))
    _reset_settings_cache()
    from backend.app import create_app
    return create_app()


def _fake_estimate(paper_id: str = "test_paper") -> dict:
    return {
        "paper": {"id": paper_id, "title": paper_id, "sha256": "abc" * 20 + "abcd"},
        "gpu": {
            "sku_id": "rtx4090",
            "label": "RTX4090 (RunPod COMMUNITY)",
            "usd_per_hour": 0.34,
            "estimated_hours": {"p50": 4.5, "p90": 6.3},
            "usd_total": {"p50": 1.53, "p90": 2.14},
        },
        "api": [
            {
                "provider": "openai",
                "model_id": "gpt-5",
                "input_tokens": 200000,
                "output_tokens": 60000,
                "usd": 0.85,
                "is_subscription": False,
                "subscription_note": None,
            }
        ],
        "recipes": {
            "strict": {
                "label": "Strict reproduction",
                "description": "Paper recipe.",
                "gpu_usd": 1.53,
                "api_usd_best": 0.20,
                "api_usd_worst": 9.75,
                "wall_clock_hours_p50": 4.5,
                "fidelity_label": "high",
                "declared_reductions": [],
            }
        },
        "calibration_metadata": {
            "based_on_n_preserved_runs": 3,
            "precision_window_pct": 85,
            "catalog_schema_version": 1,
            "calibration_schema_version": 1,
            "estimated_at_utc": "2026-05-25T12:00:00+00:00",
        },
        "estimate_id": "abcdefgh_strict_1_1",
    }


# ---------------------------------------------------------------------------
# JSON body path
# ---------------------------------------------------------------------------

def test_json_arxiv_id_returns_estimate_shape(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("1412.6980")),
    ):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "arxiv_id", "source": "1412.6980"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "paper" in data
    assert data["paper"]["id"] == "1412.6980"
    assert "gpu" in data
    assert "api" in data
    assert "recipes" in data
    assert "calibration_metadata" in data
    assert "estimate_id" in data


def test_json_default_recipe_mode_is_both(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    captured_kwargs: dict = {}

    async def _capture(source, *, source_kind, recipe_mode, **kw):
        captured_kwargs["recipe_mode"] = recipe_mode
        return _fake_estimate("default-mode")

    with patch("backend.routes.estimate.estimate_paper_budget", new=_capture):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "arxiv_id", "source": "1234.5678"},
        )

    assert resp.status_code == 200
    assert captured_kwargs.get("recipe_mode") == "both"


# ---------------------------------------------------------------------------
# Multipart upload path
# ---------------------------------------------------------------------------

def _minimal_pdf_bytes() -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000068 00000 n\n"
        b"0000000125 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
    )


def test_multipart_pdf_upload_returns_estimate_shape(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("uploaded")),
    ):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            files={"paper": ("paper.pdf", _minimal_pdf_bytes(), "application/pdf")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "estimate_id" in data


# ---------------------------------------------------------------------------
# Invariant 10: failure returns 200 + error field, not 500
# ---------------------------------------------------------------------------

def test_estimator_failure_returns_200_with_error(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    async def _raise(*args, **kw):
        raise RuntimeError("Simulated network failure")

    with patch("backend.routes.estimate.estimate_paper_budget", new=_raise):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "arxiv_id", "source": "0000.0000"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert data["fallback_available"] is True


# ---------------------------------------------------------------------------
# Invariant 7: route path must be mounted in the app
# ---------------------------------------------------------------------------

def test_estimate_route_is_mounted(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    # getattr guard: newer Starlette (CI may pip-resolve a version newer than the
    # local pin) exposes non-Route entries (mounts / included routers) on
    # app.routes that have no `.path`; skip them instead of raising AttributeError.
    estimate_paths = [
        getattr(r, "path", "")
        for r in app.routes
        if "estimate" in getattr(r, "path", "")
    ]
    assert estimate_paths, (
        "POST /paper/estimate must be mounted in create_app(). "
        f"Routes with 'estimate' in path: {estimate_paths}"
    )
