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

    # Assert via the OpenAPI schema (the public route contract), not by walking
    # app.routes: newer Starlette (CI may pip-resolve a version newer than the
    # local pin) nests included-router routes under a Mount, so the estimate path
    # no longer appears as a flat app.routes[*].path (the POST tests above still
    # pass — the route is reachable, just not flat-introspectable). openapi()
    # reflects all mounted routes regardless of nesting.
    estimate_paths = [p for p in app.openapi().get("paths", {}) if "estimate" in p]
    assert estimate_paths, (
        "POST /paper/estimate must be mounted in create_app(). "
        f"OpenAPI paths with 'estimate': {estimate_paths}"
    )


# ---------------------------------------------------------------------------
# Security fix (2026-07-13): demo-secret gate now applies to /paper/estimate
# ---------------------------------------------------------------------------

def test_json_request_rejected_without_demo_secret_when_configured(monkeypatch, tmp_path):
    """The route previously never applied the demo-secret gate at all --
    an unauthenticated caller must now be rejected with 401 when a secret
    is configured, exactly like POST /runs and POST /runs/upload."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", "topsecret")
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("gated")),
    ) as mock_estimate:
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "arxiv_id", "source": "1234.5678"},
        )

    assert resp.status_code == 401
    mock_estimate.assert_not_called()


def test_json_request_rejected_with_wrong_demo_secret(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", "topsecret")
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("gated")),
    ) as mock_estimate:
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "arxiv_id", "source": "1234.5678"},
            headers={"X-Demo-Secret": "wrong"},
        )

    assert resp.status_code == 401
    mock_estimate.assert_not_called()


def test_json_request_succeeds_with_correct_demo_secret(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", "topsecret")
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("gated-ok")),
    ):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "arxiv_id", "source": "1234.5678"},
            headers={"X-Demo-Secret": "topsecret"},
        )

    assert resp.status_code == 200
    assert resp.json()["paper"]["id"] == "gated-ok"


# ---------------------------------------------------------------------------
# Security fix (2026-07-13): pdf_path containment (unauthenticated arbitrary
# file read). runs_root is the allowed root; anything outside it is a 400,
# never a 200-with-error-string, and the file must never be opened.
# ---------------------------------------------------------------------------

def test_pdf_path_outside_runs_root_rejected_and_not_read(monkeypatch, tmp_path):
    """The vulnerability, verbatim: {"source_kind": "pdf_path", "source":
    "<path outside runs_root>"} must be rejected with a 4xx BEFORE the file
    is ever opened, and the rejection must not echo the file back."""
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_file = outside_dir / "secret.env"
    secret_file.write_text("API_KEY=super-secret-value\n")

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("should-not-run")),
    ) as mock_estimate:
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "pdf_path", "source": str(secret_file)},
        )

    assert 400 <= resp.status_code < 500
    assert resp.status_code != 200
    mock_estimate.assert_not_called()
    assert "super-secret-value" not in resp.text


def test_pdf_path_traversal_outside_runs_root_rejected(monkeypatch, tmp_path):
    """A '..'-traversal path must resolve before the containment check runs
    (a naive startswith(str(runs_root)) string check would be fooled by
    this; Path.resolve() is not)."""
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.pdf").write_bytes(b"%PDF-1.4\n%%EOF")
    traversal_source = str(runs_root / ".." / "outside" / "secret.pdf")

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("should-not-run")),
    ) as mock_estimate:
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "pdf_path", "source": traversal_source},
        )

    assert resp.status_code == 400
    mock_estimate.assert_not_called()


def test_pdf_path_inside_runs_root_succeeds(monkeypatch, tmp_path):
    """A legitimate in-root pdf_path must still work end-to-end."""
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    project_dir = runs_root / "prj_test"
    project_dir.mkdir()
    pdf_path = project_dir / "paper.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes())

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("in-root")),
    ) as mock_estimate:
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "pdf_path", "source": str(pdf_path)},
        )

    assert resp.status_code == 200
    assert resp.json()["paper"]["id"] == "in-root"
    mock_estimate.assert_called_once()
    called_source = mock_estimate.call_args.args[0]
    assert Path(called_source).resolve() == pdf_path.resolve()


def test_pdf_path_in_root_estimator_failure_still_returns_200(monkeypatch, tmp_path):
    """Invariant 10, precisely distinguished from the fix above: once a
    pdf_path passes containment, a genuine estimator failure (not a security
    rejection) still returns 200 + error, never a 500 or a 400."""
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    app = _fresh_app(monkeypatch, runs_root)

    project_dir = runs_root / "prj_test"
    project_dir.mkdir()
    pdf_path = project_dir / "paper.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes())

    async def _raise(*args, **kw):
        raise RuntimeError("Simulated parse failure")

    with patch("backend.routes.estimate.estimate_paper_budget", new=_raise):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            json={"source_kind": "pdf_path", "source": str(pdf_path)},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert data["fallback_available"] is True


# ---------------------------------------------------------------------------
# Multipart upload path still works under the new gate + containment
# ---------------------------------------------------------------------------

def test_multipart_upload_rejected_without_demo_secret_when_configured(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", "topsecret")
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("uploaded")),
    ) as mock_estimate:
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            files={"paper": ("paper.pdf", _minimal_pdf_bytes(), "application/pdf")},
        )

    assert resp.status_code == 401
    mock_estimate.assert_not_called()


def test_multipart_upload_succeeds_with_correct_demo_secret(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", "topsecret")
    app = _fresh_app(monkeypatch, runs_root)

    with patch(
        "backend.routes.estimate.estimate_paper_budget",
        new=AsyncMock(return_value=_fake_estimate("uploaded-ok")),
    ):
        client = TestClient(app)
        resp = client.post(
            "/paper/estimate",
            files={"paper": ("paper.pdf", _minimal_pdf_bytes(), "application/pdf")},
            headers={"X-Demo-Secret": "topsecret"},
        )

    assert resp.status_code == 200
    assert resp.json()["estimate_id"]


def test_multipart_temp_upload_is_cleaned_up(monkeypatch, tmp_path):
    """The server-generated scratch file under runs_root/.estimate_uploads
    must not survive past the request (no litter in the runs directory)."""
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
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
    uploads_dir = runs_root / ".estimate_uploads"
    leftover = list(uploads_dir.glob("*.pdf")) if uploads_dir.exists() else []
    assert leftover == [], f"Temp upload(s) not cleaned up: {leftover}"
