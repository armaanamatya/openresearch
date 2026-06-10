"""
Tests for Issue #7: Backend app skeleton and shared config bootstrap.
Run: pytest tests/test_issue7_app_skeleton.py -v
"""
import importlib
import os

import pytest


# -- 1. Module structure exists ----------------------------------------------

REQUIRED_MODULES = [
    "backend",
    "backend.services.ingestion",
    "backend.services.orchestration",
    "backend.services.context",
    "backend.services.runtime",
    "backend.services.verification",
    "backend.services.events",
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_module_importable(module):
    """Every service module from the PRD service map must be importable."""
    importlib.import_module(module)


# -- 2. Config loading -------------------------------------------------------

def test_config_loads_defaults():
    """Config module loads with sane defaults when no env vars are set."""
    from backend.config import get_settings
    settings = get_settings()
    assert settings.database_url is not None
    assert settings.environment in ("development", "testing", "production")


def test_config_respects_env_override(monkeypatch):
    """Config values can be overridden via environment variables."""
    monkeypatch.setenv("REPROLAB_ENVIRONMENT", "testing")
    from backend.config import get_settings
    # Force re-creation if cached
    settings = get_settings(_force_reload=True) if hasattr(get_settings, '_force_reload') else get_settings()
    assert settings.environment == "testing"


# -- 3. App entrypoint boots -------------------------------------------------

def test_app_creates_without_error():
    """FastAPI app object can be instantiated without raising."""
    from backend.app import create_app
    app = create_app()
    assert app is not None


# -- 4. Health / readiness endpoint ------------------------------------------

@pytest.fixture(scope="module")
def client():
    from backend.app import create_app
    from starlette.testclient import TestClient
    app = create_app()
    return TestClient(app)


def test_health_endpoint_returns_200(client):
    """GET /health returns 200 with status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_health_contains_version(client):
    """Health response includes a version string."""
    resp = client.get("/health")
    body = resp.json()
    assert "version" in body


# -- 5. Local dev bootstrap --------------------------------------------------

def test_requirements_file_exists():
    """A requirements.txt or pyproject.toml exists so devs can install deps."""
    assert os.path.isfile("requirements.txt") or os.path.isfile("pyproject.toml")


def test_readme_or_setup_docs_exist():
    """Developer bootstrap instructions exist."""
    assert (
        os.path.isfile("docs/setup-guide.md")
        or os.path.isfile("README.md")
    )
