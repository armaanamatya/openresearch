"""Unit tests for apply_sandbox_override and _enforce_demo_gate."""

import pytest
from fastapi import HTTPException

from backend.app import _enforce_demo_gate
from backend.services.events.live_runs import StartRunRequest, apply_sandbox_override


def test_no_override_returns_request_unchanged():
    """Empty force_sandbox leaves the request untouched."""
    request = StartRunRequest(sandbox="runpod")
    result = apply_sandbox_override(request, "")
    assert result.sandbox == "runpod"


def test_override_replaces_sandbox():
    """force_sandbox='local' overrides a runpod request to local."""
    request = StartRunRequest(sandbox="runpod")
    result = apply_sandbox_override(request, "local")
    assert result.sandbox == "local"


def test_override_idempotent():
    """Applying force_sandbox='local' to a local request still yields local."""
    request = StartRunRequest(sandbox="local")
    result = apply_sandbox_override(request, "local")
    assert result.sandbox == "local"


def test_original_request_not_mutated():
    """The original request object is not mutated by the override."""
    request = StartRunRequest(sandbox="runpod")
    apply_sandbox_override(request, "local")
    assert request.sandbox == "runpod"


# --------------------------------------------------------------------------- #
# _enforce_demo_gate unit tests
# --------------------------------------------------------------------------- #


def test_gate_disabled_accepts_none():
    """Gate is off when configured_secret is empty — None secret passes."""
    _enforce_demo_gate(None, "")  # must not raise


def test_gate_disabled_accepts_any_value():
    """Gate is off when configured_secret is empty — any value passes."""
    _enforce_demo_gate("anything", "")  # must not raise


def test_gate_rejects_missing_secret():
    """Gate raises 403 when configured_secret is set and no secret provided."""
    with pytest.raises(HTTPException) as exc_info:
        _enforce_demo_gate(None, "topsecret")
    assert exc_info.value.status_code == 403


def test_gate_rejects_wrong_secret():
    """Gate raises 403 when provided secret does not match."""
    with pytest.raises(HTTPException) as exc_info:
        _enforce_demo_gate("wrong", "topsecret")
    assert exc_info.value.status_code == 403


def test_gate_accepts_correct_secret():
    """Gate passes when provided secret matches configured secret."""
    _enforce_demo_gate("topsecret", "topsecret")  # must not raise


# --------------------------------------------------------------------------- #
# Integration test: Header wiring in FastAPI
# --------------------------------------------------------------------------- #


def test_runs_endpoint_rejects_missing_secret(monkeypatch):
    monkeypatch.setenv("REPROLAB_DEMO_SECRET", "topsecret")
    from backend.config import get_settings
    get_settings(_force_reload=True)
    try:
        from backend.app import create_app
        from starlette.testclient import TestClient

        class _FakeRunService:
            async def start_run(self, request):
                return {"projectId": "x", "status": "queued"}

        client = TestClient(create_app(run_service=_FakeRunService()))
        assert client.post("/runs", json={"mode": "sdk"}).status_code == 403
        assert client.post(
            "/runs", json={"mode": "sdk"}, headers={"X-Demo-Secret": "topsecret"}
        ).status_code == 202
    finally:
        monkeypatch.undo()
        get_settings(_force_reload=True)  # reset the module-level settings cache
