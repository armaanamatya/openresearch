"""Unit tests for apply_sandbox_override and _enforce_demo_gate."""

import pytest
from fastapi import HTTPException

from backend.app import _enforce_demo_gate
from backend.services.events.live_runs import (
    StartRunRequest,
    apply_provider_override,
    apply_sandbox_override,
)


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
# apply_provider_override unit tests
# --------------------------------------------------------------------------- #


def test_provider_no_override_returns_request_unchanged():
    """Empty force_llm_provider leaves the request untouched."""
    request = StartRunRequest(provider="anthropic")
    result = apply_provider_override(request, "")
    assert result.provider == "anthropic"


def test_provider_override_replaces_provider():
    """force_llm_provider='openai' overrides an anthropic request to openai."""
    request = StartRunRequest(provider="anthropic")
    result = apply_provider_override(request, "openai")
    assert result.provider == "openai"


def test_provider_override_idempotent():
    """Applying force_llm_provider='openai' to an openai request still yields openai."""
    request = StartRunRequest(provider="openai")
    result = apply_provider_override(request, "openai")
    assert result.provider == "openai"


def test_provider_override_invalid_value_is_ignored():
    """Unknown force values pass through (defensive — never silently set garbage)."""
    request = StartRunRequest(provider="anthropic")
    result = apply_provider_override(request, "groq")
    assert result.provider == "anthropic"


def test_provider_original_request_not_mutated():
    """The original request object is not mutated by the override."""
    request = StartRunRequest(provider="anthropic")
    apply_provider_override(request, "openai")
    assert request.provider == "anthropic"


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
    """Gate raises 401 when configured_secret is set and no secret provided."""
    with pytest.raises(HTTPException) as exc_info:
        _enforce_demo_gate(None, "topsecret")
    assert exc_info.value.status_code == 401


def test_gate_rejects_wrong_secret():
    """Gate raises 401 when provided secret does not match."""
    with pytest.raises(HTTPException) as exc_info:
        _enforce_demo_gate("wrong", "topsecret")
    assert exc_info.value.status_code == 401


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
        assert client.post("/runs", json={"mode": "rlm"}).status_code == 401
        assert client.post(
            "/runs", json={"mode": "rlm"}, headers={"X-Demo-Secret": "topsecret"}
        ).status_code == 202
    finally:
        monkeypatch.undo()
        get_settings(_force_reload=True)  # reset the module-level settings cache


def test_delete_run_rejects_missing_secret(monkeypatch):
    monkeypatch.setenv("REPROLAB_DEMO_SECRET", "topsecret")
    from backend.config import get_settings
    get_settings(_force_reload=True)
    try:
        from backend.app import create_app
        from starlette.testclient import TestClient

        class _FakeRunService:
            async def stop_run(self, project_id):
                return {"projectId": project_id, "status": "stopped"}

        client = TestClient(create_app(run_service=_FakeRunService()))
        assert client.delete("/runs/prj_test").status_code == 401
        assert client.delete(
            "/runs/prj_test", headers={"X-Demo-Secret": "topsecret"}
        ).status_code == 200
    finally:
        monkeypatch.undo()
        get_settings(_force_reload=True)


def test_phase2_mutating_route_rejects_missing_secret(monkeypatch):
    monkeypatch.setenv("REPROLAB_DEMO_SECRET", "topsecret")
    from backend.config import get_settings
    get_settings(_force_reload=True)
    try:
        from backend.app import create_app
        from starlette.testclient import TestClient

        client = TestClient(create_app())
        payload = {
            "project_id": "prj_test",
            "stage": "experiment",
            "command": "python train.py",
            "exit_code": 1,
            "stdout": "",
            "stderr": "boom",
            "timed_out": False,
            "cause_kind": "runtime_error",
            "artifact_refs": [],
        }
        assert client.post("/phase2/failures/diagnose", json=payload).status_code == 401
    finally:
        monkeypatch.undo()
        get_settings(_force_reload=True)


def test_unhandled_route_error_returns_json(monkeypatch):
    monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)
    from backend.config import get_settings
    get_settings(_force_reload=True)
    try:
        from backend.app import create_app
        from starlette.testclient import TestClient

        app = create_app()

        @app.get("/_test/boom")
        async def boom():
            raise RuntimeError("synthetic failure")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/_test/boom")
        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"detail": "synthetic failure"}
    finally:
        get_settings(_force_reload=True)
