"""Unit tests for apply_sandbox_override (pure function, no I/O)."""

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
