"""GKE alias for the gcp sandbox token (SCOPE 1).

Hermetic: no live cloud calls. ensure_gcp_available is patched to a no-op.
Asserts gke->GkeJobBackend on BOTH the enum boundary and FORCE_SANDBOX, the
build_environment no-op, and gcp+aks byte-for-byte regression parity.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.agents.execution import SandboxMode, resolve_sandbox_mode


def test_sandbox_mode_gke_resolves_to_gcp_member():
    assert SandboxMode("gke") is SandboxMode.gcp


def test_sandbox_mode_gke_case_insensitive():
    assert SandboxMode("GKE") is SandboxMode.gcp
    assert SandboxMode(" gke ") is SandboxMode.gcp


def test_gcp_token_unchanged():
    assert SandboxMode("gcp") is SandboxMode.gcp
    assert SandboxMode.gcp.value == "gcp"


def test_unknown_token_still_raises():
    with pytest.raises(ValueError):
        SandboxMode("gkeXYZ")


def test_resolve_sandbox_mode_gke_explicit(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    assert resolve_sandbox_mode("gke", pipeline_mode="rlm") is SandboxMode.gcp


def test_force_sandbox_gke_override(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_FORCE_SANDBOX", "gke")
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.gcp


def test_gke_token_constructs_gke_backend():
    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode("gke"), run_budget=None)
        assert isinstance(backend, GkeJobBackend)


def test_force_sandbox_gke_threads_run_budget():
    budget = SimpleNamespace(name="fake_budget")
    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode("gke"), run_budget=budget)
        assert isinstance(backend, GkeJobBackend)
        assert backend._run_budget is budget


def test_build_environment_noop_for_gke_via_gcp_member():
    assert SandboxMode("gke").value == "gcp"


@pytest.mark.parametrize("token", ["gcp", "azure", "runpod", "local", "docker"])
def test_existing_tokens_round_trip_identically(token):
    assert SandboxMode(token).value == token


def test_aks_path_still_resolves_to_aks_backend():
    with patch("backend.services.runtime.ensure_azure_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.aks_job_backend import AksJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.azure, run_budget=None)
        assert isinstance(backend, AksJobBackend)


def test_gke_check_script_exists_and_executable():
    import os
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent.parent
    script = repo / "scripts" / "gke_check.sh"
    assert script.is_file(), "scripts/gke_check.sh must exist"
    assert os.access(script, os.X_OK), "scripts/gke_check.sh must be executable"


def test_start_sh_preflights_gke():
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent.parent
    start = (repo / "start.sh").read_text(encoding="utf-8")
    assert "gke_check.sh" in start
    assert "gcp" in start and "gke" in start


def test_default_sandbox_setting_accepts_gke():
    """OPENRESEARCH_DEFAULT_SANDBOX=gke must boot (Settings Literal accepts it)
    so the start.sh gcp/gke preflight branch is reachable and gke is a
    first-class default, not just a CLI/enum alias."""
    from backend.config import Settings
    s = Settings(_env_file=None, default_sandbox="gke")
    assert s.default_sandbox == "gke"
    # And it aliases to the gcp member downstream.
    assert SandboxMode(s.default_sandbox) is SandboxMode.gcp


def test_force_sandbox_setting_accepts_gke():
    from backend.config import Settings
    s = Settings(_env_file=None, force_sandbox="gke")
    assert s.force_sandbox == "gke"
    assert SandboxMode(s.force_sandbox) is SandboxMode.gcp


def test_http_start_run_request_rejects_gke_accepts_gcp():
    """Pin the REAL HTTP contract: `StartRunRequest.sandbox` is typed
    `SandboxMode`, which Pydantic v2 validates by enum member-VALUE set — so the
    `gke` alias (resolved only at `SandboxMode._missing_`, i.e. on direct enum
    construction) does NOT apply over the wire and `gke` is rejected with a 422.
    The UI emits `gcp`. This guards against anyone "fixing" the 422 by accident.
    """
    import pydantic

    from backend.services.events.live_runs import StartRunRequest

    with pytest.raises(pydantic.ValidationError):
        StartRunRequest(sandbox="gke")

    # `gcp` is a real enum member value → accepted over the wire.
    assert StartRunRequest(sandbox="gcp").sandbox == SandboxMode.gcp
