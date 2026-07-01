"""The OPENRESEARCH_CLOUD_FAILOVER live-path wiring (Phase 1c, Unit A Step 5).

`primitives._resolve_run_backend` is the single seam that routes a cloud run
through the GCP<->Azure failover selector when the flag is set, and is
byte-identical to the direct `_backend_for_sandbox_mode` call when it isn't.
"""
from __future__ import annotations

import backend.agents.rlm.primitives as P


def test_flag_off_is_byte_identical_direct_selection(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CLOUD_FAILOVER", raising=False)
    calls = []
    monkeypatch.setattr(
        P, "_backend_for_sandbox_mode",
        lambda mode, **kw: calls.append(("direct", mode, kw)) or "direct-backend",
    )
    out = P._resolve_run_backend("gcp", run_budget="B", gpu_plan="G")
    assert out == "direct-backend"
    assert calls == [("direct", "gcp", {"run_budget": "B", "gpu_plan": "G"})]


def test_flag_on_cloud_routes_through_failover(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CLOUD_FAILOVER", "gcp,azure")
    import backend.services.runtime.cloud_failover as CF

    seen = {}

    def _fake_select(pref, **kw):
        seen["pref"] = pref
        seen["kw"] = kw
        return type("_Sel", (), {"backend": f"failover:{pref}"})()

    monkeypatch.setattr(CF, "select_backend_with_failover", _fake_select)
    out = P._resolve_run_backend("gcp", run_budget="B", gpu_plan="G")
    assert out == "failover:['gcp', 'azure']"
    assert seen["pref"] == ["gcp", "azure"]
    assert seen["kw"] == {"run_budget": "B", "gpu_plan": "G"}


def test_flag_on_noncloud_mode_skips_failover(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CLOUD_FAILOVER", "gcp,azure")
    calls = []
    monkeypatch.setattr(
        P, "_backend_for_sandbox_mode",
        lambda mode, **kw: calls.append(mode) or "direct",
    )
    # A non-cloud sandbox (local/docker/runpod) must ignore the failover flag.
    out = P._resolve_run_backend("local", run_budget=None, gpu_plan=None)
    assert out == "direct" and calls == ["local"]
