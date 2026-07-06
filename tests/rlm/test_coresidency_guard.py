"""T6: the anthropic-foundry / claude-oauth co-residency guard, plus
integration coverage that the Foundry patch/executor-runtime/sub-role-backend
wiring in run.py actually connects (a Wave-1 review found the consumption
layer for opus-foundry/sonnet-foundry was missing).
"""
from types import SimpleNamespace

import pytest

import backend.agents.rlm.run as run_mod
from backend.agents.rlm.role_models import RoleSpec
from backend.agents.rlm.run import assert_no_foundry_oauth_coresidency


class _Spec:
    def __init__(self, provider): self.provider = provider


class _Sel:
    def __init__(self, **roles): self.__dict__.update(roles)
    def specs(self): return [v for v in self.__dict__.values() if v is not None]


# ---------------------------------------------------------------------------
# assert_no_foundry_oauth_coresidency — plan Task 6 Step 1 (verbatim)
# ---------------------------------------------------------------------------

def test_guard_raises_on_mixed_foundry_and_oauth():
    sel = _Sel(executor=_Spec("anthropic-foundry"), grader=_Spec("anthropic-oauth"))
    with pytest.raises(ValueError, match="co-resident"):
        assert_no_foundry_oauth_coresidency("opus-foundry", sel)


def test_guard_allows_all_foundry():
    sel = _Sel(executor=_Spec("anthropic-foundry"), grader=_Spec("anthropic-foundry"))
    assert assert_no_foundry_oauth_coresidency("opus-foundry", sel) is None


def test_guard_allows_all_oauth_no_foundry():
    sel = _Sel(executor=_Spec("anthropic-oauth"))
    assert assert_no_foundry_oauth_coresidency("claude-oauth", sel) is None


# ---------------------------------------------------------------------------
# The guard must also work against the REAL RoleSelection shape (which exposes
# explicit_subroles, not .specs()) — else it silently degrades to root-key-only
# detection in production and never catches a mixed sub-role pick.
# ---------------------------------------------------------------------------

def test_guard_raises_against_real_role_selection_explicit_subroles():
    executor = RoleSpec(role="executor", token="sonnet-foundry",
                         provider="anthropic-foundry", model="claude-sonnet-5", family="claude")
    grader = RoleSpec(role="grader", token="claude-oauth",
                       provider="anthropic-oauth", model=None, family="claude")
    real_sel = SimpleNamespace(
        executor=executor, verifier=None, grader=grader, validator=None,
    )
    # Mirror RoleSelection.explicit_subroles: the non-None sub-role picks.
    real_sel.explicit_subroles = {"executor": executor, "grader": grader}
    with pytest.raises(ValueError, match="co-resident"):
        assert_no_foundry_oauth_coresidency("gpt-5", real_sel)


def test_guard_allows_real_role_selection_all_foundry():
    executor = RoleSpec(role="executor", token="sonnet-foundry",
                         provider="anthropic-foundry", model="claude-sonnet-5", family="claude")
    real_sel = SimpleNamespace(executor=executor, verifier=None, grader=None, validator=None)
    real_sel.explicit_subroles = {"executor": executor}
    assert assert_no_foundry_oauth_coresidency("gpt-5", real_sel) is None


# ---------------------------------------------------------------------------
# Integration: importing backend.agents.rlm.run applies the foundry patch to
# rlm.clients.get_client (C1) — else RLM(backend="anthropic-foundry", ...) at
# root-model-construction time crashes "Unknown backend: anthropic-foundry".
# ---------------------------------------------------------------------------

def test_importing_run_applies_foundry_patch(monkeypatch):
    from rlm import clients

    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-test")

    # run_mod is already imported (module-level, above) — its import-time
    # apply_anthropic_foundry_backend_patch() call is what we're asserting on.
    assert run_mod is not None
    client = clients.get_client("anthropic-foundry", {"model_name": "claude-opus-4-8"})
    base_url = str(client.client.base_url)
    assert "r.services.ai.azure.com" in base_url
    assert "/anthropic/v1" in base_url


# ---------------------------------------------------------------------------
# Integration: _resolve_agent_runtime returns an anthropic runtime carrying
# the foundry subprocess_env for an anthropic-foundry executor spec (C2).
# ---------------------------------------------------------------------------

def test_resolve_agent_runtime_foundry_executor_carries_subprocess_env(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-test")

    exec_spec = RoleSpec(role="executor", token="sonnet-foundry",
                          provider="anthropic-foundry", model="claude-sonnet-5", family="claude")
    role_selection = SimpleNamespace(executor=exec_spec)

    runtime, agent_model, label = run_mod._resolve_agent_runtime(None, None, role_selection)

    assert runtime.provider_name == "anthropic"
    assert runtime.subprocess_env == {
        "ANTHROPIC_BASE_URL": "https://r.services.ai.azure.com/anthropic/v1",
        "ANTHROPIC_API_KEY": "k-test",
    }
    assert runtime.agent_model == "claude-sonnet-5"
    assert agent_model == "claude-sonnet-5"
    assert "role:executor" in label
    # Never leak the credential into process-global os.environ.
    import os
    assert os.environ.get("ANTHROPIC_BASE_URL") != "https://r.services.ai.azure.com/anthropic/v1"


def test_resolve_agent_runtime_foundry_executor_missing_creds_raises(monkeypatch):
    from backend.agents.runtime import foundry_anthropic as fa
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_ANTHROPIC_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setattr(fa, "_env_or_settings", lambda *_a, **_kw: "")

    exec_spec = RoleSpec(role="executor", token="sonnet-foundry",
                          provider="anthropic-foundry", model="claude-sonnet-5", family="claude")
    role_selection = SimpleNamespace(executor=exec_spec)

    from backend.agents.runtime.base import ProviderConfigurationError
    with pytest.raises(ProviderConfigurationError):
        run_mod._resolve_agent_runtime(None, None, role_selection)


# ---------------------------------------------------------------------------
# Integration: _subrole_backend returns "anthropic-foundry" for a foundry
# sub-role, never remapping it to oauth/anthropic (C3).
# ---------------------------------------------------------------------------

def test_subrole_backend_returns_anthropic_foundry_for_foundry_spec():
    spec = RoleSpec(role="verifier", token="sonnet-foundry", provider="anthropic-foundry",
                     model="claude-sonnet-5", family="claude")
    assert run_mod._subrole_backend(spec) == "anthropic-foundry"


def test_subrole_backend_still_resolves_plain_claude_oauth(monkeypatch):
    """Regression: a plain Claude sub-role pick must still go through
    resolve_anthropic_subrole_backend() (oauth vs API key), unaffected by the
    new foundry special-case."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec = RoleSpec(role="verifier", token="sonnet", provider="anthropic-oauth",
                     model=None, family="claude")
    assert run_mod._subrole_backend(spec) in ("oauth", "anthropic")
