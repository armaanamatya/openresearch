"""Per-role model resolution — contract tests for the FROZEN public surface.

The module under test (``backend.agents.rlm.role_models``) is a pure resolver:
no ``os.environ`` reads, no client construction. Every input is passed in, so
these tests need no network and no env mutation. They pin the load-bearing
contracts: back-compat inherit, the de-collapsed opus/sonnet split, mixed
providers across roles, cli>json>legacy precedence, the legacy feeders,
advisory fidelity warnings, error shape, and stamp formatting.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.role_models import (
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_OAUTH,
    PROVIDER_AZURE,
    PROVIDER_OPENAI,
    SUBROLE_PROVIDERS,
    RoleModelError,
    RoleSelection,
    RoleSpec,
    parse_model_spec,
    resolve_role_models,
    supported_tokens,
)


# ---------------------------------------------------------------------------
# 1. Back-compat inherit — bare planner token, every sub-role None.
# ---------------------------------------------------------------------------
def test_back_compat_inherit_planner_only():
    sel = resolve_role_models(planner_token="claude-oauth")
    assert isinstance(sel, RoleSelection)
    assert sel.planner.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.planner.model == "claude-sonnet-4-6"
    assert sel.executor is None
    assert sel.verifier is None
    assert sel.grader is None
    assert sel.explicit_subroles == {}


def test_back_compat_stamp_shape():
    sel = resolve_role_models(planner_token="claude-oauth")
    assert sel.stamp() == {
        "planner": "anthropic-oauth:claude-sonnet-4-6",
        "executor": None,
        "verifier": None,
        "grader": None,
    }


# ---------------------------------------------------------------------------
# 2. De-collapsed opus vs sonnet — the module must NOT collapse them.
# ---------------------------------------------------------------------------
def test_opus_vs_sonnet_de_collapsed_as_planner():
    opus = resolve_role_models(planner_token="opus").planner
    sonnet = resolve_role_models(planner_token="sonnet").planner
    assert opus.model == "claude-opus-4-7"
    assert sonnet.model == "claude-sonnet-4-6"
    assert opus.model != sonnet.model
    # Both are the validated Claude provider.
    assert opus.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sonnet.provider == PROVIDER_ANTHROPIC_OAUTH


def test_opus_vs_sonnet_de_collapsed_as_subrole():
    opus = parse_model_spec("opus", role="executor")
    sonnet = parse_model_spec("sonnet", role="executor")
    assert opus.model == "claude-opus-4-7"
    assert sonnet.model == "claude-sonnet-4-6"
    assert opus.model != sonnet.model


# ---------------------------------------------------------------------------
# 3. Mixed providers via cli_models across all three sub-roles.
# ---------------------------------------------------------------------------
def test_mixed_providers_via_cli_models():
    sel = resolve_role_models(
        planner_token="opus",
        cli_models="executor=gpt-4o-azure,verifier=sonnet,grader=o4-mini",
    )
    assert sel.executor is not None
    assert (sel.executor.provider, sel.executor.model) == (PROVIDER_AZURE, "gpt-4o")
    assert sel.verifier is not None
    assert (sel.verifier.provider, sel.verifier.model) == (
        PROVIDER_ANTHROPIC_OAUTH,
        "claude-sonnet-4-6",
    )
    assert sel.grader is not None
    assert (sel.grader.provider, sel.grader.model) == (PROVIDER_OPENAI, "o4-mini")


# ---------------------------------------------------------------------------
# 4. Precedence — cli_models overrides role_models_json key-by-key.
# ---------------------------------------------------------------------------
def test_cli_overrides_json_key_by_key():
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="executor=opus",
        role_models_json='{"executor":"sonnet","grader":"o4-mini"}',
    )
    # CLI wins for executor.
    assert sel.executor is not None
    assert sel.executor.model == "claude-opus-4-7"
    # JSON-only key survives (not overridden by CLI).
    assert sel.grader is not None
    assert sel.grader.model == "o4-mini"


def test_planner_is_single_sourced_from_planner_token():
    # The planner is authoritative from planner_token (the resolved root key); a
    # `planner` key in the unified map is NOT re-applied here — run.py folds it
    # into resolve_root_model first, so planner_token already reflects it and the
    # stamp matches the root collapse (opus→claude-oauth→Sonnet at the root).
    sel = resolve_role_models(
        planner_token="claude-oauth",
        role_models_json='{"planner":"opus"}',
    )
    assert sel.planner.model == "claude-sonnet-4-6"  # planner_token wins, not the map


def test_planner_token_from_surface_extracts_raw_token():
    from backend.agents.rlm.role_models import planner_token_from_surface
    assert planner_token_from_surface('{"planner":"opus","executor":"sonnet"}') == "opus"
    assert planner_token_from_surface("planner=gpt-5,grader=o4-mini") == "gpt-5"
    assert planner_token_from_surface('{"executor":"sonnet"}') is None  # no planner key
    assert planner_token_from_surface(None) is None
    assert planner_token_from_surface("not json {{{") is None  # malformed → None, no raise


# ---------------------------------------------------------------------------
# 5. JSON and k=v forms both parse; malformed forms raise; unknown roles ignored.
# ---------------------------------------------------------------------------
def test_kv_and_json_forms_equivalent():
    via_kv = resolve_role_models(
        planner_token="opus", cli_models="executor=sonnet"
    )
    via_json = resolve_role_models(
        planner_token="opus", role_models_json='{"executor":"sonnet"}'
    )
    assert via_kv.executor is not None and via_json.executor is not None
    assert via_kv.executor.model == via_json.executor.model == "claude-sonnet-4-6"


def test_malformed_kv_entry_raises():
    with pytest.raises(RoleModelError):
        resolve_role_models(planner_token="opus", cli_models="executor")


def test_invalid_json_raises():
    with pytest.raises(RoleModelError):
        resolve_role_models(planner_token="opus", role_models_json="{not json}")


def test_non_object_json_raises():
    with pytest.raises(RoleModelError):
        resolve_role_models(planner_token="opus", role_models_json='["opus"]')


def test_unknown_role_keys_ignored_forward_compat():
    # An unrecognised role key (e.g. a future "navigator") is silently dropped.
    sel = resolve_role_models(
        planner_token="opus",
        cli_models="navigator=gpt-4o,executor=sonnet",
    )
    assert sel.executor is not None
    assert sel.executor.model == "claude-sonnet-4-6"
    # No surprise attribute leaked onto the selection.
    assert sel.get("navigator") is None


# ---------------------------------------------------------------------------
# 6. Legacy feeders.
# ---------------------------------------------------------------------------
def test_executor_only_from_unified_surface():
    # Executor has NO legacy feeder by design: the legacy OPENRESEARCH_EXECUTOR
    # flag keeps its own graceful-fallback path (executor.resolve_executor), so
    # it is not a resolver input. Executor is set only via the unified surface.
    assert resolve_role_models(planner_token="opus").executor is None
    sel = resolve_role_models(planner_token="opus", cli_models="executor=gpt-4o-azure")
    assert sel.executor is not None
    assert sel.executor.provider == PROVIDER_AZURE


def test_executor_env_is_not_a_parameter():
    # Guard the deliberate removal: re-adding a silent executor_env feeder would
    # re-route OPENRESEARCH_EXECUTOR through the fail-fast make_runtime branch and
    # change its missing-creds behaviour. Passing it must error, not no-op.
    with pytest.raises(TypeError):
        resolve_role_models(planner_token="opus", executor_env="azure")


def test_grader_backend_openai_with_model():
    sel = resolve_role_models(
        planner_token="opus",
        grader_backend_env="openai",
        grader_model_env="o4-mini",
    )
    assert sel.grader is not None
    assert (sel.grader.provider, sel.grader.model) == (PROVIDER_OPENAI, "o4-mini")


def test_grader_backend_azure_maps_to_azure():
    sel = resolve_role_models(planner_token="opus", grader_backend_env="azure")
    assert sel.grader is not None
    assert sel.grader.provider == PROVIDER_AZURE


@pytest.mark.parametrize("gb", ["oauth", "anthropic", ""])
def test_grader_backend_inherit_stays_none(gb):
    sel = resolve_role_models(planner_token="opus", grader_backend_env=gb)
    assert sel.grader is None


def test_verifier_model_setting_sets_verifier():
    sel = resolve_role_models(
        planner_token="opus", verifier_model_setting="sonnet"
    )
    assert sel.verifier is not None
    assert sel.verifier.model == "claude-sonnet-4-6"


@pytest.mark.parametrize("vm", ["", None])
def test_verifier_model_setting_blank_stays_none(vm):
    sel = resolve_role_models(planner_token="opus", verifier_model_setting=vm)
    assert sel.verifier is None


def test_unified_cli_beats_env_json():
    # cli_models wins over role_models_json key-by-key.
    sel = resolve_role_models(
        planner_token="opus",
        role_models_json='{"executor":"sonnet"}',
        cli_models="executor=opus",
    )
    assert sel.executor is not None
    assert sel.executor.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.executor.model == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# 7. Fidelity warnings — advisory, one per non-Claude EXPLICIT sub-role.
# ---------------------------------------------------------------------------
def test_fidelity_warnings_empty_when_not_critical():
    sel = resolve_role_models(
        planner_token="gpt-5",
        cli_models="executor=gpt-4o-azure,grader=o4-mini",
    )
    assert sel.fidelity_warnings(fidelity_critical=False) == []


def test_fidelity_warnings_one_per_non_claude_subrole():
    sel = resolve_role_models(
        planner_token="gpt-5",  # planner is never warned even if non-Claude
        cli_models="executor=gpt-4o-azure,verifier=sonnet,grader=o4-mini",
    )
    warnings = sel.fidelity_warnings(fidelity_critical=True)
    # executor (azure) + grader (openai) warn; verifier (sonnet=Claude) does not;
    # planner (gpt-5) never warns.
    assert len(warnings) == 2
    joined = " ".join(warnings)
    assert "executor" in joined
    assert "grader" in joined
    assert "verifier" not in joined
    assert "planner" not in joined


def test_fidelity_warnings_none_when_all_claude_subroles():
    sel = resolve_role_models(
        planner_token="gpt-5",
        cli_models="executor=opus,verifier=sonnet,grader=haiku",
    )
    assert sel.fidelity_warnings(fidelity_critical=True) == []


def test_fidelity_warnings_empty_when_no_explicit_subroles():
    sel = resolve_role_models(planner_token="gpt-5")
    assert sel.fidelity_warnings(fidelity_critical=True) == []


# ---------------------------------------------------------------------------
# 8. Errors — unknown token, root-only-on-subrole, unsupported tokens.
# ---------------------------------------------------------------------------
def test_unknown_token_raises_with_supported_hint():
    # Sub-roles stay strict; the planner is lenient by design (resolve_root_model
    # owns root validation), so the helpful "Supported: …" hint is asserted on a
    # sub-role where an unknown token must still be rejected.
    with pytest.raises(RoleModelError, match="Supported"):
        parse_model_spec("llama-70b", role="executor")


def test_qwen_not_supported_for_subroles():
    # qwen is a root-only model: it parses for the planner as a passthrough
    # (resolve_root_model owns root validation) but is rejected for sub-roles.
    for role in ("executor", "verifier", "grader"):
        with pytest.raises(RoleModelError):
            parse_model_spec("qwen", role=role)
    assert "qwen" not in supported_tokens()


def test_root_only_provider_parses_for_planner_but_rejected_for_subrole():
    # ``claude`` -> provider anthropic (a valid sub-role provider), so to exercise
    # the lenient-planner-vs-strict-subrole branch we need a provider in the
    # vocab that is NOT in SUBROLE_PROVIDERS. The current vocab has none, so we
    # assert the contract directly: every parseable token resolves for the
    # planner, and any token whose provider is not a sub-role provider would
    # raise for a sub-role.
    for tok in supported_tokens():
        spec = parse_model_spec(tok, role="planner")
        assert isinstance(spec, RoleSpec)
        if spec.provider not in SUBROLE_PROVIDERS:
            with pytest.raises(RoleModelError):
                parse_model_spec(tok, role="executor")


def test_empty_token_raises():
    with pytest.raises(RoleModelError):
        parse_model_spec("", role="planner")
    with pytest.raises(RoleModelError):
        parse_model_spec("   ", role="executor")


def test_resolve_root_only_planner_token_passes_through():
    # The planner token is the already-validated root key (resolve_root_model
    # owns validation); resolve_role_models stamps it as a passthrough rather
    # than re-rejecting root-only keys (the qwen/kimi/foundry regression fix).
    sel = resolve_role_models(planner_token="llama-70b")
    assert sel.planner.stamp == "root:llama-70b"


# ---------------------------------------------------------------------------
# 9. RoleSpec.stamp + is_claude formatting.
# ---------------------------------------------------------------------------
def test_stamp_provider_model_format():
    spec = parse_model_spec("o4-mini", role="grader")
    assert spec.stamp == "openai:o4-mini"


def test_stamp_azure_bare_deployment_placeholder():
    spec = parse_model_spec("azure", role="executor")
    assert spec.model is None
    assert spec.stamp == "azure:<deployment>"


def test_stamp_azure_named_deployment():
    spec = parse_model_spec("gpt-4o-azure", role="executor")
    assert spec.stamp == "azure:gpt-4o"


@pytest.mark.parametrize(
    "token,expected_is_claude",
    [
        ("opus", True),
        ("sonnet", True),
        ("haiku", True),
        ("claude", True),  # anthropic (API key) is still a validated Claude
        ("gpt-4o", False),
        ("o4-mini", False),
        ("azure", False),
        ("gpt-4o-azure", False),
    ],
)
def test_is_claude_classification(token, expected_is_claude):
    spec = parse_model_spec(token, role="planner")
    assert spec.is_claude is expected_is_claude


# ---------------------------------------------------------------------------
# Misc surface guards (frozen dataclasses, token lower-casing, supported set).
# ---------------------------------------------------------------------------
def test_rolespec_is_frozen():
    spec = parse_model_spec("opus", role="planner")
    with pytest.raises(Exception):
        spec.model = "mutated"  # type: ignore[misc]


def test_roleselection_is_frozen():
    sel = resolve_role_models(planner_token="opus")
    with pytest.raises(Exception):
        sel.executor = None  # type: ignore[misc]


def test_token_is_lowercased_and_trimmed():
    spec = parse_model_spec("  OPUS  ", role="planner")
    assert spec.token == "opus"
    assert spec.model == "claude-opus-4-7"


def test_get_planner_returns_spec():
    sel = resolve_role_models(planner_token="opus")
    assert sel.get("planner") is sel.planner


def test_supported_tokens_sorted_and_includes_known():
    toks = supported_tokens()
    assert toks == sorted(toks)
    for known in ("opus", "sonnet", "gpt-4o", "o4-mini", "azure", "gpt-4o-azure"):
        assert known in toks


def test_provider_constants_are_expected_strings():
    assert PROVIDER_ANTHROPIC_OAUTH == "anthropic-oauth"
    assert PROVIDER_ANTHROPIC == "anthropic"
    assert PROVIDER_OPENAI == "openai"
    assert PROVIDER_AZURE == "azure"


# ---------------------------------------------------------------------------
# Planner leniency — root-only keys parse for stamping (qwen/kimi/foundry)
# ---------------------------------------------------------------------------


class TestPlannerLeniencyForRootOnlyKeys:
    """A planner token that is a root-only registry key must NOT raise.

    resolve_root_model validates the root upstream; resolve_role_models is then
    called with the resolved key. Root-only keys (qwen3-coder, kimi-k2.5,
    qwen3-coder-featherless, azure-foundry) are absent from the sub-role vocab,
    so without leniency they crashed the whole run (regression repro).
    """

    @pytest.mark.parametrize(
        "root_key",
        # Root-only keys absent from the sub-role vocab. (azure-foundry IS now a
        # vocab token — a real sub-role provider — so it stamps via the vocab,
        # not the passthrough; covered by the foundry sub-role tests instead.)
        ["qwen3-coder", "kimi-k2.5", "qwen3-coder-featherless"],
    )
    def test_planner_passthrough_does_not_raise(self, root_key):
        from backend.agents.rlm.role_models import (
            PROVIDER_ROOT,
            parse_model_spec,
            resolve_role_models,
        )

        spec = parse_model_spec(root_key, role="planner")
        assert spec.provider == PROVIDER_ROOT
        assert spec.model == root_key
        assert spec.stamp == f"root:{root_key}"

        sel = resolve_role_models(planner_token=root_key)
        assert sel.planner.stamp == f"root:{root_key}"

    def test_known_planner_token_still_maps_via_vocab(self):
        from backend.agents.rlm.role_models import (
            PROVIDER_ANTHROPIC_OAUTH,
            parse_model_spec,
        )

        spec = parse_model_spec("opus", role="planner")
        assert spec.provider == PROVIDER_ANTHROPIC_OAUTH
        assert spec.model == "claude-opus-4-7"

    @pytest.mark.parametrize("role", ["executor", "verifier", "grader"])
    def test_subroles_stay_strict_on_unknown_token(self, role):
        from backend.agents.rlm.role_models import RoleModelError, parse_model_spec

        with pytest.raises(RoleModelError):
            parse_model_spec("qwen3-coder", role=role)


# ---------------------------------------------------------------------------
# Azure AI Foundry (grok) as a real sub-role provider — executor/grader/verifier
# can all run on it (model None ⇒ AZURE_FOUNDRY_DEPLOYMENT at build time), so a
# fully OAuth-free run is possible and any role is interchangeable grok⇄…
# ---------------------------------------------------------------------------


class TestAzureFoundrySubrole:
    @pytest.mark.parametrize("role", ["executor", "verifier", "grader"])
    @pytest.mark.parametrize("token", ["grok", "azure-foundry", "foundry", "grok-4.3"])
    def test_grok_token_resolves_to_foundry_provider_model_none(self, role, token):
        from backend.agents.rlm.role_models import (
            PROVIDER_AZURE_FOUNDRY,
            parse_model_spec,
        )

        spec = parse_model_spec(token, role=role)
        assert spec.provider == PROVIDER_AZURE_FOUNDRY
        assert spec.model is None  # None ⇒ use AZURE_FOUNDRY_DEPLOYMENT at build
        assert spec.stamp == "azure-foundry:<deployment>"

    def test_resolve_role_models_executor_and_grader_grok(self):
        from backend.agents.rlm.role_models import (
            PROVIDER_AZURE_FOUNDRY,
            resolve_role_models,
        )

        sel = resolve_role_models(
            planner_token="claude-oauth",
            cli_models="executor=grok,grader=grok",
        )
        assert sel.executor is not None
        assert sel.executor.provider == PROVIDER_AZURE_FOUNDRY
        assert sel.executor.model is None
        assert sel.grader is not None
        assert sel.grader.provider == PROVIDER_AZURE_FOUNDRY
        assert sel.grader.model is None
        # foundry is a non-Claude sub-role → not the validated baseline.
        assert sel.executor.is_claude is False
        assert sel.grader.is_claude is False
