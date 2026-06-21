"""Validator role + RoleSpec.family + three-tier separation_strength.

Contract tests for the external-adversarial-validator surface added in Task
P2.1 (spec 2026-06-20 §7.2/§7.4). Like ``test_role_models.py`` these exercise a
pure resolver — no network, no env mutation; every input is passed in. They pin:

  * the model-lineage family classifier (claude == oauth, azure == gpt, the
    Foundry grok/kimi disambiguation, qwen, and the un-classifiable None);
  * the de-collapsed opus≠sonnet split still maps BOTH to family "claude";
  * the three-tier separation ladder (independent / weak / degraded / unavailable)
    incl. BOTH funded panels (cross-family independent + same-provider weak);
  * the validator sub-role resolves from the unified surface + the legacy
    ``validator_model_setting`` feeder, and ``RoleSelection.stamp()`` gains it;
  * grader/verifier behaviour is unchanged by the new role (no regression).
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.role_models import (
    ROLES,
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_OAUTH,
    PROVIDER_AZURE,
    PROVIDER_AZURE_FOUNDRY,
    PROVIDER_OPENAI,
    PROVIDER_ROOT,
    RoleSpec,
    _classify_model_family,
    parse_model_spec,
    resolve_role_models,
    separation_strength,
)


# ---------------------------------------------------------------------------
# 1. ROLES / _SUBROLES include validator.
# ---------------------------------------------------------------------------
def test_validator_is_a_known_role():
    assert "validator" in ROLES


def test_validator_is_a_subrole():
    # Strict-parse path: an unknown token for the validator sub-role must raise
    # (same contract as executor/verifier/grader), proving it is a sub-role.
    with pytest.raises(Exception):
        parse_model_spec("not-a-real-token", role="validator")


# ---------------------------------------------------------------------------
# 2. _classify_model_family — lineage, NOT transport. Cover the whole vocab.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "provider,model,expected",
    [
        # Claude family — oauth and api are the SAME lineage.
        (PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6", "claude"),
        (PROVIDER_ANTHROPIC_OAUTH, "claude-opus-4-7", "claude"),
        (PROVIDER_ANTHROPIC_OAUTH, "claude-haiku-4-5-20251001", "claude"),
        (PROVIDER_ANTHROPIC, "claude-opus-4-7", "claude"),
        # GPT family — OpenAI-direct AND classic Azure OpenAI.
        (PROVIDER_OPENAI, "gpt-5", "gpt"),
        (PROVIDER_OPENAI, "gpt-4o", "gpt"),
        (PROVIDER_OPENAI, "gpt-4o-mini", "gpt"),
        (PROVIDER_OPENAI, "o4-mini", "gpt"),
        (PROVIDER_AZURE, "gpt-4o", "gpt"),
        (PROVIDER_AZURE, None, "gpt"),  # azure default deployment is still gpt.
        # Azure Foundry — disambiguated by served model name.
        (PROVIDER_AZURE_FOUNDRY, "grok-4.3", "grok"),
        (PROVIDER_AZURE_FOUNDRY, "Kimi-K2.6", "kimi"),
        (PROVIDER_AZURE_FOUNDRY, None, "foundry"),  # un-named foundry model.
        (PROVIDER_AZURE_FOUNDRY, "some-other-deployment", "foundry"),
    ],
)
def test_classify_model_family_vocab(provider, model, expected):
    assert _classify_model_family(provider, model) == expected


def test_classify_family_key_invariant_azure_gpt_ne_claude():
    # The load-bearing invariant the separation ladder relies on.
    assert _classify_model_family(PROVIDER_AZURE, "gpt-4o") == "gpt"
    assert _classify_model_family(PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6") == "claude"
    assert _classify_model_family(PROVIDER_AZURE, "gpt-4o") != _classify_model_family(
        PROVIDER_ANTHROPIC_OAUTH, "claude-sonnet-4-6"
    )


@pytest.mark.parametrize(
    "root_token,expected",
    [
        # PROVIDER_ROOT passthrough (planner-stamped root keys) classifies by token.
        ("qwen3-coder", "qwen"),
        ("qwen3-coder-featherless", "qwen"),
        ("kimi-k2.5", "kimi"),
        ("grok-4.3-root", "grok"),
        ("gpt-5", "gpt"),
        ("claude-sonnet", "claude"),
        ("llama-70b", None),  # un-classifiable → None, never crashes.
    ],
)
def test_classify_root_passthrough_token(root_token, expected):
    assert _classify_model_family(PROVIDER_ROOT, root_token) == expected


# ---------------------------------------------------------------------------
# 3. RoleSpec.family is populated by parse_model_spec.
# ---------------------------------------------------------------------------
def test_role_spec_family_set_by_parse():
    assert parse_model_spec("sonnet", role="validator").family == "claude"
    assert parse_model_spec("gpt-4o-azure", role="validator").family == "gpt"
    # The bare "grok" token resolves to (azure-foundry, model=None) — the served
    # model lives in AZURE_FOUNDRY_DEPLOYMENT, unknown at descriptor parse time —
    # so the token-level family is "foundry". (A Foundry spec WITH a concrete
    # grok* model classifies as "grok"; covered by the vocab parametrize above.)
    assert parse_model_spec("grok", role="validator").family == "foundry"


def test_role_spec_family_defaults_none_when_hand_built():
    # A bare hand-built RoleSpec (older callers / fixtures) still constructs.
    spec = RoleSpec(role="executor", token="x", provider="azure", model="gpt-4o")
    assert spec.family is None


def test_de_collapse_opus_sonnet_both_claude_family():
    opus = parse_model_spec("opus", role="validator")
    sonnet = parse_model_spec("sonnet", role="validator")
    # Distinct models (de-collapse holds) ...
    assert opus.model != sonnet.model
    # ... but the SAME lineage family (separation keys on "claude", not the model).
    assert opus.family == sonnet.family == "claude"


# ---------------------------------------------------------------------------
# 4. separation_strength — the three-tier ladder + the two funded panels.
# ---------------------------------------------------------------------------
def test_validator_independent_oauth_exec_azure_val():
    # tier: independent (cross-family — the strongest panel).
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="executor=sonnet,validator=gpt-4o-azure",
    )
    assert sel.executor.family == "claude" and sel.validator.family == "gpt"
    assert separation_strength(sel.executor, sel.validator) == "independent"


def test_validator_weak_azure_exec_azure_val_diff_model():
    # tier: weak (the operator's same-provider-different-deployment ask).
    sel = resolve_role_models(
        planner_token="azure-gpt-4o",
        cli_models="executor=gpt-4o-azure,validator=azure",
    )
    # Same family, distinct model/deployment (gpt-4o vs the bare azure default).
    assert sel.executor.family == sel.validator.family == "gpt"
    assert sel.executor.model != sel.validator.model
    assert separation_strength(sel.executor, sel.validator) == "weak"  # SUPPORTED.


def test_validator_weak_explicit_distinct_deployments():
    # Even when both sides are concrete azure deployments, distinct model strings
    # (the OPENRESEARCH_VALIDATOR_MODEL=<deploymentB> case) read as "weak".
    exec_a = RoleSpec(
        role="executor", token="azure", provider=PROVIDER_AZURE, model="gpt-4o", family="gpt"
    )
    val_b = RoleSpec(
        role="validator", token="azure", provider=PROVIDER_AZURE, model="gpt-4.1", family="gpt"
    )
    assert separation_strength(exec_a, val_b) == "weak"


def test_validator_degraded_same_model():
    # tier: degraded — same family AND same model (seed-only).
    spec_a = RoleSpec(
        role="executor", token="azure", provider=PROVIDER_AZURE, model="gpt-4o", family="gpt"
    )
    assert separation_strength(spec_a, spec_a) == "degraded"


def test_validator_unavailable_when_none():
    spec_a = RoleSpec(
        role="executor", token="azure", provider=PROVIDER_AZURE, model="gpt-4o", family="gpt"
    )
    assert separation_strength(spec_a, None) == "unavailable"


def test_validator_independent_when_no_executor_peer():
    # A selected validator with no executor pick to compare against (legacy
    # executor path) is treated as independent by construction.
    val = parse_model_spec("gpt-4o-azure", role="validator")
    assert separation_strength(None, val) == "independent"


# ---------------------------------------------------------------------------
# 5. validator sub-role resolution + stamp + legacy feeder.
# ---------------------------------------------------------------------------
def test_validator_inherits_none_by_default():
    sel = resolve_role_models(planner_token="claude-oauth")
    assert sel.validator is None
    assert sel.stamp()["validator"] is None


def test_validator_from_unified_surface():
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="validator=gpt-4o-azure"
    )
    assert sel.validator is not None
    assert sel.validator.provider == PROVIDER_AZURE
    assert sel.validator.family == "gpt"
    assert sel.validator in sel.explicit_subroles.values()


def test_validator_legacy_model_setting_feeder():
    # OPENRESEARCH_VALIDATOR_MODEL (threaded as validator_model_setting) sets it
    # when the unified surface does not.
    sel = resolve_role_models(
        planner_token="claude-oauth", validator_model_setting="sonnet"
    )
    assert sel.validator is not None
    assert sel.validator.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.validator.family == "claude"


@pytest.mark.parametrize("vm", ["", "   ", None])
def test_validator_model_setting_blank_stays_none(vm):
    sel = resolve_role_models(planner_token="claude-oauth", validator_model_setting=vm)
    assert sel.validator is None


def test_validator_unified_beats_legacy_setting():
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="validator=gpt-4o-azure",
        validator_model_setting="sonnet",
    )
    # CLI surface wins over the legacy feeder.
    assert sel.validator.provider == PROVIDER_AZURE


def test_validator_stamp_present_in_full_shape():
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="validator=gpt-4o-azure"
    )
    stamp = sel.stamp()
    assert "validator" in stamp
    assert stamp["validator"] == "azure:gpt-4o"


# ---------------------------------------------------------------------------
# 5b. §4.7 stamp fix — the validator stamp prefers OPENRESEARCH_VALIDATOR_MODEL.
# ---------------------------------------------------------------------------
def test_validator_stamp_prefers_validator_model_env(monkeypatch):
    # A bridged azure-foundry validator (model None) would otherwise stamp the
    # global AZURE_FOUNDRY_DEPLOYMENT (the executor's model). With
    # OPENRESEARCH_VALIDATOR_MODEL set, the stamp must name the validator model.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "grok-4.3-val")
    spec = parse_model_spec("grok", role="validator")
    assert spec.provider == PROVIDER_AZURE_FOUNDRY
    assert spec.stamp == "azure-foundry:grok-4.3-val"


def test_validator_model_env_does_not_leak_to_other_roles(monkeypatch):
    # The env preference is keyed on role == "validator" only — a grader/verifier
    # foundry RoleSpec must NOT pick up OPENRESEARCH_VALIDATOR_MODEL.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "grok-4.3-val")
    grader = parse_model_spec("grok", role="grader")
    assert grader.role == "grader"
    assert grader.stamp != "azure-foundry:grok-4.3-val"


def test_validator_stamp_ignores_blank_validator_model_env(monkeypatch):
    # A blank/whitespace OPENRESEARCH_VALIDATOR_MODEL falls through to the normal
    # deployment-resolution path (no spurious "azure-foundry:" stamp).
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "   ")
    spec = parse_model_spec("gpt-4o-azure", role="validator")
    # Concrete model present → normal stamp, env ignored.
    assert spec.stamp == "azure:gpt-4o"


def test_roleselection_validator_stamp_inherits_env_preference(monkeypatch):
    # The aggregator delegates to RoleSpec.stamp, so the §4.7 preference flows
    # through RoleSelection.stamp() too.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-valB")
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="validator=gpt-4o-azure"
    )
    assert sel.stamp()["validator"] == "azure:gpt-4o-valB"


# ---------------------------------------------------------------------------
# 6. No regression — adding validator must not change grader/verifier/executor.
# ---------------------------------------------------------------------------
def test_grader_verifier_unchanged_when_no_validator():
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="verifier=sonnet,grader=gpt-4o-azure",
    )
    assert sel.verifier is not None and sel.verifier.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.grader is not None and sel.grader.provider == PROVIDER_AZURE
    assert sel.validator is None


def test_back_compat_stamp_shape_now_carries_validator_key():
    # The legacy 4-key stamp gains a 5th "validator" key (None when unselected);
    # the four existing keys are byte-identical to before.
    sel = resolve_role_models(planner_token="claude-oauth")
    assert sel.stamp() == {
        "planner": "anthropic-oauth:claude-sonnet-4-6",
        "executor": None,
        "verifier": None,
        "grader": None,
        "validator": None,
    }


def test_fidelity_warning_covers_a_non_claude_validator():
    # The validator participates in the advisory fidelity surface like the other
    # sub-roles (explicit_subroles now includes it).
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="validator=gpt-4o-azure"
    )
    warnings = sel.fidelity_warnings(fidelity_critical=True)
    assert any("validator" in w for w in warnings)
    # ... and is silent on a Claude validator.
    sel_claude = resolve_role_models(
        planner_token="claude-oauth", cli_models="validator=sonnet"
    )
    assert sel_claude.fidelity_warnings(fidelity_critical=True) == []
